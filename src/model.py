# ==============================================================================
# Module : Entraînement, évaluation et sauvegarde du modèle de prédiction
# ==============================================================================
# Objectif du module :
#     Ce module contient la logique d'entraînement du modèle Random Forest
#     utilisé pour prédire la possession d'un compte bancaire.
#
#     Il effectue les actions suivantes :
#     - Filtre et nettoie les valeurs cibles invalides.
#     - Encode les variables binaires (target et accès téléphonique).
#     - Sépare les features (X) de la variable cible (y), sans l'identifiant.
#     - Évalue le modèle sur un jeu de test mis de côté (rappel, AUC...).
#     - Entraîne le modèle final et le sauvegarde, pour que l'application
#       le charge au lieu de le réentraîner à chaque démarrage.
#
# Exemple d'utilisation :
#     python -m src.model                  # contrôle, évalue, entraîne, sauvegarde
#     from src.model import load_model
# ==============================================================================


import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data_processing import DEFAULT_DATA_PATH
from src.quality_checks import assert_clean_data, check_model_features


# ==============================================================================
# Constantes
# ==============================================================================

TARGET_COLUMN      = "has_a_bank_account"
PHONE_COLUMN       = "cell_phone_access"
ID_COLUMN          = "uniqueid"
VALID_BINARY_VALUES = ["Yes", "No"]

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "random_state": 42
}

TEST_SIZE    = 0.2
SPLIT_SEED   = 42
MODEL_PATH   = Path("models/model.joblib")
METRICS_PATH = Path("models/metrics.json")


# ==============================================================================
# Entraînement du modèle
# ==============================================================================

def train_model(df: pd.DataFrame) -> tuple:
    """
    Prépare les données et entraîne un classifieur Random Forest.

    Étapes :
        1. Filtre les lignes avec des valeurs invalides sur la target et le téléphone.
        2. Encode les colonnes binaires (Yes/No → 1/0).
        3. Sépare X et y (sans l'identifiant), applique le one-hot encoding sur X.
        4. Entraîne le modèle Random Forest.

    Paramètres :
        df : DataFrame nettoyé issu de load_clean_data()

    Retourne :
        Tuple (modèle entraîné, colonnes de référence du DataFrame X)
    """
    X, y = _prepare_features_target(df)

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X, y)

    return model, X.columns


# ==============================================================================
# Évaluation du modèle
# ==============================================================================

def evaluate_model(df: pd.DataFrame) -> dict:
    """
    Évalue le modèle sur un jeu de test stratifié mis de côté.

    La précision globale seule est trompeuse : avec une minorité de comptes,
    répondre toujours « non » a déjà raison le plus souvent. Le rappel et
    l'AUC disent si le modèle retrouve vraiment les personnes avec un compte.

    Paramètres :
        df : DataFrame nettoyé issu de load_clean_data()

    Retourne :
        Dictionnaire des mesures, arrondies à 4 décimales
    """
    X, y = _prepare_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED
    )

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "rows":               int(len(X)),
        "features":           int(X.shape[1]),
        "test_rows":          int(len(y_test)),
        "test_positives":     int(y_test.sum()),
        "accuracy":           round(accuracy_score(y_test, predictions), 4),
        "baseline_accuracy":  round(float(1 - y_test.mean()), 4),
        "recall":             round(recall_score(y_test, predictions), 4),
        "precision":          round(precision_score(y_test, predictions), 4),
        "auc":                round(roc_auc_score(y_test, probabilities), 4),
    }


# ==============================================================================
# Sauvegarde et chargement
# ==============================================================================

def save_model(model: RandomForestClassifier, reference_columns: pd.Index, data_hash: str, path: Path = MODEL_PATH) -> None:
    """
    Sauvegarde le modèle entraîné, ses colonnes de référence et l'empreinte
    du CSV sur lequel il a été entraîné.

    Paramètres :
        model             : Modèle entraîné
        reference_columns : Colonnes utilisées lors de l'entraînement
        data_hash         : Empreinte du CSV d'entraînement (hash_data_file)
        path              : Fichier de destination (par défaut : MODEL_PATH)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "reference_columns": list(reference_columns), "data_hash": data_hash}, path, compress=3)


def load_model(path: Path = MODEL_PATH) -> tuple:
    """
    Charge un modèle sauvegardé par save_model().

    Paramètres :
        path : Fichier du modèle (par défaut : MODEL_PATH)

    Retourne :
        Tuple (modèle entraîné, colonnes de référence, empreinte du CSV d'entraînement)
    """
    saved = joblib.load(path)
    return saved["model"], pd.Index(saved["reference_columns"]), saved.get("data_hash")


def hash_data_file(path: str = DEFAULT_DATA_PATH) -> str:
    """
    Calcule l'empreinte SHA-256 d'un fichier de données : un modèle entraîné
    sur un autre CSV que celui en place est périmé.

    Paramètres :
        path : Chemin du CSV (par défaut : DEFAULT_DATA_PATH)

    Retourne :
        Empreinte hexadécimale du fichier
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ==============================================================================
# Fonctions internes
# ==============================================================================

def _prepare_features_target(df: pd.DataFrame) -> tuple:
    """
    Filtre, encode et sépare les données, puis vérifie que l'identifiant
    ne fait pas partie des variables.

    Paramètres :
        df : DataFrame nettoyé

    Retourne :
        Tuple (X encodé, y)
    """
    df = _filter_valid_rows(df)
    df = _encode_binary_columns(df)
    X, y = _split_features_target(df)

    leaked = check_model_features(X.columns)
    if leaked:
        raise ValueError(f"L'identifiant fait partie des variables : {leaked[:5]}")
    return X, y


def _filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve uniquement les lignes avec des valeurs valides
    sur la target et la colonne d'accès téléphonique.

    Paramètres :
        df : DataFrame source

    Retourne :
        DataFrame filtré
    """
    df = df.copy()
    df = df[df[TARGET_COLUMN].isin(VALID_BINARY_VALUES)]
    df = df[df[PHONE_COLUMN].isin(VALID_BINARY_VALUES)]
    return df


def _encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes binaires Yes/No en valeurs numériques 1/0.

    Paramètres :
        df : DataFrame filtré

    Retourne :
        DataFrame avec les colonnes binaires encodées
    """
    binary_map = {"Yes": 1, "No": 0}
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(binary_map)
    df[PHONE_COLUMN]  = df[PHONE_COLUMN].map(binary_map)
    return df


def _split_features_target(df: pd.DataFrame) -> tuple:
    """
    Sépare les features (X) de la variable cible (y)
    et applique le one-hot encoding sur X.

    L'identifiant du répondant est exclu : propre à chaque ligne,
    il n'apporte aucune information et n'existe pas dans la saisie utilisateur.

    Paramètres :
        df : DataFrame avec colonnes encodées

    Retourne :
        Tuple (X encodé, y)
    """
    X = pd.get_dummies(df.drop(columns=[TARGET_COLUMN, ID_COLUMN]))
    y = df[TARGET_COLUMN]
    return X, y


# ==============================================================================
# Point d'entrée : contrôler, évaluer, entraîner une fois, sauvegarder
# ==============================================================================

if __name__ == "__main__":
    from src.data_processing import load_clean_data

    sys.stdout.reconfigure(encoding="utf-8")
    df = load_clean_data()
    assert_clean_data(df)
    print("✅ Contrôles qualité")

    metrics = evaluate_model(df)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"✅ Évaluation : {json.dumps(metrics)} → {METRICS_PATH}")

    model, reference_columns = train_model(df)
    save_model(model, reference_columns, hash_data_file())
    print(f"✅ Modèle entraîné sur {metrics['rows']} lignes → {MODEL_PATH}")
