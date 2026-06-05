# ==============================================================================
# Module : Entraînement du modèle de prédiction
# ==============================================================================
# Objectif du module :
#     Ce module contient la logique d'entraînement du modèle Random Forest
#     utilisé pour prédire la possession d'un compte bancaire.
#
#     Il effectue les actions suivantes :
#     - Filtre et nettoie les valeurs cibles invalides.
#     - Encode les variables binaires (target et accès téléphonique).
#     - Sépare les features (X) de la variable cible (y).
#     - Entraîne un classifieur Random Forest.
#
# Exemple d'utilisation :
#     from src.model import train_model
# ==============================================================================


import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ==============================================================================
# Constantes
# ==============================================================================

TARGET_COLUMN      = "has_a_bank_account"
PHONE_COLUMN       = "cell_phone_access"
VALID_BINARY_VALUES = ["Yes", "No"]

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "random_state": 42
}


# ==============================================================================
# Entraînement du modèle
# ==============================================================================

def train_model(df: pd.DataFrame) -> tuple:
    """
    Prépare les données et entraîne un classifieur Random Forest.

    Étapes :
        1. Filtre les lignes avec des valeurs invalides sur la target et le téléphone.
        2. Encode les colonnes binaires (Yes/No → 1/0).
        3. Sépare X et y, applique le one-hot encoding sur X.
        4. Entraîne le modèle Random Forest.

    Paramètres :
        df : DataFrame nettoyé issu de load_clean_data()

    Retourne :
        Tuple (modèle entraîné, colonnes de référence du DataFrame X)
    """
    df = _filter_valid_rows(df)
    df = _encode_binary_columns(df)

    X, y = _split_features_target(df)

    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X, y)

    return model, X.columns


# ==============================================================================
# Fonctions internes
# ==============================================================================

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

    Paramètres :
        df : DataFrame avec colonnes encodées

    Retourne :
        Tuple (X encodé, y)
    """
    X = pd.get_dummies(df.drop(TARGET_COLUMN, axis=1))
    y = df[TARGET_COLUMN]
    return X, y
