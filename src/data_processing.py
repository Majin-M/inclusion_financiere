# ==============================================================================
# Module : Chargement et prétraitement des données
# ==============================================================================
# Objectif du module :
#     Ce module fournit les fonctions de chargement et de prétraitement
#     des données pour le modèle de prédiction d'inclusion bancaire.
#
#     Il effectue les actions suivantes :
#     - Charge le fichier CSV nettoyé issu de l'EDA.
#     - Encode les variables catégorielles via one-hot encoding.
#     - Aligne les colonnes de la saisie utilisateur sur celles du modèle.
#
# Exemple d'utilisation :
#     from src.data_processing import load_clean_data, preprocess_user_input
# ==============================================================================


import pandas as pd


# ==============================================================================
# Constantes
# ==============================================================================

DEFAULT_DATA_PATH = "data/financial_inclusion_clean.csv"


# ==============================================================================
# Chargement des données
# ==============================================================================

def load_clean_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Charge le fichier CSV nettoyé issu de l'analyse exploratoire (EDA).

    Paramètres :
        path : Chemin vers le fichier CSV (par défaut : DEFAULT_DATA_PATH)

    Retourne :
        DataFrame contenant les données nettoyées
    """
    return pd.read_csv(path)


# ==============================================================================
# Prétraitement des données utilisateur
# ==============================================================================

def preprocess_user_input(
    user_df: pd.DataFrame,
    reference_columns: pd.Index
) -> pd.DataFrame:
    """
    Encode et aligne les données saisies par l'utilisateur
    sur les colonnes de référence du modèle entraîné.

    Paramètres :
        user_df           : DataFrame d'une ligne issu du formulaire
        reference_columns : Colonnes utilisées lors de l'entraînement du modèle

    Retourne :
        DataFrame encodé et aligné, prêt pour la prédiction
    """
    user_encoded = pd.get_dummies(user_df)
    return user_encoded.reindex(columns=reference_columns, fill_value=0)
