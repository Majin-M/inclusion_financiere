# ==============================================================================
# Module : Prédiction d'inclusion bancaire (Interface Streamlit)
# ==============================================================================
# Objectif du module :
#     Ce module fournit une interface interactive permettant de prédire
#     la probabilité qu'un individu possède un compte bancaire,
#     à partir de ses caractéristiques socio-démographiques.
#
#     Il effectue les actions suivantes :
#     - Charge les données nettoyées et entraîne le modèle au démarrage.
#     - Affiche un formulaire de saisie pour les caractéristiques de l'individu.
#     - Retourne la probabilité de possession d'un compte bancaire.
#
# Paramètres configurables (via l'interface) :
#     country          : Pays de résidence (Afrique de l'Est)
#     year             : Année de l'enquête
#     respondent_age   : Âge du répondant
#     household_size   : Taille du foyer
#     gender           : Genre du répondant
#     location         : Type de localisation (Urbain / Rural)
#     education        : Niveau d'éducation
#     job              : Type d'emploi
#     marital_status   : Situation matrimoniale
#     relation         : Relation avec le chef du foyer
#     cell_phone       : Accès à un téléphone mobile
#
# Exemple d'utilisation :
#     streamlit run app.py
# ==============================================================================


import streamlit as st
import pandas as pd

from src.data_processing import load_clean_data, preprocess_user_input
from src.model import train_model


# ==============================================================================
# Configuration de la page
# ==============================================================================

st.set_page_config(
    page_title="💳 Prédicteur de compte bancaire",
    page_icon="💳",
    layout="wide"
)


# ==============================================================================
# Constantes — Traduction des labels
# ==============================================================================

GENDER_LABELS   = {"Male": "Homme",  "Female": "Femme"}
LOCATION_LABELS = {"Urban": "Urbain", "Rural": "Rural"}
PHONE_LABELS    = {"Yes": "Oui",     "No": "Non"}


# ==============================================================================
# Chargement des données et du modèle
# ==============================================================================

@st.cache_resource
def load_model_and_data() -> tuple:
    """
    Charge les données nettoyées et entraîne le modèle Random Forest.

    Retourne :
        Tuple (DataFrame, modèle entraîné, colonnes de référence)
    """
    df = load_clean_data()
    model, reference_columns = train_model(df)
    return df, model, reference_columns


df, model, reference_columns = load_model_and_data()


# ==============================================================================
# Fonctions utilitaires
# ==============================================================================

def reverse_label(label_dict: dict, display_value: str) -> str:
    """
    Retrouve la valeur originale à partir du label affiché.

    Paramètres :
        label_dict    : Dictionnaire {valeur_originale: label_affiché}
        display_value : Label affiché sélectionné par l'utilisateur

    Retourne :
        Valeur originale correspondante
    """
    return [key for key, value in label_dict.items() if value == display_value][0]


def build_user_input(form_values: dict) -> pd.DataFrame:
    """
    Construit un DataFrame d'une ligne à partir des valeurs du formulaire.

    Paramètres :
        form_values : Dictionnaire des valeurs saisies dans le formulaire

    Retourne :
        DataFrame formaté pour le prétraitement
    """
    return pd.DataFrame([form_values])


# ==============================================================================
# Interface Streamlit
# ==============================================================================

def render_form() -> dict | None:
    """
    Affiche le formulaire de saisie et retourne les valeurs soumises.

    Retourne :
        Dictionnaire des valeurs du formulaire si soumis, None sinon
    """
    st.title("💳 Prédicteur de compte bancaire")
    st.caption("Données issues des enquêtes FinScope — Afrique de l'Est")

    with st.form("prediction_form"):
        st.subheader("Informations personnelles")
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Pays", df["country"].unique())
            year    = st.slider(
                "Année",
                int(df["year"].min()),
                int(df["year"].max()),
                int(df["year"].median())
            )
            age = st.slider(
                "Âge",
                int(df["respondent_age"].min()),
                int(df["respondent_age"].max()),
                30
            )

        with col2:
            household_size = st.slider(
                "Taille du foyer 👨‍👩‍👧‍👦",
                int(df["household_size"].min()),
                int(df["household_size"].max()),
                3,
                help="Nombre de personnes dans le foyer"
            )
            gender   = st.selectbox("Genre",        list(GENDER_LABELS.values()))
            location = st.selectbox("Localisation",  list(LOCATION_LABELS.values()))

        with st.expander("Informations socio-professionnelles"):
            education = st.selectbox("Niveau d'éducation",              df["level_of_educuation"].unique())
            job       = st.selectbox("Type d'emploi",                   df["type_of_job"].unique())
            marital   = st.selectbox("Situation matrimoniale",          df["marital_status"].unique())
            relation  = st.selectbox("Relation avec le chef du foyer",  df["the_relathip_with_head"].unique())
            phone     = st.selectbox("Accès téléphonique 📱",           list(PHONE_LABELS.values()))

        submitted = st.form_submit_button("💡 Prédire")

    if not submitted:
        return None

    return {
        "country":               country,
        "year":                  year,
        "respondent_age":        age,
        "household_size":        household_size,
        "gender_of_respondent":  reverse_label(GENDER_LABELS,   gender),
        "type_of_location":      reverse_label(LOCATION_LABELS, location),
        "level_of_educuation":   education,
        "type_of_job":           job,
        "marital_status":        marital,
        "the_relathip_with_head": relation,
        "cell_phone_access":     reverse_label(PHONE_LABELS,    phone)
    }


def render_prediction(form_values: dict) -> None:
    """
    Prétraite les données saisies et affiche la probabilité prédite.

    Paramètres :
        form_values : Dictionnaire des valeurs soumises par l'utilisateur
    """
    user_df      = build_user_input(form_values)
    user_encoded = preprocess_user_input(user_df, reference_columns)
    probability  = model.predict_proba(user_encoded)[0][1]

    st.success(f"💳 Probabilité de posséder un compte bancaire : **{probability:.1%}**")


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    form_values = render_form()

    if form_values is not None:
        render_prediction(form_values)
