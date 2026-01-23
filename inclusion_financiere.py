import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config("💳 Prédicteur de compte en banque", "💳")

# Définir les noms de colonnes car le CSV n'a pas de header
COLUMNS = [
    "country", "year", "uniqueid", "bank_account", "location_type", "cellphone_access",
    "household_size", "age_of_respondent", "gender_of_respondent",
    "relationship_with_head", "marital_status", "education_level", "job_type"
]

@st.cache_data
def load_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")
    df = df.drop(columns=["uniqueid"], errors="ignore")  # ignore si colonne absente

    # Forcer les colonnes numériques
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["age_of_respondent"] = pd.to_numeric(df["age_of_respondent"], errors="coerce")
    df["household_size"] = pd.to_numeric(df["household_size"], errors="coerce")

    # Supprimer les lignes avec valeurs manquantes après conversion
    df = df.dropna(subset=["year", "age_of_respondent", "household_size"])

    return df

@st.cache_resource
def train_model(df):
    # Encoder les colonnes binaires
    df["bank_account"] = df["bank_account"].map({"Yes": 1, "No": 0})
    df["cellphone_access"] = df["cellphone_access"].map({"Yes": 1, "No": 0})

    # Encoder les colonnes catégorielles
    X = pd.get_dummies(df.drop("bank_account", axis=1))
    y = df["bank_account"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, X.columns

# Charger les données et entraîner le modèle
df = load_data()
@st.cache_resource
def train_model(df):
    # Supprimer les lignes où la target ou le téléphone est manquant
    df = df.dropna(subset=["bank_account", "cellphone_access"])

    # Encoder les colonnes binaires
    df["bank_account"] = df["bank_account"].map({"Yes": 1, "No": 0})
    df["cellphone_access"] = df["cellphone_access"].map({"Yes": 1, "No": 0})

    # Encoder les colonnes catégorielles
    X = pd.get_dummies(df.drop("bank_account", axis=1))
    y = df["bank_account"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, X.columns


# -----------------------------------
# UI
# -----------------------------------
st.title("💳 Prédicteur de compte en banque")

with st.form("Formulaire prédiction"):
    country = st.selectbox("Pays", df["country"].unique())
    year = st.slider("Année", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
    age = st.slider("Âge", int(df["age_of_respondent"].min()), int(df["age_of_respondent"].max()), 30)
    household = st.slider("Taille du foyer", int(df["household_size"].min()), int(df["household_size"].max()), 3)
    gender = st.selectbox("Genre", df["gender_of_respondent"].unique())
    location = st.selectbox("Localisation", df["location_type"].unique())
    education = st.selectbox("Niveau d'étude", df["education_level"].unique())
    job = st.selectbox("Type d'emploi", df["job_type"].unique())
    marital = st.selectbox("Situation matrimoniale", df["marital_status"].unique())
    relation = st.selectbox("Relation avec le chef du foyer", df["relationship_with_head"].unique())
    phone = st.selectbox("Accès téléphonique", ["Yes", "No"])

    submit = st.form_submit_button("Prédire")

if submit:
    # Créer un dataframe utilisateur
    user = pd.DataFrame([{
        "country": country,
        "year": year,
        "age_of_respondent": age,
        "household_size": household,
        "gender_of_respondent": gender,
        "location_type": location,
        "education_level": education,
        "job_type": job,
        "marital_status": marital,
        "relationship_with_head": relation,
        "cellphone_access": 1 if phone == "Yes" else 0
    }])

    # Encoder comme lors de l'entraînement
    user_encoded = pd.get_dummies(user)
    user_encoded = user_encoded.reindex(columns=X_columns, fill_value=0)

    # Prédiction
    proba = model.predict_proba(user_encoded)[0][1]

    st.success(f"Probabilité de posséder un compte en banque : **{proba:.1%}**")
