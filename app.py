import streamlit as st
import pandas as pd
from src.data_processing import load_clean_data, preprocess_user_input
from src.model import train_model

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config("💳 Prédicteur de compte bancaire", "💳", layout="wide")

# ----------------------------
# CHARGEMENT DES DONNÉES & MODÈLE
# ----------------------------
@st.cache_resource
def load_model_and_data():
    df = load_clean_data()
    model, columns = train_model(df)
    return df, model, columns

df, model, X_columns = load_model_and_data()

# ----------------------------
# DICTIONNAIRES POUR LABELS FR
# ----------------------------
gender_labels = {"Male": "Homme", "Female": "Femme"}
location_labels = {"Urban": "Urbain", "Rural": "Rural"}
phone_labels = {"Yes": "Oui", "No": "Non"}

# ----------------------------
# FORMULAIRE UTILISATEUR
# ----------------------------
st.title("💳 Prédicteur de compte bancaire")

with st.form("prediction_form"):
    st.subheader("Informations personnelles")
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("Pays", df["country"].unique())
        year = st.slider("Année", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
        age = st.slider("Âge", int(df["respondent_age"].min()), int(df["respondent_age"].max()), 30)

    with col2:
        household = st.slider(
            "Taille du foyer 👨‍👩‍👧‍👦",
            int(df["household_size"].min()), 
            int(df["household_size"].max()), 
            3,
            help="Nombre de personnes dans le foyer"
        )
        gender = st.selectbox("Genre", list(gender_labels.values()))
        location = st.selectbox("Localisation", list(location_labels.values()))
    
    with st.expander("Informations socio-professionnelles"):
        education = st.selectbox("Niveau d'étude", df["level_of_educuation"].unique())
        job = st.selectbox("Type d'emploi", df["type_of_job"].unique())
        marital = st.selectbox("Situation matrimoniale", df["marital_status"].unique())
        relation = st.selectbox("Relation avec le chef du foyer", df["the_relathip_with_head"].unique())
        phone = st.selectbox("Accès téléphonique 📱", list(phone_labels.values()))
    
    submit = st.form_submit_button("💡 Prédire")

# ----------------------------
# PREDICTION
# ----------------------------
if submit:
    # Traduire labels français en valeur originale
    gender_val = [k for k,v in gender_labels.items() if v == gender][0]
    location_val = [k for k,v in location_labels.items() if v == location][0]
    phone_val = [k for k,v in phone_labels.items() if v == phone][0]

    user = pd.DataFrame([{
        "country": country,
        "year": year,
        "respondent_age": age,
        "household_size": household,
        "gender_of_respondent": gender_val,
        "type_of_location": location_val,
        "level_of_educuation": education,
        "type_of_job": job,
        "marital_status": marital,
        "the_relathip_with_head": relation,
        "cell_phone_access": phone_val
    }])

    user_encoded = preprocess_user_input(user, X_columns)
    proba = model.predict_proba(user_encoded)[0][1]

    st.success(f"💳 Probabilité de posséder un compte bancaire : {proba:.1%}")
