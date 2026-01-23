import streamlit as st
import pandas as pd

from src.data_processing import load_clean_data, preprocess_user_input
from src.model import train_model

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config("💳 Prédicteur de compte en banque", "💳")

# -----------------------------------
# LOAD DATA & MODEL
# -----------------------------------
@st.cache_resource
def load_model_and_data():
    df = load_clean_data()
    model, columns = train_model(df)
    return df, model, columns

df, model, X_columns = load_model_and_data()

# -----------------------------------
# UI
# -----------------------------------
st.title("💳 Prédicteur de compte en banque")

with st.form("prediction_form"):
    country = st.selectbox("Pays", df["country"].unique())
    year = st.slider("Année", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
    age = st.slider("Âge", int(df["respondent_age"].min()), int(df["respondent_age"].max()), 30)
    household = st.slider("Taille du foyer", int(df["household_size"].min()), int(df["household_size"].max()), 3)
    gender = st.selectbox("Genre", df["gender_of_respondent"].unique())
    location = st.selectbox("Localisation", df["type_of_location"].unique())
    education = st.selectbox("Niveau d'étude", df["level_of_educuation"].unique())
    job = st.selectbox("Type d'emploi", df["type_of_job"].unique())
    marital = st.selectbox("Situation matrimoniale", df["marital_status"].unique())
    relation = st.selectbox("Relation avec le chef du foyer", df["the_relathip_with_head"].unique())
    phone = st.selectbox("Accès téléphonique", ["Yes", "No"])

    submit = st.form_submit_button("Prédire")

# -----------------------------------
# PREDICTION
# -----------------------------------
if submit:
    user = pd.DataFrame([{
        "country": country,
        "year": year,
        "respondent_age": age,
        "household_size": household,
        "gender_of_respondent": gender,
        "type_of_location": location,
        "level_of_educuation": education,
        "type_of_job": job,
        "marital_status": marital,
        "the_relathip_with_head": relation,
        "cell_phone_access": phone
    }])

    user_encoded = preprocess_user_input(user, X_columns)

    proba = model.predict_proba(user_encoded)[0][1]

    st.success(f"💳 Probabilité de posséder un compte bancaire : **{proba:.1%}**")
