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

# LABELS
phone_labels = {"Oui": "Yes", "Non": "No"}
gender_labels = {"Homme": "Male", "Femme": "Female"}
location_labels = {"Urbain": "Urban", "Rural": "Rural"}

with st.form("prediction_form"):
    country = st.selectbox("Pays", df["country"].unique())
    year = st.slider("Année", int(df["year"].min()), int(df["year"].max()), int(df["year"].median()))
    age = st.slider("Âge", int(df["respondent_age"].min()), int(df["respondent_age"].max()), 30)
    household = st.slider("Taille du foyer", int(df["household_size"].min()), int(df["household_size"].max()), 3)

    gender_ui = st.selectbox("Genre", gender_labels.keys())
    gender = gender_labels[gender_ui]

    location_ui = st.selectbox("Localisation", location_labels.keys())
    location = location_labels[location_ui]

    phone_ui = st.selectbox("Accès téléphonique", phone_labels.keys())
    phone = phone_labels[phone_ui]

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
