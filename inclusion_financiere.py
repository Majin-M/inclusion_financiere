import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config("Prédicteur de compte en banque", "💳")

@st.cache_data
def load_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")
    df = df.drop(columns=["uniqueid"])
    return df

@st.cache_resource
def train_model(df):
    df["bank_account"] = df["bank_account"].map({"Yes": 1, "No": 0})
    df["cellphone_access"] = df["cellphone_access"].map({"Yes": 1, "No": 0})

    X = pd.get_dummies(df.drop("bank_account", axis=1))
    y = df["bank_account"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, X.columns

df = load_data()
model, X_columns = train_model(df)

# -----------------------------------
# UI
# -----------------------------------
st.title("💳 Prédicteur de compte en banque")


with st.form("Formulaire prédiction"):
    country = st.selectbox("Pays", df["country"].unique())
    year = st.slider("Année", 2016, 2018, 2017)
    age = st.slider("Age", 16, 100, 30)
    household = st.slider("Taille du foyer", 1, 20, 3)
    gender = st.selectbox("genre", df["gender_of_respondent"].unique())
    location = st.selectbox("Localisation", df["location_type"].unique())
    education = st.selectbox("Etude", df["education_level"].unique())
    job = st.selectbox("Travail", df["job_type"].unique())
    marital = st.selectbox("Situation matrimoniale", df["marital_status"].unique())
    relation = st.selectbox("relation avec le chef", df["relationship_with_head"].unique())
    phone = st.selectbox("Accès téléphonique", ["Yes", "No"])

    submit = st.form_submit_button("Prédiction")

if submit:
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

    user = pd.get_dummies(user)
    user = user.reindex(columns=X_columns, fill_value=0)

    proba = model.predict_proba(user)[0][1]

    st.success(f"Probabilités de posséder un compte en banque: **{proba:.1%}**")
