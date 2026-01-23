from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def train_model(df):
    df = df.copy()

    # Nettoyage explicite de la target
    df = df[df["has_a_bank_account"].isin(["Yes", "No"])]
    df = df[df["cell_phone_access"].isin(["Yes", "No"])]

    df["has_a_bank_account"] = df["has_a_bank_account"].map({"Yes": 1, "No": 0})
    df["cell_phone_access"] = df["cell_phone_access"].map({"Yes": 1, "No": 0})

    # Séparation X / y
    X = pd.get_dummies(df.drop("has_a_bank_account", axis=1))
    y = df["has_a_bank_account"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    return model, X.columns
