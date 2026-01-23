import pandas as pd

def load_clean_data(path="data/financial_inclusion_clean.csv"):
    return pd.read_csv(path)


def preprocess_user_input(user_df, reference_columns):
    user_encoded = pd.get_dummies(user_df)
    return user_encoded.reindex(columns=reference_columns, fill_value=0)
