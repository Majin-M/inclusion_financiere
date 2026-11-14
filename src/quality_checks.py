# ==============================================================================
# Module : Contrôles qualité du dataset nettoyé
# ==============================================================================
# Objectif du module :
#     Ce module transforme les corrections du nettoyage en contrôles
#     automatiques, relancés à chaque exécution : à la fin de l'EDA avant
#     l'export, et avant chaque entraînement du modèle.
#
#     Chaque contrôle retourne la liste des écarts trouvés (vide si tout va
#     bien), pour qu'un défaut qui revient soit signalé au lieu de passer.
#
# Exemple d'utilisation :
#     python -m src.quality_checks
#     from src.quality_checks import assert_clean_data
# ==============================================================================


import sys

import pandas as pd


# ==============================================================================
# Constantes
# ==============================================================================

SURVEY_YEARS        = [2016, 2017, 2018]
VALID_TARGET_VALUES = ["Yes", "No"]
UNKNOWN_VALUE       = "Unknown"
EDUCATION_LEVELS    = [
    "No formal education",
    "Primary education",
    "Secondary education",
    "Vocational/Specialised training",
    "Tertiary education",
    "Other/Dont know/RTA",
    UNKNOWN_VALUE,
]
ID_COLUMN           = "uniqueid"


# ==============================================================================
# Contrôles du dataset nettoyé
# ==============================================================================

def check_clean_data(df: pd.DataFrame) -> dict:
    """
    Lance tous les contrôles sur le dataset nettoyé.

    Paramètres :
        df : DataFrame nettoyé (noms de colonnes en snake_case)

    Retourne :
        Dictionnaire {intitulé du contrôle: liste des écarts trouvés}
    """
    return {
        "Aucune valeur manquante":                          _check_no_missing(df),
        "Années d'enquête de 2016 à 2018":                  _check_survey_years(df),
        "Une seule année d'enquête par pays":               _check_one_year_per_country(df),
        "Aucun pays inconnu":                               _check_known_country(df),
        "Cible connue (Yes ou No)":                         _check_known_target(df),
        "Aucun doublon":                                    _check_no_duplicates(df),
        "Niveaux d'éducation nommés":                       _check_education_levels(df),
        "Urbains plus souvent titulaires d'un compte":      _check_urban_above_rural(df),
    }


def check_model_features(columns: pd.Index) -> list:
    """
    Vérifie que l'identifiant du répondant n'est pas une variable du modèle.

    Paramètres :
        columns : Colonnes encodées utilisées pour l'entraînement

    Retourne :
        Liste des colonnes issues de l'identifiant (vide si aucune)
    """
    return [col for col in columns if col == ID_COLUMN or col.startswith(f"{ID_COLUMN}_")]


def assert_clean_data(df: pd.DataFrame) -> None:
    """
    Lève une erreur si un contrôle échoue, en listant tous les écarts.

    Paramètres :
        df : DataFrame nettoyé
    """
    failures = {name: issues for name, issues in check_clean_data(df).items() if issues}
    if failures:
        details = "\n".join(f"- {name} : {'; '.join(map(str, issues))}" for name, issues in failures.items())
        raise ValueError(f"Contrôles qualité en échec :\n{details}")


# ==============================================================================
# Fonctions internes
# ==============================================================================

def _check_no_missing(df: pd.DataFrame) -> list:
    missing = df.isna().sum()
    return [f"{col} : {count}" for col, count in missing[missing > 0].items()]


def _check_survey_years(df: pd.DataFrame) -> list:
    return sorted(int(year) for year in df.loc[~df["year"].isin(SURVEY_YEARS), "year"].unique())


def _check_one_year_per_country(df: pd.DataFrame) -> list:
    years = df.groupby("country")["year"].nunique()
    return [f"{country} : {count} années" for country, count in years[years > 1].items()]


def _check_known_country(df: pd.DataFrame) -> list:
    count = int((df["country"] == UNKNOWN_VALUE).sum())
    return [f"{count} lignes"] if count else []


def _check_known_target(df: pd.DataFrame) -> list:
    invalid = df.loc[~df["has_a_bank_account"].isin(VALID_TARGET_VALUES), "has_a_bank_account"]
    return [f"{value} : {count}" for value, count in invalid.value_counts().items()]


def _check_no_duplicates(df: pd.DataFrame) -> list:
    count = int(df.duplicated().sum())
    return [f"{count} lignes"] if count else []


def _check_education_levels(df: pd.DataFrame) -> list:
    return sorted(set(df["level_of_educuation"]) - set(EDUCATION_LEVELS))


def _check_urban_above_rural(df: pd.DataFrame) -> list:
    known = df[df["type_of_location"].isin(["Urban", "Rural"]) & df["has_a_bank_account"].isin(VALID_TARGET_VALUES)]
    rates = known.assign(account=known["has_a_bank_account"].eq("Yes")).pivot_table(
        index="country", columns="type_of_location", values="account", aggfunc="mean"
    )
    return [f"{country} : urbains {row.Urban:.1%}, ruraux {row.Rural:.1%}" for country, row in rates.iterrows() if row.Urban <= row.Rural]


# ==============================================================================
# Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    from src.data_processing import load_clean_data

    sys.stdout.reconfigure(encoding="utf-8")
    results = check_clean_data(load_clean_data())
    for name, issues in results.items():
        print(f"{'✅' if not issues else '❌'} {name}" + (f" : {'; '.join(map(str, issues))}" if issues else ""))
    sys.exit(1 if any(results.values()) else 0)
