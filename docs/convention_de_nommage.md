# **Conventions de Nommage**

Ce document décrit les conventions de nommage utilisées pour les modules, fonctions, variables, constantes et autres objets dans le projet de prédiction d'inclusion bancaire.

## **Table des matières**

1. [Principes généraux](#principes-généraux)
2. [Conventions de nommage des fichiers](#conventions-de-nommage-des-fichiers)
3. [Conventions de nommage des fonctions](#conventions-de-nommage-des-fonctions)
   - [Règles de chargement](#règles-de-chargement)
   - [Règles de prétraitement](#règles-de-prétraitement)
   - [Règles d'entraînement](#règles-dentraînement)
   - [Règles internes](#règles-internes)
   - [Règles d'interface](#règles-dinterface)
4. [Conventions de nommage des variables](#conventions-de-nommage-des-variables)
   - [Constantes](#constantes)
   - [Variables de données](#variables-de-données)
   - [Variables de modèle](#variables-de-modèle)
5. [Glossaire des préfixes de fonctions](#glossaire-des-préfixes-de-fonctions)

---

## **Principes généraux**

- **Conventions de nommage** : Utiliser le `snake_case` pour les fonctions, variables et modules ; le `SCREAMING_SNAKE_CASE` pour les constantes.
- **Langue** : Utiliser l'anglais pour tous les noms d'objets Python.
- **Éviter les abréviations ambiguës** : Préférer des noms explicites et descriptifs à des raccourcis cryptiques.
- **Conformité PEP 8** : Respecter les conventions officielles de style Python.
- **Fonctions internes** : Préfixer avec `_` les fonctions non destinées à être appelées depuis l'extérieur du module.

---

## **Conventions de nommage des fichiers**

- Tous les fichiers Python doivent utiliser le `snake_case` avec l'extension `.py`.
- **`<rôle>.py`** à la racine pour le point d'entrée ; **`<domaine>_<rôle>.py`** dans `src/` pour les modules.

| Fichier | Rôle |
|---|---|
| `app.py` | Point d'entrée — interface Streamlit et orchestration |
| `src/data_processing.py` | Chargement et prétraitement des données |
| `src/model.py` | Entraînement du modèle de prédiction |
| `notebooks/EDA.ipynb` | Analyse exploratoire — nettoyage et export du dataset |

---

## **Conventions de nommage des fonctions**

### **Règles de chargement**
- Les fonctions de chargement d'une ressource doivent commencer par le préfixe `load_`.
- **`load_<ressource>`**
  - Exemple : `load_clean_data` → Charge le CSV nettoyé issu de l'EDA.
  - Exemple : `load_model_and_data` → Charge les données et entraîne le modèle (Streamlit cache).

### **Règles de prétraitement**
- Les fonctions de transformation des données utilisateur doivent utiliser le préfixe `preprocess_`.
- **`preprocess_<cible>`**
  - Exemple : `preprocess_user_input` → Encode et aligne les données saisies sur les colonnes du modèle.

### **Règles d'entraînement**
- Les fonctions d'entraînement de modèle doivent utiliser le préfixe `train_`.
- **`train_<modèle>`**
  - Exemple : `train_model` → Entraîne le classifieur Random Forest.

### **Règles internes**
- Les fonctions internes à un module (non exposées) doivent commencer par `_`.
- **`_<verbe>_<objet>`**
  - Exemple : `_filter_valid_rows` → Filtre les lignes avec des valeurs invalides.
  - Exemple : `_encode_binary_columns` → Encode les colonnes binaires Yes/No.
  - Exemple : `_split_features_target` → Sépare X et y pour l'entraînement.

### **Règles d'interface**
- Les fonctions de rendu Streamlit doivent commencer par le préfixe `render_`.
- **`render_<composant>`**
  - Exemple : `render_form` → Affiche le formulaire de saisie utilisateur.
  - Exemple : `render_prediction` → Affiche le résultat de la prédiction.

---

## **Conventions de nommage des variables**

### **Constantes**
- Toutes les constantes doivent utiliser le `SCREAMING_SNAKE_CASE` et être définies en haut du module.

| Constante | Module | Description |
|---|---|---|
| `DEFAULT_DATA_PATH` | `data_processing.py` | Chemin par défaut vers le CSV nettoyé |
| `TARGET_COLUMN` | `model.py` | Nom de la colonne cible (`has_a_bank_account`) |
| `PHONE_COLUMN` | `model.py` | Nom de la colonne d'accès téléphonique |
| `VALID_BINARY_VALUES` | `model.py` | Valeurs acceptées pour les colonnes binaires |
| `RANDOM_FOREST_PARAMS` | `model.py` | Hyperparamètres du classifieur |
| `GENDER_LABELS` | `app.py` | Dictionnaire de traduction des genres |
| `LOCATION_LABELS` | `app.py` | Dictionnaire de traduction des localisations |
| `PHONE_LABELS` | `app.py` | Dictionnaire de traduction des accès téléphoniques |

### **Variables de données**

| Nom | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | DataFrame principal chargé depuis le CSV |
| `user_df` | `pd.DataFrame` | DataFrame d'une ligne issu du formulaire |
| `user_encoded` | `pd.DataFrame` | DataFrame encodé et aligné sur le modèle |
| `num_cols` | `list` | Liste des colonnes numériques |
| `cat_cols` | `list` | Liste des colonnes catégorielles |
| `reference_columns` | `pd.Index` | Colonnes utilisées lors de l'entraînement |

### **Variables de modèle**

| Nom | Type | Description |
|---|---|---|
| `model` | `RandomForestClassifier` | Modèle entraîné |
| `X` | `pd.DataFrame` | Features encodées pour l'entraînement |
| `y` | `pd.Series` | Variable cible encodée (0/1) |
| `probability` | `float` | Probabilité prédite de posséder un compte |

---

## **Glossaire des préfixes de fonctions**

| Préfixe | Signification | Exemple(s) |
|---|---|---|
| `load_` | Chargement d'une ressource ou d'un modèle | `load_clean_data`, `load_model_and_data` |
| `preprocess_` | Transformation et encodage des données | `preprocess_user_input` |
| `train_` | Entraînement d'un modèle ML | `train_model` |
| `render_` | Rendu d'un composant d'interface Streamlit | `render_form`, `render_prediction` |
| `build_` | Construction d'un objet de données | `build_user_input` |
| `reverse_` | Inversion d'un mapping clé/valeur | `reverse_label` |
| `_` (préfixe) | Fonction interne au module | `_filter_valid_rows`, `_encode_binary_columns` |
