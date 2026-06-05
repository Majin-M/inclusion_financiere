# Application de Prédiction d'Inclusion Bancaire
Bienvenue dans le dépôt de l'**Application de Prédiction d'Inclusion Bancaire** ! 🚀  
Ce projet présente une application interactive de machine learning permettant de prédire la probabilité qu'un individu possède un compte bancaire, à partir de ses caractéristiques socio-démographiques. Conçu comme projet de portfolio, il met en avant les bonnes pratiques en data engineering et développement Python : pipeline de données structuré, clean code, séparation des responsabilités et documentation complète.

## 🌍 Contexte

Les données proviennent des enquêtes **FinScope** menées en **Afrique de l'Est** (Kenya, Rwanda, Tanzanie, Ouganda). L'inclusion bancaire y reste un enjeu majeur : une large partie de la population n'a pas accès aux services financiers formels. Ce projet modélise les facteurs socio-démographiques qui influencent la possession d'un compte bancaire.

---

## 🚀 Exigences du projet

### Pipeline de données (Data Engineering)

#### Objectif
Construire un pipeline structuré de bout en bout : ingestion des données brutes, analyse exploratoire, nettoyage, et mise à disposition pour la modélisation.

#### Spécifications
- **Source de données** : Dataset CSV brut issu des enquêtes FinScope (`Financial_inclusion_dataset.csv`).
- **Analyse exploratoire** : Profilage des données, détection des valeurs manquantes et des doublons, analyse des distributions — documentée dans `notebooks/EDA.ipynb`.
- **Qualité des données** : Traitement des valeurs manquantes (médiane pour le numérique, `Unknown` pour le catégoriel), suppression des doublons, normalisation des noms de colonnes en `snake_case`.
- **Output** : Dataset nettoyé exporté vers `data/financial_inclusion_clean.csv`.
- **Documentation** : Convention de nommage disponible dans `docs/naming_conventions.md`.

---

### Modélisation & Interface (Machine Learning + Streamlit)

#### Objectif
Entraîner un modèle de classification et le rendre accessible via une interface interactive permettant une prédiction en temps réel.

#### Fonctionnalités
- **Modèle** : Classifieur Random Forest (200 estimateurs) entraîné sur les données nettoyées.
- **Variables** : Pays, âge, genre, localisation, niveau d'éducation, type d'emploi, situation matrimoniale, taille du foyer, accès téléphonique.
- **Interface** : Formulaire interactif avec sliders, selectbox et labels traduits en français.
- **Résultat** : Probabilité de possession d'un compte bancaire affichée en temps réel.

Pour plus de détails sur les dépendances, consultez [requirements.txt](requirements.txt).

---

## 🗂️ Structure du projet

```
financial-inclusion/
│
├── app.py                          # Point d'entrée — streamlit run app.py
│
├── src/
│   ├── data_processing.py          # Chargement et prétraitement des données
│   └── model.py                    # Entraînement du modèle Random Forest
│
├── data/
│   ├── Financial_inclusion_dataset.csv   # Dataset brut (non versionné)
│   └── financial_inclusion_clean.csv     # Dataset nettoyé (généré par l'EDA)
│
├── notebooks/
│   └── EDA.ipynb                   # Analyse exploratoire des données
│
├── docs/
│   └── naming_conventions.md       # Conventions de nommage du projet
│
├── requirements.txt                # Dépendances du projet
└── README.md                       # Présentation du projet
```

---

## ⚙️ Installation et utilisation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Majin-M/financial-inclusion.git
cd financial-inclusion
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Générer le dataset nettoyé
Ouvrir et exécuter `notebooks/EDA.ipynb` dans Jupyter pour produire `data/financial_inclusion_clean.csv`.

### 4. Lancer l'application
```bash
streamlit run app.py
```

---

## 🛡️ Licence
Ce projet est sous licence [MIT](LICENSE). Vous êtes libre de l'utiliser, le modifier et le partager avec attribution appropriée.

## 👤 À propos de moi
Je suis Steven Mouthoud, Data Engineer passionné par la construction de pipelines de données robustes et le développement d'applications orientées data.
