# Projet de développement IA - Formation Alyra

[![CI](https://github.com/matrixise/alyra-ai-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/matrixise/alyra-ai-ml/actions/workflows/ci.yml)

Projet d'analyse de données dans le cadre de la formation en développement IA d'Alyra.

## À propos

Ce projet réalise une analyse exploratoire de données (EDA) et une Analyse Factorielle des Correspondances (AFC) sur un dataset de rétention des étudiants dans l'enseignement supérieur.

### Dataset

- **Source** : [Kaggle - Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **Emplacement** : `data/dataset.csv`
- **Contenu** : Données sur les facteurs prédictifs de la rétention des étudiants (statut marital, mode d'inscription, cursus, notes, taux de chômage, etc.)

### Objectifs

1. **EDA (Exploratory Data Analysis)** : Exploration et visualisation des données
2. **AFC (Analyse Factorielle des Correspondances)** : Analyse des relations entre variables catégorielles
3. **Sélection de modèles ML** : Évaluation et sélection d'un ou plusieurs modèles de machine learning
4. **API FastAPI** : Exposition du modèle via une API REST
5. **Interface Streamlit** (optionnel) : Interface web pour interagir avec le modèle

## Prérequis

- Python 3.12 ou supérieur
- [asdf](https://asdf-vm.com/) (optionnel, pour la gestion des versions)
- [Task](https://taskfile.dev/) (task runner)
- [act](https://github.com/nektos/act) (optionnel, pour tester les GitHub Actions localement)

## Installation

### 1. Cloner le repository

```bash
git clone <repository-url>
cd alyra-ai-ml
```

### 2. Installer les versions d'outils (avec asdf)

```bash
asdf install
```

### 3. Créer l'environnement virtuel

```bash
task venv
```

### 4. Installer les dépendances

```bash
task install
```

### 5. Installer les hooks pre-commit

```bash
task pre-commit-install
```

## Utilisation

### Commandes Task disponibles

```bash
# Créer l'environnement virtuel
task venv

# Installer les dépendances du projet
task install

# Ajouter une dépendance de production
task add -- pandas numpy

# Ajouter une dépendance de développement
task add-dev -- pytest black

# Installer les hooks pre-commit
task pre-commit-install

# Exécuter pre-commit sur tous les fichiers
task pre-commit-run

# Lister toutes les tâches disponibles
task --list
```

### Lancer Jupyter Notebook

```bash
source .venv/bin/activate
jupyter notebook
```

### Tester les GitHub Actions localement

Le projet est configuré avec `.actrc` pour utiliser l'architecture `linux/amd64` par défaut :

```bash
# Tester tous les workflows
act

# Tester un événement spécifique (ex: push)
act push

# Lister les workflows disponibles
act -l
```

## Structure du projet

```
.
├── data/               # Datasets
│   └── dataset.csv     # Dataset de rétention étudiante
├── notebooks/          # Jupyter notebooks d'analyse
├── .venv/              # Environnement virtuel Python
├── .actrc              # Configuration act (GitHub Actions locales)
├── .pre-commit-config.yaml  # Configuration pre-commit hooks
├── pyproject.toml      # Configuration du projet et dépendances
├── Taskfile.yml        # Définition des tâches Task
└── README.md           # Ce fichier
```

## Technologies utilisées

### Langage et runtime
- **Python 3.12.12** (géré via asdf)

### Bibliothèques principales
- **pandas** - Manipulation et analyse de données
- **matplotlib** - Visualisations
- **seaborn** - Visualisations statistiques avancées
- **scikit-learn** - Machine learning et analyses statistiques
- **fanalysis** - Analyse Factorielle des Correspondances (AFC)

### Outils de développement
- **uv** - Gestionnaire de paquets Python rapide
- **Task** - Task runner
- **asdf** - Gestionnaire de versions d'outils
- **act** - Test local des GitHub Actions
- **pre-commit** - Git hooks pour la qualité du code
- **ipython** - Shell Python interactif amélioré
- **ruff** - Linter et formateur Python
- **isort** - Tri des imports
- **nbstripout** - Nettoyage des notebooks Jupyter

## Licence

Projet académique - Formation Alyra

## Auteur

Étudiant en développement IA - Alyra
