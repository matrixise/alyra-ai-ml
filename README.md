# Projet de développement IA - Formation Alyra

[![Tests](https://github.com/matrixise/alyra-ai-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/matrixise/alyra-ai-ml/actions/workflows/tests.yml)

Projet d'analyse de données et de prédiction du risque d'abandon scolaire dans l'enseignement supérieur.

## À propos

Ce projet réalise une analyse exploratoire de données (EDA) et de la modélisation machine learning sur un dataset de rétention des étudiants dans l'enseignement supérieur.

### Dataset

- **Source** : [Kaggle - Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **Emplacement** : `data/dataset.csv`
- **Contenu** : Données sur les facteurs prédictifs de la rétention des étudiants (statut marital, mode d'inscription, cursus, notes, taux de chômage, etc.)

### Objectifs

1. **EDA (Exploratory Data Analysis)** : Exploration et visualisation des données
2. **Feature Engineering** : Création et sélection de features pertinentes
3. **Modélisation ML** : Prédiction du risque de dropout étudiant avec classification des niveaux de risque

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

### 3. Créer l'environnement virtuel et installer les dépendances

```bash
task venv
task install
```

### 4. Installer les hooks pre-commit

```bash
task pre-commit-install
```

## Utilisation

### Pipeline ML

#### Entraîner le modèle

```bash
# Entraînement avec les paramètres par défaut
task ml:train

# Entraînement avec options personnalisées
task ml:train -- --test-size 0.3 --seed 123

# Voir toutes les options
task ml:train -- --help
```

Options disponibles :
- `-d, --data` : Chemin vers le fichier CSV de données (défaut: `data/dataset.csv`)
- `-o, --output` : Chemin de sortie pour le modèle (défaut: `models/dropout_predictor.pkl`)
- `-t, --test-size` : Proportion des données pour le test (0.1-0.5, défaut: 0.2)
- `-s, --seed` : Graine aléatoire pour la reproductibilité (défaut: 42)

#### Faire des prédictions

```bash
# Prédictions avec affichage console
task ml:predict -- data/dataset.csv

# Sauvegarder les prédictions dans un fichier
task ml:predict -- data/dataset.csv -o predictions.csv

# Utiliser un modèle spécifique
task ml:predict -- data/dataset.csv -m models/mon_modele.pkl

# Voir toutes les options
task ml:predict -- --help
```

#### Pipeline complet

```bash
# Entraîner et prédire en une commande
task ml:pipeline
```

### Niveaux de risque

Le modèle classifie les étudiants en 4 niveaux de risque :

| Niveau | Probabilité | Action recommandée |
|--------|-------------|-------------------|
| **Bas** | < 25% | Suivi standard |
| **Moyen** | 25-50% | Vigilance accrue, entretien de suivi |
| **Élevé** | 50-75% | Intervention proactive, accompagnement personnalisé |
| **Critique** | > 75% | Action urgente, mobilisation immédiate |

### Gestion des dépendances

```bash
# Vérifier les dépendances obsolètes
task outdated

# Mettre à jour les dépendances
task upgrade

# Mettre à jour et commiter automatiquement
task upgrade-commit
```

### CI/CD

```bash
# Lister les workflows GitHub Actions
task ci:workflows

# Tester les workflows localement avec act
task ci:act
```

### Autres commandes Task

```bash
# Lister toutes les tâches disponibles
task --list

# Exécuter les tests
task test

# Formater le code
task format

# Exécuter pre-commit sur tous les fichiers
task pre-commit-run

# Ajouter une dépendance
task add -- nom-package

# Ajouter une dépendance de développement
task add-dev -- nom-package
```

### Lancer Jupyter Notebook

```bash
source .venv/bin/activate
jupyter notebook
```

### Tester les GitHub Actions localement

```bash
# Tester tous les workflows
act

# Tester un événement spécifique
act push

# Lister les workflows disponibles
act -l
```

## Structure du projet

```
.
├── data/                   # Datasets
│   └── dataset.csv         # Dataset de rétention étudiante
├── models/                 # Modèles ML entraînés
│   └── dropout_predictor.pkl
├── notebooks/              # Jupyter notebooks d'analyse
│   └── eda_minimal.ipynb   # EDA principal
├── src/alyra_ai_ml/        # Module partagé
│   ├── __init__.py         # Exports du module
│   ├── constants.py        # Constantes partagées
│   └── features.py         # Feature engineering
├── tests/                  # Tests unitaires
│   ├── conftest.py
│   ├── test_train_model.py
│   └── test_predict.py
├── train_model.py          # Script d'entraînement (CLI Typer)
├── predict.py              # Script de prédiction (CLI Typer)
├── pyproject.toml          # Configuration du projet
├── Taskfile.yml            # Définition des tâches Task
└── README.md
```

## Technologies utilisées

### Langage et runtime

- **Python 3.12.12** (géré via asdf)

### Bibliothèques principales

- **pandas** - Manipulation et analyse de données
- **scikit-learn** - Machine learning (Logistic Regression, Pipeline, preprocessing)
- **joblib** - Sérialisation de modèles ML
- **typer** - CLI moderne avec auto-completion
- **rich** - Affichage console formaté (tables, couleurs)
- **matplotlib/seaborn** - Visualisations

### Outils de développement

- **uv** - Gestionnaire de paquets Python rapide
- **Task** - Task runner
- **asdf** - Gestionnaire de versions d'outils
- **pytest** - Framework de tests
- **ruff** - Linter et formateur Python
- **pre-commit** - Git hooks pour la qualité du code

## Licence

Projet académique - Formation Alyra

## Auteur

Stéphane Wirtel <stephane@wirtel.be>
