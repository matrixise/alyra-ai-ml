"""
Script d'entraînement du modèle de prédiction d'abandon scolaire

Ce script reproduit le pipeline d'entraînement du notebook eda_minimal.ipynb
avec le même feature engineering et la même régression logistique optimisée.

Colonnes requises dans le CSV d'entrée : voir docs/csv_schema.md
"""

from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Configuration globale
RANDOM_SEED = 42
TEST_SIZE = 0.2
DATA_PATH = Path('data/dataset.csv')
MODEL_OUTPUT_PATH = Path('models/dropout_predictor.pkl')

# Définir les graines aléatoires pour la reproductibilité
np.random.seed(RANDOM_SEED)

# Rich console pour l'affichage
console = Console()

app = typer.Typer(help="Entraînement du modèle de prédiction d'abandon scolaire")


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Charge le dataset depuis le fichier CSV.

    Args:
        data_path: Chemin vers le fichier CSV

    Returns:
        DataFrame contenant les données brutes
    """
    console.print(f"Chargement des données depuis [cyan]{data_path}[/cyan]...")
    df = pd.read_csv(data_path, delimiter=';')
    console.print(f"[green]✓[/green] Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible binaire Dropout_Binary.

    Args:
        df: DataFrame avec la colonne 'Target'

    Returns:
        DataFrame avec la nouvelle colonne 'Dropout_Binary'
    """
    df['Dropout_Binary'] = df['Target'].map({'Dropout': 1, 'Graduate': 0, 'Enrolled': 0})
    console.print("[green]✓[/green] Variable cible créée. Distribution:")
    console.print(f"    {df['Dropout_Binary'].value_counts().to_dict()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'ingénierie des caractéristiques au DataFrame.

    Cette fonction crée les caractéristiques identiques au notebook eda_minimal.ipynb :

    **Performance académique :**
    - Success_Rate_Sem1/Sem2 : Taux de réussite par semestre (corrélation forte avec dropout)
    - Avg_Grade : Note moyenne sur l'année
    - Total_Approved : Total d'unités validées
    - Performance_Trend : Évolution entre sem1 et sem2

    **Profil sociodémographique :**
    - Age_Group : Tranches d'âge [17-20, 21-25, 26-35, 36+]
    - Marital_Binary : Solo vs Couple
    - Education_Level : Secondaire, Supérieur, Autre
    - Course_Domain : Tech, Santé, Business, etc.

    Args:
        df: DataFrame avec les données brutes

    Returns:
        DataFrame avec les nouvelles caractéristiques
    """
    # 1. Taux de réussite du 1er semestre
    df['Success_Rate_Sem1'] = (
        df['Curricular units 1st sem (approved)'] /
        df['Curricular units 1st sem (enrolled)'].replace(0, np.nan)
    )

    # 2. Taux de réussite du 2ème semestre
    df['Success_Rate_Sem2'] = (
        df['Curricular units 2nd sem (approved)'] /
        df['Curricular units 2nd sem (enrolled)'].replace(0, np.nan)
    )

    # 3. Note moyenne annuelle
    df['Avg_Grade'] = (
        df['Curricular units 1st sem (grade)'] +
        df['Curricular units 2nd sem (grade)']
    ) / 2

    # 4. Total d'unités validées
    df['Total_Approved'] = (
        df['Curricular units 1st sem (approved)'] +
        df['Curricular units 2nd sem (approved)']
    )

    # 5. Évolution de performance entre semestres
    df['Performance_Trend'] = (
        df['Curricular units 2nd sem (grade)'] -
        df['Curricular units 1st sem (grade)']
    )

    # 6. Statut marital binaire (Solo vs Couple)
    solo_categories = [1, 3, 4, 6]  # Célibataire, Veuf, Divorcé, Séparé
    df['Marital_Binary'] = df['Marital status'].apply(
        lambda x: 'Solo' if x in solo_categories else 'Couple'
    )

    # 7. Niveau d'éducation antérieur
    secondaire = [1, 9, 10, 12, 14, 15, 19, 38]
    superieur = [2, 3, 4, 5, 6, 39, 40, 42, 43]
    df['Education_Level'] = df['Previous qualification'].apply(
        lambda x: 'Secondaire' if x in secondaire else 'Supérieur' if x in superieur else 'Autre'
    )

    # 8. Domaine d'études
    course_domains = {
        33: 'Tech', 171: 'Arts', 8014: 'Social', 9003: 'Sciences',
        9070: 'Arts', 9085: 'Santé', 9119: 'Tech', 9130: 'Sciences',
        9147: 'Business', 9238: 'Social', 9254: 'Business', 9500: 'Santé',
        9556: 'Santé', 9670: 'Business', 9773: 'Arts', 9853: 'Education',
        9991: 'Business',
    }
    df['Course_Domain'] = df['Course'].map(course_domains)

    # 9. Groupes d'âge
    df['Age_Group'] = pd.cut(
        df['Age at enrollment'],
        bins=[0, 20, 25, 35, 100],
        labels=['17-20', '21-25', '26-35', '36+']
    )

    return df


def get_feature_sets() -> tuple[list[str], list[str], list[str], str]:
    """
    Définit les ensembles de caractéristiques selon le notebook eda_minimal.ipynb.

    Returns:
        Tuple (numeric_features, categorical_features, binary_features, target)
    """
    # Features numériques (7 features)
    numeric_features = [
        'Success_Rate_Sem1',
        'Success_Rate_Sem2',
        'Avg_Grade',
        'Total_Approved',
        'Age at enrollment',
        'Admission grade',
        'Performance_Trend',
    ]

    # Features catégorielles créées par feature engineering (4 features)
    categorical_features = [
        'Age_Group',
        'Course_Domain',
        'Marital_Binary',
        'Education_Level',
    ]

    # Features binaires déjà encodées 0/1 (5 features)
    binary_features = [
        'Tuition fees up to date',
        'Scholarship holder',
        'Debtor',
        'Gender',
        'Displaced',
    ]

    target = 'Dropout_Binary'

    return numeric_features, categorical_features, binary_features, target


def create_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str]
) -> ColumnTransformer:
    """
    Crée le pipeline de prétraitement avec StandardScaler, OneHotEncoder et passthrough.

    Architecture identique au notebook eda_minimal.ipynb :
    - StandardScaler pour les features numériques
    - OneHotEncoder pour les features catégorielles (avec handle_unknown='ignore')
    - Passthrough pour les features binaires (déjà 0/1)

    Args:
        numeric_features: Liste des noms de caractéristiques numériques
        categorical_features: Liste des noms de caractéristiques catégorielles
        binary_features: Liste des noms de caractéristiques binaires

    Returns:
        ColumnTransformer configuré
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_features),
            ('bin', 'passthrough', binary_features),
        ],
        remainder='drop'
    )

    console.print("[green]✓[/green] Pipeline de prétraitement créé")
    console.print(f"    StandardScaler sur {len(numeric_features)} features numériques")
    console.print(f"    OneHotEncoder sur {len(categorical_features)} features catégorielles")
    console.print(f"    Passthrough sur {len(binary_features)} features binaires")

    return preprocessor


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer
) -> Pipeline:
    """
    Entraîne le modèle de régression logistique.

    Args:
        X_train: Données d'entraînement
        y_train: Cible d'entraînement
        preprocessor: Pipeline de prétraitement

    Returns:
        Pipeline entraîné
    """
    console.print("\nEntraînement du modèle de régression logistique...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])

    pipeline.fit(X_train, y_train)
    console.print("[green]✓[/green] Modèle entraîné avec succès")

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> tuple[np.ndarray, float]:
    """
    Évalue le modèle sur l'ensemble de test.

    Args:
        pipeline: Pipeline entraîné
        X_test: Données de test
        y_test: Cible de test

    Returns:
        Prédictions et précision
    """
    console.print("\nÉvaluation du modèle sur l'ensemble de test...")

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    console.print(f"\n[bold]Précision: {accuracy:.4f}[/bold]")

    # Rapport de classification
    console.print("\n[bold]Rapport de Classification:[/bold]")
    console.print(classification_report(y_test, y_pred))

    # Matrice de confusion
    console.print("[bold]Matrice de Confusion:[/bold]")
    cm = confusion_matrix(y_test, y_pred)

    cm_table = Table(show_header=True, header_style="bold")
    cm_table.add_column("")
    cm_table.add_column("Prédit: 0", style="cyan")
    cm_table.add_column("Prédit: 1", style="cyan")
    cm_table.add_row("Réel: 0", str(cm[0][0]), str(cm[0][1]))
    cm_table.add_row("Réel: 1", str(cm[1][0]), str(cm[1][1]))
    console.print(cm_table)

    return y_pred, accuracy


def save_model(pipeline: Pipeline, output_path: Path) -> None:
    """
    Sauvegarde le modèle entraîné.

    Args:
        pipeline: Pipeline entraîné
        output_path: Chemin de sortie pour le modèle
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    console.print(f"\n[green]✓[/green] Modèle sauvegardé dans [cyan]{output_path}[/cyan]")


@app.command()
def main(
    data_path: Annotated[
        Path,
        typer.Option(
            '-d', '--data',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Chemin vers le fichier CSV de données"
        )
    ] = DATA_PATH,
    output: Annotated[
        Path,
        typer.Option(
            '-o', '--output',
            help="Chemin de sortie pour le modèle entraîné"
        )
    ] = MODEL_OUTPUT_PATH,
    test_size: Annotated[
        float,
        typer.Option(
            '-t', '--test-size',
            min=0.1,
            max=0.5,
            help="Proportion des données pour le test (0.1-0.5)"
        )
    ] = TEST_SIZE,
    seed: Annotated[
        int,
        typer.Option(
            '-s', '--seed',
            help="Graine aléatoire pour la reproductibilité"
        )
    ] = RANDOM_SEED,
) -> None:
    """Entraîne le modèle de prédiction d'abandon scolaire."""
    console.print("\n[bold blue]PIPELINE D'ENTRAÎNEMENT - PRÉDICTION D'ABANDON SCOLAIRE[/bold blue]\n")

    # Mise à jour de la graine si nécessaire
    if seed != RANDOM_SEED:
        np.random.seed(seed)

    # 1. Chargement des données
    df = load_data(data_path)

    # 2. Création de la variable cible
    df = create_target_variable(df)

    # 3. Ingénierie des caractéristiques (AVANT le split pour éviter data leakage)
    console.print("\nApplication de l'ingénierie des caractéristiques...")
    df = engineer_features(df)
    console.print(f"[green]✓[/green] Caractéristiques créées. Nouvelle forme: {df.shape}")

    # 4. Définir les ensembles de caractéristiques
    numeric_features, categorical_features, binary_features, target = get_feature_sets()
    all_features = numeric_features + categorical_features + binary_features

    # 5. Préparer X et y
    X = df[all_features].copy()
    y = df[target]

    # 6. Gérer les valeurs manquantes
    missing_before = X.isnull().sum().sum()
    if missing_before > 0:
        console.print(f"\n[yellow]⚠[/yellow] {missing_before} valeurs manquantes détectées...")
        # Remplir les features numériques avec 0
        for col in numeric_features:
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(0)
        # Remplir les features catégorielles avec le mode
        for col in categorical_features:
            if col in X.columns and X[col].isnull().any():
                X[col] = X[col].fillna(X[col].mode()[0])
        console.print("[green]✓[/green] Valeurs manquantes traitées")

    # 7. Division train-test (stratifiée)
    console.print(f"\nDivision des données (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    console.print(f"[green]✓[/green] Ensemble d'entraînement: {X_train.shape[0]} lignes")
    console.print(f"[green]✓[/green] Ensemble de test: {X_test.shape[0]} lignes")

    # Table des formes
    shape_table = Table(title="Formes des données")
    shape_table.add_column("Dataset", style="cyan")
    shape_table.add_column("Forme", style="white")
    shape_table.add_column("Détails", style="white")

    shape_table.add_row(
        "X_train",
        str(X_train.shape),
        f"{len(numeric_features)} num + {len(categorical_features)} cat + {len(binary_features)} bin"
    )
    shape_table.add_row("y_train", str(y_train.shape), "")
    shape_table.add_row("X_test", str(X_test.shape), "")
    shape_table.add_row("y_test", str(y_test.shape), "")

    console.print(shape_table)

    # 8. Créer le pipeline de prétraitement
    preprocessor = create_preprocessing_pipeline(
        numeric_features, categorical_features, binary_features
    )

    # 9. Entraîner le modèle
    pipeline = train_model(X_train, y_train, preprocessor)

    # 10. Évaluer le modèle
    _, accuracy = evaluate_model(pipeline, X_test, y_test)

    # 11. Sauvegarder le modèle
    save_model(pipeline, output)

    console.print(f"\n[bold green]ENTRAÎNEMENT TERMINÉ - Précision finale: {accuracy:.4f}[/bold green]\n")


if __name__ == "__main__":
    app()
