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

from alyra_ai_ml import (
    DATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    engineer_features,
    get_feature_sets,
)


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
    preprocessor: ColumnTransformer,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """
    Entraîne le modèle de régression logistique.

    Args:
        X_train: Données d'entraînement
        y_train: Cible d'entraînement
        preprocessor: Pipeline de prétraitement
        seed: Graine aléatoire

    Returns:
        Pipeline entraîné
    """
    console.print("\nEntraînement du modèle de régression logistique...")

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=seed))
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
    ] = MODEL_PATH,
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
    numeric_features, categorical_features, binary_features, target = get_feature_sets(
        include_target=True
    )
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
    pipeline = train_model(X_train, y_train, preprocessor, seed)

    # 10. Évaluer le modèle
    _, accuracy = evaluate_model(pipeline, X_test, y_test)

    # 11. Sauvegarder le modèle
    save_model(pipeline, output)

    console.print(f"\n[bold green]ENTRAÎNEMENT TERMINÉ - Précision finale: {accuracy:.4f}[/bold green]\n")


if __name__ == "__main__":
    app()
