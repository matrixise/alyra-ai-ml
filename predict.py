"""
Script de prédiction du risque d'abandon scolaire.

Ce script charge un modèle entraîné et prédit le risque de dropout
pour de nouveaux étudiants à partir d'un fichier CSV.

Usage:
    .venv/bin/python predict.py data/nouveaux_etudiants.csv
    .venv/bin/python predict.py data/nouveaux_etudiants.csv --output predictions.csv
"""

from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sklearn.pipeline import Pipeline


# Configuration
MODEL_PATH = Path('models/dropout_predictor.pkl')

# Rich console pour l'affichage
console = Console()

app = typer.Typer(help="Prédiction du risque d'abandon scolaire")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le même feature engineering que l'entraînement.

    IMPORTANT: Cette fonction doit être identique à celle de train_model.py
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
    solo_categories = [1, 3, 4, 6]
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


def get_feature_sets() -> tuple[list[str], list[str], list[str]]:
    """Retourne les listes de features (identique à train_model.py)."""
    numeric_features = [
        'Success_Rate_Sem1',
        'Success_Rate_Sem2',
        'Avg_Grade',
        'Total_Approved',
        'Age at enrollment',
        'Admission grade',
        'Performance_Trend',
    ]

    categorical_features = [
        'Age_Group',
        'Course_Domain',
        'Marital_Binary',
        'Education_Level',
    ]

    binary_features = [
        'Tuition fees up to date',
        'Scholarship holder',
        'Debtor',
        'Gender',
        'Displaced',
    ]

    return numeric_features, categorical_features, binary_features


def load_model(model_path: Path) -> Pipeline:
    """Charge le modèle depuis le disque."""
    return joblib.load(model_path)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les features pour la prédiction."""
    # Feature engineering
    df = engineer_features(df)

    # Récupérer les listes de features
    numeric_features, categorical_features, binary_features = get_feature_sets()
    all_features = numeric_features + categorical_features + binary_features

    # Vérifier les colonnes manquantes
    missing_cols = [col for col in all_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes après feature engineering: {missing_cols}")

    # Sélectionner les features
    X = df[all_features].copy()

    # Gérer les valeurs manquantes
    for col in numeric_features:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(0)
    for col in categorical_features:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

    return X


def predict(df: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Fait des prédictions sur un DataFrame.

    Returns:
        DataFrame avec colonnes ajoutées:
        - Dropout_Prediction: 0 (Non-Dropout) ou 1 (Dropout)
        - Dropout_Probability: Probabilité de dropout (0-1)
        - Risk_Level: Bas, Moyen, Élevé, Critique
    """
    X = prepare_features(df)

    # Prédictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Ajouter les résultats au DataFrame original
    result = df.copy()
    result['Dropout_Prediction'] = predictions
    result['Dropout_Probability'] = probabilities

    # Classifier le niveau de risque
    result['Risk_Level'] = classify_risk_level(probabilities)

    return result


def classify_risk_level(probabilities: np.ndarray) -> list[str]:
    """
    Classifie les probabilités de dropout en niveaux de risque.

    Seuils choisis pour un contexte éducatif :
    - Bas (< 25%)      : Suivi standard, pas d'intervention particulière
    - Moyen (25-50%)   : Vigilance accrue, entretien de suivi recommandé
    - Élevé (50-75%)   : Intervention proactive, accompagnement personnalisé
    - Critique (> 75%) : Action urgente, mobilisation immédiate des ressources

    Args:
        probabilities: Array numpy de probabilités (0-1)

    Returns:
        Liste de niveaux de risque ('Bas', 'Moyen', 'Élevé', 'Critique')
    """
    levels = []
    for prob in probabilities:
        if prob < 0.25:
            levels.append('Bas')
        elif prob < 0.50:
            levels.append('Moyen')
        elif prob < 0.75:
            levels.append('Élevé')
        else:
            levels.append('Critique')
    return levels


@app.command()
def main(
    input_file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Fichier CSV contenant les données des étudiants"
        )
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            '-o', '--output',
            help="Fichier CSV de sortie (défaut: affichage console)"
        )
    ] = None,
    model: Annotated[
        Path,
        typer.Option(
            '-m', '--model',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Chemin vers le modèle"
        )
    ] = MODEL_PATH,
) -> None:
    """Prédiction du risque d'abandon scolaire."""
    console.print("\n[bold blue]PRÉDICTION DU RISQUE D'ABANDON SCOLAIRE[/bold blue]\n")

    # Charger le modèle
    console.print(f"Chargement du modèle depuis [cyan]{model}[/cyan]...")
    loaded_model = load_model(model)
    console.print("[green]✓[/green] Modèle chargé")

    # Charger les données
    console.print(f"Chargement des données depuis [cyan]{input_file}[/cyan]...")
    df = pd.read_csv(input_file, delimiter=';')
    console.print(f"[green]✓[/green] {len(df)} étudiants chargés")

    # Faire les prédictions
    console.print("Analyse en cours...")
    results = predict(df, loaded_model)
    console.print("[green]✓[/green] Prédictions terminées\n")

    # Résumé
    n_dropout = (results['Dropout_Prediction'] == 1).sum()
    n_safe = (results['Dropout_Prediction'] == 0).sum()
    avg_prob = results['Dropout_Probability'].mean()

    # Table de résumé
    summary_table = Table(title="Résumé des Prédictions", show_header=False)
    summary_table.add_column("Métrique", style="cyan")
    summary_table.add_column("Valeur", style="white")

    summary_table.add_row("Total étudiants analysés", str(len(results)))
    summary_table.add_row("Risque de dropout", f"{n_dropout} ({n_dropout/len(results)*100:.1f}%)")
    summary_table.add_row("Faible risque", f"{n_safe} ({n_safe/len(results)*100:.1f}%)")
    summary_table.add_row("Probabilité moyenne", f"{avg_prob:.1%}")

    console.print(summary_table)

    # Distribution par niveau de risque
    if 'Risk_Level' in results.columns and results['Risk_Level'].notna().any():
        risk_table = Table(title="\nDistribution par niveau de risque")
        risk_table.add_column("Niveau", style="cyan")
        risk_table.add_column("Nombre", style="white")
        risk_table.add_column("Pourcentage", style="white")

        for level, color in [('Critique', 'red'), ('Élevé', 'yellow'), ('Moyen', 'blue'), ('Bas', 'green')]:
            count = (results['Risk_Level'] == level).sum()
            if count > 0:
                risk_table.add_row(
                    f"[{color}]{level}[/{color}]",
                    str(count),
                    f"{count/len(results)*100:.1f}%"
                )

        console.print(risk_table)

    # Sortie
    if output:
        results.to_csv(output, index=False, sep=';')
        console.print(f"\n[green]✓[/green] Résultats sauvegardés dans [cyan]{output}[/cyan]")
    else:
        at_risk = results[results['Dropout_Prediction'] == 1].sort_values(
            'Dropout_Probability', ascending=False
        )
        if len(at_risk) > 0:
            detail_table = Table(title="\nDétail des étudiants à risque (Top 10)")
            detail_table.add_column("Probabilité", style="red")
            detail_table.add_column("Niveau", style="yellow")
            detail_table.add_column("Âge", style="cyan")
            detail_table.add_column("Réussite Sem1", style="white")
            detail_table.add_column("Réussite Sem2", style="white")

            for _, row in at_risk.head(10).iterrows():
                detail_table.add_row(
                    f"{row['Dropout_Probability']:.2%}",
                    row['Risk_Level'],
                    str(int(row['Age at enrollment'])) if pd.notna(row.get('Age at enrollment')) else 'N/A',
                    f"{row['Success_Rate_Sem1']:.2f}" if pd.notna(row.get('Success_Rate_Sem1')) else 'N/A',
                    f"{row['Success_Rate_Sem2']:.2f}" if pd.notna(row.get('Success_Rate_Sem2')) else 'N/A',
                )

            console.print(detail_table)
        else:
            console.print("\n[green]Aucun étudiant à risque détecté.[/green]")

    console.print()


if __name__ == "__main__":
    app()
