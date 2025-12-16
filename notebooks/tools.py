"""
Utilitaires pour l'analyse exploratoire des données (EDA).

Ce module fournit des fonctions d'aide pour :
- Afficher des résumés de datasets (train/test splits)
- Détecter les valeurs aberrantes (outliers) via la méthode IQR
"""

import numpy as np
import pandas as pd


def print_dataset_summary(name: str, X: pd.DataFrame, y: pd.Series, total_size: int) -> None:
    """
    Affiche un résumé statistique d'un dataset (train/test).

    Args:
        name: Nom du dataset (ex: "Train", "Test")
        X: Features du dataset
        y: Variable cible (0 = non-dropout, 1 = dropout)
        total_size: Taille totale pour calculer le pourcentage
    """
    n_samples = len(X)
    dropout_count = y.sum()
    dropout_rate = y.mean()

    print(f"\n  {name}: {n_samples} samples ({n_samples / total_size:.0%})")
    print(f"    - Dropout: {dropout_count} ({dropout_rate:.1%})")
    print(f"    - Non-Dropout: {n_samples - dropout_count} ({1 - dropout_rate:.1%})")


def count_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[int, float, float]:
    """
    Détecte les outliers d'une colonne selon la méthode IQR (Interquartile Range).

    La méthode IQR considère comme outliers les valeurs situées à plus de
    1.5 × IQR en dessous de Q1 ou au-dessus de Q3.

    Args:
        df: DataFrame contenant les données
        column: Nom de la colonne à analyser

    Returns:
        tuple contenant:
            - nombre d'outliers détectés
            - borne inférieure (Q1 - 1.5 × IQR)
            - borne supérieure (Q3 + 1.5 × IQR)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1  # Écart interquartile

    # Bornes de Tukey (facteur 1.5 = outliers modérés)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrage des valeurs hors bornes
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound


def prepare_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["Success_Rate_Sem1"] = df["Curricular units 1st sem (approved)"] / df[
        "Curricular units 1st sem (enrolled)"
    ].replace(0, np.nan)
    df["Success_Rate_Sem2"] = df["Curricular units 2nd sem (approved)"] / df[
        "Curricular units 2nd sem (enrolled)"
    ].replace(0, np.nan)
    df["Avg_Grade"] = (
        df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]
    ) / 2
    df["Total_Approved"] = (
        df["Curricular units 1st sem (approved)"] + df["Curricular units 2nd sem (approved)"]
    )
    df["Performance_Trend"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )
    solo = [1, 3, 4, 6]  # Célibataire, Veuf, Divorcé, Séparé
    df["Marital_Binary"] = df["Marital status"].apply(lambda x: "Solo" if x in solo else "Couple")

    secondaire = [1, 9, 10, 12, 14, 15, 19, 38]
    superieur = [2, 3, 4, 5, 6, 39, 40, 42, 43]
    df["Education_Level"] = df["Previous qualification"].apply(
        lambda x: "Secondaire" if x in secondaire else "Supérieur" if x in superieur else "Autre"
    )

    course_domains = {
        33: "Tech",
        171: "Arts",
        8014: "Social",
        9003: "Sciences",
        9070: "Arts",
        9085: "Santé",
        9119: "Tech",
        9130: "Sciences",
        9147: "Business",
        9238: "Social",
        9254: "Business",
        9500: "Santé",
        9556: "Santé",
        9670: "Business",
        9773: "Arts",
        9853: "Education",
        9991: "Business",
    }
    df["Course_Domain"] = df["Course"].map(course_domains)

    return df
