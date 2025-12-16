"""
Utilitaires pour l'analyse exploratoire des données (EDA).

Ce module fournit des fonctions d'aide pour :
- Afficher des résumés de datasets (train/test splits)
- Détecter les valeurs aberrantes (outliers) via la méthode IQR
"""

import numpy as np
import pandas as pd


# Palette de couleurs pour éviter les warnings
BINARY_TARGET_COLORS = ["#e74c3c", "#2ecc71"]
BINARY_TARGET_ORDER = ["Dropout", "Non-Dropout"]
MAP_BINARY_TARGET_COLORS = dict(zip(BINARY_TARGET_ORDER, BINARY_TARGET_COLORS))

TERNARY_TARGET_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb"]


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

    print(f"{name}: {n_samples} samples ({n_samples / total_size:.0%})")
    print(f"- Dropout: {dropout_count} ({dropout_rate:.1%})")
    print(f"- Non-Dropout: {n_samples - dropout_count} ({1 - dropout_rate:.1%})")


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
    """
    Crée des features dérivées pour améliorer la prédiction du dropout.

    Features créées et leur intérêt :

    **Performance académique :**
    - Success_Rate_Sem1/Sem2 : Taux de réussite par semestre. Un taux faible
      est un signal fort de risque d'abandon (corrélation directe avec le dropout).
    - Avg_Grade : Note moyenne sur l'année. Synthétise la performance globale.
    - Total_Approved : Total d'unités validées. Mesure la progression concrète.
    - Performance_Trend : Évolution entre sem1 et sem2. Une tendance négative
      peut indiquer un décrochage progressif.

    **Profil sociodémographique :**
    - Marital_Binary : Solo vs Couple. Les étudiants seuls peuvent avoir moins
      de support familial, facteur de risque potentiel.
    - Education_Level : Niveau d'études antérieur regroupé. Les étudiants venant
      du secondaire vs supérieur ont des profils de risque différents.
    - Course_Domain : Domaine d'études (Tech, Santé, Business...). Certains
      domaines ont des taux d'abandon plus élevés.
    - Age_Group : Tranches d'âge. Les étudiants plus âgés (>25 ans) ont souvent
      des contraintes professionnelles/familiales augmentant le risque.

    Args:
        df: DataFrame avec les colonnes brutes du dataset

    Returns:
        DataFrame enrichi avec les nouvelles features
    """
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

    df["Age_Group"] = pd.cut(
        df["Age at enrollment"],
        bins=[0, 20, 25, 35, 100],
        labels=["17-20", "21-25", "26-35", "36+"],
    )
    return df


def barh_metric(ax, df, metric, title, xlim, text_offset, colors):
    """
    Affiche un bar chart horizontal avec barres d'erreur pour comparer des modèles.

    Args:
        ax: Axes matplotlib sur lequel dessiner
        df: DataFrame avec colonnes 'model', '{metric}-mean', '{metric}-std'
        metric: Nom de la métrique (ex: 'accuracy', 'f1')
        title: Titre du graphique
        xlim: Tuple (min, max) pour l'axe X
        text_offset: Décalage pour les annotations de valeurs
        colors: Liste de couleurs pour les barres
    """
    y = df["model"]
    mean = df[f"{metric}-mean"]
    std = df[f"{metric}-std"]

    bars = ax.barh(
        y,
        mean,
        xerr=std,
        color=colors[: len(df)],
        alpha=0.8,
        capsize=5,
    )

    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(*xlim)

    # Annotations
    for i, (v, s) in enumerate(zip(mean, std)):
        ax.text(v + s + text_offset, i, f"{v:.3f}", va="center", fontsize=10)

    return bars
