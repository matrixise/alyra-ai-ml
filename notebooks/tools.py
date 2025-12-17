"""
Utilitaires pour l'analyse exploratoire des données (EDA).

Ce module fournit des fonctions d'aide pour :
- Afficher des résumés de datasets (train/test splits)
- Détecter les valeurs aberrantes (outliers) via la méthode IQR
"""

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
