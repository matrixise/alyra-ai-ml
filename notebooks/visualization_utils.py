"""
Fonctions utilitaires pour la visualisation des données de prédiction d'abandon scolaire.

Ce module contient des fonctions réutilisables pour créer des visualisations
comparant différentes variables avec la variable cible (abandon/non-abandon).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_numeric_vs_target(
    df, columns, target_col="Dropout_Binary", plot_type="violin", colors=None, figsize_per_col=5
):
    """
    Créer des graphiques en violon ou en boîte pour les variables numériques vs variable cible.

    Paramètres:
    -----------
    df : DataFrame
        DataFrame d'entrée
    columns : list
        Liste des noms de colonnes numériques à tracer
    target_col : str
        Nom de la colonne de la variable cible
    plot_type : str
        Type de graphique ('violin' ou 'box')
    colors : dict
        Dictionnaire de palette de couleurs
    figsize_per_col : int
        Largeur de figure par colonne
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        if plot_type == "violin":
            sns.violinplot(
                data=df,
                x=target_col,
                y=col,
                hue=target_col,
                palette=colors,
                legend=False,
                ax=axes[idx],
            )
        else:
            sns.boxplot(
                data=df,
                x=target_col,
                y=col,
                hue=target_col,
                palette=colors,
                legend=False,
                ax=axes[idx],
            )

        axes[idx].set_title(f"{col} vs {target_col}")
        axes[idx].set_xlabel("")

    plt.tight_layout()
    plt.show()


def plot_categorical_vs_target(
    df, columns, target_col="Dropout_Binary", colors=None, figsize_per_col=5
):
    """
    Créer des graphiques de comptage pour les variables catégorielles vs variable cible.

    Paramètres:
    -----------
    df : DataFrame
        DataFrame d'entrée
    columns : list
        Liste des noms de colonnes catégorielles à tracer
    target_col : str
        Nom de la colonne de la variable cible
    colors : dict
        Dictionnaire de palette de couleurs
    figsize_per_col : int
        Largeur de figure par colonne
    """
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize_per_col * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        # Créer un tableau croisé
        ct = pd.crosstab(df[col], df[target_col], normalize="index") * 100

        # Tracer
        ct.plot(
            kind="bar",
            stacked=False,
            ax=axes[idx],
            color=[colors.get(c, "gray") for c in ct.columns],
        )
        axes[idx].set_title(f"{col} vs {target_col}")
        axes[idx].set_ylabel("Pourcentage")
        axes[idx].legend(title=target_col)
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def create_correlation_heatmap(df, figsize=(12, 10), annot=True, cmap="coolwarm"):
    """
    Créer une carte de chaleur de corrélation pour les variables numériques.

    Paramètres:
    -----------
    df : DataFrame
        DataFrame d'entrée
    figsize : tuple
        Taille de la figure
    annot : bool
        Si True, annoter les cellules avec les valeurs de corrélation
    cmap : str
        Carte de couleurs

    Retourne:
    ---------
    corr_matrix : DataFrame
        Matrice de corrélation des variables numériques
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])

    # Calculer la matrice de corrélation
    corr_matrix = numeric_df.corr()

    # Créer la carte de chaleur
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matrice de Corrélation - Caractéristiques Numériques")
    plt.tight_layout()
    plt.show()

    return corr_matrix
