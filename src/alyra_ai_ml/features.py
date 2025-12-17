"""Feature engineering pour la prédiction d'abandon scolaire.

Ce module contient les fonctions de transformation des données brutes
en features utilisables par le modèle de machine learning.
"""

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'ingénierie des caractéristiques au DataFrame.

    Cette fonction crée les caractéristiques suivantes :

    **Performance académique :**
    - Success_Rate_Sem1/Sem2 : Taux de réussite par semestre
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
    df["Success_Rate_Sem1"] = df["Curricular units 1st sem (approved)"] / df[
        "Curricular units 1st sem (enrolled)"
    ].replace(0, np.nan)

    # 2. Taux de réussite du 2ème semestre
    df["Success_Rate_Sem2"] = df["Curricular units 2nd sem (approved)"] / df[
        "Curricular units 2nd sem (enrolled)"
    ].replace(0, np.nan)

    # 3. Note moyenne annuelle
    df["Avg_Grade"] = (
        df["Curricular units 1st sem (grade)"] + df["Curricular units 2nd sem (grade)"]
    ) / 2

    # 4. Total d'unités validées
    df["Total_Approved"] = (
        df["Curricular units 1st sem (approved)"]
        + df["Curricular units 2nd sem (approved)"]
    )

    # 5. Évolution de performance entre semestres
    df["Performance_Trend"] = (
        df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    )

    # 6. Statut marital binaire (Solo vs Couple)
    solo_categories = [1, 3, 4, 6]  # Célibataire, Veuf, Divorcé, Séparé
    df["Marital_Binary"] = df["Marital status"].apply(
        lambda x: "Solo" if x in solo_categories else "Couple"
    )

    # 7. Niveau d'éducation antérieur
    secondaire = [1, 9, 10, 12, 14, 15, 19, 38]
    superieur = [2, 3, 4, 5, 6, 39, 40, 42, 43]
    df["Education_Level"] = df["Previous qualification"].apply(
        lambda x: "Secondaire"
        if x in secondaire
        else "Supérieur"
        if x in superieur
        else "Autre"
    )

    # 8. Domaine d'études
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

    # 9. Groupes d'âge
    df["Age_Group"] = pd.cut(
        df["Age at enrollment"],
        bins=[0, 20, 25, 35, 100],
        labels=["17-20", "21-25", "26-35", "36+"],
    )

    return df


def get_feature_sets(
    *, include_target: bool = False
) -> tuple[list[str], list[str], list[str]] | tuple[list[str], list[str], list[str], str]:
    """
    Retourne les ensembles de caractéristiques pour le modèle.

    Args:
        include_target: Si True, inclut le nom de la variable cible dans le retour.
                       Utilisé par train_model.py.

    Returns:
        Si include_target=False: (numeric_features, categorical_features, binary_features)
        Si include_target=True: (numeric_features, categorical_features, binary_features, target)
    """
    # Features numériques (7 features)
    numeric_features = [
        "Success_Rate_Sem1",
        "Success_Rate_Sem2",
        "Avg_Grade",
        "Total_Approved",
        "Age at enrollment",
        "Admission grade",
        "Performance_Trend",
    ]

    # Features catégorielles créées par feature engineering (4 features)
    categorical_features = [
        "Age_Group",
        "Course_Domain",
        "Marital_Binary",
        "Education_Level",
    ]

    # Features binaires déjà encodées 0/1 (5 features)
    binary_features = [
        "Tuition fees up to date",
        "Scholarship holder",
        "Debtor",
        "Gender",
        "Displaced",
    ]

    if include_target:
        return numeric_features, categorical_features, binary_features, TARGET_COLUMN

    return numeric_features, categorical_features, binary_features
