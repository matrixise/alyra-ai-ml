"""
Pandera DataFrameSchema pour valider le dataset Students Dropout.

Usage:
    from schemas import StudentDataSchema

    df = pd.read_csv("data/dataset.csv", sep=";")
    validated_df = StudentDataSchema.validate(df)
"""

import pandera.pandas as pa

# from pandera import Column, Check, DataFrameSchema

# Valeurs valides pour les variables catégorielles
MARITAL_STATUS_VALUES = [1, 2, 3, 4, 5, 6]

APPLICATION_MODE_VALUES = [
    1,
    2,
    5,
    7,
    10,
    15,
    16,
    17,
    18,
    26,
    27,
    39,
    42,
    43,
    44,
    51,
    53,
    57,
]

COURSE_VALUES = [
    33,
    171,
    8014,
    9003,
    9070,
    9085,
    9119,
    9130,
    9147,
    9238,
    9254,
    9500,
    9556,
    9670,
    9773,
    9853,
    9991,
]

PREVIOUS_QUALIFICATION_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    10,
    12,
    14,
    15,
    19,
    38,
    39,
    40,
    42,
    43,
]

NATIONALITY_VALUES = [
    1,
    2,
    6,
    11,
    13,
    14,
    17,
    21,
    22,
    24,
    25,
    26,
    32,
    41,
    62,
    100,
    101,
    103,
    105,
    108,
    109,
]

PARENT_QUALIFICATION_VALUES = [
    1,
    2,
    3,
    4,
    5,
    6,
    9,
    10,
    11,
    12,
    14,
    18,
    19,
    22,
    26,
    27,
    29,
    30,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
]

PARENT_OCCUPATION_VALUES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    90,
    99,
    122,
    123,
    125,
    131,
    132,
    134,
    141,
    143,
    144,
    151,
    152,
    153,
    171,
    173,
    175,
    191,
    192,
    193,
    194,
    195,
]

TARGET_VALUES = ["Dropout", "Enrolled", "Graduate"]


StudentDataSchema = pa.DataFrameSchema(
    columns={
        # Variables Catégorielles
        "Marital status": pa.Column(
            int,
            pa.Check.isin(MARITAL_STATUS_VALUES),
            nullable=False,
            description="État civil de l'étudiant",
        ),
        "Application mode": pa.Column(
            int,
            pa.Check.isin(APPLICATION_MODE_VALUES),
            nullable=False,
            description="Mode de candidature",
        ),
        "Application order": pa.Column(
            int,
            pa.Check.in_range(0, 9),
            nullable=False,
            description="Ordre de candidature (0=premier choix, 9=dernier)",
        ),
        "Course": pa.Column(
            int,
            pa.Check.isin(COURSE_VALUES),
            nullable=False,
            description="Programme d'études",
        ),
        "Daytime/evening attendance": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Jour (1) ou Soir (0)",
        ),
        "Previous qualification": pa.Column(
            int,
            pa.Check.isin(PREVIOUS_QUALIFICATION_VALUES),
            nullable=False,
            description="Qualification antérieure",
        ),
        "Nacionality": pa.Column(
            int,
            pa.Check.isin(NATIONALITY_VALUES),
            nullable=False,
            description="Nationalité",
            # Note: Le CSV original a "Nacionality" avec une faute, fixé lors du loading
        ),
        "Mother's qualification": pa.Column(
            int,
            pa.Check.isin(PARENT_QUALIFICATION_VALUES),
            nullable=False,
            description="Qualification de la mère",
        ),
        "Father's qualification": pa.Column(
            int,
            pa.Check.isin(PARENT_QUALIFICATION_VALUES),
            nullable=False,
            description="Qualification du père",
        ),
        "Mother's occupation": pa.Column(
            int,
            pa.Check.isin(PARENT_OCCUPATION_VALUES),
            nullable=False,
            description="Profession de la mère",
        ),
        "Father's occupation": pa.Column(
            int,
            pa.Check.isin(PARENT_OCCUPATION_VALUES),
            nullable=False,
            description="Profession du père",
        ),
        # Variables Binaires
        "Displaced": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Déplacé de résidence",
        ),
        "Educational special needs": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Besoins éducatifs spéciaux",
        ),
        "Debtor": pa.Column(
            int, pa.Check.isin([0, 1]), nullable=False, description="Débiteur"
        ),
        "Tuition fees up to date": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Frais de scolarité à jour",
        ),
        "Gender": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Sexe (0=Femme, 1=Homme)",
        ),
        "Scholarship holder": pa.Column(
            int, pa.Check.isin([0, 1]), nullable=False, description="Boursier"
        ),
        "International": pa.Column(
            int,
            pa.Check.isin([0, 1]),
            nullable=False,
            description="Étudiant international",
        ),
        # Variables Numériques - Admission
        "Previous qualification (grade)": pa.Column(
            float,
            pa.Check.in_range(0, 200),
            nullable=False,
            description="Note de la qualification antérieure (0-200)",
        ),
        "Admission grade": pa.Column(
            float,
            pa.Check.in_range(0, 200),
            nullable=False,
            description="Note d'admission (0-200)",
        ),
        "Age at enrollment": pa.Column(
            int,
            pa.Check.in_range(17, 100),
            nullable=False,
            description="Âge à l'inscription",
        ),
        # Variables Numériques - Performance 1er Semestre
        "Curricular units 1st sem (credited)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités créditées sem 1"
        ),
        "Curricular units 1st sem (enrolled)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités inscrites sem 1"
        ),
        "Curricular units 1st sem (evaluations)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités évaluées sem 1"
        ),
        "Curricular units 1st sem (approved)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités validées sem 1"
        ),
        "Curricular units 1st sem (grade)": pa.Column(
            float,
            pa.Check.in_range(0, 20),
            nullable=False,
            description="Note moyenne sem 1 (0-20)",
        ),
        "Curricular units 1st sem (without evaluations)": pa.Column(
            int,
            pa.Check.ge(0),
            nullable=False,
            description="Unités sans évaluation sem 1",
        ),
        # === Variables Numériques - Performance 2ème Semestre ===
        "Curricular units 2nd sem (credited)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités créditées sem 2"
        ),
        "Curricular units 2nd sem (enrolled)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités inscrites sem 2"
        ),
        "Curricular units 2nd sem (evaluations)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités évaluées sem 2"
        ),
        "Curricular units 2nd sem (approved)": pa.Column(
            int, pa.Check.ge(0), nullable=False, description="Unités validées sem 2"
        ),
        "Curricular units 2nd sem (grade)": pa.Column(
            float,
            pa.Check.in_range(0, 20),
            nullable=False,
            description="Note moyenne sem 2 (0-20)",
        ),
        "Curricular units 2nd sem (without evaluations)": pa.Column(
            int,
            pa.Check.ge(0),
            nullable=False,
            description="Unités sans évaluation sem 2",
        ),
        # === Indicateurs Macroéconomiques ===
        "Unemployment rate": pa.Column(
            float,
            pa.Check.in_range(0, 100),
            nullable=False,
            description="Taux de chômage (%)",
        ),
        "Inflation rate": pa.Column(
            float,
            pa.Check.in_range(-10, 20),
            nullable=False,
            description="Taux d'inflation (%)",
        ),
        "GDP": pa.Column(
            float, pa.Check.in_range(-10, 10), nullable=False, description="PIB"
        ),
        # === Variable Cible ===
        "Target": pa.Column(
            str,
            pa.Check.isin(TARGET_VALUES),
            nullable=False,
            description="Variable cible (Dropout, Enrolled, Graduate)",
        ),
    },
    # Vérifie qu'il n'y a pas de colonnes supplémentaires
    strict=True,
    # Vérifie l'ordre des colonnes
    ordered=False,
    # Nom du schéma
    name="StudentDropoutSchema",
    # Description
    description="Schéma de validation pour le dataset Students Dropout and Academic Success",
    drop_invalid_rows=True,
)


# Schéma alternatif après renommage de "Nacionality" -> "Nationality"
StudentDataSchemaRenamed = StudentDataSchema.remove_columns(
    ["Nacionality"]
).add_columns(
    {
        "Nationality": pa.Column(
            int,
            pa.Check.isin(NATIONALITY_VALUES),
            nullable=False,
            description="Nationalité",
        ),
    }
)


# Schéma avec Target encodé (après LabelEncoder)
StudentDataSchemaEncoded = StudentDataSchemaRenamed.update_column(
    "Target",
    dtype=int,
    checks=pa.Check.isin([0, 1, 2]),
    description="Variable cible encodée (0=Dropout, 1=Enrolled, 2=Graduate)",
)


if __name__ == "__main__":
    import pandas as pd

    # Test de validation
    df = pd.read_csv("data/dataset.csv", sep=";")
    print(f"Shape: {df.shape}")

    # Strip des noms de colonnes
    df.columns = df.columns.str.strip()

    try:
        validated_df = StudentDataSchema.validate(df, lazy=True)
        print("Validation réussie!")
        print(f"Shape: {validated_df.shape}")
    except pa.errors.SchemaError as e:
        print(f"Erreur de validation: {e}")
