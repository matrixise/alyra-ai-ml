"""
Pandera DataFrameSchema pour valider le dataset Students Dropout.

Usage:
    from schemas import StudentDataSchema

    df = pd.read_csv("data/dataset.csv", sep=";")
    validated_df = StudentDataSchema.validate(df)
"""

import pandera.pandas as pa


# Valeurs valides pour les variables catégorielles
MARITAL_STATUS_LABELS = {
    1: "Single (Célibataire)",
    2: "Married (Marié)",
    3: "Widower (Veuf)",
    4: "Divorced (Divorcé)",
    5: "Facto union (Union de fait)",
    6: "Legally separated (Séparé légalement)",
}
MARITAL_STATUS_VALUES = MARITAL_STATUS_LABELS.keys()

APPLICATION_MODE_LABELS = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)",
}
APPLICATION_MODE_VALUES = APPLICATION_MODE_LABELS.keys()

COURSE_LABELS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening attendance)",
}
COURSE_VALUES = COURSE_LABELS.keys()

DAYTIME_EVENING_LABELS = {
    1: "Daytime (Jour)",
    0: "Evening (Soir)",
}

PREVIOUS_QUALIFICATION_LABELS = {
    1: "Secondary education (Enseignement secondaire)",
    2: "Higher education - bachelor's degree",
    3: "Higher education - degree",
    4: "Higher education - master's",
    5: "Higher education - doctorate",
    6: "Frequency of higher education",
    9: "12th year of schooling - not completed",
    10: "11th year of schooling - not completed",
    12: "Other - 11th year of schooling",
    14: "10th year of schooling",
    15: "10th year of schooling - not completed",
    19: "Basic education 3rd cycle (9th/10th/11th year)",
    38: "Basic education 2nd cycle (6th/7th/8th year)",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education - master (2nd cycle)",
}

PREVIOUS_QUALIFICATION_VALUES = PREVIOUS_QUALIFICATION_LABELS.keys()

NATIONALITY_LABELS = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova (Republic of)",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian",
}

NATIONALITY_VALUES = NATIONALITY_LABELS.keys()


PARENT_QUALIFICATION_LABELS = {
    1: "Secondary Education - 12th Year or Eq.",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year - Not Completed",
    10: "11th Year - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year",
    14: "10th Year",
    18: "General commerce course",
    19: "Basic Education 3rd Cycle",
    22: "Technical-professional course",
    26: "7th year of schooling",
    27: "2nd cycle of the general high school",
    29: "9th Year - Not Completed",
    30: "8th year of schooling",
    34: "Unknown",
    35: "Can't read or write",
    36: "Can read without having a 4th year",
    37: "Basic education 1st cycle (4th year)",
    38: "Basic Education 2nd Cycle",
    39: "Technological specialization course",
    40: "Higher education - degree (1st cycle)",
    41: "Specialized higher studies course",
    42: "Professional higher technical course",
    43: "Higher Education - Master (2nd cycle)",
    44: "Higher Education - Doctorate (3rd cycle)",
}

PARENT_QUALIFICATION_VALUES = PARENT_QUALIFICATION_LABELS.keys()

PARENT_OCCUPATION_LABELS = {
    0: "Student",
    1: "Representatives of the Legislative Power and Executive Bodies",
    2: "Specialists in Intellectual and Scientific Activities",
    3: "Intermediate Level Technicians and Professions",
    4: "Administrative staff",
    5: "Personal Services, Security and Safety Workers",
    6: "Farmers and Skilled Workers in Agriculture",
    7: "Skilled Workers in Industry, Construction",
    8: "Installation and Machine Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "(blank)",
    122: "Health professionals",
    123: "Teachers",
    125: "ICT specialists",
    131: "Science and engineering technicians",
    132: "Health associate professionals",
    134: "Legal, social, sports professionals",
    141: "Office workers, secretaries",
    143: "Data, accounting workers",
    144: "Other administrative support",
    151: "Personal service workers",
    152: "Sellers",
    153: "Personal care workers",
    171: "Construction workers",
    173: "Printing, precision workers",
    175: "Food processing workers",
    191: "Cleaning workers",
    192: "Unskilled agriculture workers",
    193: "Unskilled industry workers",
    194: "Meal preparation assistants",
    195: "Street vendors",
}

PARENT_OCCUPATION_VALUES = PARENT_OCCUPATION_LABELS.keys()

TARGET_LABELS = {
    "Dropout": "Abandon des études",
    "Enrolled": "Toujours inscrit",
    "Graduate": "Diplômé",
}

TARGET_VALUES = TARGET_LABELS.keys()

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
        "Debtor": pa.Column(int, pa.Check.isin([0, 1]), nullable=False, description="Débiteur"),
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
            pa.Check.in_range(12, 100),
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
        "GDP": pa.Column(float, pa.Check.in_range(-10, 10), nullable=False, description="PIB"),
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
StudentDataSchemaRenamed = StudentDataSchema.remove_columns(["Nacionality"]).add_columns(
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
