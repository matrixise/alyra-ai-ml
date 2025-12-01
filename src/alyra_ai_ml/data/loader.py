"""Data loading and preprocessing functions."""

from pathlib import Path

import pandas as pd

from alyra_ai_ml.schemas import StudentDataSchema, StudentDataSchemaRenamed

# Default dataset path relative to project root
DEFAULT_DATASET_PATH = (
    Path(__file__).parent.parent.parent.parent.parent / "data" / "dataset.csv"
)


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with stripped column names
    """
    df.columns = df.columns.str.strip()
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to fix typos in original dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with corrected column names
    """
    return df.rename(columns={"Nacionality": "Nationality"})


def load_dataset(
    path: str | Path | None = None,
    sep: str = ";",
    fix_columns: bool = True,
) -> pd.DataFrame:
    """Load the student dropout dataset.

    Args:
        path: Path to the CSV file. Defaults to data/dataset.csv
        sep: CSV separator. Defaults to ";"
        fix_columns: Whether to strip and rename columns. Defaults to True

    Returns:
        Loaded DataFrame
    """
    if path is None:
        path = DEFAULT_DATASET_PATH

    df = pd.read_csv(path, sep=sep)

    if fix_columns:
        df = df.pipe(strip_column_names).pipe(rename_columns)

    return df


def validate_dataset(
    df: pd.DataFrame,
    renamed: bool = True,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate the dataset against the Pandera schema.

    Args:
        df: DataFrame to validate
        renamed: Whether to use the schema with renamed columns. Defaults to True
        lazy: Whether to use lazy validation. Defaults to True

    Returns:
        Validated DataFrame (invalid rows are dropped)
    """
    schema = StudentDataSchemaRenamed if renamed else StudentDataSchema
    return schema.validate(df, lazy=lazy)


def load_and_validate(
    path: str | Path | None = None,
    sep: str = ";",
) -> pd.DataFrame:
    """Load and validate the dataset in one step.

    Args:
        path: Path to the CSV file. Defaults to data/dataset.csv
        sep: CSV separator. Defaults to ";"

    Returns:
        Loaded and validated DataFrame
    """
    df = load_dataset(path=path, sep=sep, fix_columns=True)
    return validate_dataset(df, renamed=True, lazy=True)
