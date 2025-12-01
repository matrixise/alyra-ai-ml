"""Pytest fixtures and configuration."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root: Path) -> Path:
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def dataset_path(data_dir: Path) -> Path:
    """Return the path to the main dataset."""
    return data_dir / "dataset.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Marital status": [1, 2, 1],
            "Application mode": [1, 17, 39],
            "Application order": [1, 2, 1],
            "Course": [9003, 9070, 9500],
            "Daytime/evening attendance": [1, 1, 0],
            "Previous qualification": [1, 1, 1],
            "Previous qualification (grade)": [130.0, 140.0, 120.0],
            "Nacionality": [1, 1, 1],
            "Mother's qualification": [37, 19, 1],
            "Father's qualification": [37, 19, 1],
            "Mother's occupation": [5, 9, 3],
            "Father's occupation": [9, 5, 3],
            "Admission grade": [125.0, 135.0, 145.0],
            "Displaced": [1, 0, 1],
            "Educational special needs": [0, 0, 0],
            "Debtor": [0, 0, 1],
            "Tuition fees up to date": [1, 1, 0],
            "Gender": [1, 0, 1],
            "Scholarship holder": [0, 1, 0],
            "Age at enrollment": [19, 20, 25],
            "International": [0, 0, 0],
            "Curricular units 1st sem (credited)": [0, 0, 0],
            "Curricular units 1st sem (enrolled)": [6, 5, 6],
            "Curricular units 1st sem (evaluations)": [8, 6, 10],
            "Curricular units 1st sem (approved)": [6, 5, 3],
            "Curricular units 1st sem (grade)": [13.5, 14.0, 10.0],
            "Curricular units 1st sem (without evaluations)": [0, 0, 1],
            "Curricular units 2nd sem (credited)": [0, 0, 0],
            "Curricular units 2nd sem (enrolled)": [6, 5, 6],
            "Curricular units 2nd sem (evaluations)": [8, 6, 10],
            "Curricular units 2nd sem (approved)": [6, 5, 2],
            "Curricular units 2nd sem (grade)": [13.0, 13.5, 9.0],
            "Curricular units 2nd sem (without evaluations)": [0, 0, 2],
            "Unemployment rate": [10.8, 13.9, 15.5],
            "Inflation rate": [1.4, -0.3, 2.8],
            "GDP": [1.74, 0.79, -4.06],
            "Target": ["Graduate", "Graduate", "Dropout"],
        }
    )
