"""Tests for data loader module."""

from pathlib import Path

import pandas as pd
import pytest

from alyra_ai_ml.data.loader import (
    load_dataset,
    rename_columns,
    strip_column_names,
    validate_dataset,
)


class TestStripColumnNames:
    """Tests for strip_column_names function."""

    def test_strips_whitespace(self) -> None:
        """Test that whitespace is stripped from column names."""
        df = pd.DataFrame({"  col1  ": [1], "col2\t": [2]})
        result = strip_column_names(df)
        assert list(result.columns) == ["col1", "col2"]


class TestRenameColumns:
    """Tests for rename_columns function."""

    def test_renames_nacionality(self) -> None:
        """Test that Nacionality is renamed to Nationality."""
        df = pd.DataFrame({"Nacionality": [1, 2, 3]})
        result = rename_columns(df)
        assert "Nationality" in result.columns
        assert "Nacionality" not in result.columns


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_existing_dataset(self, dataset_path: Path) -> None:
        """Test loading the actual dataset."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")

        df = load_dataset(dataset_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_with_fixed_columns(self, dataset_path: Path) -> None:
        """Test that columns are fixed when loading."""
        if not dataset_path.exists():
            pytest.skip("Dataset not found")

        df = load_dataset(dataset_path, fix_columns=True)
        assert "Nationality" in df.columns


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_valid_data(self, sample_df: pd.DataFrame) -> None:
        """Test validating valid data."""
        # Rename for the renamed schema
        df = sample_df.rename(columns={"Nacionality": "Nationality"})
        validated = validate_dataset(df, renamed=True)
        assert len(validated) == len(df)
