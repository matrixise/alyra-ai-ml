"""Tests for Pandera schemas."""

import pandas as pd

from alyra_ai_ml.schemas import StudentDataSchema


class TestStudentDataSchema:
    """Tests for StudentDataSchema."""

    def test_valid_data_passes(self, sample_df: pd.DataFrame) -> None:
        """Test that valid data passes validation."""
        validated = StudentDataSchema.validate(sample_df, lazy=True)
        assert len(validated) > 0

    def test_invalid_marital_status_fails(self, sample_df: pd.DataFrame) -> None:
        """Test that invalid marital status fails validation."""
        df = sample_df.copy()
        df.loc[0, "Marital status"] = 99  # Invalid value

        # Should drop the invalid row
        validated = StudentDataSchema.validate(df, lazy=True)
        assert len(validated) == len(sample_df) - 1

    def test_invalid_target_fails(self, sample_df: pd.DataFrame) -> None:
        """Test that invalid target value fails validation."""
        df = sample_df.copy()
        df.loc[0, "Target"] = "Invalid"  # Invalid value

        # Should drop the invalid row
        validated = StudentDataSchema.validate(df, lazy=True)
        assert len(validated) == len(sample_df) - 1
