"""Tests for train_model.py"""

import pandas as pd
import pytest

from alyra_ai_ml import engineer_features, get_feature_sets
from train_model import create_target_variable


class TestCreateTargetVariable:
    """Tests for create_target_variable function."""

    def test_creates_dropout_binary_column(self, sample_df: pd.DataFrame) -> None:
        """Should add Dropout_Binary column to DataFrame."""
        result = create_target_variable(sample_df.copy())
        assert "Dropout_Binary" in result.columns

    def test_maps_dropout_to_1(self, sample_df: pd.DataFrame) -> None:
        """Should map 'Dropout' to 1."""
        result = create_target_variable(sample_df.copy())
        dropout_rows = result[result["Target"] == "Dropout"]
        assert (dropout_rows["Dropout_Binary"] == 1).all()

    def test_maps_graduate_to_0(self, sample_df: pd.DataFrame) -> None:
        """Should map 'Graduate' to 0."""
        result = create_target_variable(sample_df.copy())
        graduate_rows = result[result["Target"] == "Graduate"]
        assert (graduate_rows["Dropout_Binary"] == 0).all()

    def test_maps_enrolled_to_0(self) -> None:
        """Should map 'Enrolled' to 0."""
        df = pd.DataFrame({"Target": ["Enrolled", "Enrolled"]})
        result = create_target_variable(df)
        assert (result["Dropout_Binary"] == 0).all()


class TestEngineerFeatures:
    """Tests for engineer_features function."""

    def test_creates_success_rate_sem1(self, sample_df: pd.DataFrame) -> None:
        """Should create Success_Rate_Sem1 column."""
        result = engineer_features(sample_df.copy())
        assert "Success_Rate_Sem1" in result.columns

    def test_success_rate_sem1_calculation(self, sample_df: pd.DataFrame) -> None:
        """Should calculate Success_Rate_Sem1 correctly."""
        result = engineer_features(sample_df.copy())
        # First row: 6 approved / 6 enrolled = 1.0
        assert result.loc[0, "Success_Rate_Sem1"] == pytest.approx(1.0)
        # Third row: 3 approved / 6 enrolled = 0.5
        assert result.loc[2, "Success_Rate_Sem1"] == pytest.approx(0.5)

    def test_creates_avg_grade(self, sample_df: pd.DataFrame) -> None:
        """Should create Avg_Grade column."""
        result = engineer_features(sample_df.copy())
        assert "Avg_Grade" in result.columns

    def test_avg_grade_calculation(self, sample_df: pd.DataFrame) -> None:
        """Should calculate Avg_Grade correctly."""
        result = engineer_features(sample_df.copy())
        # First row: (13.5 + 13.0) / 2 = 13.25
        assert result.loc[0, "Avg_Grade"] == pytest.approx(13.25)

    def test_creates_total_approved(self, sample_df: pd.DataFrame) -> None:
        """Should create Total_Approved column."""
        result = engineer_features(sample_df.copy())
        assert "Total_Approved" in result.columns
        # First row: 6 + 6 = 12
        assert result.loc[0, "Total_Approved"] == 12

    def test_creates_performance_trend(self, sample_df: pd.DataFrame) -> None:
        """Should create Performance_Trend column."""
        result = engineer_features(sample_df.copy())
        assert "Performance_Trend" in result.columns
        # First row: 13.0 - 13.5 = -0.5
        assert result.loc[0, "Performance_Trend"] == pytest.approx(-0.5)

    def test_creates_marital_binary(self, sample_df: pd.DataFrame) -> None:
        """Should create Marital_Binary column with Solo/Couple values."""
        result = engineer_features(sample_df.copy())
        assert "Marital_Binary" in result.columns
        assert set(result["Marital_Binary"].unique()).issubset({"Solo", "Couple"})

    def test_marital_binary_solo_mapping(self, sample_df: pd.DataFrame) -> None:
        """Should map marital status 1 (single) to Solo."""
        result = engineer_features(sample_df.copy())
        single_rows = result[sample_df["Marital status"] == 1]
        assert (single_rows["Marital_Binary"] == "Solo").all()

    def test_creates_education_level(self, sample_df: pd.DataFrame) -> None:
        """Should create Education_Level column."""
        result = engineer_features(sample_df.copy())
        assert "Education_Level" in result.columns
        assert set(result["Education_Level"].unique()).issubset(
            {"Secondaire", "Supérieur", "Autre"}
        )

    def test_creates_course_domain(self, sample_df: pd.DataFrame) -> None:
        """Should create Course_Domain column."""
        result = engineer_features(sample_df.copy())
        assert "Course_Domain" in result.columns

    def test_creates_age_group(self, sample_df: pd.DataFrame) -> None:
        """Should create Age_Group column with correct bins."""
        result = engineer_features(sample_df.copy())
        assert "Age_Group" in result.columns
        # Age 19 -> 17-20, Age 20 -> 17-20, Age 25 -> 21-25
        assert result.loc[0, "Age_Group"] == "17-20"
        assert result.loc[2, "Age_Group"] == "21-25"

    def test_handles_zero_enrolled(self) -> None:
        """Should handle division by zero for enrolled units."""
        df = pd.DataFrame({
            "Curricular units 1st sem (approved)": [5],
            "Curricular units 1st sem (enrolled)": [0],
            "Curricular units 2nd sem (approved)": [5],
            "Curricular units 2nd sem (enrolled)": [5],
            "Curricular units 1st sem (grade)": [10.0],
            "Curricular units 2nd sem (grade)": [12.0],
            "Marital status": [1],
            "Previous qualification": [1],
            "Course": [9003],
            "Age at enrollment": [20],
        })
        result = engineer_features(df)
        assert pd.isna(result.loc[0, "Success_Rate_Sem1"])


class TestGetFeatureSets:
    """Tests for get_feature_sets function."""

    def test_returns_three_elements_by_default(self) -> None:
        """Should return tuple of 3 elements by default."""
        result = get_feature_sets()
        assert len(result) == 3

    def test_returns_four_elements_with_target(self) -> None:
        """Should return tuple of 4 elements when include_target=True."""
        result = get_feature_sets(include_target=True)
        assert len(result) == 4

    def test_numeric_features_count(self) -> None:
        """Should return 7 numeric features."""
        numeric, _, _ = get_feature_sets()
        assert len(numeric) == 7

    def test_categorical_features_count(self) -> None:
        """Should return 4 categorical features."""
        _, categorical, _ = get_feature_sets()
        assert len(categorical) == 4

    def test_binary_features_count(self) -> None:
        """Should return 5 binary features."""
        _, _, binary = get_feature_sets()
        assert len(binary) == 5

    def test_target_name(self) -> None:
        """Should return correct target name."""
        _, _, _, target = get_feature_sets(include_target=True)
        assert target == "Dropout_Binary"

    def test_numeric_features_content(self) -> None:
        """Should contain expected numeric features."""
        numeric, _, _ = get_feature_sets()
        assert "Success_Rate_Sem1" in numeric
        assert "Avg_Grade" in numeric
        assert "Age at enrollment" in numeric

    def test_categorical_features_content(self) -> None:
        """Should contain expected categorical features."""
        _, categorical, _ = get_feature_sets()
        assert "Age_Group" in categorical
        assert "Course_Domain" in categorical

    def test_binary_features_content(self) -> None:
        """Should contain expected binary features."""
        _, _, binary = get_feature_sets()
        assert "Gender" in binary
        assert "Scholarship holder" in binary
