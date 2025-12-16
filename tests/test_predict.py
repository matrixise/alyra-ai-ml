"""Tests for predict.py"""

import numpy as np
import pandas as pd
import pytest

from predict import classify_risk_level, get_feature_sets


class TestClassifyRiskLevel:
    """Tests for classify_risk_level function."""

    def test_low_risk_below_25_percent(self) -> None:
        """Should classify probability < 0.25 as 'Bas'."""
        probs = np.array([0.0, 0.1, 0.24])
        result = classify_risk_level(probs)
        assert result == ["Bas", "Bas", "Bas"]

    def test_medium_risk_25_to_50_percent(self) -> None:
        """Should classify probability 0.25-0.50 as 'Moyen'."""
        probs = np.array([0.25, 0.35, 0.49])
        result = classify_risk_level(probs)
        assert result == ["Moyen", "Moyen", "Moyen"]

    def test_high_risk_50_to_75_percent(self) -> None:
        """Should classify probability 0.50-0.75 as 'Élevé'."""
        probs = np.array([0.50, 0.60, 0.74])
        result = classify_risk_level(probs)
        assert result == ["Élevé", "Élevé", "Élevé"]

    def test_critical_risk_above_75_percent(self) -> None:
        """Should classify probability >= 0.75 as 'Critique'."""
        probs = np.array([0.75, 0.90, 1.0])
        result = classify_risk_level(probs)
        assert result == ["Critique", "Critique", "Critique"]

    def test_boundary_values(self) -> None:
        """Should handle exact boundary values correctly."""
        probs = np.array([0.25, 0.50, 0.75])
        result = classify_risk_level(probs)
        assert result == ["Moyen", "Élevé", "Critique"]

    def test_empty_array(self) -> None:
        """Should handle empty array."""
        probs = np.array([])
        result = classify_risk_level(probs)
        assert result == []

    def test_single_value(self) -> None:
        """Should handle single value array."""
        probs = np.array([0.42])
        result = classify_risk_level(probs)
        assert result == ["Moyen"]


class TestGetFeatureSets:
    """Tests for get_feature_sets function in predict module."""

    def test_returns_three_elements(self) -> None:
        """Should return tuple of 3 elements (no target)."""
        result = get_feature_sets()
        assert len(result) == 3

    def test_feature_lists_are_lists(self) -> None:
        """Should return lists of strings."""
        numeric, categorical, binary = get_feature_sets()
        assert isinstance(numeric, list)
        assert isinstance(categorical, list)
        assert isinstance(binary, list)

    def test_total_features_count(self) -> None:
        """Should return 16 total features."""
        numeric, categorical, binary = get_feature_sets()
        total = len(numeric) + len(categorical) + len(binary)
        assert total == 16

    def test_consistency_with_train_model(self) -> None:
        """Feature sets should match train_model.py (without target)."""
        from train_model import get_feature_sets as train_get_feature_sets

        predict_numeric, predict_cat, predict_bin = get_feature_sets()
        train_numeric, train_cat, train_bin, _ = train_get_feature_sets()

        assert predict_numeric == train_numeric
        assert predict_cat == train_cat
        assert predict_bin == train_bin
