"""Alyra AI/ML - Student Dropout Prediction."""

from alyra_ai_ml.constants import (
    DATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)
from alyra_ai_ml.features import engineer_features, get_feature_sets


__version__ = "0.1.0"

__all__ = [
    # Constants
    "DATA_PATH",
    "MODEL_PATH",
    "RANDOM_SEED",
    "TARGET_COLUMN",
    "TEST_SIZE",
    # Features
    "engineer_features",
    "get_feature_sets",
]
