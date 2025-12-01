"""Model prediction module."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def predict(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Make predictions with a trained model.

    Args:
        model: Trained model
        X: Features to predict

    Returns:
        Predictions array
    """
    return model.predict(X)


def predict_proba(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """Get prediction probabilities.

    Args:
        model: Trained model (must support predict_proba)
        X: Features to predict

    Returns:
        Probability array of shape (n_samples, n_classes)
    """
    return model.predict_proba(X)


def predict_single(
    model: BaseEstimator,
    features: dict[str, Any],
    feature_columns: list[str],
) -> tuple[int, np.ndarray | None]:
    """Predict for a single sample from a dictionary of features.

    Args:
        model: Trained model
        features: Dictionary of feature values
        feature_columns: List of feature column names in expected order

    Returns:
        Tuple of (prediction, probabilities or None)
    """
    # Create DataFrame from single sample
    X = pd.DataFrame([features])[feature_columns]

    prediction = model.predict(X)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

    return prediction, probabilities
