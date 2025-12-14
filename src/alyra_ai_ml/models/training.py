"""Model training module."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Default models directory
MODELS_DIR = Path(__file__).parent.parent.parent.parent.parent / "models"


def prepare_features(
    df: pd.DataFrame,
    target_column: str = "Target",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for training.

    Args:
        df: Input DataFrame
        target_column: Name of the target column

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def encode_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode target variable.

    Args:
        y: Target Series

    Returns:
        Tuple of (encoded target, fitted LabelEncoder)
    """
    encoder = LabelEncoder()
    y_encoded = pd.Series(encoder.fit_transform(y), index=y.index)
    return y_encoded, encoder


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """Train a scikit-learn model.

    Args:
        model: Scikit-learn model instance
        X_train: Training features
        y_train: Training target

    Returns:
        Fitted model
    """
    model.fit(X_train, y_train)
    return model


def save_model(
    model: BaseEstimator,
    name: str,
    models_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a trained model to disk.

    Args:
        model: Trained model
        name: Model name (without extension)
        models_dir: Directory to save the model. Defaults to models/
        metadata: Optional metadata to save with the model

    Returns:
        Path to the saved model
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"

    # Save model with metadata
    data = {"model": model}
    if metadata:
        data["metadata"] = metadata

    joblib.dump(data, model_path)
    return model_path


def load_model(
    name: str,
    models_dir: Path | None = None,
) -> tuple[BaseEstimator, dict[str, Any] | None]:
    """Load a trained model from disk.

    Args:
        name: Model name (without extension)
        models_dir: Directory containing the model. Defaults to models/

    Returns:
        Tuple of (model, metadata)
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    model_path = models_dir / f"{name}.joblib"
    data = joblib.load(model_path)

    if isinstance(data, dict):
        return data["model"], data.get("metadata")
    return data, None
