"""Constantes partagées pour le projet de prédiction d'abandon scolaire."""

from pathlib import Path


# Reproductibilité
RANDOM_SEED = 42

# Split train/test
TEST_SIZE = 0.2

# Chemins par défaut
DATA_PATH = Path("data/dataset.csv")
MODEL_PATH = Path("models/dropout_predictor.pkl")

# Variable cible
TARGET_COLUMN = "Dropout_Binary"
