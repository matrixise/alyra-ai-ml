# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Read MEMORY.md First

**Always read `MEMORY.md` when starting a new session** to understand the conversation history, context, and decisions made in previous sessions.

**Periodically update `MEMORY.md`** during your session to preserve important context, decisions, and conversation history for future sessions.

## Project Overview

This is a Machine Learning project for predicting student dropout risk in higher education. The project uses Python 3.12 with scikit-learn for model training and Typer/Rich for CLI interfaces.

## Development Environment

- **Python Version**: 3.12.12 (managed via asdf, see `.tool-versions`)
- **Task Runner**: Task 3.45.5 (managed via asdf)
- **Version Manager**: asdf
- **Package Manager**: uv (fast Python package installer)
- **IDE**: PyCharm/IntelliJ IDEA with Python plugin

## Project Structure

```
.
├── data/                   # Datasets
│   └── dataset.csv         # Student retention dataset
├── models/                 # Trained ML models
│   └── dropout_predictor.pkl
├── notebooks/              # Jupyter notebooks for EDA
│   └── eda_minimal.ipynb
├── tests/                  # Unit tests
│   ├── conftest.py         # Pytest fixtures
│   ├── test_train_model.py
│   └── test_predict.py
├── docs/                   # Documentation
├── train_model.py          # Training script (Typer CLI)
├── predict.py              # Prediction script (Typer CLI)
├── pyproject.toml          # Project config and dependencies
├── Taskfile.yml            # Task runner definitions
└── uv.lock                 # Dependency lockfile
```

## Development Commands

### Setup

```bash
# Install correct tool versions
asdf install

# Create venv and install dependencies
task venv
task install
```

### Task Runner

This project uses [Task](https://taskfile.dev/) as a task runner.

```bash
# List available tasks
task --list
```

### ML Pipeline Tasks

```bash
# Train the dropout prediction model
task ml:train

# Train with custom parameters
task ml:train -- --test-size 0.3 --seed 123

# Run predictions on a CSV file
task ml:predict -- data/dataset.csv

# Save predictions to file
task ml:predict -- data/dataset.csv -o predictions.csv

# Run full pipeline (train + predict)
task ml:pipeline
```

### Testing

```bash
# Run all tests
task test

# Run specific test file
task test -- tests/test_train_model.py

# Run with coverage (if configured)
task test -- --cov
```

### Code Quality

```bash
# Format code
task format

# Run pre-commit hooks
task pre-commit-run
```

#### Task Conventions

- **Task descriptions**: Keep descriptions concise and focused on what the task does
- **Do NOT include usage examples** in the `desc` field
- Use `ml:` namespace for ML-related tasks

## Architecture Notes

### ML Pipeline

The project implements a binary classification pipeline for student dropout prediction:

1. **Feature Engineering** (`engineer_features()`):
   - Academic performance: `Success_Rate_Sem1/2`, `Avg_Grade`, `Total_Approved`, `Performance_Trend`
   - Demographics: `Age_Group`, `Marital_Binary`, `Education_Level`, `Course_Domain`

2. **Preprocessing** (`ColumnTransformer`):
   - `StandardScaler` for 7 numeric features
   - `OneHotEncoder` for 4 categorical features
   - Passthrough for 5 binary features

3. **Model**: Logistic Regression (scikit-learn Pipeline)

4. **Risk Classification** (`classify_risk_level()`):
   - Bas (< 25%), Moyen (25-50%), Élevé (50-75%), Critique (> 75%)

### CLI Architecture

Both scripts use Typer with `Annotated` types for modern CLI definition:
- `train_model.py`: Training with configurable data path, output, test size, seed
- `predict.py`: Prediction with input file, optional output, model path

### Key Files

| File | Purpose |
|------|---------|
| `train_model.py` | Training pipeline with feature engineering |
| `predict.py` | Inference script with risk level classification |
| `notebooks/eda_minimal.ipynb` | Exploratory data analysis |
| `data/dataset.csv` | Source dataset (semicolon-delimited) |
