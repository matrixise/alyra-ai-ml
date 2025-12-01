"""CLI application using Typer."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="alyra-ml",
    help="Alyra AI/ML - Student Dropout Prediction CLI",
    add_completion=False,
)


@app.command()
def train(
    model_name: Annotated[
        str,
        typer.Option("--model", "-m", help="Model type to train"),
    ] = "decision_tree",
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output model name"),
    ] = Path("model"),
    test_size: Annotated[
        float,
        typer.Option("--test-size", "-t", help="Test set proportion"),
    ] = 0.2,
    random_state: Annotated[
        int,
        typer.Option("--seed", "-s", help="Random seed"),
    ] = 42,
) -> None:
    """Train a machine learning model."""
    typer.echo(f"Training {model_name} model...")
    typer.echo(f"Test size: {test_size}")
    typer.echo(f"Random state: {random_state}")

    # TODO: Implement actual training logic
    typer.echo("Training not yet implemented. Coming soon!")


@app.command()
def predict(
    model_path: Annotated[
        Path,
        typer.Argument(help="Path to the trained model"),
    ],
    input_file: Annotated[
        Path,
        typer.Option("--input", "-i", help="Input CSV file"),
    ] = Path("data/input.csv"),
) -> None:
    """Make predictions with a trained model."""
    typer.echo(f"Loading model from {model_path}...")
    typer.echo(f"Reading input from {input_file}...")

    # TODO: Implement actual prediction logic
    typer.echo("Prediction not yet implemented. Coming soon!")


@app.command()
def evaluate(
    model_path: Annotated[
        Path,
        typer.Argument(help="Path to the trained model"),
    ],
) -> None:
    """Evaluate a trained model."""
    typer.echo(f"Evaluating model {model_path}...")

    # TODO: Implement actual evaluation logic
    typer.echo("Evaluation not yet implemented. Coming soon!")


@app.command()
def api(
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind"),
    ] = 8000,
    reload: Annotated[
        bool,
        typer.Option("--reload", "-r", help="Enable auto-reload"),
    ] = False,
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    typer.echo(f"Starting API server on {host}:{port}...")
    uvicorn.run(
        "alyra_ai_ml.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
