"""FastAPI application entry point."""

from fastapi import FastAPI

from alyra_ai_ml.api.routes import router

app = FastAPI(
    title="Alyra AI/ML API",
    description="API for Student Dropout Prediction",
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Welcome to Alyra AI/ML API"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
