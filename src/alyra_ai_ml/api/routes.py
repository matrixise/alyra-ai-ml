"""API routes for predictions."""

from fastapi import APIRouter, HTTPException

from alyra_ai_ml.api.schemas import PredictionRequest, PredictionResponse


router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a prediction for a student.

    Args:
        request: Student features

    Returns:
        Prediction result
    """
    # TODO: Implement actual prediction logic
    # This is a placeholder that will be implemented when the model is trained
    raise HTTPException(
        status_code=501,
        detail="Prediction endpoint not yet implemented. Train a model first.",
    )
