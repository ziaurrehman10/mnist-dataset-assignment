"""
app/schemas/prediction.py
Request / response Pydantic models.
"""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    digit: int = Field(..., ge=0, le=9, description="Predicted digit (0–9)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability of the top class")
    probabilities: list[float] = Field(..., description="Softmax probabilities for all 10 classes")


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class ModelInfoResponse(BaseModel):
    input_shape: list
    output_shape: list
    total_params: int
    layers: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
