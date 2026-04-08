from datetime import datetime
from pydantic import BaseModel


class LotPrediction(BaseModel):
    lot: str
    target_time: datetime
    model_tier: str
    prediction: float
    confidence_low: float
    confidence_high: float


class PredictionResponse(BaseModel):
    generated_at: datetime
    count: int
    predictions: list[LotPrediction]


class HealthResponse(BaseModel):
    status: str
    csv_coverage: dict
