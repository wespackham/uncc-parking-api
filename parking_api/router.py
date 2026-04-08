from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Header

from .schemas import PredictionResponse, HealthResponse, LotPrediction
from .supabase_client import fetch_predictions
from .enrichment import get_coverage
from .predict import run_predictions
from .config import API_KEY

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        csv_coverage=get_coverage(),
    )


@router.get("/predictions", response_model=PredictionResponse)
def get_predictions(lot: str | None = None, from_time: str | None = None, to_time: str | None = None):
    rows = fetch_predictions(lot=lot, from_time=from_time, to_time=to_time)
    return PredictionResponse(
        generated_at=datetime.now(timezone.utc),
        count=len(rows),
        predictions=[LotPrediction(**r) for r in rows],
    )


@router.post("/predict")
def trigger_predict(authorization: str = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API_KEY not configured")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    run_predictions()
    return {"status": "ok", "message": "Prediction run completed"}
