"""Supabase helpers for reading parking data and writing predictions."""

from datetime import datetime, timezone

from supabase import create_client

from .config import SUPABASE_URL, SUPABASE_KEY, TABLE_PARKING_DATA, TABLE_PREDICTIONS

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def fetch_recent_rows(n: int = 20) -> list[dict]:
    """Fetch the most recent N rows from parking_data, ordered newest-first."""
    client = _get_client()
    result = (
        client.table(TABLE_PARKING_DATA)
        .select("created_at, data")
        .order("created_at", desc=True)
        .limit(n)
        .execute()
    )
    return result.data


def write_predictions(predictions: list[dict]):
    """Delete existing future predictions and insert new ones.

    Each prediction dict should have: target_time, lot, model_tier, prediction,
    confidence_low, confidence_high.
    """
    client = _get_client()
    now = datetime.now(timezone.utc).isoformat()

    # Delete future predictions
    client.table(TABLE_PREDICTIONS).delete().gte("target_time", now).execute()

    # Batch insert in chunks of 500
    for i in range(0, len(predictions), 500):
        batch = predictions[i:i + 500]
        client.table(TABLE_PREDICTIONS).insert(batch).execute()


def fetch_predictions(lot: str | None = None, from_time: str | None = None, to_time: str | None = None) -> list[dict]:
    """Read predictions from parking_predictions table."""
    client = _get_client()
    query = client.table(TABLE_PREDICTIONS).select("*").order("target_time")

    if lot:
        query = query.eq("lot", lot)
    if from_time:
        query = query.gte("target_time", from_time)
    if to_time:
        query = query.lte("target_time", to_time)

    result = query.execute()
    return result.data
