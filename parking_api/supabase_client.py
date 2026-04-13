"""Supabase helpers for reading parking data and writing predictions."""

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
        .gt("created_at", "2020-01-01")
        .order("created_at", desc=True)
        .limit(n)
        .execute()
    )
    return result.data


def write_predictions(predictions: list[dict]):
    """Insert predictions, preserving all historical runs for accuracy comparison.

    Each prediction dict should have: target_time, lot, model_tier, prediction,
    confidence_low, confidence_high.
    """
    client = _get_client()

    # Batch insert in chunks of 500
    for i in range(0, len(predictions), 500):
        batch = predictions[i:i + 500]
        client.table(TABLE_PREDICTIONS).insert(batch).execute()


def fetch_predictions(lot: str | None = None, from_time: str | None = None, to_time: str | None = None) -> list[dict]:
    """Read predictions from parking_predictions table (denormalized JSONB schema).

    The table stores one row per (target_time, model_tier, run) with a JSONB
    ``data`` column containing all lots. This function explodes the JSONB into
    flat per-lot rows for backward compatibility with callers that expect
    (lot, prediction, confidence_low, confidence_high) dicts.
    """
    client = _get_client()
    query = client.table(TABLE_PREDICTIONS).select("created_at, target_time, model_tier, data").order("target_time")

    if from_time:
        query = query.gte("target_time", from_time)
    if to_time:
        query = query.lte("target_time", to_time)

    result = query.execute()

    # Explode JSONB → flat per-lot rows
    rows = []
    for row in result.data:
        data = row.get("data") or {}
        for lot_name, vals in data.items():
            if lot and lot_name != lot:
                continue
            rows.append({
                "created_at": row["created_at"],
                "target_time": row["target_time"],
                "model_tier": row["model_tier"],
                "lot": lot_name,
                "prediction": vals["prediction"],
                "confidence_low": vals["confidence_low"],
                "confidence_high": vals["confidence_high"],
            })

    return rows
