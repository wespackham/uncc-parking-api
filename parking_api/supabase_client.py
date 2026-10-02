"""Supabase helpers for reading parking data and writing predictions."""

import json
import logging
from datetime import datetime, timedelta, timezone

from supabase import create_client

from .config import PREDICTIONS_BUFFER_FILE, SUPABASE_URL, SUPABASE_KEY, TABLE_PARKING_DATA, TABLE_PREDICTIONS

log = logging.getLogger(__name__)

WRITE_CHUNK_SIZE = 500

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


def _load_buffered_predictions() -> list[dict]:
    if not PREDICTIONS_BUFFER_FILE.exists():
        return []

    records = []
    for raw_line in PREDICTIONS_BUFFER_FILE.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping malformed buffered prediction line")
            continue
        if isinstance(record, dict) and "created_at" in record and "target_time" in record:
            records.append(record)
    return records


def _write_buffered_predictions(records: list[dict]) -> None:
    PREDICTIONS_BUFFER_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        if PREDICTIONS_BUFFER_FILE.exists():
            PREDICTIONS_BUFFER_FILE.unlink()
        return

    tmp_path = PREDICTIONS_BUFFER_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="ascii") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    tmp_path.replace(PREDICTIONS_BUFFER_FILE)


def _insert_predictions(batch: list[dict]) -> None:
    # Insert-or-skip on the natural key (uq_pred_run) so replays never duplicate.
    _get_client().table(TABLE_PREDICTIONS).upsert(
        batch, on_conflict="created_at,target_time,model_tier", ignore_duplicates=True
    ).execute()


def flush_buffered_predictions() -> tuple[int, int]:
    """Replay locally buffered predictions in order, in chunks.

    Returns (inserted_count, remaining_count).
    Stops at the first failed chunk to preserve ordering, and re-raises so the
    caller can alert. Unsent rows stay in the buffer for the next run.
    """
    records = _load_buffered_predictions()
    inserted = 0
    try:
        for i in range(0, len(records), WRITE_CHUNK_SIZE):
            _insert_predictions(records[i:i + WRITE_CHUNK_SIZE])
            inserted = min(i + WRITE_CHUNK_SIZE, len(records))
    finally:
        _write_buffered_predictions(records[inserted:])
    return inserted, len(records) - inserted


def write_predictions(predictions: list[dict]):
    """Insert predictions, preserving all historical runs for accuracy comparison.

    Each prediction dict should have: target_time, model_tier, data.

    Rows are stamped with created_at and appended to a local JSONL buffer before
    inserting, so a failed write is retried on the next run instead of lost.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    stamped = [{**p, "created_at": p.get("created_at") or created_at} for p in predictions]
    _write_buffered_predictions(_load_buffered_predictions() + stamped)
    inserted, _ = flush_buffered_predictions()
    if inserted > len(stamped):
        log.info("Replayed %s previously buffered prediction row(s)", inserted - len(stamped))


def delete_old_predictions(days: int = 7) -> int:
    """Delete predictions older than `days` days. Returns number of rows deleted."""
    client = _get_client()
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    result = client.table(TABLE_PREDICTIONS).delete().lt("created_at", cutoff).execute()
    return len(result.data) if result.data else 0


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
