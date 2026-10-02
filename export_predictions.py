"""Export all rows from parking_predictions to a local Parquet file.

Usage:
    python export_predictions.py
    python export_predictions.py --out ~/Downloads/parking_predictions.parquet
    python export_predictions.py --format csv --out ~/Downloads/parking_predictions.csv

Fetches in 3-day chunks (no ORDER BY) to avoid Supabase statement timeouts.
Keeps the raw JSONB schema (one row per target_time × model_tier × run).
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TABLE = "parking_predictions"
PAGE_SIZE = 1000
CHUNK_DAYS = 3


def _fetch_chunk(client, from_dt: datetime, to_dt: datetime) -> list[dict]:
    rows = []
    offset = 0
    while True:
        result = (
            client.table(TABLE)
            .select("id, created_at, target_time, model_tier, data")
            .gte("created_at", from_dt.isoformat())
            .lt("created_at", to_dt.isoformat())
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        batch = result.data or []
        rows.extend(batch)
        print(f"  fetched {len(rows):,} rows (chunk {from_dt.date()} – {to_dt.date()})...", end="\r")
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def export(out_path: Path, fmt: str):
    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Find date range
    first = client.table(TABLE).select("created_at").order("created_at").limit(1).execute()
    last = client.table(TABLE).select("created_at").order("created_at", desc=True).limit(1).execute()

    if not first.data or not last.data:
        print("Table is empty.")
        return

    start = datetime.fromisoformat(first.data[0]["created_at"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(last.data[0]["created_at"].replace("Z", "+00:00")).replace(tzinfo=timezone.utc) + timedelta(seconds=1)

    print(f"Exporting {start.date()} → {end.date()} in {CHUNK_DAYS}-day chunks...")

    all_rows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end)
        chunk = _fetch_chunk(client, chunk_start, chunk_end)
        all_rows.extend(chunk)
        chunk_start = chunk_end

    print(f"\nTotal rows fetched: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], format="ISO8601", utc=True)

    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"Saved {len(df):,} rows → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="~/Downloads/parking_predictions.parquet")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()
    export(Path(args.out), args.format)
