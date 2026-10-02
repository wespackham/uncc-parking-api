"""Weekly archive job: export predictions older than 7 days to Parquet, then delete them.

Run as CLI:
    python -m parking_api.archive
    python -m parking_api.archive --days 7 --archive-dir /home/ubuntu/parking_exports
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from postgrest.types import CountMethod, ReturnMethod
from supabase import create_client

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TABLE = "parking_predictions"
PAGE_SIZE = 1000
CHUNK_DAYS = 1


def _get_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


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
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def run_archive(days: int = 7, archive_dir: Path = Path("/home/ubuntu/parking_exports")):
    client = _get_client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Find oldest row to know how far back to fetch
    first = client.table(TABLE).select("created_at").order("created_at").limit(1).execute()
    if not first.data:
        log.info("Table is empty — nothing to archive.")
        return

    start = datetime.fromisoformat(first.data[0]["created_at"].replace("Z", "+00:00"))
    if start >= cutoff:
        log.info("No rows older than %d days — nothing to archive.", days)
        return

    log.info("Archiving rows from %s → %s", start.date(), cutoff.date())

    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    # One chunk at a time: write its Parquet file, then delete exactly that
    # range. Keeps memory bounded (~1 day of rows) and a failed run resumable.
    total = 0
    chunk_start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    while chunk_start < cutoff:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), cutoff)
        rows = _fetch_chunk(client, chunk_start, chunk_end)
        if rows:
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True)
            df["target_time"] = pd.to_datetime(df["target_time"], format="ISO8601", utc=True)
            out_path = archive_dir / f"parking_predictions_{chunk_start:%Y-%m-%dT%H%M}_{chunk_end:%Y-%m-%dT%H%M}.parquet"
            df.to_parquet(out_path, index=False)

            result = (
                client.table(TABLE)
                .delete(count=CountMethod.exact, returning=ReturnMethod.minimal)
                .gte("created_at", chunk_start.isoformat())
                .lt("created_at", chunk_end.isoformat())
                .execute()
            )
            total += len(df)
            log.info("  %s → %s: saved %d rows to %s, deleted %s", chunk_start, chunk_end, len(df), out_path.name, result.count)
        chunk_start = chunk_end

    log.info("Archived and deleted %d rows older than %s.", total, cutoff.date())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--archive-dir", default="/home/ubuntu/parking_exports")
    args = parser.parse_args()
    run_archive(days=args.days, archive_dir=Path(args.archive_dir))
