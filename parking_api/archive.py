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
import pyarrow.parquet as pq
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
DELETE_BATCH = 150  # ids per DELETE request (keeps the URL short)


class ArchiveIntegrityError(RuntimeError):
    """A chunk failed verification; nothing was deleted for it."""


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
            .order("id")  # stable paging: without an order, pages may skip/repeat rows
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        batch = result.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def _count_range(client, from_dt: datetime, to_dt: datetime) -> int:
    result = (
        client.table(TABLE)
        .select("id", count=CountMethod.exact, head=True)
        .gte("created_at", from_dt.isoformat())
        .lt("created_at", to_dt.isoformat())
        .execute()
    )
    return int(result.count or 0)


def _delete_ids(client, ids: list[str]) -> int:
    """Delete exactly these rows (never a time range), returning how many were deleted."""
    deleted = 0
    for i in range(0, len(ids), DELETE_BATCH):
        result = (
            client.table(TABLE)
            .delete(count=CountMethod.exact, returning=ReturnMethod.minimal)
            .in_("id", ids[i:i + DELETE_BATCH])
            .execute()
        )
        deleted += int(result.count or 0)
    return deleted


def _verify_parquet(path: Path, ids: set[str]) -> None:
    back = pq.read_table(path, columns=["id"])["id"].to_pylist()
    if len(back) != len(ids) or set(back) != ids:
        raise ArchiveIntegrityError(f"{path.name}: read back {len(back)} rows, expected {len(ids)}")


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

    # One chunk at a time. Every row must be saved before it is deleted:
    #   1. fetch the chunk (stable order) and check it matches the DB's own count
    #   2. write Parquet and read it back
    #   3. delete exactly the saved ids
    # Any mismatch raises before deleting, so the chunk stays in the DB for the next run.
    total = 0
    chunk_start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
    while chunk_start < cutoff:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), cutoff)
        rows = _fetch_chunk(client, chunk_start, chunk_end)
        if rows:
            ids = [row["id"] for row in rows]
            unique = set(ids)
            expected = _count_range(client, chunk_start, chunk_end)
            if len(unique) != len(ids) or len(unique) != expected:
                raise ArchiveIntegrityError(
                    f"{chunk_start} → {chunk_end}: fetched {len(ids)} rows ({len(unique)} unique), DB has {expected}"
                )
            df = pd.DataFrame(rows)
            df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True)
            df["target_time"] = pd.to_datetime(df["target_time"], format="ISO8601", utc=True)
            out_path = archive_dir / f"parking_predictions_{chunk_start:%Y-%m-%dT%H%M}_{chunk_end:%Y-%m-%dT%H%M}.parquet"
            df.to_parquet(out_path, index=False)
            _verify_parquet(out_path, unique)

            deleted = _delete_ids(client, sorted(unique))
            total += len(df)
            log.info("  %s → %s: saved %d rows to %s, deleted %d", chunk_start, chunk_end, len(df), out_path.name, deleted)
            if deleted != len(unique):
                # Every row is already saved; a lower count only means some ids were gone already.
                log.warning("  deleted %d of %d saved rows — check for concurrent deletes", deleted, len(unique))
        chunk_start = chunk_end

    log.info("Archived and deleted %d rows older than %s.", total, cutoff.date())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--archive-dir", default="/home/ubuntu/parking_exports")
    args = parser.parse_args()
    try:
        run_archive(days=args.days, archive_dir=Path(args.archive_dir))
    except Exception as exc:
        log.exception("Archive run failed")
        hook = os.environ.get("DISCORD_WEBHOOK_URL")
        label = os.environ.get("DISCORD_LABEL")
        if hook:
            import httpx

            msg = f"⚠️ Prediction archive failed (no unsaved rows were deleted): {exc}"
            httpx.post(hook, json={"content": f"**[{label}]** {msg}" if label else msg}, timeout=10)
        raise SystemExit(1)
