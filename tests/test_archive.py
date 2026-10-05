"""Tests for the chunked weekly archive job: every row is saved before it is deleted."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from parking_api import archive


def _client_with_oldest(days_back: int) -> MagicMock:
    client = MagicMock()
    oldest = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"created_at": oldest}
    ]
    return client


def _rows(from_dt, n=3):
    return [{"id": f"{from_dt:%Y%m%d%H%M%S}-{i}", "created_at": from_dt.isoformat(),
             "target_time": from_dt.isoformat(), "model_tier": "lgb",
             "data": {"CRI": {"prediction": 0.5}}} for i in range(n)]


@pytest.fixture
def db(monkeypatch):
    """In-memory stand-in: chunk rows, the DB's count for each range, and recorded deletes."""
    state = {"extra_in_db": 0, "duplicate": False, "deleted": [], "fetched": {}}

    def fetch(_client, from_dt, to_dt):
        rows = _rows(from_dt)
        if state["duplicate"]:
            rows.append(dict(rows[0]))
        state["fetched"][from_dt] = rows
        return rows

    def count(_client, from_dt, to_dt):
        return len({r["id"] for r in state["fetched"][from_dt]}) + state["extra_in_db"]

    def delete(_client, ids):
        state["deleted"].extend(ids)
        return len(ids)

    monkeypatch.setattr(archive, "_fetch_chunk", fetch)
    monkeypatch.setattr(archive, "_count_range", count)
    monkeypatch.setattr(archive, "_delete_ids", delete)
    monkeypatch.setattr(archive, "_get_client", lambda: _client_with_oldest(9))
    return state


def test_deletes_exactly_the_saved_rows(tmp_path, db):
    archive.run_archive(days=7, archive_dir=tmp_path)

    saved = pd.concat(pd.read_parquet(f) for f in tmp_path.glob("*.parquet"))
    assert len(db["fetched"]) >= 2  # 2 days back, plus a sliver from clock drift
    assert sorted(db["deleted"]) == sorted(saved["id"])
    assert saved["id"].is_unique


def test_count_mismatch_aborts_before_deleting(tmp_path, db):
    db["extra_in_db"] = 1  # DB has a row the fetch didn't return (e.g. unstable paging)

    with pytest.raises(archive.ArchiveIntegrityError):
        archive.run_archive(days=7, archive_dir=tmp_path)

    assert db["deleted"] == []


def test_duplicate_rows_in_fetch_abort_before_deleting(tmp_path, db):
    db["duplicate"] = True

    with pytest.raises(archive.ArchiveIntegrityError):
        archive.run_archive(days=7, archive_dir=tmp_path)

    assert db["deleted"] == []


def test_corrupt_parquet_aborts_before_deleting(tmp_path, db, monkeypatch):
    def broken_verify(path, ids):
        raise archive.ArchiveIntegrityError("read back 0 rows")

    monkeypatch.setattr(archive, "_verify_parquet", broken_verify)

    with pytest.raises(archive.ArchiveIntegrityError):
        archive.run_archive(days=7, archive_dir=tmp_path)

    assert db["deleted"] == []


def test_verify_parquet_checks_ids(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame({"id": ["a", "b"]}).to_parquet(path, index=False)

    archive._verify_parquet(path, {"a", "b"})
    with pytest.raises(archive.ArchiveIntegrityError):
        archive._verify_parquet(path, {"a", "b", "c"})


def test_delete_ids_targets_ids_in_batches(monkeypatch):
    monkeypatch.setattr(archive, "DELETE_BATCH", 2)
    client = MagicMock()
    client.table.return_value.delete.return_value.in_.return_value.execute.return_value = MagicMock(count=2)

    deleted = archive._delete_ids(client, ["a", "b", "c", "d"])

    in_calls = client.table.return_value.delete.return_value.in_.call_args_list
    assert [c.args for c in in_calls] == [("id", ["a", "b"]), ("id", ["c", "d"])]
    assert deleted == 4
