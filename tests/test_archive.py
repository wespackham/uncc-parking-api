"""Tests for the chunked weekly archive job."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd

from parking_api import archive


def test_archives_and_deletes_one_chunk_at_a_time(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    oldest = (now - timedelta(days=9)).isoformat()
    client = MagicMock()
    client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
        {"created_at": oldest}
    ]
    events = []

    def fake_fetch(_client, from_dt, to_dt):
        events.append(("fetch", from_dt))
        return [{"id": "x", "created_at": from_dt.isoformat(), "target_time": from_dt.isoformat(), "model_tier": "lgb", "data": {"CRI": {"prediction": 0.5}}}]

    deleter = client.table.return_value.delete.return_value.gte.return_value.lt.return_value
    deleter.execute.side_effect = lambda: events.append(("delete", None)) or MagicMock(count=1)
    monkeypatch.setattr(archive, "_get_client", lambda: client)
    monkeypatch.setattr(archive, "_fetch_chunk", fake_fetch)

    archive.run_archive(days=7, archive_dir=tmp_path)

    kinds = [k for k, _ in events]
    assert kinds == ["fetch", "delete"] * (len(kinds) // 2)
    chunks = len(kinds) // 2
    assert chunks >= 2  # 2 days back, plus a sliver from clock drift between test and job
    files = sorted(tmp_path.glob("*.parquet"))
    assert len(files) == chunks
    assert len(pd.read_parquet(files[0])) == 1
