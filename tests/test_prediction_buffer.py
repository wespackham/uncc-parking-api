"""Tests for buffer-and-retry in supabase_client.write_predictions."""

import json

import pytest

from parking_api import supabase_client


def _pred(target_time: str, tier: str = "lgb") -> dict:
    return {"target_time": target_time, "model_tier": tier, "data": {"CRI": {"prediction": 0.5}}}


@pytest.fixture
def buffer_file(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "pending_predictions.jsonl"
    monkeypatch.setattr(supabase_client, "PREDICTIONS_BUFFER_FILE", path)
    return path


@pytest.fixture
def inserts(monkeypatch):
    calls = []
    monkeypatch.setattr(supabase_client, "_insert_predictions", lambda batch: calls.append(list(batch)))
    return calls


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_success_stamps_created_at_and_clears_buffer(buffer_file, inserts):
    supabase_client.write_predictions([_pred("t1"), _pred("t2")])

    assert len(inserts) == 1
    rows = inserts[0]
    assert [r["target_time"] for r in rows] == ["t1", "t2"]
    assert rows[0]["created_at"] == rows[1]["created_at"]
    assert not buffer_file.exists()


def test_failure_keeps_rows_buffered_and_reraises(buffer_file, monkeypatch):
    def boom(batch):
        raise RuntimeError("db down")

    monkeypatch.setattr(supabase_client, "_insert_predictions", boom)

    with pytest.raises(RuntimeError):
        supabase_client.write_predictions([_pred("t1")])

    buffered = _read(buffer_file)
    assert [r["target_time"] for r in buffered] == ["t1"]
    assert buffered[0]["created_at"]


def test_next_run_replays_buffer_first_with_original_created_at(buffer_file, inserts):
    supabase_client._write_buffered_predictions([{**_pred("old"), "created_at": "2026-10-01T00:00:00+00:00"}])

    supabase_client.write_predictions([_pred("new")])

    rows = inserts[0]
    assert [r["target_time"] for r in rows] == ["old", "new"]
    assert rows[0]["created_at"] == "2026-10-01T00:00:00+00:00"
    assert rows[1]["created_at"] != rows[0]["created_at"]
    assert not buffer_file.exists()


def test_stops_at_first_failed_chunk_and_keeps_tail(buffer_file, monkeypatch):
    monkeypatch.setattr(supabase_client, "WRITE_CHUNK_SIZE", 2)
    sent = []

    def flaky(batch):
        if sent:
            raise RuntimeError("timeout")
        sent.append(list(batch))

    monkeypatch.setattr(supabase_client, "_insert_predictions", flaky)

    with pytest.raises(RuntimeError):
        supabase_client.write_predictions([_pred(f"t{i}") for i in range(5)])

    assert [r["target_time"] for r in sent[0]] == ["t0", "t1"]
    assert [r["target_time"] for r in _read(buffer_file)] == ["t2", "t3", "t4"]


def test_malformed_buffer_lines_are_skipped(buffer_file, inserts):
    buffer_file.parent.mkdir(parents=True)
    good = {**_pred("ok"), "created_at": "2026-10-01T00:00:00+00:00"}
    buffer_file.write_text("not json\n" + json.dumps(good) + "\n")

    supabase_client.write_predictions([])

    assert [r["target_time"] for r in inserts[0]] == ["ok"]


def test_insert_is_idempotent_upsert_on_natural_key(monkeypatch):
    from unittest.mock import MagicMock

    client = MagicMock()
    monkeypatch.setattr(supabase_client, "_get_client", lambda: client)

    supabase_client._insert_predictions([_pred("t1")])

    client.table.return_value.upsert.assert_called_once_with(
        [_pred("t1")], on_conflict="created_at,target_time,model_tier", ignore_duplicates=True
    )
