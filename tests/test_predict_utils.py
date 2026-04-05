"""Tests for prediction utility functions in predict.py."""

import json
import pytest

from parking_api.predict import _extract_recent_values


def _make_row(lot_values: dict) -> dict:
    """Build a fake parking_data row."""
    return {"data": lot_values, "created_at": "2026-04-04T19:00:00+00:00"}


def _make_rows(values: list[float], lot: str = "CRI") -> list[dict]:
    """Build a list of rows newest-first with sequential occupancy values."""
    return [_make_row({lot: v}) for v in values]


# ── Step-3 sampling (15-min interval reconstruction) ────────────────────────

def test_step3_picks_correct_indices():
    """step=3 means indices 0, 3, 6, 9 — matching 15-min cadence from 5-min rows."""
    rows = _make_rows([0.90, 0.85, 0.83, 0.80, 0.78, 0.75, 0.70, 0.65, 0.63, 0.60])
    result = _extract_recent_values(rows, "CRI", n=4)
    assert result == [0.90, 0.80, 0.70, 0.60]  # indices 0, 3, 6, 9


def test_step3_first_value_is_newest():
    rows = _make_rows([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    result = _extract_recent_values(rows, "CRI", n=4)
    assert result[0] == 1.0


def test_step3_four_values_needs_at_least_10_rows():
    """Indices 0, 3, 6, 9 require 10 rows for n=4."""
    rows = _make_rows([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    result = _extract_recent_values(rows, "CRI", n=4)
    assert len(result) == 4


def test_step3_fewer_rows_returns_what_is_available():
    """Only 7 rows: indices 0, 3, 6 available; index 9 missing → returns 3 values."""
    rows = _make_rows([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    result = _extract_recent_values(rows, "CRI", n=4)
    assert result == [0.9, 0.6, 0.3]


def test_step3_only_one_row():
    rows = _make_rows([0.5])
    result = _extract_recent_values(rows, "CRI", n=4)
    assert result == [0.5]


def test_step3_empty_rows():
    result = _extract_recent_values([], "CRI", n=4)
    assert result == []


# ── Lot lookup ───────────────────────────────────────────────────────────────

def test_lot_not_in_row_skipped():
    """Rows missing the requested lot should be skipped cleanly."""
    rows = [_make_row({"ED1": 0.5})] * 10  # no CRI
    result = _extract_recent_values(rows, "CRI", n=4)
    assert result == []


def test_lot_with_space():
    rows = _make_rows([0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01], lot="CD FS")
    result = _extract_recent_values(rows, "CD FS", n=2)
    assert result[0] == 0.7
    assert result[1] == 0.4  # index 3


def test_lot_with_slash():
    rows = _make_rows([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01], lot="ED2/3")
    result = _extract_recent_values(rows, "ED2/3", n=2)
    assert result[0] == 0.8
    assert result[1] == 0.5  # index 3


# ── Data formats ─────────────────────────────────────────────────────────────

def test_data_as_json_string():
    """Supabase can return JSONB data as a raw string — should be parsed."""
    rows = [{"data": json.dumps({"CRI": 0.6}), "created_at": "2026-04-04T19:00:00+00:00"}] * 10
    result = _extract_recent_values(rows, "CRI", n=1)
    assert result == [0.6]


def test_data_as_dict():
    rows = [{"data": {"CRI": 0.75}, "created_at": "2026-04-04T19:00:00+00:00"}] * 10
    result = _extract_recent_values(rows, "CRI", n=1)
    assert result == [0.75]


def test_values_cast_to_float():
    rows = [_make_row({"CRI": "0.55"})] * 10
    result = _extract_recent_values(rows, "CRI", n=1)
    assert result == [0.55]
    assert isinstance(result[0], float)
