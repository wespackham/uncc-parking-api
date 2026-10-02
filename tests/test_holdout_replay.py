from datetime import date

import pandas as pd
import pytest

from parking_api.holdout_replay import (
    build_horizon_rows,
    explode_prediction_rows,
    local_window_bounds,
    rows_for_run,
)


def test_local_window_bounds_same_day_to_utc():
    bounds = local_window_bounds(date(2026, 4, 20), "06:00", "21:00")
    assert bounds.start_local.isoformat() == "2026-04-20T06:00:00-04:00"
    assert bounds.end_local.isoformat() == "2026-04-20T21:00:00-04:00"
    assert bounds.start_utc.isoformat() == "2026-04-20T10:00:00+00:00"
    assert bounds.end_utc.isoformat() == "2026-04-21T01:00:00+00:00"


def test_local_window_bounds_wraps_midnight():
    bounds = local_window_bounds(date(2026, 4, 20), "23:55", "00:10")
    assert bounds.start_local.isoformat() == "2026-04-20T23:55:00-04:00"
    assert bounds.end_local.isoformat() == "2026-04-21T00:10:00-04:00"


def test_rows_for_run_returns_newest_first():
    parking_rows = [
        {"created_at": "2026-04-20T10:00:00+00:00", "data": {"CRI": 0.10}},
        {"created_at": "2026-04-20T10:05:00+00:00", "data": {"CRI": 0.20}},
        {"created_at": "2026-04-20T10:10:00+00:00", "data": {"CRI": 0.30}},
        {"created_at": "2026-04-20T10:15:00+00:00", "data": {"CRI": 0.40}},
    ]
    parking_ts = [pd.Timestamp(row["created_at"]) for row in parking_rows]

    out = rows_for_run(parking_rows, parking_ts, pd.Timestamp("2026-04-20T10:12:00+00:00"), 3)

    assert [row["data"]["CRI"] for row in out] == [0.30, 0.20, 0.10]


def test_explode_prediction_rows_normalizes_created_at():
    raw_rows = [{
        "created_at": "2026-04-20T10:00:59+00:00",
        "target_time": "2026-04-20T10:05:00+00:00",
        "model_tier": "lgb",
        "data": {
            "CRI": {"prediction": 0.45, "confidence_low": 0.40, "confidence_high": 0.50},
        },
    }]

    df = explode_prediction_rows(raw_rows, normalize_created_at=True)

    assert len(df) == 1
    assert df.iloc[0]["created_at"] == pd.Timestamp("2026-04-20T10:00:00+00:00")


def test_build_horizon_rows_picks_lower_mae():
    matched = pd.DataFrame({
        "created_at": pd.to_datetime([
            "2026-04-20T10:00:00+00:00",
            "2026-04-20T10:00:00+00:00",
            "2026-04-20T10:00:00+00:00",
            "2026-04-20T10:00:00+00:00",
        ], utc=True),
        "target_time": pd.to_datetime([
            "2026-04-20T10:05:00+00:00",
            "2026-04-20T10:10:00+00:00",
            "2026-04-20T10:05:00+00:00",
            "2026-04-20T10:10:00+00:00",
        ], utc=True),
        "model_tier": ["lgb", "lgb", "lgb_v3_replay", "lgb_v3_replay"],
        "lot": ["CRI", "CRI", "CRI", "CRI"],
        "prediction": [0.50, 0.62, 0.48, 0.58],
        "confidence_low": [0.40, 0.52, 0.38, 0.48],
        "confidence_high": [0.60, 0.72, 0.58, 0.68],
        "actual": [0.55, 0.70, 0.50, 0.60],
    })

    rows = build_horizon_rows(matched, [("lgb", "lgb"), ("lgb_v3_replay", "v3_replay")], 5, 10)

    assert rows[0]["best"] == "v3_replay"
    assert rows[1]["best"] == "v3_replay"
