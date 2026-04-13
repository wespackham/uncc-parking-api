"""Tests for LightGBM prediction functions in predict.py."""

import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parking_api.predict import _extract_lgb_deltas, _run_lgb_predictions
from parking_api.config import LGB_MODELS_DIR, LOTS

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_row(lot_values: dict) -> dict:
    return {"data": lot_values, "created_at": "2026-04-08T14:00:00+00:00"}


def _make_rows(n: int = 10, lot: str = "CRI", value: float = 0.5) -> list[dict]:
    """Build n identical rows newest-first, all lots set to `value`."""
    all_lots = {l: value for l in LOTS}
    return [_make_row(all_lots) for _ in range(n)]


def _make_weather_df() -> pd.DataFrame:
    """Minimal weather DataFrame matching what fetch_forecast_sync() returns.

    fetch_forecast_sync() strips timezone (parse_forecast strips tz at line 18 in weather.py),
    so datetimes here must also be tz-naive to match get_weather_for_time() comparisons.
    """
    now = pd.Timestamp.now().floor("h")  # tz-naive, matching parse_forecast output
    datetimes = pd.date_range(now, periods=48, freq="1h")
    return pd.DataFrame({
        "datetime": datetimes,
        "temperature_f": [65.0] * 48,
        "humidity": [55.0] * 48,
        "precipitation_in": [0.0] * 48,
    })


@pytest.fixture(scope="module")
def lgb_models():
    """Load the real LGB models; skip the test module if they're not on disk."""
    try:
        with open(LGB_MODELS_DIR / "lgb_point.pkl", "rb") as f:
            point = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_lower.pkl", "rb") as f:
            lower = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_upper.pkl", "rb") as f:
            upper = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_config.pkl", "rb") as f:
            config = pickle.load(f)
        return point, lower, upper, config
    except FileNotFoundError:
        pytest.skip("LGB model files not found in models_lgb/")


# ── _extract_lgb_deltas ───────────────────────────────────────────────────────

def test_deltas_exact_indices():
    """current=0.8, lag5=idx1=0.7, lag15=idx3=0.6, lag30=idx6=0.5."""
    rows = [_make_row({"CRI": v}) for v in [0.8, 0.7, 0.75, 0.6, 0.65, 0.62, 0.5]]
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CRI")
    assert cap == pytest.approx(0.8)
    assert d5  == pytest.approx(0.8 - 0.7)
    assert d15 == pytest.approx(0.8 - 0.6)
    assert d30 == pytest.approx(0.8 - 0.5)


def test_deltas_fallback_when_few_rows():
    """With only 2 rows, lag15 and lag30 fall back to lag5."""
    rows = [_make_row({"CRI": 0.6}), _make_row({"CRI": 0.5})]
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CRI")
    assert cap == pytest.approx(0.6)
    assert d5  == pytest.approx(0.1)   # 0.6 - 0.5
    assert d15 == pytest.approx(0.1)   # falls back to lag5
    assert d30 == pytest.approx(0.1)   # falls back to lag5


def test_deltas_single_row_all_deltas_zero():
    rows = [_make_row({"CRI": 0.7})]
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CRI")
    assert cap == pytest.approx(0.7)
    assert d5  == pytest.approx(0.0)
    assert d15 == pytest.approx(0.0)
    assert d30 == pytest.approx(0.0)


def test_deltas_empty_rows_returns_zeros():
    cap, d5, d15, d30 = _extract_lgb_deltas([], "CRI")
    assert cap == 0.0
    assert d5  == 0.0
    assert d15 == 0.0
    assert d30 == 0.0


def test_deltas_lot_missing_falls_back_to_zero():
    rows = [_make_row({"ED1": 0.5})] * 10  # no CRI
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CRI")
    assert cap == 0.0


def test_deltas_json_string_data():
    rows = [{"data": json.dumps({"CRI": 0.55}), "created_at": "2026-04-08T14:00:00+00:00"}] * 10
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CRI")
    assert cap == pytest.approx(0.55)
    assert d5 == 0.0  # all same value


def test_deltas_lot_with_space():
    rows = [_make_row({"CD FS": 0.4})] * 10
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "CD FS")
    assert cap == pytest.approx(0.4)


def test_deltas_lot_with_slash():
    rows = [_make_row({"ED2/3": 0.65})] * 10
    cap, d5, d15, d30 = _extract_lgb_deltas(rows, "ED2/3")
    assert cap == pytest.approx(0.65)


# ── _run_lgb_predictions ──────────────────────────────────────────────────────

def test_lgb_output_row_count(lgb_models):
    """One row per horizon (not per lot) — 36 rows for the 3h model."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert len(result) == len(config["horizons"])


def test_lgb_all_predictions_in_unit_interval(lgb_models):
    """All prediction, confidence_low, confidence_high values must be in [0, 1]."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    for rec in result:
        for lot_data in rec["data"].values():
            assert 0.0 <= lot_data["prediction"]      <= 1.0
            assert 0.0 <= lot_data["confidence_low"]  <= 1.0
            assert 0.0 <= lot_data["confidence_high"] <= 1.0


def test_lgb_model_tier_is_lgb(lgb_models):
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert all(r["model_tier"] == "lgb" for r in result)


def test_lgb_all_lots_present(lgb_models):
    """Every row's data dict must have all 10 lots as keys."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    expected_lots = set(config["lots"])
    for rec in result:
        assert set(rec["data"].keys()) == expected_lots, \
            f"Missing lots in {rec['target_time']}: {expected_lots - set(rec['data'].keys())}"


def test_lgb_all_horizons_present(lgb_models):
    """One unique target_time per horizon — 36 distinct target_times for the 3h model."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    unique_targets = {r["target_time"] for r in result}
    assert len(unique_targets) == len(config["horizons"])


def test_lgb_target_times_are_in_future(lgb_models):
    """target_time should be strictly after now_utc for all records."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    for rec in result:
        tgt = pd.Timestamp(rec["target_time"])
        assert tgt > pd.Timestamp(now_utc), f"target_time not in future: {rec}"


def test_lgb_values_are_rounded_to_4dp(lgb_models):
    """All numeric values in data must be rounded to 4 decimal places."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    for rec in result:
        for lot, lot_data in rec["data"].items():
            for key in ("prediction", "confidence_low", "confidence_high"):
                val = lot_data[key]
                assert val == round(val, 4), f"{lot}.{key} not rounded to 4dp: {val}"


def test_lgb_works_with_minimal_rows(lgb_models):
    """Should not crash when only 1 row of parking data is available."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows(n=1)
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert len(result) == len(config["horizons"])


# ── Model loading ─────────────────────────────────────────────────────────────

def test_lgb_model_files_exist():
    """All four LGB model files must be present."""
    for name in ("lgb_point.pkl", "lgb_lower.pkl", "lgb_upper.pkl", "lgb_config.pkl"):
        path = LGB_MODELS_DIR / name
        assert path.exists(), f"Missing model file: {path}"


def test_lgb_config_has_required_keys():
    """Config must contain lots, horizons, and features lists."""
    try:
        with open(LGB_MODELS_DIR / "lgb_config.pkl", "rb") as f:
            config = pickle.load(f)
    except FileNotFoundError:
        pytest.skip("lgb_config.pkl not found")
    assert "lots" in config
    assert "horizons" in config
    assert "features" in config
    assert len(config["lots"]) == 10
    assert len(config["horizons"]) == 36
    assert len(config["features"]) == 35


# ── JSON output format ───────────────────────────────────────────────────────

def test_lgb_output_has_required_top_level_keys(lgb_models):
    """Each row must have exactly target_time, model_tier, and data."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        assert set(rec.keys()) == {"target_time", "model_tier", "data"}, \
            f"Unexpected top-level keys: {rec.keys()}"


def test_lgb_output_no_lot_at_top_level(lgb_models):
    """The old 'lot' key must not appear at the top level of any row."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        assert "lot" not in rec, f"'lot' found at top level in: {rec}"


def test_lgb_output_no_flat_prediction_keys(lgb_models):
    """prediction/confidence_low/confidence_high must not appear at top level."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        for key in ("prediction", "confidence_low", "confidence_high"):
            assert key not in rec, f"'{key}' found at top level — should be inside data[lot]"


def test_lgb_data_field_is_dict(lgb_models):
    """data must be a plain dict (serialisable as JSONB)."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        assert isinstance(rec["data"], dict), f"data is not a dict: {type(rec['data'])}"


def test_lgb_lot_entry_has_required_keys(lgb_models):
    """Each lot entry inside data must have prediction, confidence_low, confidence_high."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    expected = {"prediction", "confidence_low", "confidence_high"}
    for rec in result:
        for lot, lot_data in rec["data"].items():
            assert set(lot_data.keys()) == expected, \
                f"{lot} entry has unexpected keys: {lot_data.keys()}"


def test_lgb_data_lot_count(lgb_models):
    """data must contain exactly 10 lots per row."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        assert len(rec["data"]) == 10, \
            f"Expected 10 lots in data, got {len(rec['data'])}: {list(rec['data'].keys())}"


def test_lgb_no_duplicate_target_times(lgb_models):
    """Each target_time must appear exactly once — no duplicate rows."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    target_times = [r["target_time"] for r in result]
    assert len(target_times) == len(set(target_times)), "Duplicate target_times in output"


def test_lgb_lot_names_match_config(lgb_models):
    """Lot keys in data must exactly match the lots list from config."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    expected_lots = set(config["lots"])
    for rec in result:
        assert set(rec["data"].keys()) == expected_lots


def test_lgb_row_count_is_horizons_not_lots_times_horizons(lgb_models):
    """Sanity-check the 10x row reduction: 36 rows, not 360."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    assert len(result) == len(config["horizons"])
    assert len(result) != len(config["lots"]) * len(config["horizons"])


# ── Target time snapping ──────────────────────────────────────────────────────

def test_lgb_target_times_all_multiples_of_5(lgb_models):
    """Every target_time must land on a :X0 or :X5 minute, second=0, microsecond=0."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 7, 32, 123456, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        tgt = pd.Timestamp(rec["target_time"])
        assert tgt.minute % 5 == 0,   f"minute not multiple of 5: {tgt}"
        assert tgt.second == 0,        f"second nonzero: {tgt}"
        assert tgt.microsecond == 0,   f"microsecond nonzero: {tgt}"


def test_lgb_target_snap_mid_interval(lgb_models):
    """At 14:07:32, base floors to 14:05:00; first target (h=5) must be 14:10:00."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 7, 32, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    unique_targets = sorted({pd.Timestamp(r["target_time"]) for r in result})
    assert unique_targets[0] == pd.Timestamp("2026-04-08T14:10:00+00:00")


def test_lgb_target_snap_on_boundary(lgb_models):
    """At exactly 14:05:00 (already on boundary), first target must be 14:10:00."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 5, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    unique_targets = sorted({pd.Timestamp(r["target_time"]) for r in result})
    assert unique_targets[0] == pd.Timestamp("2026-04-08T14:10:00+00:00")


def test_lgb_target_snap_top_of_hour(lgb_models):
    """At 14:00:00, base is 14:00:00; first target (h=5) must be 14:05:00."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    unique_targets = sorted({pd.Timestamp(r["target_time"]) for r in result})
    assert unique_targets[0] == pd.Timestamp("2026-04-08T14:05:00+00:00")


def test_lgb_target_snap_end_of_interval(lgb_models):
    """At 14:09:59, base floors to 14:05:00; first target must be 14:10:00."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 9, 59, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    unique_targets = sorted({pd.Timestamp(r["target_time"]) for r in result})
    assert unique_targets[0] == pd.Timestamp("2026-04-08T14:10:00+00:00")


def test_lgb_target_snap_with_seconds_and_microseconds(lgb_models):
    """Targets are clean even when now_utc has sub-minute noise."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 3, 47, 999999, tzinfo=timezone.utc)
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    for rec in result:
        tgt = pd.Timestamp(rec["target_time"])
        assert tgt.second == 0
        assert tgt.microsecond == 0


def test_lgb_target_snap_targets_cover_full_horizon_range(lgb_models):
    """Snapped targets should still cover all expected horizon offsets."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 7, 32, tzinfo=timezone.utc)
    base = datetime(2026, 4, 8, 14, 5, 0, tzinfo=timezone.utc)  # expected floor
    result = _run_lgb_predictions(now_utc, _make_rows(), _make_weather_df(),
                                  point, lower, upper, config)
    unique_targets = sorted({pd.Timestamp(r["target_time"]) for r in result})
    expected = sorted(
        pd.Timestamp(base + timedelta(minutes=h)) for h in config["horizons"]
    )
    assert unique_targets == expected
