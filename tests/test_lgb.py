"""Tests for LightGBM prediction functions in predict.py."""

import json
import pickle
from datetime import datetime, timezone
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
    """10 lots × 36 horizons = 360 rows."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert len(result) == len(config["lots"]) * len(config["horizons"])


def test_lgb_all_predictions_in_unit_interval(lgb_models):
    """All prediction, confidence_low, confidence_high values must be in [0, 1]."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    for rec in result:
        assert 0.0 <= rec["prediction"]      <= 1.0, f"prediction out of range: {rec}"
        assert 0.0 <= rec["confidence_low"]  <= 1.0, f"confidence_low out of range: {rec}"
        assert 0.0 <= rec["confidence_high"] <= 1.0, f"confidence_high out of range: {rec}"


def test_lgb_model_tier_is_lgb(lgb_models):
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert all(r["model_tier"] == "lgb" for r in result)


def test_lgb_all_lots_present(lgb_models):
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    result_lots = {r["lot"] for r in result}
    assert result_lots == set(config["lots"])


def test_lgb_all_horizons_present(lgb_models):
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    # Each lot should cover all horizons
    cri_results = [r for r in result if r["lot"] == "CRI"]
    assert len(cri_results) == len(config["horizons"])


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
    """Predictions should be rounded to 4 decimal places."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows()
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    for rec in result:
        for key in ("prediction", "confidence_low", "confidence_high"):
            val = rec[key]
            assert val == round(val, 4), f"{key} not rounded to 4dp: {val}"


def test_lgb_works_with_minimal_rows(lgb_models):
    """Should not crash when only 1 row of parking data is available."""
    point, lower, upper, config = lgb_models
    now_utc = datetime(2026, 4, 8, 14, 0, 0, tzinfo=timezone.utc)
    rows = _make_rows(n=1)
    weather_df = _make_weather_df()

    result = _run_lgb_predictions(now_utc, rows, weather_df, point, lower, upper, config)
    assert len(result) == len(config["lots"]) * len(config["horizons"])


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
