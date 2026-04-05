"""Tests for ModelRegistry loading, prediction ranges, and confidence bands."""

import pytest
from datetime import datetime
from unittest.mock import patch

from parking_api.models import ModelRegistry
from parking_api.features import build_feature_vector
from parking_api.config import LOTS

MOCK_CAL    = {"is_class_day": 1, "is_break": 0, "is_finals": 0, "is_commencement": 0, "is_holiday": 0}
MOCK_SPORTS = {"home_game_count": 0, "has_basketball": 0, "has_baseball": 0, "has_softball": 0, "has_lacrosse": 0, "high_impact_game": 0}
MOCK_DIS    = {"condition_level": 0, "is_remote": 0, "is_cancelled": 0}
WEATHER     = {"temperature_f": 68.0, "humidity": 55.0, "precipitation_in": 0.0}

@pytest.fixture(autouse=True)
def mock_enrichment():
    # side_effect returns a fresh dict copy each call — build_calendar_features
    # mutates the returned dict in-place for weekend override.
    with patch("parking_api.features.get_calendar", side_effect=lambda *_: dict(MOCK_CAL)), \
         patch("parking_api.features.get_sports", side_effect=lambda *_: dict(MOCK_SPORTS)), \
         patch("parking_api.features.get_disruptions", side_effect=lambda *_: dict(MOCK_DIS)):
        yield


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry()


# ── Registry loading ─────────────────────────────────────────────────────────

def test_total_models_loaded(registry):
    assert registry.list_models()["total"] == 30


def test_all_lots_present(registry):
    loaded = {m["lot"] for m in registry.list_models()["models"]}
    assert loaded == set(LOTS)


def test_all_horizons_present(registry):
    loaded = {m["horizon"] for m in registry.list_models()["models"]}
    assert loaded == {"baseline", "30min", "60min"}


def test_feature_names_baseline_length(registry):
    names = registry.get_feature_names("CRI", "baseline")
    assert len(names) == 25


def test_feature_names_horizon_length(registry):
    for lot in LOTS:
        names_30 = registry.get_feature_names(lot, "30min")
        names_60 = registry.get_feature_names(lot, "60min")
        assert len(names_30) == 30, f"{lot} 30min: expected 30, got {len(names_30)}"
        assert len(names_60) == 30, f"{lot} 60min: expected 30, got {len(names_60)}"


def test_feature_names_horizon_contains_lot_lags(registry):
    from parking_api.config import safe_name
    for lot in LOTS:
        sn = safe_name(lot)
        names = registry.get_feature_names(lot, "30min")
        assert f"{sn}_now" in names
        assert f"{sn}_lag_5" in names
        assert f"{sn}_lag_15" in names
        assert f"{sn}_delta_5" in names


# ── Prediction range ─────────────────────────────────────────────────────────

def test_baseline_prediction_in_unit_range(registry):
    fn = registry.get_feature_names("CRI", "baseline")
    X = build_feature_vector(dt=datetime(2026, 3, 25, 10, 30), weather_row=WEATHER, feature_names=fn)
    mean, low, high = registry.predict("CRI", "baseline", X)
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_confidence_band_ordering(registry):
    """low <= mean <= high must always hold."""
    fn = registry.get_feature_names("CRI", "30min")
    X = build_feature_vector(
        dt=datetime(2026, 3, 25, 11, 0), weather_row=WEATHER,
        lot="CRI", recent_values=[0.7, 0.65, 0.60, 0.55], feature_names=fn,
    )
    mean, low, high = registry.predict("CRI", "30min", X)
    assert low <= mean <= high


def test_confidence_band_clamped(registry):
    """Bands must never exceed [0, 1] even for extreme lag inputs."""
    fn = registry.get_feature_names("ED1", "30min")
    X = build_feature_vector(
        dt=datetime(2026, 3, 25, 8, 0), weather_row=WEATHER,
        lot="ED1", recent_values=[1.0, 1.0, 1.0, 1.0], feature_names=fn,
    )
    mean, low, high = registry.predict("ED1", "30min", X)
    assert low >= 0.0
    assert high <= 1.0


def test_all_lots_baseline(registry):
    dt = datetime(2026, 3, 25, 14, 0)
    for lot in LOTS:
        fn = registry.get_feature_names(lot, "baseline")
        X = build_feature_vector(dt=dt, weather_row=WEATHER, feature_names=fn)
        mean, low, high = registry.predict(lot, "baseline", X)
        assert 0.0 <= mean <= 1.0, f"{lot} baseline out of range: {mean}"
        assert low <= mean <= high, f"{lot} baseline band ordering failed"


def test_all_lots_30min(registry):
    dt = datetime(2026, 3, 25, 14, 0)
    recent = [0.5, 0.5, 0.5, 0.5]
    for lot in LOTS:
        fn = registry.get_feature_names(lot, "30min")
        X = build_feature_vector(dt=dt, weather_row=WEATHER, lot=lot, recent_values=recent, feature_names=fn)
        mean, low, high = registry.predict(lot, "30min", X)
        assert 0.0 <= mean <= 1.0, f"{lot} 30min out of range: {mean}"
        assert low <= mean <= high, f"{lot} 30min band ordering failed"


def test_all_lots_60min(registry):
    dt = datetime(2026, 3, 25, 14, 0)
    recent = [0.5, 0.5, 0.5, 0.5]
    for lot in LOTS:
        fn = registry.get_feature_names(lot, "60min")
        X = build_feature_vector(dt=dt, weather_row=WEATHER, lot=lot, recent_values=recent, feature_names=fn)
        mean, low, high = registry.predict(lot, "60min", X)
        assert 0.0 <= mean <= 1.0, f"{lot} 60min out of range: {mean}"
        assert low <= mean <= high, f"{lot} 60min band ordering failed"


# ── Prediction sensitivity ────────────────────────────────────────────────────

def test_high_occupancy_lags_push_prediction_up(registry):
    """Full lags should produce higher prediction than empty lags for the same time."""
    dt = datetime(2026, 3, 25, 14, 0)
    fn = registry.get_feature_names("CRI", "30min")

    X_full = build_feature_vector(dt=dt, weather_row=WEATHER, lot="CRI",
                                   recent_values=[0.95, 0.95, 0.95, 0.95], feature_names=fn)
    X_empty = build_feature_vector(dt=dt, weather_row=WEATHER, lot="CRI",
                                    recent_values=[0.05, 0.05, 0.05, 0.05], feature_names=fn)
    mean_full, _, _ = registry.predict("CRI", "30min", X_full)
    mean_empty, _, _ = registry.predict("CRI", "30min", X_empty)
    assert mean_full > mean_empty, "High occupancy lags should predict higher than low occupancy lags"


def test_weekend_vs_weekday_baseline(registry):
    """Weekday midday (19:00 UTC = 3pm EDT) should predict higher occupancy than same weekend hour.
    Training data uses UTC hours — 19:00 UTC is afternoon peak, a clear weekday/weekend split.
    """
    fn = registry.get_feature_names("CRI", "baseline")
    weekday = datetime(2026, 3, 25, 19, 0)  # Wednesday 3pm EDT
    weekend = datetime(2026, 3, 28, 19, 0)  # Saturday 3pm EDT

    X_wd = build_feature_vector(dt=weekday, weather_row=WEATHER, feature_names=fn)
    X_we = build_feature_vector(dt=weekend, weather_row=WEATHER, feature_names=fn)
    mean_wd, _, _ = registry.predict("CRI", "baseline", X_wd)
    mean_we, _, _ = registry.predict("CRI", "baseline", X_we)
    # Availability ratio: 1=empty, 0=full. Weekday has more cars → lower availability.
    assert mean_we > mean_wd, (
        f"CRI weekend 3pm EDT ({mean_we:.3f}) should be more available than weekday ({mean_wd:.3f}) — campus lots nearly empty on weekends"
    )
