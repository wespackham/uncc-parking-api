"""Verify feature engineering produces correct output for known inputs."""

import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from parking_api.features import (
    build_time_features,
    build_calendar_features,
    build_sports_features,
    build_disruption_features,
    build_weather_features,
    build_lag_features,
    build_feature_vector,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

MOCK_CAL     = {"is_class_day": 1, "is_break": 0, "is_finals": 0, "is_commencement": 0, "is_holiday": 0}
MOCK_SPORTS  = {"home_game_count": 0, "has_basketball": 0, "has_baseball": 0, "has_softball": 0, "has_lacrosse": 0, "high_impact_game": 0}
MOCK_DIS     = {"condition_level": 0, "is_remote": 0, "is_cancelled": 0}
MOCK_WEATHER = {"temperature_f": 65.0, "humidity": 50.0, "precipitation_in": 0.0}

BASELINE_FEATURE_NAMES = [
    "hour", "minute", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_class_day", "is_break", "is_finals", "is_commencement", "is_holiday",
    "home_game_count", "has_basketball", "has_baseball",
    "has_softball", "has_lacrosse", "high_impact_game",
    "condition_level", "is_remote", "is_cancelled",
    "temperature_f", "humidity", "precipitation_in",
]


@pytest.fixture(autouse=True)
def mock_enrichment():
    """Patch all CSV-dependent enrichment calls for every test in this file.
    Use side_effect (not return_value) so each call gets a fresh dict copy —
    build_calendar_features mutates the returned dict in-place for weekend override.
    """
    with patch("parking_api.features.get_calendar", side_effect=lambda *_: dict(MOCK_CAL)), \
         patch("parking_api.features.get_sports", side_effect=lambda *_: dict(MOCK_SPORTS)), \
         patch("parking_api.features.get_disruptions", side_effect=lambda *_: dict(MOCK_DIS)):
        yield


# ── Time features ────────────────────────────────────────────────────────────

def test_time_features_weekday():
    dt = datetime(2026, 3, 25, 10, 30)  # Wednesday
    f = build_time_features(dt)
    assert f["hour"] == 10
    assert f["minute"] == 30
    assert f["day_of_week"] == 2
    assert f["is_weekend"] == 0
    assert np.isclose(f["hour_sin"], np.sin(2 * np.pi * 10 / 24))
    assert np.isclose(f["hour_cos"], np.cos(2 * np.pi * 10 / 24))
    assert np.isclose(f["dow_sin"], np.sin(2 * np.pi * 2 / 7))
    assert np.isclose(f["dow_cos"], np.cos(2 * np.pi * 2 / 7))


def test_time_features_weekend_saturday():
    dt = datetime(2026, 3, 28, 14, 0)  # Saturday
    f = build_time_features(dt)
    assert f["day_of_week"] == 5
    assert f["is_weekend"] == 1


def test_time_features_weekend_sunday():
    dt = datetime(2026, 3, 29, 9, 0)  # Sunday
    f = build_time_features(dt)
    assert f["day_of_week"] == 6
    assert f["is_weekend"] == 1


def test_time_features_midnight():
    dt = datetime(2026, 3, 25, 0, 0)
    f = build_time_features(dt)
    assert f["hour"] == 0
    assert f["minute"] == 0
    assert np.isclose(f["hour_sin"], 0.0, atol=1e-10)
    assert np.isclose(f["hour_cos"], 1.0, atol=1e-10)


def test_time_features_end_of_day():
    dt = datetime(2026, 3, 25, 23, 59)
    f = build_time_features(dt)
    assert f["hour"] == 23
    assert f["minute"] == 59


def test_time_features_cyclical_range():
    for hour in range(24):
        dt = datetime(2026, 4, 1, hour, 0)
        f = build_time_features(dt)
        assert -1.0 <= f["hour_sin"] <= 1.0
        assert -1.0 <= f["hour_cos"] <= 1.0


def test_time_features_utc_aware():
    """UTC-aware datetime should use UTC hour — this is what training used."""
    dt = datetime(2026, 3, 25, 19, 0, tzinfo=timezone.utc)  # 19:00 UTC = 3pm EDT
    f = build_time_features(dt)
    assert f["hour"] == 19  # Must be 19 (UTC), not 15 (EDT)


# ── Calendar features ────────────────────────────────────────────────────────

def test_calendar_features_weekend_forces_no_class():
    f = build_calendar_features("2026-03-28", is_weekend=1)
    assert f["is_class_day"] == 0


def test_calendar_features_weekday_preserves_class():
    f = build_calendar_features("2026-03-25", is_weekend=0)
    assert f["is_class_day"] == 1  # from MOCK_CAL


def test_calendar_features_passthrough():
    f = build_calendar_features("2026-03-25", is_weekend=0)
    assert f["is_break"] == 0
    assert f["is_finals"] == 0
    assert f["is_commencement"] == 0
    assert f["is_holiday"] == 0


# ── Sports features ──────────────────────────────────────────────────────────

def test_sports_features_no_game():
    f = build_sports_features("2099-01-01")
    assert f["home_game_count"] == 0
    assert f["has_basketball"] == 0
    assert f["has_baseball"] == 0
    assert f["has_softball"] == 0
    assert f["has_lacrosse"] == 0
    assert f["high_impact_game"] == 0


# ── Disruption features ──────────────────────────────────────────────────────

def test_disruption_features_no_disruption():
    f = build_disruption_features("2099-01-01")
    assert f["condition_level"] == 0
    assert f["is_remote"] == 0
    assert f["is_cancelled"] == 0


# ── Weather features ─────────────────────────────────────────────────────────

def test_weather_features_full():
    f = build_weather_features({"temperature_f": 72.5, "humidity": 80.0, "precipitation_in": 0.25})
    assert f["temperature_f"] == 72.5
    assert f["humidity"] == 80.0
    assert f["precipitation_in"] == 0.25


def test_weather_features_empty_defaults_to_zero():
    f = build_weather_features({})
    assert f["temperature_f"] == 0.0
    assert f["humidity"] == 0.0
    assert f["precipitation_in"] == 0.0


def test_weather_features_partial():
    f = build_weather_features({"temperature_f": 55.0})
    assert f["temperature_f"] == 55.0
    assert f["humidity"] == 0.0


# ── Lag features ─────────────────────────────────────────────────────────────

def test_lag_features_all_values():
    f = build_lag_features([0.75, 0.70, 0.65, 0.60], "CRI")
    assert f["CRI_now"] == 0.75
    assert f["CRI_lag_5"] == 0.70
    assert f["CRI_lag_10"] == 0.65
    assert f["CRI_lag_15"] == 0.60
    assert np.isclose(f["CRI_delta_5"], 0.05)


def test_lag_features_decreasing():
    f = build_lag_features([0.40, 0.60, 0.70, 0.75], "ED1")
    assert np.isclose(f["ED1_delta_5"], -0.20)


def test_lag_features_one_value_fallback():
    f = build_lag_features([0.8], "ED1")
    assert f["ED1_now"] == 0.8
    assert f["ED1_lag_5"] == 0.8   # falls back to now
    assert f["ED1_lag_10"] == 0.8
    assert f["ED1_lag_15"] == 0.8
    assert f["ED1_delta_5"] == 0.0


def test_lag_features_two_values_fallback():
    f = build_lag_features([0.9, 0.7], "UDL")
    assert f["UDL_lag_10"] == 0.7  # falls back to lag_5
    assert f["UDL_lag_15"] == 0.7


def test_lag_features_empty_all_zero():
    f = build_lag_features([], "WEST")
    assert f["WEST_now"] == 0.0
    assert f["WEST_lag_5"] == 0.0
    assert f["WEST_delta_5"] == 0.0


def test_lag_features_lot_with_space():
    f = build_lag_features([0.5, 0.4], "CD FS")
    assert "CD_FS_now" in f
    assert f["CD_FS_now"] == 0.5


def test_lag_features_lot_with_slash():
    f = build_lag_features([0.3, 0.2], "ED2/3")
    assert "ED2_3_now" in f
    assert f["ED2_3_now"] == 0.3


def test_lag_features_zero_occupancy():
    f = build_lag_features([0.0, 0.0, 0.0, 0.0], "CRI")
    assert f["CRI_now"] == 0.0
    assert f["CRI_delta_5"] == 0.0


# ── build_feature_vector ─────────────────────────────────────────────────────

def test_feature_vector_baseline_shape_and_columns():
    dt = datetime(2026, 3, 25, 10, 30)
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER, feature_names=BASELINE_FEATURE_NAMES)
    assert list(X.columns) == BASELINE_FEATURE_NAMES
    assert X.shape == (1, 25)


def test_feature_vector_baseline_values():
    dt = datetime(2026, 3, 25, 10, 30)
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER, feature_names=BASELINE_FEATURE_NAMES)
    assert X["hour"].iloc[0] == 10
    assert X["minute"].iloc[0] == 30
    assert X["temperature_f"].iloc[0] == 65.0


def test_feature_vector_with_lags_shape():
    horizon_features = BASELINE_FEATURE_NAMES + ["CRI_now", "CRI_lag_5", "CRI_lag_10", "CRI_lag_15", "CRI_delta_5"]
    dt = datetime(2026, 3, 25, 11, 0)
    X = build_feature_vector(
        dt=dt, weather_row=MOCK_WEATHER, lot="CRI",
        recent_values=[0.8, 0.75, 0.70, 0.65],
        feature_names=horizon_features,
    )
    assert X.shape == (1, 30)
    assert X["CRI_now"].iloc[0] == 0.8
    assert X["CRI_lag_15"].iloc[0] == 0.65


def test_feature_vector_column_order_respected():
    """feature_names order must be preserved exactly — model expects specific column order."""
    reversed_features = list(reversed(BASELINE_FEATURE_NAMES))
    dt = datetime(2026, 3, 25, 10, 0)
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER, feature_names=reversed_features)
    assert list(X.columns) == reversed_features


def test_feature_vector_unknown_column_defaults_zero():
    """Unknown feature names should silently default to 0.0."""
    features_with_unknown = BASELINE_FEATURE_NAMES + ["nonexistent_feature"]
    dt = datetime(2026, 3, 25, 10, 0)
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER, feature_names=features_with_unknown)
    assert X["nonexistent_feature"].iloc[0] == 0.0


def test_feature_vector_no_feature_names_returns_all():
    dt = datetime(2026, 3, 25, 10, 0)
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER)
    assert "hour" in X.columns
    assert "temperature_f" in X.columns


def test_feature_vector_weekend_forces_no_class():
    dt = datetime(2026, 3, 28, 10, 0)  # Saturday
    X = build_feature_vector(dt=dt, weather_row=MOCK_WEATHER, feature_names=BASELINE_FEATURE_NAMES)
    assert X["is_weekend"].iloc[0] == 1
    assert X["is_class_day"].iloc[0] == 0
