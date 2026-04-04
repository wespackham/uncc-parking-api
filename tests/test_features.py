"""Verify feature engineering produces correct output for known inputs."""

import numpy as np
from datetime import datetime

from parking_api.features import (
    build_time_features,
    build_calendar_features,
    build_sports_features,
    build_disruption_features,
    build_weather_features,
    build_lag_features,
    build_feature_vector,
)


def test_time_features_weekday():
    # Wednesday March 25, 2026 at 10:30
    dt = datetime(2026, 3, 25, 10, 30)
    feats = build_time_features(dt)
    assert feats["hour"] == 10
    assert feats["minute"] == 30
    assert feats["day_of_week"] == 2  # Wednesday
    assert feats["is_weekend"] == 0
    assert np.isclose(feats["hour_sin"], np.sin(2 * np.pi * 10 / 24))
    assert np.isclose(feats["hour_cos"], np.cos(2 * np.pi * 10 / 24))


def test_time_features_weekend():
    dt = datetime(2026, 3, 28, 14, 0)  # Saturday
    feats = build_time_features(dt)
    assert feats["day_of_week"] == 5
    assert feats["is_weekend"] == 1


def test_calendar_features_weekend_override():
    # Even if CSV says is_class_day=1, weekends should be forced to 0
    feats = build_calendar_features("2026-03-28", is_weekend=1)
    assert feats["is_class_day"] == 0


def test_sports_features_no_game():
    feats = build_sports_features("2099-01-01")  # Far future, no games
    assert feats["home_game_count"] == 0
    assert feats["has_basketball"] == 0


def test_disruption_features_no_disruption():
    feats = build_disruption_features("2099-01-01")
    assert feats["condition_level"] == 0
    assert feats["is_remote"] == 0
    assert feats["is_cancelled"] == 0


def test_weather_features():
    weather = {"temperature_f": 65.0, "humidity": 70.0, "precipitation_in": 0.1}
    feats = build_weather_features(weather)
    assert feats["temperature_f"] == 65.0
    assert feats["humidity"] == 70.0
    assert feats["precipitation_in"] == 0.1


def test_weather_features_missing_keys():
    feats = build_weather_features({})
    assert feats["temperature_f"] == 0.0


def test_lag_features():
    values = [0.75, 0.70, 0.65, 0.60]  # newest-first
    feats = build_lag_features(values, "CRI")
    assert feats["CRI_now"] == 0.75
    assert feats["CRI_lag_5"] == 0.70
    assert feats["CRI_lag_10"] == 0.65
    assert feats["CRI_lag_15"] == 0.60
    assert np.isclose(feats["CRI_delta_5"], 0.05)


def test_lag_features_lot_with_space():
    values = [0.5, 0.4]
    feats = build_lag_features(values, "CD FS")
    assert "CD_FS_now" in feats
    assert feats["CD_FS_now"] == 0.5


def test_lag_features_partial():
    values = [0.8]  # only one value
    feats = build_lag_features(values, "ED1")
    assert feats["ED1_now"] == 0.8
    assert feats["ED1_lag_5"] == 0.8  # falls back to now
    assert feats["ED1_delta_5"] == 0.0


def test_feature_vector_baseline():
    baseline_features = [
        "hour", "minute", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_class_day", "is_break", "is_finals", "is_commencement", "is_holiday",
        "home_game_count", "has_basketball", "has_baseball",
        "has_softball", "has_lacrosse", "high_impact_game",
        "condition_level", "is_remote", "is_cancelled",
        "temperature_f", "humidity", "precipitation_in",
    ]
    dt = datetime(2026, 3, 25, 10, 30)
    weather = {"temperature_f": 60.0, "humidity": 50.0, "precipitation_in": 0.0}
    X = build_feature_vector(dt=dt, weather_row=weather, feature_names=baseline_features)
    assert list(X.columns) == baseline_features
    assert X.shape == (1, 25)
    assert X["hour"].iloc[0] == 10


def test_feature_vector_horizon():
    horizon_features = [
        "hour", "minute", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_class_day", "is_break", "is_finals", "is_commencement", "is_holiday",
        "home_game_count", "has_basketball", "has_baseball",
        "has_softball", "has_lacrosse", "high_impact_game",
        "condition_level", "is_remote", "is_cancelled",
        "temperature_f", "humidity", "precipitation_in",
        "CRI_now", "CRI_lag_5", "CRI_lag_10", "CRI_lag_15", "CRI_delta_5",
    ]
    dt = datetime(2026, 3, 25, 11, 0)
    weather = {"temperature_f": 60.0, "humidity": 50.0, "precipitation_in": 0.0}
    recent = [0.8, 0.75, 0.70, 0.65]
    X = build_feature_vector(
        dt=dt, weather_row=weather, lot="CRI",
        recent_values=recent, feature_names=horizon_features,
    )
    assert list(X.columns) == horizon_features
    assert X.shape == (1, 30)
    assert X["CRI_now"].iloc[0] == 0.8
