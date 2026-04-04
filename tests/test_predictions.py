"""Smoke tests: load models, predict, verify output shape and range."""

import pytest
from datetime import datetime

from parking_api.models import ModelRegistry
from parking_api.features import build_feature_vector
from parking_api.config import LOTS


@pytest.fixture(scope="module")
def registry():
    return ModelRegistry()


def test_models_loaded(registry):
    info = registry.list_models()
    # 10 lots x 3 horizons = 30 models
    assert info["total"] == 30


def test_baseline_prediction_range(registry):
    feature_names = registry.get_feature_names("CRI", "baseline")
    dt = datetime(2026, 3, 25, 10, 30)
    weather = {"temperature_f": 60.0, "humidity": 50.0, "precipitation_in": 0.0}
    X = build_feature_vector(dt=dt, weather_row=weather, feature_names=feature_names)

    mean, low, high = registry.predict("CRI", "baseline", X)
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= low <= mean
    assert mean <= high <= 1.0


def test_horizon_prediction_range(registry):
    feature_names = registry.get_feature_names("CRI", "30min")
    dt = datetime(2026, 3, 25, 11, 0)
    weather = {"temperature_f": 60.0, "humidity": 50.0, "precipitation_in": 0.0}
    recent = [0.8, 0.75, 0.70, 0.65]
    X = build_feature_vector(
        dt=dt, weather_row=weather, lot="CRI",
        recent_values=recent, feature_names=feature_names,
    )

    mean, low, high = registry.predict("CRI", "30min", X)
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= low <= mean
    assert mean <= high <= 1.0


def test_all_lots_baseline(registry):
    """Ensure every lot's baseline model loads and predicts."""
    dt = datetime(2026, 3, 25, 14, 0)
    weather = {"temperature_f": 70.0, "humidity": 60.0, "precipitation_in": 0.0}

    for lot in LOTS:
        feature_names = registry.get_feature_names(lot, "baseline")
        X = build_feature_vector(dt=dt, weather_row=weather, feature_names=feature_names)
        mean, low, high = registry.predict(lot, "baseline", X)
        assert 0.0 <= mean <= 1.0, f"Failed for {lot}: mean={mean}"


def test_all_lots_30min(registry):
    """Ensure every lot's 30-min model loads and predicts."""
    dt = datetime(2026, 3, 25, 14, 0)
    weather = {"temperature_f": 70.0, "humidity": 60.0, "precipitation_in": 0.0}
    recent = [0.5, 0.5, 0.5, 0.5]

    for lot in LOTS:
        feature_names = registry.get_feature_names(lot, "30min")
        X = build_feature_vector(
            dt=dt, weather_row=weather, lot=lot,
            recent_values=recent, feature_names=feature_names,
        )
        mean, low, high = registry.predict(lot, "30min", X)
        assert 0.0 <= mean <= 1.0, f"Failed for {lot}: mean={mean}"
