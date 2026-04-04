"""Feature engineering mirroring backtest.py exactly.

Operates on single timestamps instead of DataFrames.
Produces the same 25 baseline features + 5 lot-specific lag features for horizon models.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from .enrichment import get_calendar, get_sports, get_disruptions
from .config import safe_name


def build_time_features(dt: datetime) -> dict:
    hour = dt.hour
    minute = dt.minute
    day_of_week = dt.weekday()
    is_weekend = int(day_of_week >= 5)
    return {
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
    }


def build_calendar_features(date_str: str, is_weekend: int) -> dict:
    cal = get_calendar(date_str)
    if is_weekend:
        cal["is_class_day"] = 0
    return cal


def build_sports_features(date_str: str) -> dict:
    return get_sports(date_str)


def build_disruption_features(date_str: str) -> dict:
    return get_disruptions(date_str)


def build_weather_features(weather_row: dict) -> dict:
    # Planned features (not yet in trained models — add after retraining):
    #   wind_speed_mph   — from Open-Meteo windspeed_10m (convert km/h * 0.621371)
    #   cloud_cover      — from Open-Meteo cloudcover (0–100 int; could bucket into
    #                      0=clear, 1=partly_cloudy, 2=overcast for a categorical encoding)
    return {
        "temperature_f": weather_row.get("temperature_f", 0.0),
        "humidity": weather_row.get("humidity", 0.0),
        "precipitation_in": weather_row.get("precipitation_in", 0.0),
    }


def build_lag_features(recent_values: list[float], lot: str) -> dict:
    """Build lag features from the most recent occupancy values for a lot.

    recent_values should be ordered newest-first: [now, lag_5, lag_10, lag_15].
    These correspond to 5-minute intervals (the scraper's cadence).
    The model features are named {LOT}_now, {LOT}_lag_5, {LOT}_lag_10, {LOT}_lag_15, {LOT}_delta_5.
    """
    sn = safe_name(lot)

    now = recent_values[0] if len(recent_values) > 0 else 0.0
    lag_5 = recent_values[1] if len(recent_values) > 1 else now
    lag_10 = recent_values[2] if len(recent_values) > 2 else lag_5
    lag_15 = recent_values[3] if len(recent_values) > 3 else lag_10
    delta_5 = now - lag_5

    return {
        f"{sn}_now": now,
        f"{sn}_lag_5": lag_5,
        f"{sn}_lag_10": lag_10,
        f"{sn}_lag_15": lag_15,
        f"{sn}_delta_5": delta_5,
    }


def build_feature_vector(
    dt: datetime,
    weather_row: dict,
    lot: str | None = None,
    recent_values: list[float] | None = None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single-row DataFrame with features in the model-expected column order.

    For baseline models: lot and recent_values can be None.
    For horizon models: provide lot and recent_values (newest-first occupancy list).
    feature_names: the ordered list from the model's features.pkl or {LOT}_{horizon}_features.pkl.
    """
    date_str = dt.strftime("%Y-%m-%d")
    time_feats = build_time_features(dt)
    is_weekend = time_feats["is_weekend"]

    features = {}
    features.update(time_feats)
    features.update(build_calendar_features(date_str, is_weekend))
    features.update(build_sports_features(date_str))
    features.update(build_disruption_features(date_str))
    features.update(build_weather_features(weather_row))

    if lot is not None and recent_values is not None:
        features.update(build_lag_features(recent_values, lot))

    if feature_names is not None:
        row = {col: features.get(col, 0.0) for col in feature_names}
        return pd.DataFrame([row], columns=feature_names)

    return pd.DataFrame([features])
