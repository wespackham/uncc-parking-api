"""Open-Meteo weather client for forecast and current conditions."""

import httpx
import pandas as pd

from .config import LAT, LON

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_weather_cache: pd.DataFrame | None = None


def _parse_hourly(data: dict) -> pd.DataFrame:
    hourly = data["hourly"]
    return pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_f": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "precipitation_in": hourly["precipitation"],
    })


async def fetch_forecast(hours_ahead: int = 168) -> pd.DataFrame:
    """Fetch 7-day hourly forecast from Open-Meteo. Returns DataFrame indexed by datetime."""
    global _weather_cache
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": 7,
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(FORECAST_URL, params=params)
            resp.raise_for_status()
        df = _parse_hourly(resp.json())
        _weather_cache = df
        return df
    except Exception:
        if _weather_cache is not None:
            return _weather_cache
        raise


def fetch_forecast_sync() -> pd.DataFrame:
    """Synchronous version for CLI usage."""
    global _weather_cache
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": 7,
    }
    try:
        resp = httpx.get(FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        df = _parse_hourly(resp.json())
        _weather_cache = df
        return df
    except Exception:
        if _weather_cache is not None:
            return _weather_cache
        raise


def get_weather_for_time(weather_df: pd.DataFrame, dt) -> dict:
    """Look up weather for the nearest hour to dt."""
    target = pd.Timestamp(dt).round("h")
    if target in weather_df["datetime"].values:
        row = weather_df[weather_df["datetime"] == target].iloc[0]
    else:
        idx = (weather_df["datetime"] - target).abs().idxmin()
        row = weather_df.iloc[idx]
    return {
        "temperature_f": float(row["temperature_f"]) if pd.notna(row["temperature_f"]) else 0.0,
        "humidity": float(row["humidity"]) if pd.notna(row["humidity"]) else 0.0,
        "precipitation_in": float(row["precipitation_in"]) if pd.notna(row["precipitation_in"]) else 0.0,
    }
