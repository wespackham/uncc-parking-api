"""Open-Meteo weather client for forecast and current conditions."""

import logging
import time

import httpx
import pandas as pd

from .config import LAT, LON, WEATHER_CACHE_FILE, WEATHER_CACHE_MAX_AGE_HOURS

log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

_weather_cache: pd.DataFrame | None = None


def _parse_hourly(data: dict) -> pd.DataFrame:
    hourly = data["hourly"]
    datetimes = pd.to_datetime(hourly["time"])
    if datetimes.tz is not None:
        datetimes = datetimes.tz_convert(None)
    return pd.DataFrame({
        "datetime": datetimes,
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


def _save_disk_cache(df: pd.DataFrame) -> None:
    try:
        WEATHER_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = WEATHER_CACHE_FILE.with_suffix(".tmp")
        df.to_json(tmp, orient="records", date_format="iso")
        tmp.replace(WEATHER_CACHE_FILE)
    except Exception:
        log.warning("Could not write weather cache %s", WEATHER_CACHE_FILE, exc_info=True)


def _load_disk_cache() -> pd.DataFrame | None:
    """Last successful forecast, if it is recent enough to still cover the next day."""
    try:
        age_hours = (time.time() - WEATHER_CACHE_FILE.stat().st_mtime) / 3600
    except FileNotFoundError:
        return None
    if age_hours > WEATHER_CACHE_MAX_AGE_HOURS:
        log.warning("Weather cache is %.0f h old — too stale to use", age_hours)
        return None
    df = pd.read_json(WEATHER_CACHE_FILE, orient="records")
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    log.warning("Open-Meteo unavailable — using cached forecast from %.1f h ago", age_hours)
    return df


def fetch_forecast_sync(retries: int = 1, retry_delay: float = 3.0) -> pd.DataFrame:
    """Synchronous version for CLI usage.

    Each prediction run is a new process, so the fallback is a disk cache of the last
    successful forecast (an Open-Meteo outage otherwise aborts the whole run).
    """
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": 7,
    }
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = httpx.get(FORECAST_URL, params=params, timeout=15)
            resp.raise_for_status()
            df = _parse_hourly(resp.json())
            _save_disk_cache(df)
            return df
        except Exception as exc:
            last_exc = exc
            log.warning("Open-Meteo forecast attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries:
                time.sleep(retry_delay)
    cached = _load_disk_cache()
    if cached is not None:
        return cached
    raise last_exc


def get_weather_for_time(weather_df: pd.DataFrame, dt) -> dict:
    """Look up weather for the nearest hour to dt."""
    target = pd.Timestamp(dt).replace(tzinfo=None).round("h")
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
