"""Open-Meteo outages fall back to the last cached forecast instead of aborting the run."""

import os
import time

import httpx
import pandas as pd
import pytest

from parking_api import weather

PAYLOAD = {"hourly": {"time": ["2026-10-03T14:00", "2026-10-03T15:00"],
                      "temperature_2m": [70.0, 71.0], "relative_humidity_2m": [50, 52],
                      "precipitation": [0.0, 0.1]}}


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return PAYLOAD


@pytest.fixture
def cache_file(tmp_path, monkeypatch):
    path = tmp_path / "weather_forecast.json"
    monkeypatch.setattr(weather, "WEATHER_CACHE_FILE", path)
    return path


def _down(*args, **kwargs):
    raise httpx.ReadTimeout("timed out")


def test_success_writes_cache_and_outage_reads_it(cache_file, monkeypatch):
    monkeypatch.setattr(weather.httpx, "get", lambda *a, **k: _Resp())
    fresh = weather.fetch_forecast_sync()
    assert cache_file.exists()

    monkeypatch.setattr(weather.httpx, "get", _down)
    cached = weather.fetch_forecast_sync(retry_delay=0)
    pd.testing.assert_frame_equal(cached.reset_index(drop=True), fresh.reset_index(drop=True), check_dtype=False)
    assert cached["datetime"].dt.tz is None


def test_outage_without_cache_raises(cache_file, monkeypatch):
    monkeypatch.setattr(weather.httpx, "get", _down)
    with pytest.raises(httpx.ReadTimeout):
        weather.fetch_forecast_sync(retry_delay=0)


def test_stale_cache_is_not_used(cache_file, monkeypatch):
    monkeypatch.setattr(weather.httpx, "get", lambda *a, **k: _Resp())
    weather.fetch_forecast_sync()
    old = time.time() - 72 * 3600
    os.utime(cache_file, (old, old))

    monkeypatch.setattr(weather.httpx, "get", _down)
    with pytest.raises(httpx.ReadTimeout):
        weather.fetch_forecast_sync(retry_delay=0)


def test_retry_succeeds_on_second_attempt(cache_file, monkeypatch):
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.HTTPStatusError("503", request=None, response=None)
        return _Resp()

    monkeypatch.setattr(weather.httpx, "get", flaky)
    df = weather.fetch_forecast_sync(retry_delay=0)
    assert len(calls) == 2 and len(df) == 2
