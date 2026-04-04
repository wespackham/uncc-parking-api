"""Prediction loop: generates forecasts for all lots across all horizons.

Run as CLI: python -m parking_api.predict

Horizon strategy:
  T+30min, T+60min       → 30-min model (real lag features)
  T+90min to T+3hrs      → 60-min model (autoregressive: lags from prior predictions)
  T+3hrs to T+7days      → baseline model (no lags, 1-hour intervals)
"""

import json
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from .config import LOTS, DISCORD_WEBHOOK_URL, safe_name
from .models import ModelRegistry
from .features import build_feature_vector
from .weather import fetch_forecast_sync, get_weather_for_time
from .supabase_client import fetch_recent_rows, write_predictions


def _extract_recent_values(rows: list[dict], lot: str, n: int = 4) -> list[float]:
    """Extract the most recent N occupancy values for a lot from parking_data rows (newest-first)."""
    values = []
    for row in rows:
        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        val = data.get(lot)
        if val is not None:
            values.append(float(val))
        if len(values) >= n:
            break
    return values


def _send_discord_alert(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception:
        pass


def run_predictions():
    print("Loading models...")
    registry = ModelRegistry()

    print("Fetching recent parking data...")
    recent_rows = fetch_recent_rows(n=20)
    if not recent_rows:
        _send_discord_alert("⚠️ Prediction service: no recent parking data found")
        print("ERROR: No recent parking data. Aborting.")
        return

    # Check data freshness
    latest_time = pd.Timestamp(recent_rows[0]["created_at"])
    now = pd.Timestamp.now(tz="UTC")
    data_age_minutes = (now - latest_time).total_seconds() / 60
    stale = data_age_minutes > 15
    if stale:
        print(f"WARNING: Latest data is {data_age_minutes:.0f}min old. Using baseline models only.")

    print("Fetching weather forecast...")
    weather_df = fetch_forecast_sync()

    # Use EST for feature engineering (matching training data)
    now_est = datetime.now(timezone(timedelta(hours=-4)))

    predictions = []

    for lot in LOTS:
        sn = safe_name(lot)
        recent_values = _extract_recent_values(recent_rows, lot, n=4)

        # Track predicted values for autoregressive chaining
        predicted_chain = list(recent_values)  # newest-first

        # --- Near-term: 30-min model (T+30, T+60) ---
        if not stale:
            feat_names_30 = registry.get_feature_names(lot, "30min")
            for minutes_ahead in (30, 60):
                target_dt = now_est + timedelta(minutes=minutes_ahead)
                target_utc = (now + timedelta(minutes=minutes_ahead)).isoformat()
                weather_row = get_weather_for_time(weather_df, target_dt)
                X = build_feature_vector(
                    dt=target_dt,
                    weather_row=weather_row,
                    lot=lot,
                    recent_values=predicted_chain[:4],
                    feature_names=feat_names_30,
                )
                mean, low, high = registry.predict(lot, "30min", X)
                predictions.append({
                    "target_time": target_utc,
                    "lot": lot,
                    "model_tier": "30min",
                    "prediction": round(mean, 4),
                    "confidence_low": round(low, 4),
                    "confidence_high": round(high, 4),
                })
                # Prepend prediction for autoregressive chaining
                predicted_chain.insert(0, mean)

        # --- Medium-term: 60-min model (T+90 to T+180, every 15 min) ---
        if not stale:
            feat_names_60 = registry.get_feature_names(lot, "60min")
            for minutes_ahead in range(90, 181, 15):
                target_dt = now_est + timedelta(minutes=minutes_ahead)
                target_utc = (now + timedelta(minutes=minutes_ahead)).isoformat()
                weather_row = get_weather_for_time(weather_df, target_dt)
                X = build_feature_vector(
                    dt=target_dt,
                    weather_row=weather_row,
                    lot=lot,
                    recent_values=predicted_chain[:4],
                    feature_names=feat_names_60,
                )
                mean, low, high = registry.predict(lot, "60min", X)
                predictions.append({
                    "target_time": target_utc,
                    "lot": lot,
                    "model_tier": "60min",
                    "prediction": round(mean, 4),
                    "confidence_low": round(low, 4),
                    "confidence_high": round(high, 4),
                })
                predicted_chain.insert(0, mean)

        # --- Long-term: baseline model (T+3hrs to T+24hrs, hourly) ---
        feat_names_base = registry.get_feature_names(lot, "baseline")
        start_hours = 3 if not stale else 0
        for hours_ahead in range(start_hours, 24):
            target_dt = now_est + timedelta(hours=hours_ahead)
            target_utc = (now + timedelta(hours=hours_ahead)).isoformat()
            weather_row = get_weather_for_time(weather_df, target_dt)
            X = build_feature_vector(
                dt=target_dt,
                weather_row=weather_row,
                feature_names=feat_names_base,
            )
            mean, low, high = registry.predict(lot, "baseline", X)
            predictions.append({
                "target_time": target_utc,
                "lot": lot,
                "model_tier": "baseline",
                "prediction": round(mean, 4),
                "confidence_low": round(low, 4),
                "confidence_high": round(high, 4),
            })

    print(f"Generated {len(predictions)} predictions for {len(LOTS)} lots")

    print("Writing predictions to Supabase...")
    try:
        write_predictions(predictions)
        print("Done.")
    except Exception as e:
        msg = f"⚠️ Prediction service failed to write: {e}"
        print(f"ERROR: {msg}")
        _send_discord_alert(msg)


if __name__ == "__main__":
    run_predictions()
