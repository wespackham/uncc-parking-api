"""Prediction loop: generates LightGBM forecasts for all lots.

Run as CLI:
    python -m parking_api.predict           # 3h model (T+5-180, every 5 min)
    python -m parking_api.predict --model 24h  # 24h model (T+5-1440, every hour)

3h model: 360 predictions per run (10 lots × 36 horizons), model_tier="lgb"
24h model: 2,880 predictions per run (10 lots × 288 horizons), model_tier="lgb_24h"

If parking data is stale (>15 min old), lag features fall back to 0 and the
model relies on calendar/weather/time features only.
"""

import argparse
import json
import logging
import pickle
import traceback
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd

from .config import LOTS, DISCORD_WEBHOOK_URL, LGB_MODELS_DIR, LGB_MODELS_V2_DIR
from .features import (build_time_features, build_calendar_features,
                        build_sports_features, build_disruption_features,
                        build_weather_features)
from .weather import fetch_forecast_sync, get_weather_for_time
from .supabase_client import fetch_recent_rows, write_predictions

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _extract_lgb_deltas(rows: list[dict], lot: str) -> tuple[float, float, float, float]:
    """Extract current capacity and deltas at 5, 15, 30 min for LightGBM inference.

    Uses indices 0 (now), 1 (5 min ago), 3 (15 min ago), 6 (30 min ago) from
    newest-first 5-min scraper rows. Falls back gracefully when rows are insufficient.

    Returns (current_capacity, delta_5, delta_15, delta_30).
    """
    def get_val(idx: int) -> float | None:
        if idx >= len(rows):
            return None
        data = rows[idx]["data"]
        if isinstance(data, str):
            data = json.loads(data)
        v = data.get(lot)
        return float(v) if v is not None else None

    now   = get_val(0) or 0.0
    lag5  = get_val(1);  lag5  = lag5  if lag5  is not None else now
    lag15 = get_val(3);  lag15 = lag15 if lag15 is not None else lag5
    lag30 = get_val(6);  lag30 = lag30 if lag30 is not None else lag15
    return now, now - lag5, now - lag15, now - lag30


def _run_lgb_predictions(
    now_utc: datetime,
    recent_rows: list[dict],
    weather_df: pd.DataFrame,
    lgb_point,
    lgb_lower,
    lgb_upper,
    lgb_config: dict,
    model_tier: str = "lgb",
) -> list[dict]:
    """Build inference batch (lots × horizons) and run all 3 LGB models.

    Target-time features (calendar, sports, weather, time encodings) are computed
    at t + horizon_minutes — matching how the model was trained.
    """
    lots     = lgb_config["lots"]
    horizons = lgb_config["horizons"]
    features = lgb_config["features"]

    # Current time context (same for all rows)
    cur_dow = now_utc.weekday()
    cur_time_feats = {
        "cur_hour_sin":  np.sin(2 * np.pi * now_utc.hour / 24),
        "cur_hour_cos":  np.cos(2 * np.pi * now_utc.hour / 24),
        "cur_dow_sin":   np.sin(2 * np.pi * cur_dow / 7),
        "cur_dow_cos":   np.cos(2 * np.pi * cur_dow / 7),
        "cur_is_weekend": int(cur_dow >= 5),
    }

    # Pre-extract per-lot current state (avoids repeated JSON parsing)
    lot_state = {lot: _extract_lgb_deltas(recent_rows, lot) for lot in lots}

    rows = []
    meta = []  # (lot, target_utc) for assembling prediction records

    for h in horizons:
        target_dt  = now_utc + timedelta(minutes=h)
        target_utc = target_dt.isoformat()
        tgt_date   = target_dt.strftime("%Y-%m-%d")

        # Target-time features — computed once per horizon, shared across all lots
        tgt_time      = build_time_features(target_dt)
        tgt_is_weekend = tgt_time["is_weekend"]
        tgt_cal        = build_calendar_features(tgt_date, tgt_is_weekend)
        tgt_sports     = build_sports_features(tgt_date)
        tgt_dis        = build_disruption_features(tgt_date)
        tgt_wthr       = build_weather_features(get_weather_for_time(weather_df, target_dt))

        tgt_feats = {
            "tgt_hour_sin":        tgt_time["hour_sin"],
            "tgt_hour_cos":        tgt_time["hour_cos"],
            "tgt_minute_sin":      tgt_time["minute_sin"],
            "tgt_minute_cos":      tgt_time["minute_cos"],
            "tgt_dow_sin":         tgt_time["dow_sin"],
            "tgt_dow_cos":         tgt_time["dow_cos"],
            "tgt_is_weekend":      tgt_is_weekend,
            "tgt_is_class_day":    tgt_cal["is_class_day"],
            "tgt_is_break":        tgt_cal.get("is_break", 0),
            "tgt_is_finals":       tgt_cal.get("is_finals", 0),
            "tgt_is_commencement": tgt_cal.get("is_commencement", 0),
            "tgt_is_holiday":      tgt_cal.get("is_holiday", 0),
            "tgt_home_game_count":  tgt_sports.get("home_game_count", 0),
            "tgt_has_basketball":   tgt_sports.get("has_basketball", 0),
            "tgt_has_baseball":     tgt_sports.get("has_baseball", 0),
            "tgt_has_softball":     tgt_sports.get("has_softball", 0),
            "tgt_has_lacrosse":     tgt_sports.get("has_lacrosse", 0),
            "tgt_high_impact_game": tgt_sports.get("high_impact_game", 0),
            "tgt_condition_level":  tgt_dis.get("condition_level", 0),
            "tgt_is_remote":        tgt_dis.get("is_remote", 0),
            "tgt_is_cancelled":     tgt_dis.get("is_cancelled", 0),
            "tgt_temperature_f":    tgt_wthr.get("temperature_f", 65.0),
            "tgt_humidity":         tgt_wthr.get("humidity", 55.0),
            "tgt_precipitation_in": tgt_wthr.get("precipitation_in", 0.0),
        }

        for lot in lots:
            cap, d5, d15, d30 = lot_state[lot]
            rows.append({
                "current_capacity": cap,
                "delta_5":  d5,
                "delta_15": d15,
                "delta_30": d30,
                **cur_time_feats,
                "horizon_minutes": h,
                "deck_id": lot,
                **tgt_feats,
            })
            meta.append((lot, target_utc))

    X = pd.DataFrame(rows)[features]
    X["deck_id"] = pd.Categorical(X["deck_id"], categories=lots)

    preds = lgb_point.predict(X).clip(0, 1)
    lows  = lgb_lower.predict(X).clip(0, 1)
    highs = lgb_upper.predict(X).clip(0, 1)

    return [
        {
            "target_time":     target_utc,
            "lot":             lot,
            "model_tier":      model_tier,
            "prediction":      round(float(preds[i]), 4),
            "confidence_low":  round(float(lows[i]),  4),
            "confidence_high": round(float(highs[i]), 4),
        }
        for i, (lot, target_utc) in enumerate(meta)
    ]


def _send_discord_alert(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception:
        pass


def run_predictions(model: str = "3h"):
    models_dir = LGB_MODELS_V2_DIR if model == "24h" else LGB_MODELS_DIR
    model_tier = "lgb_24h" if model == "24h" else "lgb"
    log.info(f"Loading LGB models from {models_dir} (tier={model_tier})...")
    try:
        with open(models_dir / "lgb_point.pkl", "rb") as f:
            lgb_point = pickle.load(f)
        with open(models_dir / "lgb_lower.pkl", "rb") as f:
            lgb_lower = pickle.load(f)
        with open(models_dir / "lgb_upper.pkl", "rb") as f:
            lgb_upper = pickle.load(f)
        with open(models_dir / "lgb_config.pkl", "rb") as f:
            lgb_config = pickle.load(f)
        log.info("LGB models loaded.")
    except Exception:
        msg = f"⚠️ Prediction service: failed to load LGB {model} models"
        log.error(f"{msg}\n{traceback.format_exc()}")
        _send_discord_alert(msg)
        return

    log.info("Fetching recent parking data...")
    recent_rows = fetch_recent_rows(n=20)
    if not recent_rows:
        _send_discord_alert("⚠️ Prediction service: no recent parking data found")
        log.error("No recent parking data. Aborting.")
        return
    log.info(f"Got {len(recent_rows)} recent rows. Latest: {recent_rows[0]['created_at']}")

    latest_time = pd.Timestamp(recent_rows[0]["created_at"])
    now = pd.Timestamp.now(tz="UTC")
    data_age_minutes = (now - latest_time).total_seconds() / 60
    if data_age_minutes > 15:
        log.warning(f"Data is {data_age_minutes:.1f} min old — lag features will be zeroed")

    log.info("Fetching weather forecast...")
    weather_df = fetch_forecast_sync()

    now_utc = datetime.now(timezone.utc)

    try:
        predictions = _run_lgb_predictions(
            now_utc, recent_rows, weather_df,
            lgb_point, lgb_lower, lgb_upper, lgb_config,
            model_tier=model_tier,
        )
        log.info(f"Generated {len(predictions)} predictions")
    except Exception:
        msg = f"⚠️ LGB {model} prediction run failed"
        log.error(f"{msg}\n{traceback.format_exc()}")
        _send_discord_alert(msg)
        return

    log.info("Writing predictions to Supabase...")
    try:
        write_predictions(predictions)
        log.info("Done.")
    except Exception as e:
        msg = f"⚠️ Prediction service failed to write: {e}"
        log.error(msg)
        _send_discord_alert(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["3h", "24h"], default="3h",
                        help="Which model to run: 3h (T+5-180, every 5 min) or 24h (T+5-1440, every hour)")
    args = parser.parse_args()
    run_predictions(model=args.model)
