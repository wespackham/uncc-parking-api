"""Prediction loop: generates forecasts for all lots across all horizons.

Run as CLI: python -m parking_api.predict

Produces a prediction every 5 minutes from T+5 to T+360 (6 hours) per lot.

V1 horizon strategy (30 models):
  T+5  to T+60   (every 5 min) → 30-min model (autoregressive lag features)
  T+65 to T+180  (every 5 min) → 60-min model (autoregressive lag features)
  T+185 to T+360 (every 5 min) → baseline model (calendar + weather, no lags)
  When stale: baseline only, T+5 to T+360.

V2 shadow (runs alongside V1, tiers suffixed _v2):
  T+5  to T+60   (every 5 min) → 30min_v2  (true 5-min lags, minute_sin/cos)
  T+65 to T+360  (every 5 min) → baseline_v2
  When stale: baseline_v2 only, T+5 to T+360.
"""

import json
import logging
import pickle
import traceback
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
import pandas as pd

from .config import LOTS, DISCORD_WEBHOOK_URL, MODELS_V2_DIR, LGB_MODELS_DIR, safe_name
from .models import ModelRegistry
from .features import (build_feature_vector, build_time_features,
                        build_calendar_features, build_sports_features,
                        build_disruption_features, build_weather_features)
from .weather import fetch_forecast_sync, get_weather_for_time
from .supabase_client import fetch_recent_rows, write_predictions

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _extract_recent_values(rows: list[dict], lot: str, n: int = 4, step: int = 3) -> list[float]:
    """Extract N occupancy values spaced `step` rows apart from parking_data rows (newest-first).

    V1 (step=3): reconstructs 15-min lag spacing from 5-min scraper rows.
      index 0 = now, index 3 = 15 min ago, index 6 = 30 min ago, index 9 = 45 min ago.
      Requires fetch_recent_rows(n >= 10).

    V2 (step=1): true 5-min lags matching how v2 models were trained.
      index 0 = now, index 1 = 5 min ago, index 2 = 10 min ago, index 3 = 15 min ago.
    """
    values = []
    for i in range(n):
        idx = i * step
        if idx >= len(rows):
            break
        data = rows[idx]["data"]
        if isinstance(data, str):
            data = json.loads(data)
        val = data.get(lot)
        if val is not None:
            values.append(float(val))
    return values


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
) -> list[dict]:
    """Build 360-row inference batch (10 lots × 36 horizons) and run all 3 LGB models.

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
            "model_tier":      "lgb",
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


def run_predictions():
    log.info("Loading models...")
    registry = ModelRegistry()
    registry_v2 = ModelRegistry(MODELS_V2_DIR)
    log.info(f"V1 models loaded: {registry.list_models()['total']} entries")
    log.info(f"V2 models loaded: {registry_v2.list_models()['total']} entries")

    # Load LGB models (optional — skip gracefully if not present)
    lgb_loaded = False
    lgb_point = lgb_lower = lgb_upper = lgb_config = None
    try:
        with open(LGB_MODELS_DIR / "lgb_point.pkl", "rb") as f:
            lgb_point = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_lower.pkl", "rb") as f:
            lgb_lower = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_upper.pkl", "rb") as f:
            lgb_upper = pickle.load(f)
        with open(LGB_MODELS_DIR / "lgb_config.pkl", "rb") as f:
            lgb_config = pickle.load(f)
        lgb_loaded = True
        log.info("LGB models loaded.")
    except Exception:
        log.warning(f"LGB models not found or failed to load — skipping lgb predictions:\n{traceback.format_exc()}")

    log.info("Fetching recent parking data...")
    recent_rows = fetch_recent_rows(n=20)
    if not recent_rows:
        _send_discord_alert("⚠️ Prediction service: no recent parking data found")
        log.error("No recent parking data. Aborting.")
        return
    log.info(f"Got {len(recent_rows)} recent rows. Latest: {recent_rows[0]['created_at']}")

    # Check data freshness
    latest_time = pd.Timestamp(recent_rows[0]["created_at"])
    now = pd.Timestamp.now(tz="UTC")
    data_age_minutes = (now - latest_time).total_seconds() / 60
    stale = data_age_minutes > 15
    log.info(f"Data age: {data_age_minutes:.1f} min — {'STALE, baseline only' if stale else 'fresh'}")

    log.info("Fetching weather forecast...")
    weather_df = fetch_forecast_sync()
    log.info(f"Weather rows: {len(weather_df)}, datetime dtype: {weather_df['datetime'].dtype}, "
             f"tz: {weather_df['datetime'].dt.tz}, "
             f"sample: {weather_df['datetime'].iloc[0]}")

    # Use UTC for feature engineering — training data (backtest.py) derives hour/day_of_week
    # from created_at which Supabase returns as UTC. Models learned UTC-indexed time patterns.
    now_utc = datetime.now(timezone.utc)
    log.info(f"now_utc: {now_utc} (tzinfo={now_utc.tzinfo})")

    predictions = []

    for lot in LOTS:
        log.debug(f"Processing lot: {lot}")
        recent_values_v1 = _extract_recent_values(recent_rows, lot, n=4, step=3)
        recent_values_v2 = _extract_recent_values(recent_rows, lot, n=4, step=1)
        predicted_chain = list(recent_values_v1)

        # ── V1 predictions ────────────────────────────────────────────────────
        try:
            # --- Near-term: 30-min model (T+5 to T+60, every 5 min) ---
            if not stale:
                feat_names_30 = registry.get_feature_names(lot, "30min")
                for minutes_ahead in range(5, 61, 5):
                    target_dt = now_utc + timedelta(minutes=minutes_ahead)
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
                    predicted_chain.insert(0, mean)

            # --- Medium-term: 60-min model (T+65 to T+180, every 5 min) ---
            if not stale:
                feat_names_60 = registry.get_feature_names(lot, "60min")
                for minutes_ahead in range(65, 181, 5):
                    target_dt = now_utc + timedelta(minutes=minutes_ahead)
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

            # --- Long-term: baseline model (T+185 to T+360, every 5 min) ---
            feat_names_base = registry.get_feature_names(lot, "baseline")
            start_minutes = 185 if not stale else 5
            for minutes_ahead in range(start_minutes, 361, 5):
                target_dt = now_utc + timedelta(minutes=minutes_ahead)
                target_utc = (now + timedelta(minutes=minutes_ahead)).isoformat()
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

        except Exception:
            log.error(f"[V1] Failed on lot {lot}:\n{traceback.format_exc()}")
            _send_discord_alert(f"⚠️ Prediction failed for lot {lot}")
            continue

        # ── V2 shadow predictions (isolated — V1 failures above skip this via continue) ──
        try:
            predicted_chain_v2 = list(recent_values_v2)

            # --- V2 near-term: 30min_v2 (T+5 to T+60, every 5 min, true 5-min lags) ---
            if not stale:
                feat_names_30_v2 = registry_v2.get_feature_names(lot, "30min")
                for minutes_ahead in range(5, 61, 5):
                    target_dt = now_utc + timedelta(minutes=minutes_ahead)
                    target_utc = (now + timedelta(minutes=minutes_ahead)).isoformat()
                    weather_row = get_weather_for_time(weather_df, target_dt)
                    X = build_feature_vector(
                        dt=target_dt,
                        weather_row=weather_row,
                        lot=lot,
                        recent_values=predicted_chain_v2[:4],
                        feature_names=feat_names_30_v2,
                    )
                    mean, low, high = registry_v2.predict(lot, "30min", X)
                    predictions.append({
                        "target_time": target_utc,
                        "lot": lot,
                        "model_tier": "30min_v2",
                        "prediction": round(mean, 4),
                        "confidence_low": round(low, 4),
                        "confidence_high": round(high, 4),
                    })
                    predicted_chain_v2.insert(0, mean)

            # --- V2 long-term: baseline_v2 (T+65 to T+360, every 5 min) ---
            feat_names_base_v2 = registry_v2.get_feature_names(lot, "baseline")
            start_minutes_v2 = 65 if not stale else 5
            for minutes_ahead in range(start_minutes_v2, 361, 5):
                target_dt = now_utc + timedelta(minutes=minutes_ahead)
                target_utc = (now + timedelta(minutes=minutes_ahead)).isoformat()
                weather_row = get_weather_for_time(weather_df, target_dt)
                X = build_feature_vector(
                    dt=target_dt,
                    weather_row=weather_row,
                    feature_names=feat_names_base_v2,
                )
                mean, low, high = registry_v2.predict(lot, "baseline", X)
                predictions.append({
                    "target_time": target_utc,
                    "lot": lot,
                    "model_tier": "baseline_v2",
                    "prediction": round(mean, 4),
                    "confidence_low": round(low, 4),
                    "confidence_high": round(high, 4),
                })

        except Exception:
            log.error(f"[V2] Failed on lot {lot}:\n{traceback.format_exc()}")

    # ── LightGBM predictions (all lots × all horizons in one batch) ──────────
    if lgb_loaded and not stale:
        try:
            lgb_preds = _run_lgb_predictions(
                now_utc, recent_rows, weather_df,
                lgb_point, lgb_lower, lgb_upper, lgb_config,
            )
            predictions.extend(lgb_preds)
            log.info(f"LGB: generated {len(lgb_preds)} predictions")
        except Exception:
            log.error(f"[LGB] Failed:\n{traceback.format_exc()}")
            _send_discord_alert("⚠️ LGB prediction block failed")

    log.info(f"Generated {len(predictions)} predictions for {len(LOTS)} lots")

    log.info("Writing predictions to Supabase...")
    try:
        write_predictions(predictions)
        log.info("Done.")
    except Exception as e:
        msg = f"⚠️ Prediction service failed to write: {e}"
        log.error(msg)
        _send_discord_alert(msg)


if __name__ == "__main__":
    run_predictions()
