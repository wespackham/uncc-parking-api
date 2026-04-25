"""Prediction loop: generates LightGBM forecasts for all lots.

Run as CLI:
    python -m parking_api.predict              # 3h live model + optional v3 shadow
    python -m parking_api.predict --model 24h  # 24h model (T+5-1440, every hour)

3h live: writes model_tier="lgb"
3h shadow: writes model_tier="lgb_v3" when models exist in models_lgb_v3/
24h model: writes model_tier="lgb_24h"

If parking data is stale (>15 min old), lag-style features reuse the latest
available row so the run still completes.
"""

import argparse
import json
import logging
import pickle
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from .config import (
    DISCORD_WEBHOOK_URL,
    LGB_MODELS_DIR,
    LGB_MODELS_V2_DIR,
    LGB_MODELS_V3_DIR,
)
from .enrichment import get_semester_metadata
from .features import (
    build_calendar_features,
    build_disruption_features,
    build_event_features,
    build_semester_features,
    build_sports_features,
    build_time_features,
    build_weather_features,
)
from .supabase_client import fetch_recent_rows, write_predictions
from .weather import fetch_forecast_sync, get_weather_for_time

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMA_30_ALPHA = 2 / 7
EMA_60_ALPHA = 2 / 13


@dataclass
class LGBBundle:
    models_dir: Path
    model_tier: str
    point: object
    lower: object
    upper: object
    config: dict
    required: bool = True


def _row_data(row: dict) -> dict:
    data = row.get("data") or {}
    if isinstance(data, str):
        data = json.loads(data)
    return data


def _get_lot_value(rows: list[dict], lot: str, idx: int) -> float | None:
    if idx >= len(rows):
        return None
    value = _row_data(rows[idx]).get(lot)
    return float(value) if value is not None else None


def _ema(values_oldest_to_newest: list[float], alpha: float) -> float:
    if not values_oldest_to_newest:
        return 0.0
    ema = float(values_oldest_to_newest[0])
    for value in values_oldest_to_newest[1:]:
        ema = alpha * float(value) + (1 - alpha) * ema
    return float(ema)


def _extract_lgb_state(rows: list[dict], lot: str) -> dict:
    """Extract the live state needed by both production and v3 LGB models."""

    now = _get_lot_value(rows, lot, 0)
    now = now if now is not None else 0.0

    lag5 = _get_lot_value(rows, lot, 1)
    lag5 = lag5 if lag5 is not None else now
    lag15 = _get_lot_value(rows, lot, 3)
    lag15 = lag15 if lag15 is not None else lag5
    lag30 = _get_lot_value(rows, lot, 6)
    lag30 = lag30 if lag30 is not None else lag15
    lag60 = _get_lot_value(rows, lot, 12)
    lag60 = lag60 if lag60 is not None else lag30
    lag90 = _get_lot_value(rows, lot, 18)
    lag90 = lag90 if lag90 is not None else lag60
    lag120 = _get_lot_value(rows, lot, 24)
    lag120 = lag120 if lag120 is not None else lag90

    filled_series_newest_first = []
    last = now
    for idx in range(min(len(rows), 25)):
        value = _get_lot_value(rows, lot, idx)
        if value is None:
            value = last
        filled_series_newest_first.append(float(value))
        last = float(value)

    if not filled_series_newest_first:
        filled_series_newest_first = [0.0]

    series_oldest_to_newest = list(reversed(filled_series_newest_first))
    ema_30 = _ema(series_oldest_to_newest, EMA_30_ALPHA)
    ema_60 = _ema(series_oldest_to_newest, EMA_60_ALPHA)

    return {
        "current_capacity": now,
        "lag_5": lag5,
        "lag_15": lag15,
        "lag_30": lag30,
        "lag_60": lag60,
        "lag_90": lag90,
        "lag_120": lag120,
        "ema_30": ema_30,
        "ema_60": ema_60,
        "delta_5": now - lag5,
        "delta_15": now - lag15,
        "delta_30": now - lag30,
        "delta_60": now - lag60,
        "delta_120": now - lag120,
    }


def _extract_lgb_deltas(rows: list[dict], lot: str) -> tuple[float, float, float, float]:
    """Backward-compatible helper for legacy tests and 3h production features."""
    state = _extract_lgb_state(rows, lot)
    return (
        state["current_capacity"],
        state["delta_5"],
        state["delta_15"],
        state["delta_30"],
    )


def _build_target_feature_dict(target_dt: datetime, weather_df: pd.DataFrame, feature_names: list[str], lgb_config: dict) -> dict:
    tgt_date = target_dt.strftime("%Y-%m-%d")
    tgt_time = build_time_features(target_dt)
    tgt_is_weekend = tgt_time["is_weekend"]
    tgt_cal = build_calendar_features(tgt_date, tgt_is_weekend)
    tgt_sports = build_sports_features(tgt_date)
    tgt_dis = build_disruption_features(tgt_date)
    tgt_wthr = build_weather_features(get_weather_for_time(weather_df, target_dt))

    target_features = {
        "tgt_hour_sin": tgt_time["hour_sin"],
        "tgt_hour_cos": tgt_time["hour_cos"],
        "tgt_minute_sin": tgt_time["minute_sin"],
        "tgt_minute_cos": tgt_time["minute_cos"],
        "tgt_dow_sin": tgt_time["dow_sin"],
        "tgt_dow_cos": tgt_time["dow_cos"],
        "tgt_is_weekend": tgt_is_weekend,
        "tgt_is_class_day": tgt_cal["is_class_day"],
        "tgt_is_break": tgt_cal.get("is_break", 0),
        "tgt_is_finals": tgt_cal.get("is_finals", 0),
        "tgt_is_commencement": tgt_cal.get("is_commencement", 0),
        "tgt_is_holiday": tgt_cal.get("is_holiday", 0),
        "tgt_home_game_count": tgt_sports.get("home_game_count", 0),
        "tgt_has_basketball": tgt_sports.get("has_basketball", 0),
        "tgt_has_baseball": tgt_sports.get("has_baseball", 0),
        "tgt_has_softball": tgt_sports.get("has_softball", 0),
        "tgt_has_lacrosse": tgt_sports.get("has_lacrosse", 0),
        "tgt_high_impact_game": tgt_sports.get("high_impact_game", 0),
        "tgt_condition_level": tgt_dis.get("condition_level", 0),
        "tgt_is_remote": tgt_dis.get("is_remote", 0),
        "tgt_is_cancelled": tgt_dis.get("is_cancelled", 0),
        "tgt_temperature_f": tgt_wthr.get("temperature_f", 65.0),
        "tgt_humidity": tgt_wthr.get("humidity", 55.0),
        "tgt_precipitation_in": tgt_wthr.get("precipitation_in", 0.0),
    }

    if any(name.startswith("tgt_class_week_") for name in feature_names) or "tgt_weeks_until_finals" in feature_names:
        semester_meta = get_semester_metadata()
        first_class_date = lgb_config.get("first_class_date") or semester_meta.get("first_class_date")
        finals_start_date = lgb_config.get("finals_start_date") or semester_meta.get("finals_start_date")
        total_weeks = int(lgb_config.get("total_weeks", semester_meta.get("total_weeks", 16)))
        if first_class_date and finals_start_date:
            target_features.update(
                build_semester_features(
                    tgt_date,
                    tgt_cal,
                    first_class_date=first_class_date,
                    finals_start_date=finals_start_date,
                    total_weeks=total_weeks,
                )
            )

    if "tgt_event_max_impact" in feature_names or "tgt_event_high_count" in feature_names:
        target_features.update(build_event_features(tgt_date))

    return target_features


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
    """Build inference batch (lots × horizons) and run the requested LGB bundle."""
    lots = lgb_config["lots"]
    horizons = lgb_config["horizons"]
    feature_names = lgb_config["features"]
    target_mode = lgb_config.get("target_mode", "absolute")

    cur_dow = now_utc.weekday()
    cur_time_feats = {
        "cur_hour_sin": np.sin(2 * np.pi * now_utc.hour / 24),
        "cur_hour_cos": np.cos(2 * np.pi * now_utc.hour / 24),
        "cur_dow_sin": np.sin(2 * np.pi * cur_dow / 7),
        "cur_dow_cos": np.cos(2 * np.pi * cur_dow / 7),
        "cur_is_weekend": int(cur_dow >= 5),
    }

    lot_state = {lot: _extract_lgb_state(recent_rows, lot) for lot in lots}

    base_utc = now_utc.replace(
        minute=(now_utc.minute // 5) * 5,
        second=0,
        microsecond=0,
    )

    rows = []
    meta: list[tuple[str, str]] = []
    current_caps = []

    for horizon in horizons:
        target_dt = base_utc + timedelta(minutes=horizon)
        target_utc = target_dt.isoformat()
        target_features = _build_target_feature_dict(target_dt, weather_df, feature_names, lgb_config)

        for lot in lots:
            state = lot_state[lot]
            rows.append({
                **state,
                **cur_time_feats,
                "horizon_minutes": horizon,
                "deck_id": lot,
                **target_features,
            })
            meta.append((lot, target_utc))
            current_caps.append(state["current_capacity"])

    X = pd.DataFrame(rows)
    for feature in feature_names:
        if feature not in X.columns:
            X[feature] = 0.0
    X = X[feature_names]
    if "deck_id" in X.columns:
        X["deck_id"] = pd.Categorical(X["deck_id"], categories=lots)

    preds = np.asarray(lgb_point.predict(X), dtype=float)
    lows = np.asarray(lgb_lower.predict(X), dtype=float)
    highs = np.asarray(lgb_upper.predict(X), dtype=float)

    if target_mode == "residual":
        current_caps_arr = np.asarray(current_caps, dtype=float)
        preds = preds + current_caps_arr
        lows = lows + current_caps_arr
        highs = highs + current_caps_arr
    elif target_mode != "absolute":
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    preds = preds.clip(0, 1)
    lows = lows.clip(0, 1)
    highs = highs.clip(0, 1)

    result_map: dict[str, dict] = {}
    for idx, (lot, target_utc) in enumerate(meta):
        if target_utc not in result_map:
            result_map[target_utc] = {
                "target_time": target_utc,
                "model_tier": model_tier,
                "data": {},
            }
        result_map[target_utc]["data"][lot] = {
            "prediction": round(float(preds[idx]), 4),
            "confidence_low": round(float(lows[idx]), 4),
            "confidence_high": round(float(highs[idx]), 4),
        }

    return list(result_map.values())


def _load_lgb_bundle(models_dir: Path, model_tier: str, *, required: bool) -> LGBBundle | None:
    try:
        with open(models_dir / "lgb_point.pkl", "rb") as file:
            lgb_point = pickle.load(file)
        with open(models_dir / "lgb_lower.pkl", "rb") as file:
            lgb_lower = pickle.load(file)
        with open(models_dir / "lgb_upper.pkl", "rb") as file:
            lgb_upper = pickle.load(file)
        with open(models_dir / "lgb_config.pkl", "rb") as file:
            lgb_config = pickle.load(file)
        return LGBBundle(
            models_dir=models_dir,
            model_tier=model_tier,
            point=lgb_point,
            lower=lgb_lower,
            upper=lgb_upper,
            config=lgb_config,
            required=required,
        )
    except Exception:
        if required:
            raise
        log.warning("Skipping optional bundle %s from %s\n%s", model_tier, models_dir, traceback.format_exc())
        return None


def _bundles_for_model(model: str) -> list[LGBBundle]:
    if model == "24h":
        bundle = _load_lgb_bundle(LGB_MODELS_V2_DIR, "lgb_24h", required=True)
        return [bundle] if bundle else []

    primary = _load_lgb_bundle(LGB_MODELS_DIR, "lgb", required=True)
    bundles = [primary] if primary else []
    shadow = _load_lgb_bundle(LGB_MODELS_V3_DIR, "lgb_v3", required=False)
    if shadow is not None:
        bundles.append(shadow)
    return bundles


def _required_history_rows(bundles: list[LGBBundle]) -> int:
    needs_extended = any(
        bundle.config.get("target_mode") == "residual" or
        any(
            feature in bundle.config.get("features", [])
            for feature in ("lag_60", "lag_90", "lag_120", "ema_30", "ema_60", "delta_60", "delta_120")
        )
        for bundle in bundles
    )
    return 30 if needs_extended else 20


def _send_discord_alert(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception:
        pass


def run_predictions(model: str = "3h"):
    try:
        bundles = _bundles_for_model(model)
    except Exception:
        msg = f"⚠️ Prediction service: failed to load LGB {model} models"
        log.error("%s\n%s", msg, traceback.format_exc())
        _send_discord_alert(msg)
        return

    if not bundles:
        msg = f"⚠️ Prediction service: no runnable LGB bundles found for {model}"
        log.error(msg)
        _send_discord_alert(msg)
        return

    for bundle in bundles:
        bundle_type = "required" if bundle.required else "shadow"
        log.info("Loaded %s bundle %s from %s", bundle_type, bundle.model_tier, bundle.models_dir)

    log.info("Fetching recent parking data...")
    recent_rows = fetch_recent_rows(n=_required_history_rows(bundles))
    if not recent_rows:
        _send_discord_alert("⚠️ Prediction service: no recent parking data found")
        log.error("No recent parking data. Aborting.")
        return
    log.info("Got %s recent rows. Latest: %s", len(recent_rows), recent_rows[0]["created_at"])

    latest_time = pd.Timestamp(recent_rows[0]["created_at"])
    now = pd.Timestamp.now(tz="UTC")
    data_age_minutes = (now - latest_time).total_seconds() / 60
    if data_age_minutes > 15:
        log.warning("Data is %.1f min old — lag features will reuse the latest available row", data_age_minutes)

    log.info("Fetching weather forecast...")
    weather_df = fetch_forecast_sync()
    now_utc = datetime.now(timezone.utc)

    all_predictions = []
    for bundle in bundles:
        try:
            predictions = _run_lgb_predictions(
                now_utc,
                recent_rows,
                weather_df,
                bundle.point,
                bundle.lower,
                bundle.upper,
                bundle.config,
                model_tier=bundle.model_tier,
            )
            log.info("Generated %s prediction rows for %s", len(predictions), bundle.model_tier)
            all_predictions.extend(predictions)
        except Exception:
            msg = f"⚠️ {bundle.model_tier} prediction run failed"
            log.error("%s\n%s", msg, traceback.format_exc())
            if bundle.required:
                _send_discord_alert(msg)
                return

    if not all_predictions:
        log.warning("No predictions generated; nothing to write.")
        return

    log.info("Writing %s prediction rows to Supabase...", len(all_predictions))
    try:
        write_predictions(all_predictions)
        log.info("Done.")
    except Exception as exc:
        msg = f"⚠️ Prediction service failed to write: {exc}"
        log.error(msg)
        _send_discord_alert(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["3h", "24h"],
        default="3h",
        help="Which model to run: 3h (T+5-180, every 5 min) or 24h (T+5-1440, every hour)",
    )
    args = parser.parse_args()
    run_predictions(model=args.model)
