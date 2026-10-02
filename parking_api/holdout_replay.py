"""Replay v3 on a historical day and compare it to stored production lgb predictions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from supabase import create_client

from evaluate_predictions import add_minutes_ahead, compute_metrics, match_predictions_to_actuals

from .config import BASE_DIR, DATA_DIR, LGB_MODELS_V3_DIR, SUPABASE_KEY, SUPABASE_URL, TABLE_PARKING_DATA, TABLE_PREDICTIONS
from .predict import _load_lgb_bundle, _required_history_rows, _run_lgb_predictions

LOCAL_TZ = ZoneInfo("America/New_York")
DEFAULT_START_LOCAL = "06:00"
DEFAULT_END_LOCAL = "21:00"
DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_HORIZON_MIN = 5
DEFAULT_HORIZON_MAX = 180
WEATHER_CANDIDATES = [
    DATA_DIR / "weather.csv",
    BASE_DIR.parent / "uncc-parking-notebook" / "data" / "weather.csv",
]


@dataclass(frozen=True)
class WindowBounds:
    start_local: datetime
    end_local: datetime
    start_utc: datetime
    end_utc: datetime


def _get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env or environment.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def normalize_clock_time(clock_time: str) -> str:
    try:
        parsed = datetime.strptime(clock_time, "%H:%M")
    except ValueError as exc:
        raise SystemExit(f"Invalid clock time '{clock_time}'. Use HH:MM in 24-hour time.") from exc
    return parsed.strftime("%H:%M")


def local_window_bounds(day: date, start_clock: str, end_clock: str) -> WindowBounds:
    start_clock = normalize_clock_time(start_clock)
    end_clock = normalize_clock_time(end_clock)
    start_time = time.fromisoformat(start_clock)
    end_time = time.fromisoformat(end_clock)
    start_local = datetime.combine(day, start_time, tzinfo=LOCAL_TZ)
    end_local = datetime.combine(day, end_time, tzinfo=LOCAL_TZ)
    if end_local < start_local:
        end_local += timedelta(days=1)
    return WindowBounds(
        start_local=start_local,
        end_local=end_local,
        start_utc=start_local.astimezone(ZoneInfo("UTC")),
        end_utc=end_local.astimezone(ZoneInfo("UTC")),
    )


def candidate_local_dates(today_local: date, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[date]:
    return [today_local - timedelta(days=offset) for offset in range(1, lookback_days + 1)]


def _paged_execute(query, page_size: int = 1000) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        batch = query.range(offset, offset + page_size - 1).execute().data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def fetch_parking_rows_between(client, from_utc: datetime, to_utc: datetime) -> list[dict]:
    query = (
        client.table(TABLE_PARKING_DATA)
        .select("created_at, data")
        .gt("created_at", "2020-01-01")
        .gte("created_at", from_utc.isoformat())
        .lte("created_at", to_utc.isoformat())
        .order("created_at")
    )
    return _paged_execute(query)


def fetch_prediction_rows_between(
    client,
    *,
    model_tier: str,
    created_from_utc: datetime,
    created_to_utc: datetime,
    lot: str | None = None,
) -> pd.DataFrame:
    query = (
        client.table(TABLE_PREDICTIONS)
        .select("created_at, target_time, model_tier, data")
        .eq("model_tier", model_tier)
        .gte("created_at", created_from_utc.isoformat())
        .lte("created_at", (created_to_utc + timedelta(minutes=5)).isoformat())
        .order("created_at")
    )
    raw_rows = _paged_execute(query)
    df = explode_prediction_rows(raw_rows, lot=lot, normalize_created_at=True)
    if df.empty:
        return df
    return df[df["created_at"].between(pd.Timestamp(created_from_utc), pd.Timestamp(created_to_utc))].copy()


def explode_prediction_rows(raw_rows: list[dict], *, lot: str | None = None, normalize_created_at: bool) -> pd.DataFrame:
    rows = []
    for row in raw_rows:
        data = row.get("data") or {}
        if isinstance(data, str):
            data = json.loads(data)
        for lot_name, vals in data.items():
            if lot and lot_name != lot:
                continue
            rows.append({
                "created_at": row["created_at"],
                "target_time": row["target_time"],
                "model_tier": row["model_tier"],
                "lot": lot_name,
                "prediction": vals["prediction"],
                "confidence_low": vals["confidence_low"],
                "confidence_high": vals["confidence_high"],
            })

    if not rows:
        return pd.DataFrame(columns=[
            "created_at", "target_time", "model_tier", "lot",
            "prediction", "confidence_low", "confidence_high",
        ])

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    if normalize_created_at:
        df["created_at"] = df["created_at"].dt.floor("5min")
    return df


def build_actuals_df(raw_rows: list[dict], *, lot: str | None = None) -> pd.DataFrame:
    records = []
    for row in raw_rows:
        actual_time = pd.to_datetime(row["created_at"], utc=True)
        data = row.get("data") or {}
        if isinstance(data, str):
            data = json.loads(data)
        for lot_name, value in data.items():
            if lot and lot_name != lot:
                continue
            if value is None:
                continue
            records.append({
                "actual_time": actual_time,
                "lot": lot_name,
                "actual": float(value),
            })
    return pd.DataFrame(records)


def load_historical_weather() -> pd.DataFrame:
    weather_path = next((path for path in WEATHER_CANDIDATES if path.exists()), None)
    if weather_path is None:
        tried = ", ".join(str(path) for path in WEATHER_CANDIDATES)
        sys.exit(f"Could not find historical weather.csv. Tried: {tried}")

    weather_df = pd.read_csv(weather_path)
    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
    required = ["datetime", "temperature_f", "humidity", "precipitation_in"]
    return weather_df[required].copy()


def rows_for_run(parking_rows_sorted: list[dict], parking_ts_sorted: list[pd.Timestamp], run_time_utc: pd.Timestamp, n: int) -> list[dict]:
    idx = bisect_right(parking_ts_sorted, run_time_utc) - 1
    if idx < 0:
        return []
    start_idx = max(0, idx - n + 1)
    return list(reversed(parking_rows_sorted[start_idx:idx + 1]))


def flatten_prediction_batches(prediction_rows: list[dict], *, created_at: datetime, model_tier: str, lot: str | None = None) -> pd.DataFrame:
    rows = []
    for row in prediction_rows:
        for lot_name, vals in row["data"].items():
            if lot and lot_name != lot:
                continue
            rows.append({
                "created_at": created_at.isoformat(),
                "target_time": row["target_time"],
                "model_tier": model_tier,
                "lot": lot_name,
                "prediction": vals["prediction"],
                "confidence_low": vals["confidence_low"],
                "confidence_high": vals["confidence_high"],
            })
    if not rows:
        return pd.DataFrame(columns=[
            "created_at", "target_time", "model_tier", "lot",
            "prediction", "confidence_low", "confidence_high",
        ])
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    return df


def simulate_v3_predictions(
    *,
    parking_rows_sorted: list[dict],
    parking_ts_sorted: list[pd.Timestamp],
    weather_df: pd.DataFrame,
    run_start_utc: datetime,
    run_end_utc: datetime,
    lot: str | None,
) -> pd.DataFrame:
    bundle = _load_lgb_bundle(LGB_MODELS_V3_DIR, "lgb_v3", required=True)
    assert bundle is not None
    history_rows = _required_history_rows([bundle])

    run_times = pd.date_range(run_start_utc, run_end_utc, freq="5min", tz="UTC")
    frames = []
    for run_time in run_times:
        recent_rows = rows_for_run(parking_rows_sorted, parking_ts_sorted, run_time, history_rows)
        if not recent_rows:
            continue
        simulated = _run_lgb_predictions(
            run_time.to_pydatetime(),
            recent_rows,
            weather_df,
            bundle.point,
            bundle.lower,
            bundle.upper,
            bundle.config,
            model_tier="lgb_v3_replay",
        )
        frames.append(
            flatten_prediction_batches(
                simulated,
                created_at=run_time.to_pydatetime(),
                model_tier="lgb_v3_replay",
                lot=lot,
            )
        )

    if not frames:
        return pd.DataFrame(columns=[
            "created_at", "target_time", "model_tier", "lot",
            "prediction", "confidence_low", "confidence_high",
        ])
    return pd.concat(frames, ignore_index=True)


def filter_horizon_range(df: pd.DataFrame, horizon_min: int, horizon_max: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = add_minutes_ahead(df)
    return df[df["minutes_ahead"].between(horizon_min, horizon_max)].copy()


def build_summary_rows(matched: pd.DataFrame, tiers: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for tier, label in tiers:
        subset = matched[matched["model_tier"] == tier]
        if subset.empty:
            continue
        rows.append({"label": label, **compute_metrics(subset)})
    return rows


def build_horizon_rows(matched: pd.DataFrame, tiers: list[tuple[str, str]], horizon_min: int, horizon_max: int) -> list[dict]:
    matched = filter_horizon_range(matched, horizon_min, horizon_max)
    rows = []
    for horizon in range(horizon_min, horizon_max + 1, 5):
        row = {"horizon": horizon}
        best = None
        for tier, label in tiers:
            subset = matched[(matched["model_tier"] == tier) & (matched["minutes_ahead"] == horizon)]
            row[f"{label}_n"] = len(subset)
            if subset.empty:
                row[f"{label}_mae"] = None
                continue
            metrics = compute_metrics(subset)
            row[f"{label}_mae"] = metrics["mae"]
            if best is None or metrics["mae"] < best[0]:
                best = (metrics["mae"], label)
        row["best"] = best[1] if best else "—"
        rows.append(row)
    return rows


def _print_summary_table(title: str, rows: list[dict]):
    print(f"\n{'=' * 88}")
    print(f"  {title}")
    print(f"{'=' * 88}")
    print(f"  {'Model':<14}  {'N':>7}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'In Band':>8}")
    print(f"  {'-' * 74}")
    for row in rows:
        r2_str = f"{row['r2']:.4f}" if not np.isnan(row["r2"]) else "   N/A"
        print(
            f"  {row['label']:<14}  {row['n']:>7,}  {row['mae']:>8.4f}  {row['rmse']:>8.4f}  {r2_str:>8}  {row['within_band_pct']:>7.1f}%"
        )


def _print_horizon_table(rows: list[dict], left_label: str, right_label: str):
    print(f"\n{'=' * 88}")
    print("  Horizon Comparison")
    print(f"{'=' * 88}")
    print(f"  {'Hzn':<6}  {left_label + ' MAE':>12}  {right_label + ' MAE':>12}  {'Best':>12}")
    print(f"  {'-' * 50}")
    for row in rows:
        left = row.get(f"{left_label}_mae")
        right = row.get(f"{right_label}_mae")
        left_str = f"{left:>12.4f}" if left is not None else f"{'—':>12}"
        right_str = f"{right:>12.4f}" if right is not None else f"{'—':>12}"
        horizon_label = f"T+{row['horizon']}"
        print(f"  {horizon_label:<6}  {left_str}  {right_str}  {row['best']:>12}")


def choose_random_day(
    client,
    *,
    lookback_days: int,
    start_local: str,
    end_local: str,
    seed: int | None,
) -> date:
    today_local = datetime.now(LOCAL_TZ).date()
    candidates = candidate_local_dates(today_local, lookback_days=lookback_days)
    rng = random.Random(seed)
    rng.shuffle(candidates)

    for day in candidates:
        bounds = local_window_bounds(day, start_local, end_local)
        stored = fetch_prediction_rows_between(
            client,
            model_tier="lgb",
            created_from_utc=bounds.start_utc,
            created_to_utc=bounds.end_utc,
        )
        if not stored.empty:
            return day

    raise SystemExit("No eligible local day in the requested lookback window had stored lgb predictions.")


def run_holdout_replay(
    *,
    day: date | None,
    lookback_days: int,
    seed: int | None,
    start_local: str,
    end_local: str,
    horizon_min: int,
    horizon_max: int,
    lot: str | None,
):
    client = _get_client()
    day_source = "manual"
    if day is None:
        day = choose_random_day(
            client,
            lookback_days=lookback_days,
            start_local=start_local,
            end_local=end_local,
            seed=seed,
        )
        day_source = "random"

    bounds = local_window_bounds(day, start_local, end_local)
    history_from_utc = bounds.start_utc - timedelta(minutes=120)
    actuals_to_utc = bounds.end_utc + timedelta(minutes=horizon_max)

    print(f"Replay day: {day.isoformat()} ({day_source})")
    print(f"Run window: {bounds.start_local.strftime('%Y-%m-%d %H:%M')} to {bounds.end_local.strftime('%Y-%m-%d %H:%M')} {LOCAL_TZ.key}")
    print(f"Horizon window: T+{horizon_min} to T+{horizon_max}")

    parking_rows = fetch_parking_rows_between(client, history_from_utc, actuals_to_utc)
    if not parking_rows:
        raise SystemExit("No parking_data rows found for the requested replay window.")

    parking_rows_sorted = sorted(parking_rows, key=lambda row: row["created_at"])
    parking_ts_sorted = [pd.Timestamp(row["created_at"]) for row in parking_rows_sorted]
    weather_df = load_historical_weather()
    actuals_df = build_actuals_df(parking_rows, lot=lot)

    stored_lgb = fetch_prediction_rows_between(
        client,
        model_tier="lgb",
        created_from_utc=bounds.start_utc,
        created_to_utc=bounds.end_utc,
        lot=lot,
    )
    if stored_lgb.empty:
        raise SystemExit("No stored lgb predictions found in the requested replay window.")

    simulated_v3 = simulate_v3_predictions(
        parking_rows_sorted=parking_rows_sorted,
        parking_ts_sorted=parking_ts_sorted,
        weather_df=weather_df,
        run_start_utc=bounds.start_utc,
        run_end_utc=bounds.end_utc,
        lot=lot,
    )
    if simulated_v3.empty:
        raise SystemExit("No simulated lgb_v3 predictions were generated for the requested replay window.")

    combined_preds = pd.concat([stored_lgb, simulated_v3], ignore_index=True)
    combined_preds = filter_horizon_range(combined_preds, horizon_min, horizon_max)
    matched = match_predictions_to_actuals(combined_preds, actuals_df)
    matched = filter_horizon_range(matched, horizon_min, horizon_max)

    tiers = [("lgb", "lgb"), ("lgb_v3_replay", "v3_replay")]
    _print_summary_table("Overall Comparison", build_summary_rows(matched, tiers))
    _print_horizon_table(build_horizon_rows(matched, tiers, horizon_min, horizon_max), "lgb", "v3_replay")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Replay lgb_v3 on a historical day and compare it to stored production lgb predictions.")
    parser.add_argument("--date", help="Local date in YYYY-MM-DD. If omitted, picks a random eligible day from the last week.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="How many complete local days to consider when auto-picking a replay day.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for auto-picking a day.")
    parser.add_argument("--start-local", default=DEFAULT_START_LOCAL, help="Inclusive local replay start time in HH:MM.")
    parser.add_argument("--end-local", default=DEFAULT_END_LOCAL, help="Inclusive local replay end time in HH:MM.")
    parser.add_argument("--horizon-min", type=int, default=DEFAULT_HORIZON_MIN, help="Minimum horizon in minutes.")
    parser.add_argument("--horizon-max", type=int, default=DEFAULT_HORIZON_MAX, help="Maximum horizon in minutes.")
    parser.add_argument("--lot", help="Optional lot filter, e.g. WEST.")
    args = parser.parse_args(argv)

    replay_day = date.fromisoformat(args.date) if args.date else None
    run_holdout_replay(
        day=replay_day,
        lookback_days=args.lookback_days,
        seed=args.seed,
        start_local=args.start_local,
        end_local=args.end_local,
        horizon_min=args.horizon_min,
        horizon_max=args.horizon_max,
        lot=args.lot,
    )


if __name__ == "__main__":
    main()
