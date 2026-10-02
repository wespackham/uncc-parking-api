"""
Weekly deep evaluation: day-by-day, hour-by-hour, horizon-by-horizon, lot-by-lot.

Runs all breakdowns in one pass and flags anomalous periods where error spikes
above 2× the tier baseline — useful for identifying unmodeled events.

Usage:
    python weekly_deep_eval.py                          # last 7 days, all tiers
    python weekly_deep_eval.py --days 14
    python weekly_deep_eval.py --tier lgb               # single tier
    python weekly_deep_eval.py --csv-dir ./eval_out     # export all CSVs
    python weekly_deep_eval.py --top-errors 50          # show top N worst predictions
    python weekly_deep_eval.py --no-matrices            # skip large lot×hour/lot×horizon tables
"""

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DISPLAY_TZ_NAME = os.environ.get("EVALUATOR_TZ", "America/New_York")
DISPLAY_TZ = ZoneInfo(DISPLAY_TZ_NAME)

TIER_ORDER = ["lgb", "lgb_v3", "lgb_24h"]
MATCH_TOLERANCE_SEC = 4 * 60
ANOMALY_MULTIPLIER = 2.0  # flag when MAE exceeds this × tier baseline


# ── DB helpers ───────────────────────────────────────────────────────────────

def get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env or environment.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _fetch_predictions_chunk(client, from_dt: str, to_dt: str, tier_filter: str | None) -> list:
    """Fetch one date-bounded chunk of prediction rows (no ORDER BY to avoid statement timeouts)."""
    raw_rows = []
    page_size = 1000
    offset = 0
    while True:
        q = (
            client.table("parking_predictions")
            .select("created_at, target_time, model_tier, data")
            .gte("target_time", from_dt)
            .lt("target_time", to_dt)
            .range(offset, offset + page_size - 1)
        )
        if tier_filter:
            q = q.eq("model_tier", tier_filter)
        result = q.execute()
        batch = result.data
        raw_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return raw_rows


def fetch_predictions(client, from_dt: str, to_dt: str, tier_filter: str | None,
                      chunk_days: int = 3) -> pd.DataFrame:
    from_ts = datetime.fromisoformat(from_dt.replace("Z", "+00:00")) if from_dt.endswith("Z") else datetime.fromisoformat(from_dt)
    to_ts = datetime.fromisoformat(to_dt.replace("Z", "+00:00")) if to_dt.endswith("Z") else datetime.fromisoformat(to_dt)
    if from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=timezone.utc)
    if to_ts.tzinfo is None:
        to_ts = to_ts.replace(tzinfo=timezone.utc)

    print(f"Fetching predictions {from_dt[:10]} → {to_dt[:10]} (chunked by {chunk_days}d)...")
    all_raw = []
    chunk_start = from_ts
    while chunk_start < to_ts:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), to_ts)
        chunk_raw = _fetch_predictions_chunk(
            client, chunk_start.isoformat(), chunk_end.isoformat(), tier_filter
        )
        all_raw.extend(chunk_raw)
        print(f"  chunk {chunk_start.date()} → {chunk_end.date()}: {len(chunk_raw):,} DB rows")
        chunk_start = chunk_end

    rows = []
    for row in all_raw:
        for lot_name, vals in (row.get("data") or {}).items():
            rows.append({
                "created_at": row["created_at"],
                "target_time": row["target_time"],
                "model_tier": row["model_tier"],
                "lot": lot_name,
                "prediction": vals["prediction"],
                "confidence_low": vals["confidence_low"],
                "confidence_high": vals["confidence_high"],
            })
    print(f"  {len(rows):,} prediction rows from {len(all_raw):,} DB rows total")
    if not rows:
        sys.exit("No predictions found for the given range/tier.")
    df = pd.DataFrame(rows)
    df["target_time"] = pd.to_datetime(df["target_time"], format="ISO8601", utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], format="ISO8601", utc=True)
    return df


def fetch_actuals(client, from_dt: str, to_dt: str, chunk_days: int = 3) -> pd.DataFrame:
    from_ts = datetime.fromisoformat(from_dt.replace("Z", "+00:00")) if from_dt.endswith("Z") else datetime.fromisoformat(from_dt)
    to_ts = datetime.fromisoformat(to_dt.replace("Z", "+00:00")) if to_dt.endswith("Z") else datetime.fromisoformat(to_dt)
    if from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=timezone.utc)
    if to_ts.tzinfo is None:
        to_ts = to_ts.replace(tzinfo=timezone.utc)

    print(f"Fetching actuals {from_dt[:10]} → {to_dt[:10]} (chunked by {chunk_days}d)...")
    rows = []
    chunk_start = from_ts
    while chunk_start < to_ts:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), to_ts)
        page_size = 1000
        offset = 0
        chunk_rows = 0
        while True:
            result = (
                client.table("parking_data")
                .select("created_at, data")
                .gt("created_at", "2020-01-01")
                .gte("created_at", chunk_start.isoformat())
                .lt("created_at", chunk_end.isoformat())
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data
            rows.extend(batch)
            chunk_rows += len(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        print(f"  chunk {chunk_start.date()} → {chunk_end.date()}: {chunk_rows:,} rows")
        chunk_start = chunk_end

    print(f"  {len(rows):,} actual rows total")
    if not rows:
        sys.exit("No actual data found for the given range.")
    records = []
    for row in rows:
        ts = pd.to_datetime(row["created_at"], utc=True)
        for lot, value in (row["data"] or {}).items():
            if value is not None:
                records.append({"actual_time": ts, "lot": lot, "actual": float(value)})
    return pd.DataFrame(records)


def match_predictions_to_actuals(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    print("Matching predictions to actuals...")
    actuals_sorted = actuals.sort_values("actual_time")
    preds_sorted = preds.sort_values("target_time")
    parts = []
    for lot, lot_preds in preds_sorted.groupby("lot"):
        lot_actuals = actuals_sorted[actuals_sorted["lot"] == lot]
        if lot_actuals.empty:
            continue
        merged = pd.merge_asof(
            lot_preds,
            lot_actuals[["actual_time", "actual"]],
            left_on="target_time",
            right_on="actual_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SEC),
        )
        parts.append(merged)
    if not parts:
        sys.exit("No matched rows.")
    df = pd.concat(parts, ignore_index=True)
    matched = df.dropna(subset=["actual"])
    dropped = len(df) - len(matched)
    if dropped:
        print(f"  Dropped {dropped:,} rows with no actual within {MATCH_TOLERANCE_SEC}s")
    print(f"  {len(matched):,} matched pairs")
    return matched


# ── Derived columns ───────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw_min = (df["target_time"] - df["created_at"]).dt.total_seconds() / 60
    df["minutes_ahead"] = (raw_min / 5).round().astype(int) * 5
    local = df["target_time"].dt.tz_convert(DISPLAY_TZ)
    df["date_local"] = local.dt.date
    df["hour_local"] = local.dt.hour
    df["dow_local"] = local.dt.day_name()
    df["abs_error"] = (df["prediction"] - df["actual"]).abs()
    df["signed_error"] = df["prediction"] - df["actual"]
    df["in_band"] = (df["actual"] >= df["confidence_low"]) & (df["actual"] <= df["confidence_high"])
    return df


# ── Metric helpers ────────────────────────────────────────────────────────────

def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    pred, actual = df["prediction"].values, df["actual"].values
    low, high = df["confidence_low"].values, df["confidence_high"].values
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    in_band = float(np.mean((actual >= low) & (actual <= high)) * 100)
    return {"n": len(df), "mae": mae, "rmse": rmse, "bias": bias, "r2": r2, "in_band_pct": in_band}


# ── Print helpers ─────────────────────────────────────────────────────────────

W = 80

def section(title: str):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def _r2_str(v) -> str:
    return f"{v:.4f}" if (v is not None and not np.isnan(v)) else "   N/A"


def _fmt(v, w=8, decimals=4) -> str:
    return f"{v:>{w}.{decimals}f}" if v is not None and not np.isnan(v) else f"{'—':>{w}}"


# ── Section 1: Overall summary ────────────────────────────────────────────────

def print_overall_summary(matched: pd.DataFrame):
    section("Overall Summary by Model Tier")
    header = f"  {'Tier':<12}  {'N':>8}  {'MAE':>8}  {'RMSE':>8}  {'Bias':>8}  {'R²':>8}  {'In Band':>8}"
    print(header)
    print(f"  {'-' * 70}")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        m = metrics(sub)
        bias_str = f"{m['bias']:>+8.4f}"
        print(f"  {tier:<12}  {m['n']:>8,}  {m['mae']:>8.4f}  {m['rmse']:>8.4f}  {bias_str}  {_r2_str(m['r2']):>8}  {m['in_band_pct']:>7.1f}%")


# ── Section 2: Day-by-day ─────────────────────────────────────────────────────

def print_by_day(matched: pd.DataFrame, tier_baselines: dict[str, float]):
    section("Day-by-Day Accuracy (local time)")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        baseline = tier_baselines.get(tier, 0)
        days = sorted(sub["date_local"].unique())
        print(f"\n  ── {tier} ──")
        print(f"  {'Date':<12}  {'DoW':<9}  {'N':>7}  {'MAE':>8}  {'Bias':>8}  {'RMSE':>8}  {'In Band':>8}  {'Flag'}")
        print(f"  {'-' * 74}")
        for day in days:
            day_df = sub[sub["date_local"] == day]
            m = metrics(day_df)
            flag = " *** HIGH" if m["mae"] > ANOMALY_MULTIPLIER * baseline else ""
            dow = day_df["dow_local"].iloc[0][:3]
            print(
                f"  {str(day):<12}  {dow:<9}  {m['n']:>7,}  {m['mae']:>8.4f}  {m['bias']:>+8.4f}"
                f"  {m['rmse']:>8.4f}  {m['in_band_pct']:>7.1f}%{flag}"
            )


# ── Section 3: Hour-by-hour ───────────────────────────────────────────────────

def print_by_hour(matched: pd.DataFrame, tier_baselines: dict[str, float]):
    section(f"Hour-of-Day Accuracy ({DISPLAY_TZ_NAME})")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        baseline = tier_baselines.get(tier, 0)
        print(f"\n  ── {tier} ──")
        print(f"  {'Hour':<6}  {'N':>7}  {'MAE':>8}  {'Bias':>8}  {'RMSE':>8}  {'In Band':>8}  {'Flag'}")
        print(f"  {'-' * 62}")
        for hour in range(24):
            h_df = sub[sub["hour_local"] == hour]
            if h_df.empty:
                continue
            m = metrics(h_df)
            flag = " *** HIGH" if m["mae"] > ANOMALY_MULTIPLIER * baseline else ""
            print(
                f"  {f'{hour:02d}:00':<6}  {m['n']:>7,}  {m['mae']:>8.4f}  {m['bias']:>+8.4f}"
                f"  {m['rmse']:>8.4f}  {m['in_band_pct']:>7.1f}%{flag}"
            )


# ── Section 4: Horizon-by-horizon ────────────────────────────────────────────

def print_by_horizon(matched: pd.DataFrame, tier_baselines: dict[str, float]):
    section("Horizon-by-Horizon Accuracy")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        baseline = tier_baselines.get(tier, 0)
        steps = sorted(sub["minutes_ahead"].unique())
        if len(steps) <= 1:
            continue
        print(f"\n  ── {tier} ──")
        print(f"  {'T+min':<7}  {'N':>7}  {'MAE':>8}  {'Bias':>8}  {'RMSE':>8}  {'In Band':>8}  {'Flag'}")
        print(f"  {'-' * 62}")
        for step in steps:
            s_df = sub[sub["minutes_ahead"] == step]
            if s_df.empty:
                continue
            m = metrics(s_df)
            flag = " *** HIGH" if m["mae"] > ANOMALY_MULTIPLIER * baseline else ""
            print(
                f"  {f'T+{step}':<7}  {m['n']:>7,}  {m['mae']:>8.4f}  {m['bias']:>+8.4f}"
                f"  {m['rmse']:>8.4f}  {m['in_band_pct']:>7.1f}%{flag}"
            )


# ── Section 5: Lot-by-lot ─────────────────────────────────────────────────────

def print_by_lot(matched: pd.DataFrame, tier_baselines: dict[str, float]):
    section("Per-Lot Accuracy")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        baseline = tier_baselines.get(tier, 0)
        lots = sorted(sub["lot"].unique())
        print(f"\n  ── {tier} ──")
        print(f"  {'Lot':<10}  {'N':>7}  {'MAE':>8}  {'Bias':>8}  {'RMSE':>8}  {'In Band':>8}  {'Flag'}")
        print(f"  {'-' * 64}")
        for lot in lots:
            l_df = sub[sub["lot"] == lot]
            m = metrics(l_df)
            flag = " *** HIGH" if m["mae"] > ANOMALY_MULTIPLIER * baseline else ""
            print(
                f"  {lot:<10}  {m['n']:>7,}  {m['mae']:>8.4f}  {m['bias']:>+8.4f}"
                f"  {m['rmse']:>8.4f}  {m['in_band_pct']:>7.1f}%{flag}"
            )


# ── Section 6: Day × Lot MAE matrix ──────────────────────────────────────────

def print_day_lot_matrix(matched: pd.DataFrame):
    section("Day × Lot MAE Matrix")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        lots = sorted(sub["lot"].unique())
        days = sorted(sub["date_local"].unique())
        col_w = 8
        print(f"\n  ── {tier} ──")
        header = f"  {'Date':<12}" + "".join(f"  {lot:>{col_w}}" for lot in lots)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for day in days:
            day_df = sub[sub["date_local"] == day]
            row = f"  {str(day):<12}"
            for lot in lots:
                cell = day_df[day_df["lot"] == lot]
                if len(cell) < 2:
                    row += f"  {'—':>{col_w}}"
                else:
                    mae = float(np.mean(np.abs(cell["prediction"].values - cell["actual"].values)))
                    row += f"  {mae:>{col_w}.4f}"
            print(row)


# ── Section 7: Hour × Lot MAE matrix ─────────────────────────────────────────

def print_hour_lot_matrix(matched: pd.DataFrame):
    section(f"Hour × Lot MAE Matrix ({DISPLAY_TZ_NAME})")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        lots = sorted(sub["lot"].unique())
        col_w = 8
        print(f"\n  ── {tier} ──")
        header = f"  {'Hour':<6}" + "".join(f"  {lot:>{col_w}}" for lot in lots)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for hour in range(24):
            h_df = sub[sub["hour_local"] == hour]
            if h_df.empty:
                continue
            row = f"  {f'{hour:02d}:00':<6}"
            for lot in lots:
                cell = h_df[h_df["lot"] == lot]
                if len(cell) < 2:
                    row += f"  {'—':>{col_w}}"
                else:
                    mae = float(np.mean(np.abs(cell["prediction"].values - cell["actual"].values)))
                    row += f"  {mae:>{col_w}.4f}"
            print(row)


# ── Section 8: Horizon × Lot MAE matrix ──────────────────────────────────────

def print_horizon_lot_matrix(matched: pd.DataFrame, max_horizons: int = 36):
    section(f"Horizon × Lot MAE Matrix (first {max_horizons} steps shown)")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        lots = sorted(sub["lot"].unique())
        steps = sorted(sub["minutes_ahead"].unique())[:max_horizons]
        col_w = 8
        print(f"\n  ── {tier} ──")
        header = f"  {'T+min':<7}" + "".join(f"  {lot:>{col_w}}" for lot in lots)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for step in steps:
            s_df = sub[sub["minutes_ahead"] == step]
            row = f"  {f'T+{step}':<7}"
            for lot in lots:
                cell = s_df[s_df["lot"] == lot]
                if len(cell) < 2:
                    row += f"  {'—':>{col_w}}"
                else:
                    mae = float(np.mean(np.abs(cell["prediction"].values - cell["actual"].values)))
                    row += f"  {mae:>{col_w}.4f}"
            print(row)


# ── Section 9: Bias analysis ──────────────────────────────────────────────────

def print_bias_analysis(matched: pd.DataFrame):
    """Show mean signed error (bias) to detect systematic over/under prediction.
    Positive bias = model overpredicts occupancy. Negative = underpredicts.
    """
    section("Bias Analysis (Mean Signed Error: + = over-predict, − = under-predict)")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        print(f"\n  ── {tier} — Bias by Hour ──")
        print(f"  {'Hour':<6}  {'Bias':>8}  {'MAE':>8}  {'N':>7}")
        print(f"  {'-' * 34}")
        for hour in range(24):
            h_df = sub[sub["hour_local"] == hour]
            if h_df.empty:
                continue
            bias = float(np.mean(h_df["signed_error"]))
            mae = float(np.mean(h_df["abs_error"]))
            print(f"  {f'{hour:02d}:00':<6}  {bias:>+8.4f}  {mae:>8.4f}  {len(h_df):>7,}")

        print(f"\n  ── {tier} — Bias by Lot ──")
        print(f"  {'Lot':<10}  {'Bias':>8}  {'MAE':>8}  {'N':>7}")
        print(f"  {'-' * 38}")
        for lot in sorted(sub["lot"].unique()):
            l_df = sub[sub["lot"] == lot]
            bias = float(np.mean(l_df["signed_error"]))
            mae = float(np.mean(l_df["abs_error"]))
            print(f"  {lot:<10}  {bias:>+8.4f}  {mae:>8.4f}  {len(l_df):>7,}")

        steps = sorted(sub["minutes_ahead"].unique())
        if len(steps) > 1:
            print(f"\n  ── {tier} — Bias by Horizon (first 36 steps) ──")
            print(f"  {'T+min':<7}  {'Bias':>8}  {'MAE':>8}  {'N':>7}")
            print(f"  {'-' * 34}")
            for step in steps[:36]:
                s_df = sub[sub["minutes_ahead"] == step]
                bias = float(np.mean(s_df["signed_error"]))
                mae = float(np.mean(s_df["abs_error"]))
                print(f"  {f'T+{step}':<7}  {bias:>+8.4f}  {mae:>8.4f}  {len(s_df):>7,}")


# ── Section 10: Top-N worst predictions ──────────────────────────────────────

def print_top_errors(matched: pd.DataFrame, top_n: int = 30):
    """Show the worst individual predictions — the key view for spotting events."""
    section(f"Top {top_n} Worst Predictions (by absolute error)")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier].copy()
        if sub.empty:
            continue
        worst = sub.nlargest(top_n, "abs_error")
        print(f"\n  ── {tier} ──")
        print(
            f"  {'Target (local)':<18}  {'Lot':<8}  {'T+min':>6}  "
            f"{'Pred':>6}  {'Actual':>6}  {'Error':>7}  {'Low':>6}  {'High':>6}  {'InBand'}"
        )
        print(f"  {'-' * 80}")
        for _, row in worst.iterrows():
            ts_local = row["target_time"].tz_convert(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
            in_band_str = "yes" if row["in_band"] else " NO"
            horizon_label = f"T+{row['minutes_ahead']}"
            print(
                f"  {ts_local:<18}  {row['lot']:<8}  {horizon_label:>6}  "
                f"{row['prediction']:>6.3f}  {row['actual']:>6.3f}  {row['signed_error']:>+7.3f}"
                f"  {row['confidence_low']:>6.3f}  {row['confidence_high']:>6.3f}  {in_band_str}"
            )


# ── Section 11: Anomaly flags ─────────────────────────────────────────────────

def print_anomaly_flags(matched: pd.DataFrame, tier_baselines: dict[str, float], multiplier: float = 2.0):
    """Summarise every (tier, date, hour) cell whose MAE exceeds 2× baseline."""
    section(f"Anomaly Flags — cells with MAE > {multiplier}× tier baseline")
    any_found = False
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        baseline = tier_baselines.get(tier, 0)
        threshold = multiplier * baseline
        rows = []
        for day in sorted(sub["date_local"].unique()):
            for hour in range(24):
                cell = sub[(sub["date_local"] == day) & (sub["hour_local"] == hour)]
                if len(cell) < 5:
                    continue
                m = metrics(cell)
                if m["mae"] > threshold:
                    rows.append({
                        "tier": tier, "date": str(day), "hour": hour,
                        "n": m["n"], "mae": m["mae"], "bias": m["bias"],
                        "baseline": baseline, "ratio": m["mae"] / baseline,
                        "top_lot": cell.groupby("lot")["abs_error"].mean().idxmax(),
                    })
        if rows:
            any_found = True
            print(f"\n  ── {tier} (baseline MAE={baseline:.4f}, flag > {threshold:.4f}) ──")
            print(f"  {'Date':<12}  {'Hour':>5}  {'N':>6}  {'MAE':>8}  {'×Base':>6}  {'Bias':>8}  {'Worst Lot'}")
            print(f"  {'-' * 62}")
            for r in sorted(rows, key=lambda x: -x["ratio"]):
                print(
                    f"  {r['date']:<12}  {r['hour']:>02d}:00  {r['n']:>6,}  {r['mae']:>8.4f}"
                    f"  {r['ratio']:>5.1f}×  {r['bias']:>+8.4f}  {r['top_lot']}"
                )
    if not any_found:
        print(f"\n  No (tier, date, hour) cells exceeded {multiplier}× baseline.")


# ── Section 12: Confidence band quality ──────────────────────────────────────

def print_band_quality(matched: pd.DataFrame):
    section("Confidence Band Quality (% actual in [low, high])")
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        lots = sorted(sub["lot"].unique())
        print(f"\n  ── {tier} — In-Band % by Lot × Hour ──")
        col_w = 6
        header = f"  {'Hour':<6}" + "".join(f"  {lot:>{col_w}}" for lot in lots)
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for hour in range(24):
            h_df = sub[sub["hour_local"] == hour]
            if h_df.empty:
                continue
            row = f"  {f'{hour:02d}:00':<6}"
            for lot in lots:
                cell = h_df[h_df["lot"] == lot]
                if cell.empty:
                    row += f"  {'—':>{col_w}}"
                else:
                    pct = float(cell["in_band"].mean() * 100)
                    row += f"  {pct:>{col_w}.0f}%"
            print(row)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csvs(matched: pd.DataFrame, csv_dir: str):
    out = Path(csv_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Full matched dataset
    path = out / "matched_full.csv"
    matched.to_csv(path, index=False)
    print(f"  {path}  ({len(matched):,} rows)")

    # 2. Day × Lot × Tier summary
    records = []
    for tier in matched["model_tier"].unique():
        sub = matched[matched["model_tier"] == tier]
        for day in sorted(sub["date_local"].unique()):
            for lot in sorted(sub["lot"].unique()):
                cell = sub[(sub["date_local"] == day) & (sub["lot"] == lot)]
                if cell.empty:
                    continue
                m = metrics(cell)
                records.append({"tier": tier, "date": str(day), "lot": lot, **m})
    pd.DataFrame(records).to_csv(out / "day_lot_tier.csv", index=False)
    print(f"  {out / 'day_lot_tier.csv'}")

    # 3. Hour × Lot × Tier summary
    records = []
    for tier in matched["model_tier"].unique():
        sub = matched[matched["model_tier"] == tier]
        for hour in range(24):
            for lot in sorted(sub["lot"].unique()):
                cell = sub[(sub["hour_local"] == hour) & (sub["lot"] == lot)]
                if cell.empty:
                    continue
                m = metrics(cell)
                records.append({"tier": tier, "hour": hour, "lot": lot, **m})
    pd.DataFrame(records).to_csv(out / "hour_lot_tier.csv", index=False)
    print(f"  {out / 'hour_lot_tier.csv'}")

    # 4. Horizon × Lot × Tier summary
    records = []
    for tier in matched["model_tier"].unique():
        sub = matched[matched["model_tier"] == tier]
        for step in sorted(sub["minutes_ahead"].unique()):
            for lot in sorted(sub["lot"].unique()):
                cell = sub[(sub["minutes_ahead"] == step) & (sub["lot"] == lot)]
                if cell.empty:
                    continue
                m = metrics(cell)
                records.append({"tier": tier, "horizon_min": int(step), "lot": lot, **m})
    pd.DataFrame(records).to_csv(out / "horizon_lot_tier.csv", index=False)
    print(f"  {out / 'horizon_lot_tier.csv'}")

    # 5. Day × Hour × Tier (for time-series anomaly analysis)
    records = []
    for tier in matched["model_tier"].unique():
        sub = matched[matched["model_tier"] == tier]
        for day in sorted(sub["date_local"].unique()):
            for hour in range(24):
                cell = sub[(sub["date_local"] == day) & (sub["hour_local"] == hour)]
                if len(cell) < 2:
                    continue
                m = metrics(cell)
                records.append({"tier": tier, "date": str(day), "hour": hour, **m})
    pd.DataFrame(records).to_csv(out / "day_hour_tier.csv", index=False)
    print(f"  {out / 'day_hour_tier.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly deep evaluation of parking predictions")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--from", dest="from_dt")
    parser.add_argument("--to", dest="to_dt")
    parser.add_argument("--tier", help="Filter to a single model tier (lgb, lgb_v3, lgb_24h)")
    parser.add_argument("--csv-dir", help="Directory to export analysis CSVs")
    parser.add_argument("--top-errors", type=int, default=30, metavar="N",
                        help="Number of worst predictions to show per tier (default: 30)")
    parser.add_argument("--no-matrices", action="store_true",
                        help="Skip the Hour×Lot and Horizon×Lot matrices (faster output)")
    parser.add_argument("--anomaly-multiplier", type=float, default=ANOMALY_MULTIPLIER,
                        help=f"Flag cells with MAE > this × baseline (default: {ANOMALY_MULTIPLIER})")
    args = parser.parse_args()

    multiplier = args.anomaly_multiplier

    now = datetime.now(timezone.utc)
    if args.from_dt:
        from_dt = args.from_dt
        to_dt = args.to_dt or now.isoformat()
    else:
        from_dt = (now - timedelta(days=args.days)).isoformat()
        to_dt = now.isoformat()

    print(f"\nEvaluating {from_dt[:10]} → {to_dt[:10]}  (tz display: {DISPLAY_TZ_NAME})\n")

    client = get_client()
    preds = fetch_predictions(client, from_dt, to_dt, args.tier)
    actual_min = preds["target_time"].min().isoformat()
    actual_max = preds["target_time"].max().isoformat()
    actuals = fetch_actuals(client, actual_min, actual_max)
    matched = match_predictions_to_actuals(preds, actuals)
    matched = enrich(matched)

    tier_baselines = {}
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if not sub.empty:
            tier_baselines[tier] = float(np.mean(sub["abs_error"]))

    print_overall_summary(matched)
    print_by_day(matched, tier_baselines)
    print_by_hour(matched, tier_baselines)
    print_by_horizon(matched, tier_baselines)
    print_by_lot(matched, tier_baselines)
    print_day_lot_matrix(matched)

    if not args.no_matrices:
        print_hour_lot_matrix(matched)
        print_horizon_lot_matrix(matched)

    print_bias_analysis(matched)
    print_top_errors(matched, top_n=args.top_errors)
    print_anomaly_flags(matched, tier_baselines, multiplier)
    print_band_quality(matched)

    if args.csv_dir:
        section(f"CSV Exports → {args.csv_dir}")
        export_csvs(matched, args.csv_dir)

    print(f"\n{'═' * W}")
    print(f"  Done. {len(matched):,} matched pairs, {from_dt[:10]} → {to_dt[:10]}")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
