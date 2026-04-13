"""
Compare parking predictions to actual observed occupancy.

Fetches parking_predictions and parking_data from Supabase, joins them by
matching each prediction's target_time to the closest actual observation,
then reports MAE, RMSE, R², and within-band % broken down by model tier
and optionally by lot.

Usage:
    python evaluate_predictions.py                    # last 7 days
    python evaluate_predictions.py --days 14
    python evaluate_predictions.py --from 2025-03-01 --to 2025-03-31
    python evaluate_predictions.py --lot CRI
    python evaluate_predictions.py --by-lot           # show per-lot breakdown
"""

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

TIER_ORDER = ["lgb", "lgb_24h"]

# Maximum seconds between target_time and actual observation to count as a match
MATCH_TOLERANCE_SEC = 4 * 60  # 4 minutes


def get_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env or environment.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def fetch_predictions(client, from_dt: str, to_dt: str, lot: str | None) -> pd.DataFrame:
    print(f"Fetching predictions from {from_dt} to {to_dt}...")
    rows = []
    page_size = 1000
    offset = 0
    while True:
        query = (
            client.table("parking_predictions")
            .select("created_at, target_time, lot, model_tier, prediction, confidence_low, confidence_high")
            .gte("target_time", from_dt)
            .lte("target_time", to_dt)
            .order("target_time")
            .range(offset, offset + page_size - 1)
        )
        if lot:
            query = query.eq("lot", lot)
        result = query.execute()
        batch = result.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    print(f"  {len(rows)} prediction rows")
    if not rows:
        sys.exit("No predictions found for the given range/lot.")

    df = pd.DataFrame(rows)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def fetch_actuals(client, from_dt: str, to_dt: str) -> pd.DataFrame:
    """Fetch parking_data rows and expand the JSONB 'data' dict into long form."""
    print(f"Fetching actuals from {from_dt} to {to_dt}...")
    rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.table("parking_data")
            .select("created_at, data")
            .gt("created_at", "2020-01-01")  # exclude NULL created_at rows
            .gte("created_at", from_dt)
            .lte("created_at", to_dt)
            .order("created_at")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    print(f"  {len(rows)} actual rows")
    if not rows:
        sys.exit("No actual data found for the given range.")

    # Expand JSONB → long form: (created_at, lot, actual)
    records = []
    for row in rows:
        ts = pd.to_datetime(row["created_at"], utc=True)
        for lot, value in (row["data"] or {}).items():
            if value is not None:
                records.append({"actual_time": ts, "lot": lot, "actual": float(value)})

    return pd.DataFrame(records)


def match_predictions_to_actuals(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """
    For each prediction row, find the actual observation whose timestamp is
    closest to target_time (same lot). Drops rows where no match within tolerance.
    """
    print("Matching predictions to actuals...")

    # Sort actuals for merge_asof
    actuals_sorted = actuals.sort_values("actual_time")
    preds_sorted = preds.sort_values("target_time")

    merged_parts = []
    for lot, lot_preds in preds_sorted.groupby("lot"):
        lot_actuals = actuals_sorted[actuals_sorted["lot"] == lot].copy()
        if lot_actuals.empty:
            continue

        # merge_asof: for each prediction, find the nearest-earlier actual
        merged = pd.merge_asof(
            lot_preds,
            lot_actuals[["actual_time", "actual"]],
            left_on="target_time",
            right_on="actual_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_SEC),
        )
        merged_parts.append(merged)

    if not merged_parts:
        sys.exit("No matched rows — check that actuals exist within the prediction range.")

    df = pd.concat(merged_parts, ignore_index=True)
    matched = df.dropna(subset=["actual"])
    dropped = len(df) - len(matched)
    if dropped:
        print(f"  Dropped {dropped} predictions with no actual within {MATCH_TOLERANCE_SEC}s")
    print(f"  {len(matched)} matched pairs")
    return matched


def compute_metrics(df: pd.DataFrame) -> dict:
    pred = df["prediction"].values
    actual = df["actual"].values
    low = df["confidence_low"].values
    high = df["confidence_high"].values

    errors = pred - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    within_band = np.mean((actual >= low) & (actual <= high)) * 100

    return {
        "n": len(df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "within_band_pct": within_band,
    }


def print_table(title: str, rows: list[dict], index_label: str):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")
    header = f"  {index_label:<12}  {'N':>7}  {'MAE':>7}  {'MAE(1h)':>8}  {'RMSE':>7}  {'R²':>7}  {'In Band':>8}"
    print(header)
    print(f"  {'-' * 68}")
    for r in rows:
        r2_str = f"{r['r2']:.4f}" if not np.isnan(r['r2']) else "   N/A"
        mae_1h_str = f"{r['mae_1h']:>8.4f}" if r.get("mae_1h") is not None else "     N/A"
        print(
            f"  {r['label']:<12}  {r['n']:>7,}  {r['mae']:>7.4f}  {mae_1h_str}  {r['rmse']:>7.4f}  {r2_str:>7}  {r['within_band_pct']:>7.1f}%"
        )


def print_early_horizons(tier_subsets: dict):
    """Always-on table showing accuracy at T+5, T+10, T+15 across all tiers."""
    HORIZONS = [5, 10, 15]
    print(f"\n{'=' * 72}")
    print(f"  Early-Horizon Accuracy (T+5 / T+10 / T+15)")
    print(f"{'=' * 72}")
    print(f"  {'Tier':<12}  {'T+min':>6}  {'N':>6}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}")
    print(f"  {'-' * 52}")
    for tier, df in tier_subsets.items():
        df = df.copy()
        raw_min = (df["target_time"] - df["created_at"]).dt.total_seconds() / 60
        df["minutes_ahead"] = (raw_min / 5).round().astype(int) * 5
        for h in HORIZONS:
            sub = df[df["minutes_ahead"] == h]
            if sub.empty:
                print(f"  {tier:<12}  {f'T+{h}':>6}  {'—':>6}")
                continue
            errors = sub["prediction"].values - sub["actual"].values
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((sub["actual"].values - sub["actual"].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "   N/A"
            print(f"  {tier:<12}  {f'T+{h}':>6}  {len(sub):>6,}  {mae:>7.4f}  {rmse:>7.4f}  {r2_str:>7}")
        print(f"  {'-' * 52}")


def print_by_horizon(tier: str, df: pd.DataFrame):
    """Show MAE per prediction step (T+5, T+10, ...) to detect autoregressive error compounding."""
    df = df.copy()
    raw_min = (df["target_time"] - df["created_at"]).dt.total_seconds() / 60
    df["minutes_ahead"] = (raw_min / 5).round().astype(int) * 5

    steps = sorted(df["minutes_ahead"].unique())
    if len(steps) <= 1:
        print(f"\n  Horizon breakdown — {tier}: only one step, skipping.")
        return

    print(f"\n  Horizon breakdown — {tier}")
    print(f"  {'T+min':>6}  {'N':>6}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}")
    print(f"  {'-' * 42}")
    for step in steps:
        sub = df[df["minutes_ahead"] == step]
        if sub.empty:
            continue
        errors = sub["prediction"].values - sub["actual"].values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((sub["actual"].values - sub["actual"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "   N/A"
        print(f"  {f'T+{step}':>6}  {len(sub):>6,}  {mae:>7.4f}  {rmse:>7.4f}  {r2_str:>7}")


def print_samples(tier: str, df: pd.DataFrame, n: int = 10):
    sample = df.sample(min(n, len(df)), random_state=42)
    sample = sample.sort_values("target_time")
    print(f"\n  Sample predictions — {tier}")
    print(f"  {'target_time':<22}  {'lot':<8}  {'pred':>6}  {'actual':>6}  {'error':>7}  {'low':>6}  {'high':>6}")
    print(f"  {'-' * 72}")
    for _, row in sample.iterrows():
        ts = row["target_time"].strftime("%Y-%m-%d %H:%M UTC")
        error = row["prediction"] - row["actual"]
        print(
            f"  {ts:<22}  {row['lot']:<8}  {row['prediction']:>6.3f}  {row['actual']:>6.3f}  {error:>+7.3f}  {row['confidence_low']:>6.3f}  {row['confidence_high']:>6.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate parking prediction accuracy")
    parser.add_argument("--days", type=int, default=7, help="Number of past days to evaluate (default: 7)")
    parser.add_argument("--from", dest="from_dt", help="Start datetime (ISO 8601, e.g. 2025-03-01)")
    parser.add_argument("--to", dest="to_dt", help="End datetime (ISO 8601, e.g. 2025-03-31)")
    parser.add_argument("--lot", help="Filter to a single lot (e.g. CRI)")
    parser.add_argument("--by-lot", action="store_true", help="Show per-lot breakdown within each tier")
    parser.add_argument("--by-horizon", action="store_true", help="Show MAE per prediction step (T+5, T+10, ...) to detect error compounding")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if args.from_dt:
        from_dt = args.from_dt
        to_dt = args.to_dt or now.isoformat()
    else:
        from_dt = (now - timedelta(days=args.days)).isoformat()
        to_dt = now.isoformat()

    client = get_client()

    preds = fetch_predictions(client, from_dt, to_dt, args.lot)
    # Actuals window: target_time range (predictions reference future times)
    actual_min = preds["target_time"].min().isoformat()
    actual_max = preds["target_time"].max().isoformat()
    actuals = fetch_actuals(client, actual_min, actual_max)

    matched = match_predictions_to_actuals(preds, actuals)

    to_dt_ts = pd.Timestamp(to_dt, tz="UTC") if "+" not in to_dt and "Z" not in to_dt else pd.Timestamp(to_dt).tz_convert("UTC")
    last_1h_cutoff = to_dt_ts - pd.Timedelta(hours=1)

    # --- Summary by tier ---
    tier_rows = []
    tier_subsets = {}
    for tier in TIER_ORDER:
        subset = matched[matched["model_tier"] == tier]
        if subset.empty:
            continue
        m = compute_metrics(subset)
        subset_1h = subset[subset["target_time"] >= last_1h_cutoff]
        m["mae_1h"] = float(np.mean(np.abs(subset_1h["prediction"] - subset_1h["actual"]))) if not subset_1h.empty else None
        tier_rows.append({"label": tier, **m})
        tier_subsets[tier] = subset

    print_table("Accuracy by Model Tier", tier_rows, "Tier")
    print_early_horizons(tier_subsets)

    for tier, subset in tier_subsets.items():
        print_samples(tier, subset)

    # --- Horizon breakdown ---
    if args.by_horizon:
        for tier, subset in tier_subsets.items():
            print_by_horizon(tier, subset)

    # --- Per-lot breakdown ---
    if args.by_lot:
        for tier in TIER_ORDER:
            tier_subset = matched[matched["model_tier"] == tier]
            if tier_subset.empty:
                continue
            lot_rows = []
            for lot in sorted(tier_subset["lot"].unique()):
                subset = tier_subset[tier_subset["lot"] == lot]
                m = compute_metrics(subset)
                lot_rows.append({"label": lot, **m})
            print_table(f"Per-Lot Breakdown — {tier}", lot_rows, "Lot")

    # --- Overall ---
    overall = compute_metrics(matched)
    print(f"\n  Overall ({len(matched):,} pairs, {from_dt[:10]} to {to_dt[:10]})")
    print(f"  MAE={overall['mae']:.4f}  RMSE={overall['rmse']:.4f}  R²={overall['r2']:.4f}  In Band={overall['within_band_pct']:.1f}%\n")


if __name__ == "__main__":
    main()
