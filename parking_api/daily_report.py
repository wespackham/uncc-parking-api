"""Daily accuracy report posted to Discord at 6pm ET.

Fetches the last 24 hours of predictions and actuals from Supabase,
computes MAE/RMSE/R²/In-Band per model tier, and sends a formatted
summary to the Discord webhook.

Run as CLI: python -m parking_api.daily_report
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

TIER_ORDER = ["lgb", "lgb_24h"]
MATCH_TOLERANCE_SEC = 4 * 60


def _get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _fetch_predictions(client, from_dt: str, to_dt: str) -> pd.DataFrame:
    rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.table("parking_predictions")
            .select("created_at, target_time, lot, model_tier, prediction, confidence_low, confidence_high")
            .gte("target_time", from_dt)
            .lte("target_time", to_dt)
            .order("target_time")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def _fetch_actuals(client, from_dt: str, to_dt: str) -> pd.DataFrame:
    rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            client.table("parking_data")
            .select("created_at, data")
            .gt("created_at", "2020-01-01")
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

    records = []
    for row in rows:
        ts = pd.to_datetime(row["created_at"], utc=True)
        for lot, value in (row["data"] or {}).items():
            if value is not None:
                records.append({"actual_time": ts, "lot": lot, "actual": float(value)})
    return pd.DataFrame(records)


def _match(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    actuals_sorted = actuals.sort_values("actual_time")
    preds_sorted = preds.sort_values("target_time")
    parts = []
    for lot, lot_preds in preds_sorted.groupby("lot"):
        lot_actuals = actuals_sorted[actuals_sorted["lot"] == lot].copy()
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
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).dropna(subset=["actual"])


def _metrics(df: pd.DataFrame) -> dict:
    pred = df["prediction"].values
    actual = df["actual"].values
    errors = pred - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    in_band = np.mean((actual >= df["confidence_low"].values) & (actual <= df["confidence_high"].values)) * 100
    return {"n": len(df), "mae": mae, "rmse": rmse, "r2": r2, "in_band": in_band}


def _build_report(matched: pd.DataFrame, date_str: str) -> list[str]:
    """Return a list of Discord message strings (split to stay under 2000 chars each)."""
    messages = []

    # --- Summary table ---
    lines = [f"**Daily Accuracy Report — {date_str}**", "```"]
    lines.append(f"{'Tier':<12}  {'N':>7}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'In Band':>8}")
    lines.append("-" * 56)
    tier_subsets = {}
    for tier in TIER_ORDER:
        sub = matched[matched["model_tier"] == tier]
        if sub.empty:
            continue
        m = _metrics(sub)
        r2_str = f"{m['r2']:.4f}" if not np.isnan(m["r2"]) else "   N/A"
        lines.append(
            f"{tier:<12}  {m['n']:>7,}  {m['mae']:>7.4f}  {m['rmse']:>7.4f}  {r2_str:>7}  {m['in_band']:>7.1f}%"
        )
        tier_subsets[tier] = (sub, m)
    lines.append("```")
    messages.append("\n".join(lines))

    # --- Per-lot breakdown per tier ---
    for tier, (sub, m) in tier_subsets.items():
        lines = [f"**Per-lot breakdown — {tier}**", "```"]
        lines.append(f"{'Lot':<8}  {'N':>6}  {'MAE':>7}  {'RMSE':>7}  {'R²':>7}  {'In Band':>8}")
        lines.append("-" * 52)
        for lot in sorted(sub["lot"].unique()):
            lsub = sub[sub["lot"] == lot]
            lm = _metrics(lsub)
            r2_str = f"{lm['r2']:.4f}" if not np.isnan(lm["r2"]) else "   N/A"
            lines.append(
                f"{lot:<8}  {lm['n']:>6,}  {lm['mae']:>7.4f}  {lm['rmse']:>7.4f}  {r2_str:>7}  {lm['in_band']:>7.1f}%"
            )
        lines.append("```")
        messages.append("\n".join(lines))

    return messages


def _post(message: str):
    """Post to Discord, splitting if over the 2000-char limit."""
    while len(message) > 1990:
        split_at = message.rfind("\n", 0, 1990)
        if split_at == -1:
            split_at = 1990
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message[:split_at]}, timeout=15)
        message = message[split_at:].lstrip("\n")
    if message.strip():
        httpx.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=15)


def main():
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set, exiting.")
        sys.exit(0)

    now = datetime.now(timezone.utc)
    from_dt = (now - timedelta(hours=24)).isoformat()
    to_dt = now.isoformat()
    date_str = now.strftime("%Y-%m-%d")

    client = _get_client()

    preds = _fetch_predictions(client, from_dt, to_dt)
    if preds.empty:
        _post(f"⚠️ Daily report ({date_str}): no predictions found in the last 24h.")
        return

    actual_min = preds["target_time"].min().isoformat()
    actual_max = preds["target_time"].max().isoformat()
    actuals = _fetch_actuals(client, actual_min, actual_max)
    if actuals.empty:
        _post(f"⚠️ Daily report ({date_str}): no actuals found for the prediction window.")
        return

    matched = _match(preds, actuals)
    if matched.empty:
        _post(f"⚠️ Daily report ({date_str}): no matched prediction/actual pairs found.")
        return

    for msg in _build_report(matched, date_str):
        _post(msg)


if __name__ == "__main__":
    main()
