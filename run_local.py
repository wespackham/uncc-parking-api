"""
Local prediction preview — runs models directly without pushing to the cloud.

Shows:
  1. Current occupancy for each lot (fetched from Supabase)
  2. 30-min model predictions at T+30 and T+60
  3. 60-min model predictions at T+90, T+105, T+120
  4. Baseline model predictions every 30 minutes up to T+120

Usage:
    python run_local.py               # full run (requires .env with Supabase creds)
    python run_local.py --no-live     # skip Supabase, use 0.5 as placeholder occupancy
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# sklearn warns about feature names on every tree prediction — suppress globally
warnings.filterwarnings("ignore", category=UserWarning)

# ── Dependency check ─────────────────────────────────────────────────────────

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Missing dependencies. Run: pip install -r requirements.txt")

# ── Add repo root to path ─────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parking_api.config import LOTS
from parking_api.models import ModelRegistry
from parking_api.features import build_feature_vector
from parking_api.weather import fetch_forecast_sync, get_weather_for_time

# ── Argument parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--no-live", action="store_true", help="Skip Supabase, use placeholder occupancy")
args = parser.parse_args()

# ── Load models ───────────────────────────────────────────────────────────────

print("\nLoading models...", end=" ", flush=True)
try:
    registry = ModelRegistry()
except FileNotFoundError as e:
    sys.exit(f"\nCould not load models: {e}\nEnsure models/ directory is present with .pkl files.")
print(f"OK ({registry.list_models()['total']} models)")

# ── Fetch current occupancy ───────────────────────────────────────────────────

current_occupancy: dict[str, float] = {}

if args.no_live:
    print("Skipping live data (--no-live). Using 0.50 placeholder for all lots.")
    current_occupancy = {lot: 0.50 for lot in LOTS}
    recent_rows = []
else:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from parking_api.supabase_client import fetch_recent_rows
        print("Fetching current occupancy from Supabase...", end=" ", flush=True)
        recent_rows = fetch_recent_rows(n=20)
        if not recent_rows:
            print("No data returned. Using 0.50 placeholder.")
            current_occupancy = {lot: 0.50 for lot in LOTS}
            recent_rows = []
        else:
            latest = recent_rows[0]["data"]
            if isinstance(latest, str):
                latest = json.loads(latest)
            current_occupancy = {lot: float(latest.get(lot, 0.0)) for lot in LOTS}
            ts = recent_rows[0]["created_at"]
            print(f"OK (latest: {ts})")
    except Exception as e:
        print(f"\nSupabase fetch failed: {e}\nUsing 0.50 placeholder.")
        current_occupancy = {lot: 0.50 for lot in LOTS}
        recent_rows = []

# ── Extract lag values per lot ────────────────────────────────────────────────

def _extract_lags(rows: list[dict], lot: str, n: int = 4) -> list[float]:
    """Sample every 3rd row to get 15-min spacing (matches training cadence)."""
    values = []
    step = 3
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
    if not values:
        # Fall back to current occupancy when no rows available
        values = [current_occupancy.get(lot, 0.5)]
    return values

# ── Fetch weather ─────────────────────────────────────────────────────────────

print("Fetching weather forecast...", end=" ", flush=True)
try:
    weather_df = fetch_forecast_sync()
    print(f"OK ({len(weather_df)} hourly rows)")
except Exception as e:
    print(f"\nWeather fetch failed: {e}\nUsing zero-weather fallback.")
    weather_df = None

def _get_weather(dt: datetime) -> dict:
    if weather_df is not None:
        return get_weather_for_time(weather_df, dt)
    return {"temperature_f": 0.0, "humidity": 0.0, "precipitation_in": 0.0}

# ── Build prediction schedule ─────────────────────────────────────────────────
# All feature engineering uses UTC to match training data (backtest.py used UTC created_at)

now_utc = datetime.now(timezone.utc)
now_display = now_utc.strftime("%H:%M UTC")

# Time points for the 2-hour preview
SCHEDULE = [
    # (label, minutes_ahead, model_tier)
    ("+30m",  30,  "30min"),
    ("+60m",  60,  "30min"),
    ("+90m",  90,  "60min"),
    ("+105m", 105, "60min"),
    ("+120m", 120, "60min"),
]

BASELINE_POINTS = [
    # (label, minutes_ahead)
    ("+30m",  30),
    ("+60m",  60),
    ("+90m",  90),
    ("+120m", 120),
]

# ── Run predictions ───────────────────────────────────────────────────────────

# Structure: results[lot][label] = (mean, low, high)
horizon_results: dict[str, dict] = {lot: {} for lot in LOTS}
baseline_results: dict[str, dict] = {lot: {} for lot in LOTS}

print("Running predictions...", end=" ", flush=True)

for lot in LOTS:
    lags = _extract_lags(recent_rows, lot)
    predicted_chain = list(lags)

    # Horizon predictions (30min + 60min models)
    for label, minutes_ahead, tier in SCHEDULE:
        target_dt = now_utc + timedelta(minutes=minutes_ahead)
        weather_row = _get_weather(target_dt)
        feat_names = registry.get_feature_names(lot, tier)
        X = build_feature_vector(
            dt=target_dt,
            weather_row=weather_row,
            lot=lot,
            recent_values=predicted_chain[:4],
            feature_names=feat_names,
        )
        mean, low, high = registry.predict(lot, tier, X)
        horizon_results[lot][label] = (mean, low, high)
        predicted_chain.insert(0, mean)

    # Baseline predictions (independent of lags)
    feat_names_base = registry.get_feature_names(lot, "baseline")
    for label, minutes_ahead in BASELINE_POINTS:
        target_dt = now_utc + timedelta(minutes=minutes_ahead)
        weather_row = _get_weather(target_dt)
        X = build_feature_vector(
            dt=target_dt,
            weather_row=weather_row,
            feature_names=feat_names_base,
        )
        mean, low, high = registry.predict(lot, "baseline", X)
        baseline_results[lot][label] = (mean, low, high)

print("Done.\n")

# ── Display ───────────────────────────────────────────────────────────────────

W = 12  # column width

HORIZON_LABELS = [label for label, _, _ in SCHEDULE]
HORIZON_TIERS  = [tier  for _, _, tier  in SCHEDULE]
BASELINE_LABELS = [label for label, _ in BASELINE_POINTS]

def pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def band(low: float, high: float) -> str:
    return f"[{low*100:.0f}-{high*100:.0f}]"

divider = "─" * (10 + W * (len(HORIZON_LABELS) + 1))

print("═" * len(divider))
print(f"  UNCC PARKING — LOCAL PREDICTION PREVIEW     {now_display}")
print("═" * len(divider))

# ── Section 1: Current occupancy ─────────────────────────────────────────────

print("\n  CURRENT OCCUPANCY\n")
for lot in LOTS:
    occ = current_occupancy.get(lot, 0.0)
    bar_len = int(occ * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  {lot:<8}  {pct(occ):>6}  {bar}")

# ── Section 2: Horizon models (30min + 60min) ────────────────────────────────

print(f"\n\n  HORIZON MODELS  (30min model: T+30/60 · 60min model: T+90/105/120)\n")

# Header
tiers_row = "  " + " " * 10 + "".join(f"  {t:^{W-2}}" for t in HORIZON_TIERS)
label_row = "  " + f"{'LOT':<10}" + "".join(f"  {l:^{W-2}}" for l in HORIZON_LABELS)
print(tiers_row)
print(label_row)
print("  " + divider[2:])

for lot in LOTS:
    mean_row  = f"  {lot:<10}"
    band_row  = "  " + " " * 10
    for label in HORIZON_LABELS:
        mean, low, high = horizon_results[lot][label]
        mean_row += f"  {pct(mean):^{W-2}}"
        band_row += f"  {band(low, high):^{W-2}}"
    print(mean_row)
    print(band_row)

# ── Section 3: Baseline model comparison ─────────────────────────────────────

print(f"\n\n  BASELINE MODEL  (calendar + weather only, no lag features)\n")

label_row_b = "  " + f"{'LOT':<10}" + "".join(f"  {l:^{W-2}}" for l in BASELINE_LABELS)
divider_b = "─" * (10 + W * (len(BASELINE_LABELS) + 1))
print(label_row_b)
print("  " + divider_b[2:])

for lot in LOTS:
    mean_row = f"  {lot:<10}"
    band_row = "  " + " " * 10
    for label in BASELINE_LABELS:
        mean, low, high = baseline_results[lot][label]
        mean_row += f"  {pct(mean):^{W-2}}"
        band_row += f"  {band(low, high):^{W-2}}"
    print(mean_row)
    print(band_row)

print("\n" + "═" * len(divider))
print("  Confidence bands show ±2σ across 200 RF trees.")
print("  Horizon models use current occupancy as lag input.")
print("  All times in UTC (matching training data timezone).")
print("═" * len(divider) + "\n")
