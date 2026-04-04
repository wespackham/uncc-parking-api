# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the prediction service.

## Project Overview

Prediction service for the UNCC Parking system. Every 5 minutes, loads 30 trained RandomForest models, fetches live weather from Open-Meteo, queries recent occupancy from Supabase, and writes predictions for all 10 lots across a 7-day horizon back to Supabase. Also exposes a lightweight FastAPI server for health checks and debugging.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run prediction loop manually (requires .env with Supabase credentials)
python -m parking_api.predict

# Run API server locally
uvicorn parking_api.main:app --reload

# Run tests (no .env needed)
pytest tests/ -v
```

## Architecture

```
parking_api/
├── config.py          # Env vars, LOTS list, coordinates, safe_name()
├── enrichment.py      # CSV lookups for calendar, sports, disruptions (cached in memory)
├── features.py        # Feature engineering — mirrors backtest.py exactly
├── models.py          # ModelRegistry: loads .pkl files, predicts with confidence bands
├── weather.py         # Open-Meteo forecast client (async + sync)
├── supabase_client.py # Read parking_data, write/read parking_predictions
├── predict.py         # CLI entrypoint: full prediction loop (__main__)
├── main.py            # FastAPI app with lifespan (loads ModelRegistry on startup)
├── router.py          # GET /health, GET /predictions, POST /predict
└── schemas.py         # Pydantic response models
```

## Prediction Tiers

Each run generates ~1680 rows covering all lots across all horizons:

| Horizon | Model | Interval | Notes |
|---------|-------|----------|-------|
| T+30min, T+60min | 30-min RF | 30-min steps | Uses real lag features from Supabase |
| T+90min to T+3hrs | 60-min RF | 15-min steps | Autoregressive: lags from prior predictions |
| T+3hrs to T+7days | Baseline RF | 1-hour steps | Calendar + weather only, no lags |

If the latest `parking_data` row is >15 minutes old, horizon models are skipped and only baseline predictions are written.

## Feature Engineering

**Critical:** `features.py` must mirror `uncc-parking-notebook/backtest.py:engineer_features()` exactly. Any change to the training notebook requires updating both files.

25 baseline features (in model-expected order):
1. `hour`, `minute`, `day_of_week`, `is_weekend`
2. `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
3. `is_class_day`, `is_break`, `is_finals`, `is_commencement`, `is_holiday`
4. `home_game_count`, `has_basketball`, `has_baseball`, `has_softball`, `has_lacrosse`, `high_impact_game`
5. `condition_level`, `is_remote`, `is_cancelled`
6. `temperature_f`, `humidity`, `precipitation_in`

Horizon models add 5 lot-specific lag features: `{LOT}_now`, `{LOT}_lag_5`, `{LOT}_lag_10`, `{LOT}_lag_15`, `{LOT}_delta_5`
(5-minute intervals; the scraper runs every 5 minutes)

Weekends always force `is_class_day=0` regardless of the CSV.

## Models

- 51 `.pkl` files in `models/` tracked by Git LFS
- Trained with scikit-learn 1.8.0 — requirements.txt must stay pinned to this version
- Confidence bands: `np.array([tree.predict(X) for tree in model.estimators_])` → mean ± 2*std, clamped to [0, 1]
- Feature lists per model stored in `models/features.pkl` (baseline) and `models/{LOT}_{horizon}min_features.pkl`
- Lot name sanitization: `"CD FS"` → `"CD_FS"`, `"ED2/3"` → `"ED2_3"`

## Enrichment Data

CSVs in `data/` are copied from `uncc-parking-notebook/data/` and committed:

| File | Update cadence | Key columns |
|------|---------------|-------------|
| `academic_calendar.csv` | Each semester | `date`, `is_class_day`, `category` |
| `sports_schedule.csv` | As published | `date`, `sport`, `home_away`, `parking_impact` |
| `campus_disruptions.csv` | As needed | `date`, `condition`, `classes` |

When updating a CSV, copy it to both `uncc-parking-notebook/data/` and `uncc-parking-api/data/`.

## API Endpoints

- `GET /health` — models loaded count, CSV date coverage, Supabase status
- `GET /predictions?lot=CRI&from=...&to=...` — read from `parking_predictions` table
- `POST /predict` — manual trigger, requires `Authorization: Bearer {API_KEY}`

The dashboard reads from Supabase directly — it does not call the API.

## Supabase Table: `parking_predictions`

```sql
CREATE TABLE parking_predictions (
  id              BIGSERIAL PRIMARY KEY,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  target_time     TIMESTAMPTZ NOT NULL,
  lot             TEXT NOT NULL,
  model_tier      TEXT NOT NULL,
  prediction      REAL NOT NULL,
  confidence_low  REAL NOT NULL,
  confidence_high REAL NOT NULL
);
CREATE INDEX idx_pred_target ON parking_predictions (target_time, lot);
ALTER TABLE parking_predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read" ON parking_predictions FOR SELECT USING (true);
```

Run this in Supabase SQL editor before deploying.

## Deployment (EC2)

Three systemd units:

| Unit | Type | Schedule |
|------|------|----------|
| `parking-api.service` | long-running | always on, uvicorn port 8000 |
| `parking-predictor.service` | oneshot | triggered by timer |
| `parking-predictor.timer` | timer | `*:1/5` (fires at :01, :06, :11... — 1 min after scraper) |

CI/CD deploys on push to main via `.github/workflows/deploy-api.yml` (same SSH pattern as scraper).

## Environment Variables

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...   # optional, for failure alerts
API_KEY=your-secret-key                                     # for POST /predict
```

## Tests

```bash
pytest tests/ -v        # 17 tests, no env vars needed
```

- `tests/test_features.py` — 12 tests verifying feature output is correct
- `tests/test_predictions.py` — 5 smoke tests: all 30 models load, predictions in [0, 1]
