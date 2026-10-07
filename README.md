# UNCC Parking — Prediction Service

LightGBM forecasts of parking-deck availability for UNC Charlotte, written every 5 minutes (3h
horizon) and hourly (24h horizon) to PostgreSQL on the project's DigitalOcean droplet, via PostgREST at
`https://parking.abrupt.app`. The dashboard (`uncc-parking-dashboard`) reads them.

## What runs (systemd timers on the droplet, UTC)

| Unit | Schedule | Purpose |
|---|---|---|
| `parking-predictor` | every 5 min | 3h tiers (`lgb`, `lgb_v3`, `lgb_v4`) |
| `parking-predictor-24h` | hourly | 24h tiers (`lgb_24h`, `lgb_24h_v2`) |
| `parking-analytics` | every 10 min | refresh the precomputed dashboard analytics (`sql/analytics_snapshot.sql`) |
| `parking-daily-report` | 22:00 | Discord accuracy report |
| `parking-export` | Mon 03:00 | archive predictions older than 7 days to Parquet (every row saved and verified before deletion) |

Lots listed in `SUPPRESSED_LOTS` (broken sensor feeds) are estimated from the other decks
(`parking_api/estimate.py`, `models_west_est/`) and flagged `"estimated": true`.

## Deploy

Push to `main`. `.github/workflows/deploy-api.yml` resets the droplet checkout to `origin/main`,
installs requirements, applies `sql/analytics_snapshot.sql`, installs the units and restarts the timers.
Don't edit code on the droplet.

## Develop

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest -q          # no .env needed
venv/bin/python -m parking_api.predict             # 3h run (needs .env)
venv/bin/python -m parking_api.predict --model 24h
```

`.env`: `SUPABASE_URL` (PostgREST base URL), `SUPABASE_KEY` (JWT), `DISCORD_WEBHOOK_URL`, optional
`DISCORD_LABEL`, `SUPPRESSED_LOTS`. A laptop `.env` should use the read-only anon key.

Feature engineering in `parking_api/features.py` must mirror the training code in
`uncc-parking-notebook` (current retrain: `train_2026_10.py`). Model details: `models_*/MODEL.md`.
