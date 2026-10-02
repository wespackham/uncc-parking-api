# LGB Near-Term Model (3h)

## Overview

LightGBM single-model architecture covering all 10 lots and all near-term horizons in one model.
Trained with `train_lgb.py` in `uncc-parking-notebook/`.

## Training

| Field | Value |
|---|---|
| **Trained** | 2026-04-08 |
| **Training data** | `data/parking_data_rows.csv` (Supabase export) |
| **Data cadence** | 5-min scraper intervals |
| **Train/test split** | Last 7 days held out as test |

## Horizons

| Field | Value |
|---|---|
| **Range** | T+5 to T+180 minutes |
| **Step** | 5 minutes |
| **Total horizons** | 36 |
| **Predictions per run** | 360 (36 horizons × 10 lots) |
| **Run cadence** | Every 5 minutes |
| **Supabase `model_tier`** | `lgb` |

## Features

- `current_capacity` — current occupancy ratio (0–1) at run time
- `delta_5`, `delta_15`, `delta_30` — occupancy change over last 5/15/30 min
- `cur_hour_sin/cos`, `cur_dow_sin/cos`, `cur_is_weekend` — current time context
- `horizon_minutes` — minutes ahead being predicted
- `deck_id` — lot identifier (categorical)
- `tgt_hour_sin/cos`, `tgt_minute_sin/cos`, `tgt_dow_sin/cos`, `tgt_is_weekend` — target time encodings
- `tgt_is_class_day`, `tgt_is_break`, `tgt_is_finals`, `tgt_is_commencement`, `tgt_is_holiday`
- `tgt_home_game_count`, `tgt_has_basketball`, `tgt_has_baseball`, `tgt_has_softball`, `tgt_has_lacrosse`, `tgt_high_impact_game`
- `tgt_condition_level`, `tgt_is_remote`, `tgt_is_cancelled`
- `tgt_temperature_f`, `tgt_humidity`, `tgt_precipitation_in`

## Model Parameters

```python
LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, max_depth=8,
              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=0.1)
# Early stopping on 2-day validation slice
# Quantile models: alpha=0.025 (lower), alpha=0.85 (upper)
```

## Performance

**Test set (7-day holdout at training time):**

| Metric | Value |
|---|---|
| MAE | 0.0228 |
| RMSE | 0.0321 |
| R² | 0.9719 |
| In Band | 93.9% |

**Real-world deployment (Apr 9 – May 10 2026, 32 days, 3.2M matched pairs):**

| Metric | Semester avg | Normal days | Event days | Post-graduation |
|--------|-------------|-------------|------------|-----------------|
| MAE | 0.0303 | ~0.025 | ~0.040 | ~0.074 |
| R² | 0.9415 | — | — | negative |
| In-Band | 88.7% | ~92% | ~85% | 97%* |
| Bias | +0.0082 | — | — | — |

*97% in-band post-graduation is a false positive — upper bound degenerates to ≈1.0, so everything is "in band" even with large errors.

**Per-lot MAE (deployment):**

| Lot | MAE | Bias | In-Band |
|-----|-----|------|---------|
| WEST | 0.0423 | +0.009 | 88.4% |
| UDU | 0.0394 | +0.016 | 87.7% |
| UDL | 0.0348 | +0.007 | 89.3% |
| CD FS | 0.0357 | +0.002 | 89.5% |
| CD VS | 0.0345 | +0.010 | 88.2% |
| ED1 | 0.0285 | +0.015 | 81.6% |
| ED2/3 | 0.0258 | +0.003 | 92.4% |
| SOUTH | 0.0243 | +0.006 | 92.1% |
| CRI | 0.0216 | +0.010 | 86.8% |
| NORTH | 0.0161 | +0.005 | 91.3% |

**Horizon degradation (deployment):**
T+5: MAE=0.018, Bias=+0.008 → T+180: MAE=0.044, Bias=+0.010. Near-linear growth. Bias is nearly constant across all horizons (±0.002) — this is a global training artifact, not a real signal.

**Hour-of-day worst cells:**

| Hour | MAE | Bias |
|------|-----|------|
| 08:00 | 0.0490 | +0.034 |
| 09:00 | 0.0496 | +0.022 |
| 00:00–05:00 | 0.009–0.015 | +0.004–0.006 |

## Known Issues

- **Global positive bias** — flat +0.007–0.016 over-prediction per lot across all horizons. Can be corrected at inference without retraining by subtracting per-lot constants (UDU: −0.016, ED1: −0.015, CRI: −0.010, CD VS: −0.010, WEST: −0.009; see FALL_2026_MODEL_PLAN.md §P0-A).
- **Upper confidence bound broken** — α=0.85 quantile model (`lgb_upper.pkl`) degenerates toward 1.0 on bounded [0,1] targets. In-band% is not a reliable quality signal for this model.
- **No event features** — `campus_events.csv` is never loaded during training. Campus events (Airband, career fairs, commencement) are pure noise. Apr 24 Airband caused WEST error of +0.69.
- **No semester-active regime** — model predicts semester-level occupancy after the calendar ends (May 11+). Post-graduation MAE inflates to 0.074.
- **Commencement failure** — UDU/WEST drain to ~0.0 during commencement (May 8–9); model predicted ~0.8. Requires `tgt_commencement_drains_lot` per-lot interaction feature.

## Files

| File | Contents |
|---|---|
| `lgb_point.pkl` | Point prediction model |
| `lgb_lower.pkl` | Lower confidence bound (α=0.025) |
| `lgb_upper.pkl` | Upper confidence bound (α=0.85) |
| `lgb_config.pkl` | Feature list, lots, horizons |
