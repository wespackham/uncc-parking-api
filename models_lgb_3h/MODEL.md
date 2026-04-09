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

## Performance (test set)

| Metric | Value |
|---|---|
| MAE | 0.0228 |
| RMSE | 0.0321 |
| R² | 0.9719 |
| In Band | 93.9% |

Weakest lot: CD VS. Upper confidence bound tends toward 1.0 due to bounded [0,1] target — known limitation.

## Files

| File | Contents |
|---|---|
| `lgb_point.pkl` | Point prediction model |
| `lgb_lower.pkl` | Lower confidence bound (α=0.025) |
| `lgb_upper.pkl` | Upper confidence bound (α=0.85) |
| `lgb_config.pkl` | Feature list, lots, horizons |
