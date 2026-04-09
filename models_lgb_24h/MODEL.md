# LGB Long-Range Model (24h)

## Overview

LightGBM single-model architecture covering all 10 lots and all long-range horizons in one model.
Trained with `train_lgb_v2.py` in `uncc-parking-notebook/` on an RTX 3060 GPU.

## Training

| Field | Value |
|---|---|
| **Trained** | 2026-04-08 |
| **Training data** | `data/parking_data_rows.csv` (Supabase export) |
| **Data cadence** | 5-min scraper intervals |
| **Train/test split** | Last 7 days held out as test |
| **Training rows** | ~56.5M (after lag feature expansion across all horizons) |
| **Best iteration** | 676 trees (early stopping from 2000 max) |

## Horizons

| Field | Value |
|---|---|
| **Range** | T+5 to T+1440 minutes (24 hours) |
| **Step** | 5 minutes |
| **Total horizons** | 288 |
| **Predictions per run** | 2,880 (288 horizons × 10 lots) |
| **Run cadence** | Every 1 hour |
| **Supabase `model_tier`** | `lgb_24h` |

## Features

Same as the 3h model — lag features (delta_5/15/30) carry less signal at long horizons but are kept for consistency. The model learns to down-weight them automatically at large `horizon_minutes` values.

- `current_capacity`, `delta_5`, `delta_15`, `delta_30`
- `cur_hour_sin/cos`, `cur_dow_sin/cos`, `cur_is_weekend`
- `horizon_minutes`, `deck_id`
- `tgt_hour_sin/cos`, `tgt_minute_sin/cos`, `tgt_dow_sin/cos`, `tgt_is_weekend`
- `tgt_is_class_day`, `tgt_is_break`, `tgt_is_finals`, `tgt_is_commencement`, `tgt_is_holiday`
- `tgt_home_game_count`, `tgt_has_basketball`, `tgt_has_baseball`, `tgt_has_softball`, `tgt_has_lacrosse`, `tgt_high_impact_game`
- `tgt_condition_level`, `tgt_is_remote`, `tgt_is_cancelled`
- `tgt_temperature_f`, `tgt_humidity`, `tgt_precipitation_in`

## Model Parameters

```python
LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63, max_depth=8,
              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=0.1, device='gpu')
# float32 cast to reduce memory (~9GB peak)
# Early stopping on 2-day validation slice
# Quantile models: alpha=0.025 (lower), alpha=0.85 (upper)
# Quantile models trained on CPU (GPU quantile not supported by LightGBM)
```

## Performance (test set — last 7 days withheld)

| Horizon bucket | MAE | R² |
|---|---|---|
| T+5–30 min | 0.0262 | 0.9619 |
| T+35–60 min | 0.0309 | 0.9515 |
| T+65–180 min | 0.0380 | 0.9423 |
| T+3–6 h | 0.0432 | 0.9383 |
| T+6–12 h | 0.0460 | 0.9348 |
| T+12–24 h | 0.0479 | 0.9297 |
| **Overall** | **0.0423** | **0.9353** |

Weakest lot: CD VS (R²=0.56). WEST has highest MAE (0.077).
Upper confidence bound degenerates toward 1.0 — known limitation of quantile regression on bounded [0,1] targets.

Note: test week (early April 2026) is late spring semester — may include unusual patterns not representative of typical weeks.

## Files

| File | Contents |
|---|---|
| `lgb_point.pkl` | Point prediction model |
| `lgb_lower.pkl` | Lower confidence bound (α=0.025) |
| `lgb_upper.pkl` | Upper confidence bound (α=0.85) |
| `lgb_config.pkl` | Feature list, lots, horizons |
