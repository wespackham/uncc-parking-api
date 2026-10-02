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

## Performance

**Test set (7-day holdout at training time):**

| Horizon bucket | MAE | R² |
|---|---|---|
| T+5–30 min | 0.0262 | 0.9619 |
| T+35–60 min | 0.0309 | 0.9515 |
| T+65–180 min | 0.0380 | 0.9423 |
| T+3–6 h | 0.0432 | 0.9383 |
| T+6–12 h | 0.0460 | 0.9348 |
| T+12–24 h | 0.0479 | 0.9297 |
| **Overall** | **0.0423** | **0.9353** |

**Real-world deployment (Apr 9 – May 10 2026, 32 days, 2.1M matched pairs):**

| Metric | Semester avg | Normal days | Event days | Post-graduation |
|--------|-------------|-------------|------------|-----------------|
| MAE | 0.0595 | ~0.042 | ~0.083 | **0.2218** |
| R² | 0.8076 | — | — | **−18.0** |
| In-Band | 65.3% | ~72% | ~62% | 17.7% |
| Bias | +0.0202 | — | — | — |

The test-set/deploy gap (0.042 → 0.060) is larger than lgb's (0.020 → 0.030) because the 24h model has no live signal at long horizons and is purely calendar-driven.

**Hour-of-day worst cells (deployment):**

| Hour | MAE | Bias | Note |
|------|-----|------|------|
| 10:00 | 0.0950 | +0.062 | Peak structural over-prediction |
| 11:00 | 0.0992 | +0.057 | Worst single hour |
| 09:00 | 0.0885 | +0.061 | |
| 01:00–04:00 | 0.026–0.028 | +0.004–0.009 | Best hours |

MAE stays plateau-flat from T+60 to T+1440 (~0.059–0.065) — the model converges to a mean prediction past the first hour and horizon no longer matters.

**Per-lot MAE (deployment):**

| Lot | MAE | Bias |
|-----|-----|------|
| WEST | 0.0893 | +0.020 |
| UDU | 0.0751 | +0.023 |
| UDL | 0.0727 | +0.024 |
| CD FS | 0.0692 | +0.012 |
| CD VS | 0.0643 | +0.038 |
| ED1 | 0.0566 | +0.043 |
| ED2/3 | 0.0475 | +0.005 |
| SOUTH | 0.0451 | +0.002 |
| CRI | 0.0447 | +0.023 |
| NORTH | 0.0310 | +0.011 |

## Known Issues

- **Structural daytime over-prediction** — MAE 0.085–0.099 at 09:00–15:00 with +0.052–0.062 positive bias. The model has no live signal at long horizons and predicts semester-level occupancy regardless of current state. Fix: add `tgt_hist_mean_occupancy[lot, hour, dow]` lookup feature (see FALL_2026_MODEL_PLAN.md §P1-D).
- **Upper confidence bound broken** — α=0.85 quantile degenerates toward 1.0 on bounded [0,1] targets. In-band% is not a reliable quality signal.
- **Post-graduation regime failure** — MAE=0.22, R²=−18 post-commencement. Model predicts semester occupancy with no way to detect lot-emptying. SOUTH: MAE=0.50, in-band=0.0%. Fix: `tgt_is_semester_active` feature + lgb_24h summer suppression.
- **No event features** — identical to lgb; campus events are zero signal. CD FS, WEST, UDU event-day errors are all unmodeled.
- **CD VS structural positive bias** — +0.038 per-lot bias; strongest structural overestimation across all lots. Partially fixed by P1-D hist_mean feature.
- **No per-lot commencement response** — UDU/WEST drain to ~0.0 during commencement; 24h model predicted ~0.7–0.8. Worst cell: May 8 10:00 UDU, MAE=0.344 (5.8× baseline).

## Files

| File | Contents |
|---|---|
| `lgb_point.pkl` | Point prediction model |
| `lgb_lower.pkl` | Lower confidence bound (α=0.025) |
| `lgb_upper.pkl` | Upper confidence bound (α=0.85) |
| `lgb_config.pkl` | Feature list, lots, horizons |
