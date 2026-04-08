# LightGBM Parking Prediction Model

Single LightGBM model covering all 10 lots and all horizons T+5 to T+180 (3 hours).  
Trained in `uncc-parking-notebook/train_lgb.py`. Models saved to `models_lgb/`.

## Performance (7-day holdout, 2026-03-30 to 2026-04-06)

| Horizon     | MAE    | RMSE   | R²     | In Band |
|-------------|--------|--------|--------|---------|
| T+5–30 min  | 0.0133 | 0.0197 | 0.9939 | 96.0%   |
| T+35–60 min | 0.0172 | 0.0269 | 0.9886 | 94.7%   |
| T+65–120 min| 0.0241 | 0.0392 | 0.9757 | 94.3%   |
| T+125–180 min| 0.0317 | 0.0517 | 0.9572 | 93.9%   |
| **Overall** | **0.0236** | — | **0.9749** | **94.5%** |

Compared to RF v1 30min model: similar MAE at T+5–30, but covers 3× the horizon range
with a single model and no autoregressive error compounding.

## Compute and Memory (measured)

| Metric | LightGBM | RF v1+v2 (estimated) |
|--------|----------|----------------------|
| Model files | 3 files, **6.8 MB** | 100+ files, ~300 MB |
| Load time | ~1.8s | ~5–10s |
| RAM loaded | **56 MB** | ~300–400 MB |
| Inference (360 rows) | **8 ms** | ~100–300 ms |

LightGBM inference is a single batch call of 360 rows (10 lots × 36 horizons) against
3 models. No loops, no autoregressive chaining.

## Model Files

```
uncc-parking-api/models_lgb/
├── lgb_point.pkl     849 KB   Point prediction model (135 trees, early stopping)
├── lgb_lower.pkl     6.1 MB   Lower confidence bound (α=0.025, 1000 trees)
├── lgb_upper.pkl     6 KB     Upper confidence bound (α=0.975, 1 tree — see Known Issues)
└── lgb_config.pkl    1 KB     Feature list, lot names, horizon list
```

Load with:
```python
import pickle
with open('models_lgb/lgb_point.pkl', 'rb') as f: point = pickle.load(f)
with open('models_lgb/lgb_lower.pkl', 'rb') as f: lower = pickle.load(f)
with open('models_lgb/lgb_upper.pkl', 'rb') as f: upper = pickle.load(f)
with open('models_lgb/lgb_config.pkl', 'rb') as f: cfg = pickle.load(f)
# cfg['features'] — ordered feature list
# cfg['lots']     — ['CRI', 'ED1', ...]
# cfg['horizons'] — [5, 10, ..., 180]
```

## Feature List (35 features, in model-expected order)

### Current state (at prediction time t)
| Feature | Description |
|---------|-------------|
| `current_capacity` | Lot occupancy now (0–1) |
| `delta_5` | current_capacity − occupancy 5 min ago |
| `delta_15` | current_capacity − occupancy 15 min ago |
| `delta_30` | current_capacity − occupancy 30 min ago |

### Current time context (at t, UTC)
| Feature | Description |
|---------|-------------|
| `cur_hour_sin` | sin(2π × hour / 24) |
| `cur_hour_cos` | cos(2π × hour / 24) |
| `cur_dow_sin` | sin(2π × day_of_week / 7) |
| `cur_dow_cos` | cos(2π × day_of_week / 7) |
| `cur_is_weekend` | 1 if Saturday or Sunday |

### Horizon and lot identity
| Feature | Description |
|---------|-------------|
| `horizon_minutes` | Integer: 5, 10, 15, ..., 180 |
| `deck_id` | Categorical: lot name (CRI, ED1, UDL, UDU, WEST, CD FS, CD VS, ED2/3, NORTH, SOUTH) |

### Target-time features (at t + horizon_minutes, UTC)
These are the conditions that will exist at the moment being predicted.

**Time:**
| Feature | Description |
|---------|-------------|
| `tgt_hour_sin` | sin(2π × hour / 24) at target time |
| `tgt_hour_cos` | cos(2π × hour / 24) at target time |
| `tgt_minute_sin` | sin(2π × minute / 60) at target time |
| `tgt_minute_cos` | cos(2π × minute / 60) at target time |
| `tgt_dow_sin` | sin(2π × day_of_week / 7) at target time |
| `tgt_dow_cos` | cos(2π × day_of_week / 7) at target time |
| `tgt_is_weekend` | 1 if target falls on Saturday or Sunday |

**Academic calendar** (looked up by target date from `data/academic_calendar.csv`):
| Feature | Description |
|---------|-------------|
| `tgt_is_class_day` | 1 on instructional days (forced 0 on weekends) |
| `tgt_is_break` | 1 during spring recess |
| `tgt_is_finals` | 1 during finals period |
| `tgt_is_commencement` | 1 on commencement day |
| `tgt_is_holiday` | 1 when university is closed |

**Sports** (looked up by target date from `data/sports_schedule.csv`, home games only):
| Feature | Description |
|---------|-------------|
| `tgt_home_game_count` | Number of home games on target date |
| `tgt_has_basketball` | 1 if home basketball game |
| `tgt_has_baseball` | 1 if home baseball game |
| `tgt_has_softball` | 1 if home softball game |
| `tgt_has_lacrosse` | 1 if home lacrosse game |
| `tgt_high_impact_game` | 1 if any home game has parking_impact='high' |

**Campus disruptions** (looked up by target date from `data/campus_disruptions.csv`):
| Feature | Description |
|---------|-------------|
| `tgt_condition_level` | 0=Normal, 1=C1, 2=C2 |
| `tgt_is_remote` | 1 if classes moved remote |
| `tgt_is_cancelled` | 1 if classes cancelled |

**Weather** (looked up by target hour, rounded, from `data/weather.csv`):
| Feature | Description |
|---------|-------------|
| `tgt_temperature_f` | Temperature at target hour (°F) |
| `tgt_humidity` | Relative humidity at target hour (%) |
| `tgt_precipitation_in` | Precipitation at target hour (inches) |

## Inference: Building One Prediction Cycle

Each run generates 360 rows (10 lots × 36 horizons) in a single DataFrame:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

def build_inference_batch(now_utc, lot_occupancy, lot_deltas,
                          cal_lookup, sports_lookup, dis_lookup, weather_lookup,
                          cfg):
    """
    now_utc       — datetime (UTC) of the current observation
    lot_occupancy — dict {lot: current_capacity (0-1)}
    lot_deltas    — dict {lot: (delta_5, delta_15, delta_30)}
    *_lookup      — pre-loaded lookup dicts/DataFrames (same as training)
    cfg           — loaded from lgb_config.pkl
    """
    rows = []
    for lot in cfg['lots']:
        cap              = lot_occupancy[lot]
        d5, d15, d30     = lot_deltas[lot]
        cur_dt           = now_utc.replace(tzinfo=None)

        for h in cfg['horizons']:
            tgt_dt   = cur_dt + timedelta(minutes=h)
            tgt_date = tgt_dt.date().isoformat()
            tgt_hour = tgt_dt.replace(minute=0, second=0, microsecond=0)

            cal  = cal_lookup.get(tgt_date, {})
            spt  = sports_lookup.get(tgt_date, {})
            dis  = dis_lookup.get(tgt_date, {})
            wthr = weather_lookup.get(tgt_hour, {})

            tgt_dow = tgt_dt.weekday()
            tgt_is_weekend = int(tgt_dow >= 5)

            rows.append({
                'current_capacity':   cap,
                'delta_5':            d5,
                'delta_15':           d15,
                'delta_30':           d30,
                'cur_hour_sin':  np.sin(2*np.pi*cur_dt.hour/24),
                'cur_hour_cos':  np.cos(2*np.pi*cur_dt.hour/24),
                'cur_dow_sin':   np.sin(2*np.pi*cur_dt.weekday()/7),
                'cur_dow_cos':   np.cos(2*np.pi*cur_dt.weekday()/7),
                'cur_is_weekend': int(cur_dt.weekday() >= 5),
                'horizon_minutes': h,
                'deck_id': lot,
                'tgt_hour_sin':  np.sin(2*np.pi*tgt_dt.hour/24),
                'tgt_hour_cos':  np.cos(2*np.pi*tgt_dt.hour/24),
                'tgt_minute_sin': np.sin(2*np.pi*tgt_dt.minute/60),
                'tgt_minute_cos': np.cos(2*np.pi*tgt_dt.minute/60),
                'tgt_dow_sin':   np.sin(2*np.pi*tgt_dow/7),
                'tgt_dow_cos':   np.cos(2*np.pi*tgt_dow/7),
                'tgt_is_weekend': tgt_is_weekend,
                'tgt_is_class_day':    0 if tgt_is_weekend else int(cal.get('is_class_day', 1)),
                'tgt_is_break':        int(cal.get('is_break', 0)),
                'tgt_is_finals':       int(cal.get('is_finals', 0)),
                'tgt_is_commencement': int(cal.get('is_commencement', 0)),
                'tgt_is_holiday':      int(cal.get('is_holiday', 0)),
                'tgt_home_game_count':  int(spt.get('home_game_count', 0)),
                'tgt_has_basketball':   int(spt.get('has_basketball', 0)),
                'tgt_has_baseball':     int(spt.get('has_baseball', 0)),
                'tgt_has_softball':     int(spt.get('has_softball', 0)),
                'tgt_has_lacrosse':     int(spt.get('has_lacrosse', 0)),
                'tgt_high_impact_game': int(spt.get('high_impact_game', 0)),
                'tgt_condition_level':  int(dis.get('condition_level', 0)),
                'tgt_is_remote':        int(dis.get('is_remote', 0)),
                'tgt_is_cancelled':     int(dis.get('is_cancelled', 0)),
                'tgt_temperature_f':    wthr.get('temperature_f', 65.0),
                'tgt_humidity':         wthr.get('humidity', 55.0),
                'tgt_precipitation_in': wthr.get('precipitation_in', 0.0),
            })

    X = pd.DataFrame(rows)[cfg['features']]
    X['deck_id'] = pd.Categorical(X['deck_id'], categories=cfg['lots'])
    return X

# Run predictions
X = build_inference_batch(...)
preds = point.predict(X).clip(0, 1)
lows  = lower.predict(X).clip(0, 1)
# upper is currently broken (see Known Issues) — omit or use preds + fixed offset

# Write to Supabase (same schema as parking_predictions)
# model_tier = 'lgb' for all rows
```

## EC2 Deployment Checklist

1. **Install LightGBM on EC2:**
   ```bash
   pip install lightgbm
   # Also requires libomp: sudo apt-get install libomp-dev
   ```

2. **Copy model files** (tracked by Git LFS or scp directly):
   ```bash
   scp -r uncc-parking-api/models_lgb/ ec2-user@<host>:~/uncc-parking-api/models_lgb/
   ```
   Only 4 files needed: `lgb_point.pkl`, `lgb_lower.pkl`, `lgb_upper.pkl`, `lgb_config.pkl`

3. **Add inference code** to `parking_api/predict.py`:
   - Load 3 models + config at startup alongside existing RF registry
   - Build 360-row batch using `build_inference_batch()` above
   - Write results with `model_tier='lgb'`
   - Can run in parallel with RF as shadow tier initially

4. **Data files needed at runtime** (already on EC2):
   - `data/academic_calendar.csv`
   - `data/sports_schedule.csv`
   - `data/campus_disruptions.csv`
   - `data/weather.csv` (or live Open-Meteo forecast)

5. **Supabase schema**: uses the same `parking_predictions` table, just a new
   `model_tier` value (`'lgb'`). No schema changes needed.

## Known Issues

### Upper confidence bound is broken
`lgb_upper.pkl` trained only 1 tree and predicts ≈1.0 for everything.
The 97.5th percentile of parking occupancy is essentially always at capacity,
so LightGBM trivially learns to predict 1.0 with a single split.

**Fix options (in order of preference):**
1. Use tighter quantiles — retrain upper with `alpha=0.85` or `alpha=0.90` for a
   more informative "high estimate" rather than a true 97.5% bound
2. Compute confidence bands from residuals: fit point model, compute per-lot
   per-horizon MAE on validation set, use `prediction ± 2 * mae` as the band
3. The lower bound (`lgb_lower.pkl`) is working correctly and is the more
   useful bound for "when will the lot be at least X% available?"

### NORTH lot in-band coverage is low (82.8%)
The lower bound is too tight for NORTH. Likely due to NORTH having more
high-variance occupancy patterns. Consider per-lot residual calibration.

### Data CSVs at target time
`weather.csv` covers historical data. For live predictions beyond the current
hour, the system needs the Open-Meteo forecast (already used in the RF pipeline
via `weather.py`). The LightGBM serving code should use the same forecast fetch.
