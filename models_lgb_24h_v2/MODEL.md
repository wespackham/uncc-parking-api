# LGB Long-Range Model v2 (24h) — `lgb_24h_v2`

Retrain of `lgb_24h` with the **same 35 features** on the full Jan 27 → Oct 2 2026 history.
Trained with `uncc-parking-notebook/train_2026_10.py --model 24h` (CPU, M2 Pro, ~15 min/run).

## Training

| Field | Value |
|---|---|
| **Trained** | 2026-10-02 |
| **Data** | `parking_data_rows_2026-10-02.csv` — full `parking_data` export, 59,037 snapshots |
| **Trained through** | 2026-10-02 02:30 UTC |
| **Target** | absolute occupancy ratio |
| **Rows** | 27.6M (origins every 30 min × 288 horizons × 10 lots; WEST only through 2026-06-25) |
| **Trees** | point 1003 (eval best 912 × 1.1), quantiles 1000 each (α=0.025 / 0.85) |
| **Horizons** | T+5 → T+1440, 5-min steps (2,880 predictions per run, hourly) |

Data fixes are identical to `models_lgb_v4/MODEL.md` (feed-freeze exclusion, WEST excluded from 2026-06-25,
gap-aware grid, calendar/sports/events/weather refreshed through Dec 2026).

## Holdout (train < Sep 10, early-stop Sep 11–17, test Sep 18 → Oct 2; WEST excluded)

| Horizon | lgb_24h_v2 MAE |
|---|---|
| T+5–60 | **0.0293** |
| T+65–180 | **0.0359** |
| T+3–6h | **0.0443** |
| T+6–12h | **0.0514** |
| T+12–24h | **0.0523** |
| **Overall** | **0.0487** (bias +0.009, in-band 91.7%) vs live lgb_24h 0.0582 (bias +0.017, in-band 58.1%) |

Offline numbers use actual weather; live uses forecasts.

## Known limitations

- Small positive bias (+0.006 → +0.010 by horizon) — the Fall 2026 plan's P1-D (hour × lot
  historical mean) targets this.
- No semester/class-week features (by design of this feature set); summer vs semester comes only
  from `tgt_is_class_day`.
- Same weather-timezone convention caveat as lgb_v4.
