# LGB Near-Term Model v4 (3h, residual) — `lgb_v4`

Retrain of `lgb_v3` with the **same 64 features** on the full Jan 27 → Oct 2 2026 history.
Trained with `uncc-parking-notebook/train_2026_10.py --model 3h` (CPU, M2 Pro, ~6 min/run).

## Training

| Field | Value |
|---|---|
| **Trained** | 2026-10-02 |
| **Data** | `parking_data_rows_2026-10-02.csv` — full `parking_data` export, 59,037 snapshots |
| **Trained through** | 2026-10-02 02:30 UTC |
| **Target** | residual (`target − current_capacity`); inference adds `current_capacity` back |
| **Rows** | 10.4M (origins every 10 min × 36 horizons × 10 lots; WEST only through 2026-06-25) |
| **Trees** | point 479 (eval best 435 × 1.1), quantiles 1000 each (α=0.025 / 0.85) |
| **Config** | `semesters` list (Spring + Fall 2026) — semester-aware class week / weeks-until-finals |

## Data fixes vs lgb_v3

- Excluded whole-feed freeze 2026-06-25 → 2026-08-03 (every lot constant, then Supabase outage).
- Excluded **WEST from 2026-06-25 onward** — feed still broken (confirmed 2026-10-02). WEST is
  learned from Spring only, and the predictor skips it while `SUPPRESSED_LOTS=WEST`.
- No interpolation across gaps > 30 min.
- Calendar extended (summer = non-class days, Fall 2026, winter break); `fall_recess` counts as break.
- Weather, sports (Fall football/soccer/volleyball/basketball), campus events refreshed through Dec 2026.
- `ema_30`/`ema_60` computed over the last 25 snapshots, exactly as inference does.

## Holdout (train < Sep 11, early-stop Sep 11–17, test Sep 18 → Oct 2; WEST excluded)

| Horizon | lgb_v4 MAE | live lgb_v3 MAE (same window, excl. WEST) |
|---|---|---|
| T+5–30 | **0.0096** | — |
| T+35–60 | **0.0153** | — |
| T+65–120 | **0.0233** | — |
| T+125–180 | **0.0326** | — |
| **Overall** | **0.0227** (bias +0.001, in-band 75.3%) | 0.0277 (bias −0.005, in-band 66.6%) |

Offline numbers use actual weather; live predictions use forecasts, so expect slightly worse live.
Live lgb_v3 on this window also suffered two inference bugs fixed alongside this retrain
(Fall scored as week 16 / finals; event features always 0).

## Known limitations

- Quantile bands still under-cover (75.3% vs ~82.5% nominal).
- `semesters` ends Dec 13 2026 — add Spring 2027 to `academic_calendar.csv` and retrain (or extend
  the config) before January.
- Weather is looked up by UTC time in a local-time table (4–5 h offset), consistently in training
  and inference — fix both together in a future retrain.
