# WEST ("Central Deck") Estimator

Infers WEST availability from the other 9 decks + time/calendar context, so the dashboard can show
an **estimated** forecast while UNCC's WEST sensor is broken (`SUPPRESSED_LOTS=WEST`).
Trained with `uncc-parking-notebook/train_west_estimator.py`.

| Field | Value |
|---|---|
| **Trained** | 2026-10-02 on healthy WEST data only (2026-01-27 → 2026-06-25, 42,668 snapshots) |
| **Inputs** | other 9 decks' availability at the target time + hour/minute/day-of-week, calendar, sports, events |
| **Model** | LightGBM point regressor; bands = point + per-local-hour 10th/90th holdout residuals |

## Holdout (Apr 13 → May 10 2026: semester end, finals, commencement)

| Method | WEST MAE |
|---|---|
| WEST's own time-of-day average | 0.0875 |
| **This estimator** (given actual other-deck values) | **0.0556** (bias −0.001, 10–90 band coverage 79.5%) |

Error by local hour: ~0.013–0.025 overnight, ~0.06–0.10 from 08:00 to 21:00.

## How it's used

`parking_api/estimate.py` runs after each forecast tier: the tier's *predicted* values for the other
decks at each target time are fed in, and a `WEST` entry with `"estimated": true` is added.
Reports (`daily_report.py`, `evaluate_predictions.py`, the migration status script) skip estimated
entries.

## Caveats

- Live error is higher than the holdout: inputs are forecasts, not actual readings.
- Learned from Spring 2026. If the deck's usage changed in Fall (it was renamed Central Deck),
  the relationship may not hold — there's no working sensor to verify against.
- Remove `WEST` from `SUPPRESSED_LOTS` and the dashboard's `SENSOR_ISSUE_DECKS` once the feed is fixed.
