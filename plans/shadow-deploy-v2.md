# Shadow Deploy v2 Models + Architecture Review

> **Superseded.** This plan was for shadow-deploying RandomForest v2 models. The RF architecture was replaced in April 2026 by LightGBM (lgb / lgb_24h / lgb_v3). This document is historical only.

## Context

We retrained the 30 RandomForest models on properly-resampled 5-min data into `uncc-parking-notebook/models_v2/`. Holdout (Mar 30 – Apr 6) shows v2 dramatically beats v1 across the board: baseline R² 0.770 (v1: 0.52), 30-min R² 0.989, 60-min R² 0.970. v1 in production is even worse than its notebook score on the 60-min tier (R² 0.41) because the deployed pipeline reuses the 60-min model for T+90→T+180 with stale autoregressive lags.

Goal: run the new **30-min** and **baseline** v2 models in shadow alongside v1 for 2–4 days, write both sets of predictions to Supabase under distinct `model_tier` labels, then compare with `evaluate_predictions.py`. No user-visible behavior change; the dashboard keeps reading v1 tiers. v2 60-min is intentionally excluded — we want to first see whether v2's stronger baseline already beats the v1 60-min in the T+90→T+180 window, which would let us delete the 60-min tier entirely.

---

## Architecture Question: Microservices?

**Recommendation: no, don't split.** Reasons specific to this project:

1. **Tight coupling is intentional.** `features.py` must mirror `backtest.py` exactly — that's a correctness contract, not an accident of layering. Putting a network boundary between them doesn't decouple them; it just makes the contract harder to verify and adds a place for schema drift to hide.
2. **No independent scaling axis.** The predictor runs once every 5 minutes as a oneshot systemd unit. There is no load to shed, no separate team owning either side, no language boundary, and no need to deploy them on different cadences.
3. **Latency & failure surface get worse.** A single in-process call becomes an HTTP round trip per lot/horizon (~1700 calls per run) plus a new service to monitor, deploy, and alert on. The whole prediction loop currently fits in one EC2 box and one systemd unit — splitting it doubles ops without halving anything.
4. **The clean seam already exists at the function level.** `build_feature_vector()` and `ModelRegistry.predict()` are already separable; if we ever need to reuse feature engineering elsewhere (e.g. a notebook, a backfill job) we import the module. That's the right boundary for a project this size.
5. **When it *would* be worth splitting:** multiple model consumers with different deploy cadences, a non-Python client, a feature store shared across models/teams, or feature computation becoming expensive enough to warrant batching/caching. None apply today.

**Action:** keep the monolith. The refactor worth doing is collapsing `predict.py`'s tier loop into a single horizon-driven function so v1/v2 share scheduling code — see step 4 below.

---

## Implementation Steps

### Step 1 — Ship v2 model artifacts into the API repo
- Copy `uncc-parking-notebook/models_v2/` → `uncc-parking-api/models_v2/` (51 `.pkl` files + `features.pkl` + per-lot/horizon `*_features.pkl`).
- Track via Git LFS using the same `.gitattributes` rule that already covers `models/*.pkl`.
- Include `models_v2/training_results.json` for an in-repo record of holdout metrics.

### Step 2 — `parking_api/config.py`
Add one line:
```python
MODELS_V2_DIR = BASE_DIR / "models_v2"
```
No other config changes — `LOTS`, `safe_name`, table names are shared.

### Step 3 — `parking_api/features.py`
Add `minute_sin` and `minute_cos` to `build_time_features()`:
```python
"minute_sin": math.sin(2 * math.pi * dt.minute / 60),
"minute_cos": math.cos(2 * math.pi * dt.minute / 60),
```
Safe for v1: `build_feature_vector()` already filters by `feature_names`, so v1 models silently ignore the two new keys. No changes to `build_lag_features()` — v2 column names are identical; only the spacing of the underlying samples differs (caller concern).

### Step 4 — `parking_api/predict.py` — dual prediction loop
**a)** Add `step` parameter to `_extract_recent_values`:
```python
def _extract_recent_values(rows, lot, n=4, step=3):
```
Default of `step=3` keeps all 12 existing tests passing. v2 extraction uses `step=1` (true 5-min lags).

**b)** Load two registries:
```python
registry_v1 = ModelRegistry(MODELS_DIR)
registry_v2 = ModelRegistry(MODELS_V2_DIR)
```

**c)** Run the existing v1 loop unchanged → writes `model_tier ∈ {"30min", "60min", "baseline"}`.

**d)** Add v2 loop after the v1 loop:
- `30min_v2` for T+30 and T+60 using `_extract_recent_values(..., step=1)`
- `baseline_v2` for the **entire** T+90 → T+24h window (1-hour cadence, matching v1 baseline row spacing) — this lets us A/B v2-baseline against v1-60min in the T+90–T+180 zone

**e)** Wrap the entire v2 block in `try/except` that logs and continues. v1 must never break because v2 errored.

**Row volume impact:** ~2× per run (~3,400 rows vs ~1,700). Well within Supabase write limits.

### Step 5 — `parking_api/main.py`
Build both registries in `lifespan` and stash on `app.state`:
```python
app.state.registry = ModelRegistry(MODELS_DIR)
app.state.registry_v2 = ModelRegistry(MODELS_V2_DIR)
```
Only matters for `POST /predict`; the systemd cron path uses `predict.py`'s module-level loaders.

### Step 6 — `evaluate_predictions.py`
Extend `TIER_ORDER`:
```python
TIER_ORDER = ["30min", "30min_v2", "60min", "baseline", "baseline_v2"]
```
No other logic changes — `compute_metrics` is tier-agnostic and the per-lot loop iterates `TIER_ORDER` already.

### Step 7 — `tests/`
- `test_predict_utils.py`: add 2 tests for `step=1` (true 5-min) — one happy path verifying indices 0,1,2,3 are selected, one for short input.
- `test_features.py`: verify `build_time_features` includes `minute_sin`/`minute_cos` and they're in `[-1, 1]`.
- `test_predictions.py`: parametrize over both registries so all 30 v2 models load and predict in [0, 1].

### Step 8 — `.github/workflows/deploy-api.yml`
Add `models_v2/**` to the `paths:` trigger list. Without this, pushing v2 artifacts won't trigger a redeploy.

---

## Rollback Plan

v2 lives entirely behind new tier names. To kill it:
1. Revert the v2 block in `predict.py` (one commit).
2. Optionally: `DELETE FROM parking_predictions WHERE model_tier LIKE '%_v2'`.

v1 traffic is untouched throughout.

---

## File Change Summary

| File | Change |
|------|--------|
| `uncc-parking-api/models_v2/` | NEW — 51 `.pkl` files copied from notebook |
| `parking_api/config.py` | + `MODELS_V2_DIR` |
| `parking_api/features.py` | + `minute_sin` / `minute_cos` |
| `parking_api/predict.py` | + `step` param, + v2 loop, + v2 registry |
| `parking_api/main.py` | + second registry on `app.state` |
| `evaluate_predictions.py` | + v2 tiers in `TIER_ORDER` |
| `tests/test_predict_utils.py` | + step=1 cases |
| `tests/test_features.py` | + minute_sin/cos assertions |
| `tests/test_predictions.py` | + v2 registry parametrize |
| `.github/workflows/deploy-api.yml` | + `models_v2/**` path trigger |

---

## Verification

### Local (before pushing)
1. `pytest tests/ -v` — all existing tests pass, new tests pass.
2. `python -m parking_api.predict` against real Supabase — logs show v1 and v2 batches completing.
3. Supabase check:
   ```sql
   SELECT model_tier, count(*)
   FROM parking_predictions
   WHERE created_at > now() - interval '10 min'
   GROUP BY 1;
   ```
   Expect 5 distinct tiers.

### In production (after deploy)
1. Watch first systemd timer fire: `journalctl -u parking-predictor.service -f`. v1 must complete even if v2 errors.
2. After ~24h: `python evaluate_predictions.py --days 1 --by-lot` — confirm v2 tier rows appear with sane MAE.
3. After 2–4 days: `python evaluate_predictions.py --days 3` to compare:
   - `30min` vs `30min_v2` — validates the 5-min lag fix
   - `60min` vs `baseline_v2` over T+90–T+180 — decides whether 60-min tier survives at all
   - `baseline` vs `baseline_v2` for long-horizon (CD VS especially: holdout R² went from −1.21 → 0.55)

### Decision point
If v2 wins as expected: swap tier labels (`30min_v2` → `30min`, etc.) in a follow-up PR and delete v1 artifacts.
