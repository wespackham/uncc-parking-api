-- Migration: parking_predictions → parking_predictions_v2
--
-- Old schema (normalized): one row per (lot × target_time × model_tier × run)
-- New schema (denormalized): one row per (target_time × model_tier × run), all 10 lots in JSONB
--
-- Grouping key: date_trunc('second', created_at)
--   The 3h model inserts <500 rows in a single batch — all rows share the same exact created_at.
--   The 24h model inserts 2,880 rows across 6 batches of 500 — each batch is a separate
--   HTTP request/transaction, so created_at may differ by milliseconds between batches.
--   Truncating to the second collapses within-run batches safely.
--   This is safe because runs are at minimum 5 minutes apart, so no two different runs
--   share the same second-level timestamp.
--
-- Run this in the Supabase SQL editor. The old table is left untouched until you're
-- confident the migration is correct — rename/drop it separately.

-- ── 1. Create new table ────────────────────────────────────────────────────────

CREATE TABLE parking_predictions_v2 (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at  TIMESTAMPTZ NOT NULL,
  target_time TIMESTAMPTZ NOT NULL,
  model_tier  TEXT        NOT NULL,
  data        JSONB       NOT NULL
);

-- Dashboard fetches latest predictions by (model_tier, created_at DESC) then reads by target_time
CREATE INDEX idx_pred_v2_created ON parking_predictions_v2 (created_at DESC, model_tier);
CREATE INDEX idx_pred_v2_target  ON parking_predictions_v2 (target_time, model_tier);

ALTER TABLE parking_predictions_v2 ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Public read" ON parking_predictions_v2 FOR SELECT USING (true);


-- ── 2. Migrate data ────────────────────────────────────────────────────────────

INSERT INTO parking_predictions_v2 (created_at, target_time, model_tier, data)
SELECT
  min(created_at) AS created_at,   -- representative timestamp for this run's group
  target_time,
  model_tier,
  jsonb_object_agg(
    lot,
    jsonb_build_object(
      'prediction',      prediction,
      'confidence_low',  confidence_low,
      'confidence_high', confidence_high
    )
  ) AS data
FROM parking_predictions
GROUP BY date_trunc('second', created_at), target_time, model_tier;


-- ── 3. Verify ──────────────────────────────────────────────────────────────────
-- old total_rows / 10 should roughly equal new total_rows for each model_tier.
-- Exact match won't happen if any run was missing lots (e.g. a lot had no data).

SELECT
  'old'    AS tbl,
  model_tier,
  count(*)                   AS total_rows,
  count(*) / 10              AS expected_new_rows,
  count(DISTINCT date_trunc('second', created_at)) AS distinct_runs
FROM parking_predictions
GROUP BY model_tier

UNION ALL

SELECT
  'new'    AS tbl,
  model_tier,
  count(*)                   AS total_rows,
  NULL                       AS expected_new_rows,
  count(DISTINCT created_at) AS distinct_runs
FROM parking_predictions_v2
GROUP BY model_tier

ORDER BY tbl DESC, model_tier;


-- ── 4. Spot-check a single run ─────────────────────────────────────────────────
-- Pick any target_time and verify the JSONB has all 10 lots.

SELECT
  created_at,
  target_time,
  model_tier,
  jsonb_object_keys(data) AS lot,
  data->jsonb_object_keys(data) AS lot_data
FROM parking_predictions_v2
WHERE model_tier = 'lgb'
ORDER BY created_at DESC, target_time
LIMIT 10;


-- ── 5. Rename tables when ready ────────────────────────────────────────────────
-- Run these only after verifying the migration looks correct.
-- The API and dashboard will need to be updated to point to parking_predictions_v2
-- (or rename v2 → parking_predictions after dropping the old one).

-- ALTER TABLE parking_predictions     RENAME TO parking_predictions_old;
-- ALTER TABLE parking_predictions_v2  RENAME TO parking_predictions;
-- DROP TABLE parking_predictions_old;  -- only after new code is live and verified
