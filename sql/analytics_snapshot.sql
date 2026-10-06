-- Precomputed analytics for the dashboard's Analytics view.
--
-- The dashboard used to download every prediction row for the last 7 days (~70k rows, ~60 MB)
-- and compute these metrics in the browser. Now ops.refresh_analytics_snapshot() computes the
-- same payload inside Postgres every 10 minutes (parking-analytics.timer) and the dashboard
-- reads one small row from public.analytics_snapshot.
--
-- Mirrors uncc-parking-dashboard/src/services/AnalyticsService.js:
--   * predictions for one tier whose target_time is in [now - days, now]
--   * horizon = round((target_time - created_at) / 5 min) * 5, kept if in p_horizons
--   * excluded lots (broken feeds) and "estimated" entries are skipped
--   * each prediction is matched to the snapshot nearest its target_time, within 4 minutes
--   * MAE / RMSE / R² / in-band % by horizon and by deck; average availability by deck
--
-- Idempotent: applied on every API deploy by .github/workflows/deploy-api.yml.

SET ROLE parking_owner;

CREATE SCHEMA IF NOT EXISTS ops;  -- not in PostgREST db-schemas, so not callable over HTTP

CREATE TABLE IF NOT EXISTS public.analytics_snapshot (
  tier         text PRIMARY KEY,
  computed_at  timestamptz NOT NULL,
  payload      jsonb NOT NULL
);
-- parking_owner default privileges grant parking_anon SELECT; be explicit anyway.
GRANT SELECT ON public.analytics_snapshot TO parking_anon;
ALTER TABLE public.analytics_snapshot ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Public read" ON public.analytics_snapshot;
CREATE POLICY "Public read" ON public.analytics_snapshot FOR SELECT USING (true);

CREATE OR REPLACE FUNCTION ops.analytics_payload(
  p_tier            text,
  p_days            int         DEFAULT 7,
  p_horizons        int[]       DEFAULT '{5,10,15,30,45,60}',
  p_excluded_lots   text[]      DEFAULT '{WEST}',
  p_now             timestamptz DEFAULT now()
) RETURNS jsonb
LANGUAGE sql STABLE AS $$
WITH win AS (
  SELECT p_now - make_interval(days => p_days) AS t0, p_now AS t1
),
pred AS (
  SELECT p.target_time,
         (round(extract(epoch FROM p.target_time - p.created_at) / 300) * 5)::int AS horizon,
         d.key AS lot,
         (d.value->>'prediction')::float8      AS prediction,
         (d.value->>'confidence_low')::float8  AS lo,
         (d.value->>'confidence_high')::float8 AS hi
  FROM public.parking_predictions p, win, jsonb_each(p.data) d
  WHERE p.model_tier = p_tier
    AND p.target_time BETWEEN win.t0 AND win.t1
    AND NOT (d.key = ANY (p_excluded_lots))
    AND NOT (d.value ? 'estimated')
),
pred_h AS (
  SELECT * FROM pred WHERE horizon = ANY (p_horizons)
),
targets AS (
  SELECT DISTINCT target_time FROM pred_h
),
-- only snapshots inside the window are candidates (as in the browser version)
nearest AS (  -- snapshot closest to each target time, if within 4 minutes
  SELECT t.target_time, best.data
  FROM targets t
  CROSS JOIN win
  CROSS JOIN LATERAL (
    SELECT c.data, abs(extract(epoch FROM c.created_at - t.target_time)) AS gap
    FROM (
      (SELECT created_at, data FROM public.parking_data
        WHERE created_at <= t.target_time AND created_at > t.target_time - interval '4 minutes'
          AND created_at BETWEEN win.t0 AND win.t1
        ORDER BY created_at DESC LIMIT 1)
      UNION ALL
      (SELECT created_at, data FROM public.parking_data
        WHERE created_at > t.target_time AND created_at <= t.target_time + interval '4 minutes'
          AND created_at BETWEEN win.t0 AND win.t1
        ORDER BY created_at ASC LIMIT 1)
    ) c
    ORDER BY gap
    LIMIT 1
  ) best
),
matched AS (
  SELECT ph.horizon, ph.lot, ph.prediction, ph.lo, ph.hi, (n.data->>ph.lot)::float8 AS actual
  FROM pred_h ph JOIN nearest n USING (target_time)
  WHERE n.data ? ph.lot AND n.data->>ph.lot IS NOT NULL
),
actuals AS (
  SELECT d.key AS lot, d.value::text::float8 AS actual
  FROM public.parking_data a, win, jsonb_each(a.data) d
  WHERE a.created_at > '2020-01-01' AND a.created_at BETWEEN win.t0 AND win.t1
    AND jsonb_typeof(d.value) = 'number'
    AND NOT (d.key = ANY (p_excluded_lots))
),
avg_by_deck AS (
  SELECT lot, avg(actual) AS average, count(*) AS samples FROM actuals GROUP BY lot
),
metrics AS (  -- grouped by horizon and by lot in one pass
  SELECT horizon, lot, count(*) AS n,
         avg(abs(prediction - actual)) AS mae,
         sqrt(avg((prediction - actual) ^ 2)) AS rmse,
         CASE WHEN var_pop(actual) > 0
              THEN 1 - sum((prediction - actual) ^ 2) / (count(*) * var_pop(actual)) END AS r2,
         100.0 * avg((actual >= lo AND actual <= hi)::int) AS in_band
  FROM matched
  GROUP BY GROUPING SETS ((horizon), (lot))
),
lots AS (
  SELECT lot FROM avg_by_deck UNION SELECT lot FROM matched
)
SELECT jsonb_build_object(
  'days', p_days,
  'generatedAt', to_char(p_now AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'),
  'matchedCount', (SELECT count(*) FROM matched),
  'averageCapacity', jsonb_build_object(
    'overallAverage', (SELECT avg(actual) FROM actuals),
    'highestAverageDeck', (SELECT jsonb_build_object('lot', lot, 'average', average, 'samples', samples)
                           FROM avg_by_deck ORDER BY average DESC, lot LIMIT 1),
    'lowestAverageDeck', (SELECT jsonb_build_object('lot', lot, 'average', average, 'samples', samples)
                          FROM avg_by_deck ORDER BY average ASC, lot LIMIT 1),
    'byDeck', COALESCE((SELECT jsonb_agg(jsonb_build_object('lot', lot, 'average', average, 'samples', samples) ORDER BY lot)
                        FROM avg_by_deck), '[]'::jsonb)
  ),
  'byHorizon', (
    SELECT jsonb_agg(jsonb_build_object(
             'horizon', h, 'n', COALESCE(m.n, 0), 'mae', m.mae, 'rmse', m.rmse,
             'r2', m.r2, 'withinBandPct', m.in_band) ORDER BY h)
    FROM unnest(p_horizons) h
    LEFT JOIN metrics m ON m.horizon = h AND m.lot IS NULL
  ),
  'byDeck', COALESCE((
    SELECT jsonb_agg(jsonb_build_object(
             'lot', l.lot, 'averageCapacity', a.average, 'n', COALESCE(m.n, 0), 'mae', m.mae,
             'rmse', m.rmse, 'r2', m.r2, 'withinBandPct', m.in_band) ORDER BY l.lot)
    FROM lots l
    LEFT JOIN avg_by_deck a ON a.lot = l.lot
    LEFT JOIN metrics m ON m.lot = l.lot AND m.horizon IS NULL
  ), '[]'::jsonb)
);
$$;

CREATE OR REPLACE FUNCTION ops.refresh_analytics_snapshot(p_tier text, p_days int DEFAULT 7)
RETURNS void
LANGUAGE sql AS $$
  INSERT INTO public.analytics_snapshot (tier, computed_at, payload)
  VALUES (p_tier, now(), ops.analytics_payload(p_tier, p_days))
  ON CONFLICT (tier) DO UPDATE SET computed_at = EXCLUDED.computed_at, payload = EXCLUDED.payload;
$$;

REVOKE ALL ON FUNCTION ops.analytics_payload(text, int, int[], text[], timestamptz) FROM PUBLIC;
REVOKE ALL ON FUNCTION ops.refresh_analytics_snapshot(text, int) FROM PUBLIC;

RESET ROLE;

-- Tell PostgREST to pick up new/changed tables without a restart.
NOTIFY pgrst, 'reload schema';
