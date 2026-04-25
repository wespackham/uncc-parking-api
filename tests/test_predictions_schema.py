"""Tests for the denormalized JSONB prediction schema.

Verifies that the explode logic in evaluate_predictions, daily_report, and
supabase_client correctly converts the JSONB `data` column into flat per-lot
rows — and that downstream functions (matching, metrics) work on the result.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# ── Sample data matching the new DB schema ────────────────────────────────

SAMPLE_JSONB_ROWS = [
    {
        "created_at": "2026-04-13T12:01:00+00:00",
        "target_time": "2026-04-13T12:05:00+00:00",
        "model_tier": "lgb",
        "data": {
            "CRI": {"prediction": 0.45, "confidence_low": 0.38, "confidence_high": 0.52},
            "ED1": {"prediction": 0.72, "confidence_low": 0.65, "confidence_high": 0.80},
            "WEST": {"prediction": 0.30, "confidence_low": 0.22, "confidence_high": 0.40},
        },
    },
    {
        "created_at": "2026-04-13T12:01:00+00:00",
        "target_time": "2026-04-13T12:10:00+00:00",
        "model_tier": "lgb",
        "data": {
            "CRI": {"prediction": 0.46, "confidence_low": 0.39, "confidence_high": 0.53},
            "ED1": {"prediction": 0.71, "confidence_low": 0.64, "confidence_high": 0.79},
            "WEST": {"prediction": 0.31, "confidence_low": 0.23, "confidence_high": 0.41},
        },
    },
]

# Row with mixed timestamp formats (with and without fractional seconds)
MIXED_TIMESTAMP_ROWS = [
    {
        "created_at": "2026-04-13T12:01:00.123456+00:00",
        "target_time": "2026-04-13T12:05:00+00:00",
        "model_tier": "lgb",
        "data": {
            "CRI": {"prediction": 0.45, "confidence_low": 0.38, "confidence_high": 0.52},
        },
    },
    {
        "created_at": "2026-04-13T12:01:00+00:00",
        "target_time": "2026-04-13T12:10:00.654321+00:00",
        "model_tier": "lgb",
        "data": {
            "CRI": {"prediction": 0.46, "confidence_low": 0.39, "confidence_high": 0.53},
        },
    },
]


# ── evaluate_predictions.py ───────────────────────────────────────────────

class TestEvaluatePredictionsFetch:
    """Test the JSONB explode in evaluate_predictions.fetch_predictions."""

    def _mock_client(self, rows):
        """Build a mock Supabase client that returns the given rows."""
        client = MagicMock()
        result = MagicMock()
        result.data = rows
        # Chain: client.table().select().gte().lte().order().range().execute()
        chain = client.table.return_value.select.return_value
        chain.gte.return_value = chain
        chain.lte.return_value = chain
        chain.order.return_value = chain
        chain.range.return_value = chain
        chain.execute.return_value = result
        return client

    def test_explode_basic(self):
        """3 lots × 2 target_times = 6 flat rows."""
        from evaluate_predictions import fetch_predictions
        client = self._mock_client(SAMPLE_JSONB_ROWS)
        df = fetch_predictions(client, "2026-04-13", "2026-04-14", lot=None)
        assert len(df) == 6
        assert set(df.columns) >= {"created_at", "target_time", "model_tier", "lot", "prediction", "confidence_low", "confidence_high"}

    def test_explode_lot_filter(self):
        """Filtering by lot should only return rows for that lot."""
        from evaluate_predictions import fetch_predictions
        client = self._mock_client(SAMPLE_JSONB_ROWS)
        df = fetch_predictions(client, "2026-04-13", "2026-04-14", lot="CRI")
        assert len(df) == 2
        assert (df["lot"] == "CRI").all()

    def test_explode_lot_filter_no_match(self):
        """Filtering by a lot not in the data should exit."""
        from evaluate_predictions import fetch_predictions
        client = self._mock_client(SAMPLE_JSONB_ROWS)
        with pytest.raises(SystemExit):
            fetch_predictions(client, "2026-04-13", "2026-04-14", lot="NONEXISTENT")

    def test_explode_values(self):
        """Spot-check that prediction values are correctly extracted."""
        from evaluate_predictions import fetch_predictions
        client = self._mock_client(SAMPLE_JSONB_ROWS)
        df = fetch_predictions(client, "2026-04-13", "2026-04-14", lot="CRI")
        row = df.iloc[0]
        assert row["prediction"] == 0.45
        assert row["confidence_low"] == 0.38
        assert row["confidence_high"] == 0.52

    def test_mixed_timestamp_formats(self):
        """Timestamps with and without fractional seconds should both parse."""
        from evaluate_predictions import fetch_predictions
        client = self._mock_client(MIXED_TIMESTAMP_ROWS)
        df = fetch_predictions(client, "2026-04-13", "2026-04-14", lot=None)
        assert len(df) == 2
        assert df["target_time"].dtype == "datetime64[ns, UTC]"
        assert df["created_at"].dtype == "datetime64[ns, UTC]"

    def test_empty_data_field(self):
        """Rows with null/empty data should produce no exploded rows."""
        from evaluate_predictions import fetch_predictions
        rows = [{
            "created_at": "2026-04-13T12:01:00+00:00",
            "target_time": "2026-04-13T12:05:00+00:00",
            "model_tier": "lgb",
            "data": None,
        }]
        client = self._mock_client(rows)
        with pytest.raises(SystemExit):
            fetch_predictions(client, "2026-04-13", "2026-04-14", lot=None)

    def test_preserves_model_tier(self):
        """model_tier from the parent row should propagate to every exploded row."""
        from evaluate_predictions import fetch_predictions
        rows = [{
            "created_at": "2026-04-13T12:01:00+00:00",
            "target_time": "2026-04-13T12:05:00+00:00",
            "model_tier": "lgb_24h",
            "data": {
                "CRI": {"prediction": 0.5, "confidence_low": 0.4, "confidence_high": 0.6},
                "ED1": {"prediction": 0.6, "confidence_low": 0.5, "confidence_high": 0.7},
            },
        }]
        client = self._mock_client(rows)
        df = fetch_predictions(client, "2026-04-13", "2026-04-14", lot=None)
        assert (df["model_tier"] == "lgb_24h").all()


# ── daily_report.py ───────────────────────────────────────────────────────

class TestDailyReportFetch:
    """Test the JSONB explode in daily_report._fetch_predictions."""

    def _mock_client(self, rows):
        client = MagicMock()
        result = MagicMock()
        result.data = rows
        chain = client.table.return_value.select.return_value
        chain.gte.return_value = chain
        chain.lte.return_value = chain
        chain.order.return_value = chain
        chain.range.return_value = chain
        chain.execute.return_value = result
        return client

    def test_explode_basic(self):
        from parking_api.daily_report import _fetch_predictions
        client = self._mock_client(SAMPLE_JSONB_ROWS)
        df = _fetch_predictions(client, "2026-04-13", "2026-04-14")
        assert len(df) == 6
        assert set(df.columns) >= {"lot", "prediction", "confidence_low", "confidence_high"}

    def test_empty_returns_empty_df(self):
        from parking_api.daily_report import _fetch_predictions
        client = self._mock_client([])
        df = _fetch_predictions(client, "2026-04-13", "2026-04-14")
        assert df.empty

    def test_mixed_timestamps(self):
        from parking_api.daily_report import _fetch_predictions
        client = self._mock_client(MIXED_TIMESTAMP_ROWS)
        df = _fetch_predictions(client, "2026-04-13", "2026-04-14")
        assert len(df) == 2
        assert df["created_at"].dtype == "datetime64[ns, UTC]"


# ── supabase_client.py ────────────────────────────────────────────────────

class TestSupabaseClientFetch:
    """Test the JSONB explode in supabase_client.fetch_predictions."""

    def test_explode_all_lots(self):
        from parking_api.supabase_client import fetch_predictions

        result = MagicMock()
        result.data = SAMPLE_JSONB_ROWS

        mock_client = MagicMock()
        chain = mock_client.table.return_value.select.return_value
        chain.order.return_value = chain
        chain.gte.return_value = chain
        chain.lte.return_value = chain
        chain.execute.return_value = result

        with patch("parking_api.supabase_client._get_client", return_value=mock_client):
            rows = fetch_predictions()

        assert len(rows) == 6
        lots = {r["lot"] for r in rows}
        assert lots == {"CRI", "ED1", "WEST"}

    def test_explode_with_lot_filter(self):
        from parking_api.supabase_client import fetch_predictions

        result = MagicMock()
        result.data = SAMPLE_JSONB_ROWS

        mock_client = MagicMock()
        chain = mock_client.table.return_value.select.return_value
        chain.order.return_value = chain
        chain.gte.return_value = chain
        chain.lte.return_value = chain
        chain.execute.return_value = result

        with patch("parking_api.supabase_client._get_client", return_value=mock_client):
            rows = fetch_predictions(lot="ED1")

        assert len(rows) == 2
        assert all(r["lot"] == "ED1" for r in rows)

    def test_row_has_all_fields(self):
        from parking_api.supabase_client import fetch_predictions

        result = MagicMock()
        result.data = SAMPLE_JSONB_ROWS

        mock_client = MagicMock()
        chain = mock_client.table.return_value.select.return_value
        chain.order.return_value = chain
        chain.gte.return_value = chain
        chain.lte.return_value = chain
        chain.execute.return_value = result

        with patch("parking_api.supabase_client._get_client", return_value=mock_client):
            rows = fetch_predictions()

        required = {"created_at", "target_time", "model_tier", "lot", "prediction", "confidence_low", "confidence_high"}
        for row in rows:
            assert required <= set(row.keys())


# ── compute_metrics ───────────────────────────────────────────────────────

class TestComputeMetrics:
    """Test the metrics computation with known values."""

    def test_perfect_predictions(self):
        from evaluate_predictions import compute_metrics
        df = pd.DataFrame({
            "prediction": [0.5, 0.6, 0.7],
            "actual": [0.5, 0.6, 0.7],
            "confidence_low": [0.4, 0.5, 0.6],
            "confidence_high": [0.6, 0.7, 0.8],
        })
        m = compute_metrics(df)
        assert m["mae"] == 0.0
        assert m["rmse"] == 0.0
        assert m["r2"] == 1.0
        assert m["within_band_pct"] == 100.0

    def test_known_errors(self):
        from evaluate_predictions import compute_metrics
        df = pd.DataFrame({
            "prediction": [0.5, 0.7],
            "actual": [0.4, 0.6],
            "confidence_low": [0.3, 0.5],
            "confidence_high": [0.6, 0.8],
        })
        m = compute_metrics(df)
        assert m["mae"] == pytest.approx(0.1)
        assert m["rmse"] == pytest.approx(0.1)
        assert m["within_band_pct"] == 100.0  # both actuals within bands

    def test_outside_band(self):
        from evaluate_predictions import compute_metrics
        df = pd.DataFrame({
            "prediction": [0.5],
            "actual": [0.9],
            "confidence_low": [0.4],
            "confidence_high": [0.6],
        })
        m = compute_metrics(df)
        assert m["within_band_pct"] == 0.0


# ── horizon helpers ────────────────────────────────────────────────────────

class TestHorizonHelpers:
    def test_add_minutes_ahead_rounds_to_5min(self):
        from evaluate_predictions import add_minutes_ahead
        df = pd.DataFrame({
            "created_at": pd.to_datetime(["2026-04-13T12:01:00+00:00"], utc=True),
            "target_time": pd.to_datetime(["2026-04-13T12:16:10+00:00"], utc=True),
        })
        out = add_minutes_ahead(df)
        assert out.loc[0, "minutes_ahead"] == 15

    def test_build_horizon_comparison_pivots_tiers(self):
        from evaluate_predictions import build_horizon_metrics, build_horizon_comparison

        matched = pd.DataFrame({
            "created_at": pd.to_datetime([
                "2026-04-13T12:00:00+00:00",
                "2026-04-13T12:00:00+00:00",
                "2026-04-13T12:00:00+00:00",
                "2026-04-13T12:00:00+00:00",
            ], utc=True),
            "target_time": pd.to_datetime([
                "2026-04-13T12:05:00+00:00",
                "2026-04-13T12:10:00+00:00",
                "2026-04-13T12:05:00+00:00",
                "2026-04-13T12:10:00+00:00",
            ], utc=True),
            "model_tier": ["lgb", "lgb", "lgb_v3", "lgb_v3"],
            "lot": ["CRI", "CRI", "CRI", "CRI"],
            "prediction": [0.50, 0.62, 0.48, 0.58],
            "confidence_low": [0.40, 0.52, 0.38, 0.48],
            "confidence_high": [0.60, 0.72, 0.58, 0.68],
            "actual": [0.55, 0.70, 0.50, 0.60],
        })

        tier_subsets = {
            tier: matched[matched["model_tier"] == tier].copy()
            for tier in ["lgb", "lgb_v3"]
        }
        horizon_metrics = build_horizon_metrics(tier_subsets)
        comparison = build_horizon_comparison(horizon_metrics)

        assert list(comparison["minutes_ahead"]) == [5, 10]
        assert "lgb_mae" in comparison.columns
        assert "lgb_v3_mae" in comparison.columns
        assert comparison.loc[comparison["minutes_ahead"] == 5, "best_mae_tier"].iloc[0] == "lgb_v3"


# ── match_predictions_to_actuals ──────────────────────────────────────────

class TestMatchPredictions:
    """Test the merge_asof matching logic."""

    def test_exact_match(self):
        from evaluate_predictions import match_predictions_to_actuals
        preds = pd.DataFrame({
            "target_time": pd.to_datetime(["2026-04-13 12:05:00+00:00"], utc=True),
            "lot": ["CRI"],
            "prediction": [0.5],
            "confidence_low": [0.4],
            "confidence_high": [0.6],
            "model_tier": ["lgb"],
            "created_at": pd.to_datetime(["2026-04-13 12:01:00+00:00"], utc=True),
        })
        actuals = pd.DataFrame({
            "actual_time": pd.to_datetime(["2026-04-13 12:05:00+00:00"], utc=True),
            "lot": ["CRI"],
            "actual": [0.48],
        })
        matched = match_predictions_to_actuals(preds, actuals)
        assert len(matched) == 1
        assert matched.iloc[0]["actual"] == 0.48

    def test_within_tolerance(self):
        from evaluate_predictions import match_predictions_to_actuals
        preds = pd.DataFrame({
            "target_time": pd.to_datetime(["2026-04-13 12:05:00+00:00"], utc=True),
            "lot": ["CRI"],
            "prediction": [0.5],
            "confidence_low": [0.4],
            "confidence_high": [0.6],
            "model_tier": ["lgb"],
            "created_at": pd.to_datetime(["2026-04-13 12:01:00+00:00"], utc=True),
        })
        # Actual 2 minutes away — within 4-min tolerance
        actuals = pd.DataFrame({
            "actual_time": pd.to_datetime(["2026-04-13 12:07:00+00:00"], utc=True),
            "lot": ["CRI"],
            "actual": [0.48],
        })
        matched = match_predictions_to_actuals(preds, actuals)
        assert len(matched) == 1

    def test_outside_tolerance(self):
        from evaluate_predictions import match_predictions_to_actuals
        preds = pd.DataFrame({
            "target_time": pd.to_datetime(["2026-04-13 12:05:00+00:00"], utc=True),
            "lot": ["CRI"],
            "prediction": [0.5],
            "confidence_low": [0.4],
            "confidence_high": [0.6],
            "model_tier": ["lgb"],
            "created_at": pd.to_datetime(["2026-04-13 12:01:00+00:00"], utc=True),
        })
        # Actual 10 minutes away — outside 4-min tolerance
        actuals = pd.DataFrame({
            "actual_time": pd.to_datetime(["2026-04-13 12:15:00+00:00"], utc=True),
            "lot": ["CRI"],
            "actual": [0.48],
        })
        matched = match_predictions_to_actuals(preds, actuals)
        assert len(matched) == 0

    def test_multiple_lots(self):
        from evaluate_predictions import match_predictions_to_actuals
        preds = pd.DataFrame({
            "target_time": pd.to_datetime(["2026-04-13 12:05+00:00", "2026-04-13 12:05+00:00"], utc=True),
            "lot": ["CRI", "ED1"],
            "prediction": [0.5, 0.7],
            "confidence_low": [0.4, 0.6],
            "confidence_high": [0.6, 0.8],
            "model_tier": ["lgb", "lgb"],
            "created_at": pd.to_datetime(["2026-04-13 12:01+00:00", "2026-04-13 12:01+00:00"], utc=True),
        })
        actuals = pd.DataFrame({
            "actual_time": pd.to_datetime(["2026-04-13 12:05+00:00", "2026-04-13 12:05+00:00"], utc=True),
            "lot": ["CRI", "ED1"],
            "actual": [0.48, 0.69],
        })
        matched = match_predictions_to_actuals(preds, actuals)
        assert len(matched) == 2
        assert set(matched["lot"]) == {"CRI", "ED1"}
