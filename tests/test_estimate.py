"""Tests for estimated forecasts of suppressed lots (parking_api/estimate.py)."""

import numpy as np

from parking_api import estimate


class _FakeModel:
    def predict(self, X):
        # "WEST" = mean of the other decks' predictions
        return X[[c for c in X.columns if c.startswith("other_")]].mean(axis=1).values


def _install(monkeypatch, others=("CRI", "UDL")):
    cfg = {
        "other_lots": list(others),
        "features": [f"other_{o}" for o in others] + ["tgt_hour_sin"],
        "context_features": ["tgt_hour_sin"],
        "band_offsets_by_local_hour": {"low": {h: -0.1 for h in range(24)}, "high": {h: 0.2 for h in range(24)}},
        "band_timezone": "America/New_York",
    }
    monkeypatch.setitem(estimate._cache, "WEST", estimate.Estimator("WEST", _FakeModel(), cfg))


def _row(t, **lots):
    return {"target_time": t, "model_tier": "lgb_v4",
            "data": {k: {"prediction": v, "confidence_low": v, "confidence_high": v} for k, v in lots.items()}}


def test_adds_flagged_estimate_with_calibrated_band(monkeypatch):
    _install(monkeypatch)
    rows = [_row("2026-10-02T14:00:00+00:00", CRI=0.4, UDL=0.6)]

    estimate.add_estimates(rows, {"WEST"}, lambda dt, names: {"tgt_hour_sin": 0.0})

    west = rows[0]["data"]["WEST"]
    assert west["estimated"] is True
    assert np.isclose(west["prediction"], 0.5)
    assert np.isclose(west["confidence_low"], 0.4) and np.isclose(west["confidence_high"], 0.7)


def test_rows_missing_an_input_deck_are_left_alone(monkeypatch):
    _install(monkeypatch)
    rows = [_row("2026-10-02T14:00:00+00:00", CRI=0.4)]

    estimate.add_estimates(rows, {"WEST"}, lambda dt, names: {})

    assert "WEST" not in rows[0]["data"]


def test_no_estimator_means_lot_stays_omitted(monkeypatch):
    monkeypatch.setitem(estimate._cache, "WEST", None)
    rows = [_row("2026-10-02T14:00:00+00:00", CRI=0.4, UDL=0.6)]

    estimate.add_estimates(rows, {"WEST"}, lambda dt, names: {})

    assert "WEST" not in rows[0]["data"]
