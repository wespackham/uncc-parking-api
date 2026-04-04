"""ModelRegistry: loads all .pkl models and provides prediction with confidence bands."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOTS, MODELS_DIR, safe_name


class ModelRegistry:
    def __init__(self, models_dir: Path | str | None = None):
        models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._models = {}
        self._features = {}
        self._load_all(models_dir)

    def _load_all(self, models_dir: Path):
        # Baseline feature list (shared across all baseline models)
        with open(models_dir / "features.pkl", "rb") as f:
            baseline_features = pickle.load(f)
        self._features["baseline"] = baseline_features

        for lot in LOTS:
            sn = safe_name(lot)

            # Baseline model
            with open(models_dir / f"{sn}.pkl", "rb") as f:
                self._models[(lot, "baseline")] = pickle.load(f)

            # 30-min and 60-min horizon models
            for horizon in ("30min", "60min"):
                model_path = models_dir / f"{sn}_{horizon}.pkl"
                feat_path = models_dir / f"{sn}_{horizon}_features.pkl"
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        self._models[(lot, horizon)] = pickle.load(f)
                    with open(feat_path, "rb") as f:
                        self._features[(lot, horizon)] = pickle.load(f)

    def get_feature_names(self, lot: str, horizon: str) -> list[str]:
        if horizon == "baseline":
            return self._features["baseline"]
        return self._features[(lot, horizon)]

    def predict(self, lot: str, horizon: str, X: pd.DataFrame) -> tuple[float, float, float]:
        """Predict occupancy and return (mean, confidence_low, confidence_high).

        Confidence bands from individual tree predictions: mean ± 2*std, clamped to [0, 1].
        """
        model = self._models[(lot, horizon)]
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        mean = float(tree_preds.mean())
        std = float(tree_preds.std())
        low = max(0.0, mean - 2 * std)
        high = min(1.0, mean + 2 * std)
        mean = max(0.0, min(1.0, mean))
        return mean, low, high

    def list_models(self) -> dict:
        return {
            "total": len(self._models),
            "models": [{"lot": lot, "horizon": h} for (lot, h) in self._models.keys()],
        }
