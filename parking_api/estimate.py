"""Estimated forecasts for decks whose live feed is broken (see SUPPRESSED_LOTS).

A small model trained while the deck's sensor was healthy maps the *other* decks' availability
(plus time/calendar context) to the broken deck's availability. At inference each forecast tier's
predicted values for the other decks are fed through it, so users still get an indicative
forecast. Entries are written with ``"estimated": True``. Trained by
``uncc-parking-notebook/train_west_estimator.py``.
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ESTIMATOR_DIRS, safe_name

log = logging.getLogger(__name__)


@dataclass
class Estimator:
    lot: str
    model: object
    config: dict


_cache: dict[str, Estimator | None] = {}


def load_estimator(lot: str) -> Estimator | None:
    if lot not in _cache:
        models_dir: Path | None = ESTIMATOR_DIRS.get(lot)
        try:
            if models_dir is None:
                raise FileNotFoundError(f"no estimator configured for {lot}")
            with open(models_dir / "lgb_point.pkl", "rb") as f:
                model = pickle.load(f)
            with open(models_dir / "lgb_config.pkl", "rb") as f:
                config = pickle.load(f)
            _cache[lot] = Estimator(lot, model, config)
            log.info("Loaded %s estimator from %s", lot, models_dir)
        except Exception as exc:
            log.warning("No estimator for %s (%s) — it will be omitted", lot, exc)
            _cache[lot] = None
    return _cache[lot]


def add_estimates(predictions: list[dict], lots: set[str], context_for) -> list[dict]:
    """Add estimated entries for ``lots`` to each prediction row, in place.

    ``context_for(target_dt, feature_names)`` must return the target-time context features
    (predict._build_target_feature_dict). Rows missing any input deck are left untouched.
    """
    for lot in lots:
        est = load_estimator(lot)
        if est is None or not predictions:
            continue
        cfg = est.config
        others = cfg["other_lots"]
        ctx_names = cfg["context_features"]
        offsets = cfg["band_offsets_by_local_hour"]
        tz = ZoneInfo(cfg.get("band_timezone", "America/New_York"))

        usable = [row for row in predictions if all(o in row["data"] for o in others)]
        if not usable:
            continue
        records, hours = [], []
        for row in usable:
            target_dt = datetime.fromisoformat(row["target_time"])
            ctx = context_for(target_dt, ctx_names)
            rec = {f"other_{safe_name(o)}": row["data"][o]["prediction"] for o in others}
            rec.update({name: ctx.get(name, 0.0) for name in ctx_names})
            records.append(rec)
            hours.append(target_dt.astimezone(tz).hour)

        X = pd.DataFrame(records)[cfg["features"]].astype(np.float32)
        point = np.clip(est.model.predict(X), 0, 1)
        for row, p, hour in zip(usable, point, hours):
            lo = float(np.clip(p + offsets["low"].get(hour, offsets["low"].get(str(hour), 0.0)), 0, 1))
            hi = float(np.clip(p + offsets["high"].get(hour, offsets["high"].get(str(hour), 0.0)), 0, 1))
            row["data"][lot] = {
                "prediction": round(float(p), 4),
                "confidence_low": round(min(lo, float(p)), 4),
                "confidence_high": round(max(hi, float(p)), 4),
                "estimated": True,
            }
    return predictions
