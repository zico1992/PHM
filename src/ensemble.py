"""OOF convex blending ranked only by the official PHM score."""

from __future__ import annotations

import numpy as np

from src.calibrate import apply_calibration, fit_additive_offset
from src.scoring import score_target
from src.train import clip_pred, select_quantile_from_oof

ALPHA = 0.01
WEIGHT_STEP = 0.05


def blend(pred_a: np.ndarray, pred_b: np.ndarray, weight_a: float) -> np.ndarray:
    return weight_a * np.asarray(pred_a, dtype=float) + (1.0 - weight_a) * np.asarray(pred_b, dtype=float)


def fit_blend_and_offset(
    y_true,
    pred_lgb,
    pred_cb,
    target: str,
    beta: float,
    step: float = WEIGHT_STEP,
) -> dict:
    """Grid-search LightGBM weight in [0, 1] and an additive offset on OOF only."""
    y_true = np.asarray(y_true, dtype=float)
    pred_lgb = clip_pred(target, np.asarray(pred_lgb, dtype=float))
    pred_cb = clip_pred(target, np.asarray(pred_cb, dtype=float))
    best = {"score": np.inf, "lgb_weight": 1.0, "cb_weight": 0.0, "offset": 0.0}
    for w in np.round(np.arange(0.0, 1.0 + 1e-9, step), 4):
        mixed = blend(pred_lgb, pred_cb, w)
        offset = fit_additive_offset(y_true, mixed, target, beta)
        cal = apply_calibration(mixed, offset, target)
        sc = score_target(y_true, cal, ALPHA, beta)
        if sc < best["score"]:
            best = {
                "score": float(sc),
                "lgb_weight": float(w),
                "cb_weight": float(1.0 - w),
                "offset": float(offset),
            }
    mixed = blend(pred_lgb, pred_cb, best["lgb_weight"])
    best["prediction"] = apply_calibration(mixed, best["offset"], target)
    return best


def apply_ensemble(pred_lgb, pred_cb, lgb_weight: float, offset: float, target: str) -> np.ndarray:
    mixed = blend(pred_lgb, pred_cb, lgb_weight)
    return apply_calibration(mixed, offset, target)


__all__ = [
    "apply_ensemble",
    "blend",
    "fit_blend_and_offset",
    "select_quantile_from_oof",
]
