"""Validation-only additive calibration for asymmetric scoring."""

from __future__ import annotations

import numpy as np

from src.scoring import score_target
from src.train import clip_pred

ALPHA = 0.01


def fit_additive_offset(y_true, y_pred, target: str, beta: float, grid: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if grid is None:
        scale = max(np.std(y_true), 1.0)
        grid = np.linspace(-0.25 * scale, 0.05 * scale, 31)
    best_off, best = 0.0, np.inf
    for off in grid:
        sc = score_target(y_true, clip_pred(target, y_pred + off), ALPHA, beta)
        if sc < best:
            best, best_off = sc, float(off)
    return best_off


def apply_calibration(pred, offset: float, target: str):
    return clip_pred(target, np.asarray(pred, dtype=float) + offset)
