"""Target-specific LightGBM quantile models."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.scoring import TARGETS, score_target, target_betas

QUANTILES = [0.30, 0.35, 0.40, 0.45, 0.50]
ALPHA = 0.01

LGB_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_estimators": 400,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}


@dataclass
class TargetModel:
    target: str
    quantile: float
    model: lgb.LGBMRegressor
    feature_cols: list[str]


def train_quantile_model(x, y, quantile: float, x_val=None, y_val=None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **LGB_PARAMS)
    fit_kw = {}
    if x_val is not None:
        fit_kw["eval_X"] = x_val
        fit_kw["eval_y"] = y_val
        fit_kw["callbacks"] = [lgb.early_stopping(40, verbose=False)]
    model.fit(x, y, **fit_kw)
    return model


def clip_pred(target: str, pred: np.ndarray) -> np.ndarray:
    caps = {"Cycles_to_WW": 1300.0, "Cycles_to_HPC_SV": 12500.0, "Cycles_to_HPT_SV": 6000.0}
    return np.clip(pred, 0.0, caps[target])


def select_quantile_from_oof(oof_by_q: dict[float, np.ndarray], y_true: np.ndarray, target: str, betas: dict) -> tuple[float, np.ndarray, float]:
    best_q, best_score, best_pred = None, np.inf, None
    for q, pred in oof_by_q.items():
        pred = clip_pred(target, pred)
        sc = score_target(y_true, pred, ALPHA, betas[target])
        if sc < best_score:
            best_q, best_score, best_pred = q, sc, pred
    return best_q, best_pred, best_score
