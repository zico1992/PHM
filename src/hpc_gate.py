"""HPC short-RUL gate: pull down long predictions when degradation looks near-event.

Targets the late-tail failure mode found in OOF error analysis (true RUL <=500
but predictions in the thousands), especially ESN-104-like windows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.calibrate import apply_calibration, fit_additive_offset
from src.scoring import score_target
from src.train import clip_pred

ALPHA = 0.01
HPC_TARGET = "Cycles_to_HPC_SV"
SHORT_RUL_THRESHOLD = 500.0

# Existing causal features that separate short vs long HPC RUL on holdout.
GATE_FEATURES = [
    "resid_Sensed_T45_s4",
    "resid_Sensed_T45_s7",
    "resid_Sensed_Core_Speed_s8",
    "resid_Sensed_Core_Speed_s2",
    "Sensed_T25_s2_dlt",
    "Sensed_Core_Speed_s8_dlt",
    "Sensed_T45_s1_slope",
    "Sensed_T45_s1_dlt",
    "Sensed_Core_Speed_s2_slope",
    "Sensed_T3_s6_dlt",
    "resid_Sensed_T3_s6",
    "Sensed_T45_s4_dlt",
    "Sensed_Ps3_s4_dlt",
    "resid_Sensed_Ps3_s4",
]

PROBA_THRESHOLDS = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]
SCALE_FACTORS = [0.45, 0.55, 0.6, 0.65, 0.7, 0.75]


@dataclass
class HpcGateConfig:
    feature_cols: list[str]
    proba_threshold: float
    scale_factor: float
    post_offset: float
    short_rul_threshold: float = SHORT_RUL_THRESHOLD
    oof_score: float | None = None


def available_gate_features(columns) -> list[str]:
    cols = set(columns)
    found = [c for c in GATE_FEATURES if c in cols]
    if not found:
        raise ValueError("No HPC gate features present in frame")
    return found


def _clf() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )


def fit_short_rul_classifier(x: pd.DataFrame, y_rul: np.ndarray, threshold: float = SHORT_RUL_THRESHOLD):
    model = _clf()
    model.fit(x, (np.asarray(y_rul, dtype=float) <= threshold).astype(int))
    return model


def predict_short_proba(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def apply_scale_gate(pred: np.ndarray, proba: np.ndarray, threshold: float, scale_factor: float, target: str = HPC_TARGET) -> np.ndarray:
    out = np.asarray(pred, dtype=float).copy()
    mask = np.asarray(proba, dtype=float) >= threshold
    out[mask] = out[mask] * scale_factor
    return clip_pred(target, out)


def apply_hpc_gate(pred: np.ndarray, proba: np.ndarray, config: HpcGateConfig) -> np.ndarray:
    gated = apply_scale_gate(pred, proba, config.proba_threshold, config.scale_factor, HPC_TARGET)
    return apply_calibration(gated, config.post_offset, HPC_TARGET)


def _score(y_true, y_pred, beta: float) -> float:
    return score_target(y_true, y_pred, ALPHA, beta)


def fit_gate_params(
    y_true: np.ndarray,
    pred: np.ndarray,
    proba: np.ndarray,
    beta: float,
    thresholds: list[float] | None = None,
    scales: list[float] | None = None,
) -> HpcGateConfig:
    """Select threshold/scale + post-offset on provided OOF preds/proba."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    proba = np.asarray(proba, dtype=float)
    thresholds = thresholds or PROBA_THRESHOLDS
    scales = scales or SCALE_FACTORS
    best_score, best = np.inf, None
    for thr in thresholds:
        for scale in scales:
            gated = apply_scale_gate(pred, proba, thr, scale)
            offset = fit_additive_offset(y_true, gated, HPC_TARGET, beta)
            cal = apply_calibration(gated, offset, HPC_TARGET)
            sc = _score(y_true, cal, beta)
            if sc < best_score:
                best_score = sc
                best = HpcGateConfig(
                    feature_cols=[],
                    proba_threshold=float(thr),
                    scale_factor=float(scale),
                    post_offset=float(offset),
                    oof_score=float(sc),
                )
    assert best is not None
    return best


def oof_short_proba(samples: pd.DataFrame, feature_cols: list[str], folds, y_rul: np.ndarray) -> np.ndarray:
    """Leave-one-engine-out short-RUL probabilities (no leakage into held engine)."""
    from src.split import iter_fold_masks

    proba = np.zeros(len(samples), dtype=float)
    for _, _, tr_mask, va_mask in iter_fold_masks(samples, folds):
        model = fit_short_rul_classifier(samples.loc[tr_mask, feature_cols], y_rul[tr_mask.to_numpy()])
        proba[va_mask.to_numpy()] = predict_short_proba(model, samples.loc[va_mask, feature_cols])
    return proba


def nested_oof_gated_predictions(
    samples: pd.DataFrame,
    ensemble_pred: np.ndarray,
    folds,
    beta: float,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[dict], HpcGateConfig]:
    """Honest LOEO gate: fit classifier + params on train engines, apply to held."""
    from src.split import iter_fold_masks

    feature_cols = feature_cols or available_gate_features(samples.columns)
    y = samples[HPC_TARGET].to_numpy(dtype=float)
    pred = np.asarray(ensemble_pred, dtype=float)
    # OOF proba once (classifier LOEO)
    proba = oof_short_proba(samples, feature_cols, folds, y)

    gated = np.zeros(len(samples), dtype=float)
    fold_cfgs: list[dict] = []
    for _, held, tr_mask, va_mask in iter_fold_masks(samples, folds):
        tr = tr_mask.to_numpy()
        va = va_mask.to_numpy()
        cfg = fit_gate_params(y[tr], pred[tr], proba[tr], beta)
        gated[va] = apply_hpc_gate(pred[va], proba[va], cfg)
        fold_cfgs.append({"held_engine": int(held), **{k: v for k, v in asdict(cfg).items() if k != "feature_cols"}})

    # Production config: fit on all OOF rows (for inference metadata)
    prod = fit_gate_params(y, pred, proba, beta)
    prod.feature_cols = feature_cols
    return gated, fold_cfgs, prod


def config_to_dict(config: HpcGateConfig) -> dict:
    return asdict(config)
