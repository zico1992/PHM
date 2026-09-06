"""Official PHM North America 2025 time-weighted error scorer."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ["Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"]


def time_weighted_error(y_true, y_pred, alpha=0.02, beta=1):
    """Returns the weighted squared error for an array of predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    weight = np.where(
        error >= 0,
        2 / (1 + alpha * y_true),
        1 / (1 + alpha * y_true),
    )
    return weight * (error**2) * beta


def score_target(y_true, y_pred, alpha, beta):
    return float(np.mean(time_weighted_error(y_true, y_pred, alpha, beta)))


def target_betas(y_true_ww, y_true_hpc, y_true_hpt):
    return {
        "Cycles_to_WW": 1 / float(np.max(y_true_ww)),
        "Cycles_to_HPC_SV": 2 / float(np.max(y_true_hpc)),
        "Cycles_to_HPT_SV": 2 / float(np.max(y_true_hpt)),
    }


def score_submitted_result(df_true, df_pred):
    """Calculate the score for a single team's submission."""
    true_WW = df_true.Cycles_to_WW.values
    true_HPC = df_true.Cycles_to_HPC_SV.values
    true_HPT = df_true.Cycles_to_HPT_SV.values

    pred_WW = df_pred.Cycles_to_WW.values
    pred_HPC = df_pred.Cycles_to_HPC_SV.values
    pred_HPT = df_pred.Cycles_to_HPT_SV.values

    alpha = 0.01
    score_WW = np.mean(time_weighted_error(true_WW, pred_WW, alpha, 1 / float(max(true_WW))))
    score_HPC = np.mean(time_weighted_error(true_HPC, pred_HPC, alpha, 2 / float(max(true_HPC))))
    score_HPT = np.mean(time_weighted_error(true_HPT, pred_HPT, alpha, 2 / float(max(true_HPT))))
    score = np.mean([score_WW, score_HPC, score_HPT])
    return float(score), {
        "score": float(score),
        "score_WW": float(score_WW),
        "score_HPC": float(score_HPC),
        "score_HPT": float(score_HPT),
    }


def regression_diagnostics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error": float(np.mean(err)),
        "late_rate": float(np.mean(err > 0)),
        "early_rate": float(np.mean(err < 0)),
    }


def full_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    score, parts = score_submitted_result(df_true, df_pred)
    out = dict(parts)
    for t in TARGETS:
        out[t] = regression_diagnostics(df_true[t].values, df_pred[t].values)
    return out
