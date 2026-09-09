"""Restore the pre-low-q ensemble (HPC LGB q0.30) and re-apply the scale gate."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.calibrate import apply_calibration, fit_additive_offset
from src.data_loader import TARGETS, project_root
from src.ensemble import apply_ensemble, fit_blend_and_offset
from src.hpc_gate import (
    HPC_TARGET,
    available_gate_features,
    config_to_dict,
    fit_short_rul_classifier,
    nested_oof_gated_predictions,
    predict_short_proba,
)
from src.scoring import full_metrics, target_betas
from src.split import load_or_create_folds
from src.submission import write_submission
from src.train import clip_pred, train_catboost_quantile_model, train_quantile_model


def _load_oof(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "oof" in payload:
        return payload["oof"]
    return payload


def main() -> None:
    root = project_root()
    reports = root / "reports"
    models_dir = root / "models"
    processed = root / "data" / "processed"
    outputs = root / "outputs"

    samples = pd.read_pickle(processed / "train_windows.pkl")
    feat_cols = [c for c in samples.columns if c not in ["engine", "end_cycle"] + TARGETS]
    folds = load_or_create_folds(samples, reports / "engine_holdout_folds.csv")
    betas = target_betas(samples["Cycles_to_WW"], samples["Cycles_to_HPC_SV"], samples["Cycles_to_HPT_SV"])
    oof_lgb = _load_oof(processed / "lgb_oof.pkl")
    oof_cb = _load_oof(processed / "cb_oof.pkl")

    # Original recipe: HPC uses LGB q0.30 only (not low-q).
    recipes = {
        "Cycles_to_HPT_SV": {"lgb_q": 0.3, "cb_q": 0.3},
        "Cycles_to_HPC_SV": {"lgb_q": 0.3, "cb_q": 0.35},
        "Cycles_to_WW": {"lgb_q": 0.3, "cb_q": 0.3},
    }
    ensemble_pred = pd.DataFrame({"engine": samples["engine"].values})
    ensemble_sel = {}
    for target, rec in recipes.items():
        y = samples[target].to_numpy()
        lgb_raw = clip_pred(target, oof_lgb[target][rec["lgb_q"]])
        cb_raw = clip_pred(target, oof_cb[target][rec["cb_q"]])
        fitted = fit_blend_and_offset(y, lgb_raw, cb_raw, target, betas[target])
        ensemble_sel[target] = {
            "lgb_quantile": rec["lgb_q"],
            "cb_quantile": rec["cb_q"],
            "lgb_weight": fitted["lgb_weight"],
            "cb_weight": fitted["cb_weight"],
            "offset": fitted["offset"],
            "oof_score": fitted["score"],
        }
        ensemble_pred[target] = fitted["prediction"]
        print("ensemble", target, ensemble_sel[target])

    pre_metrics = full_metrics(samples[TARGETS], ensemble_pred[TARGETS])
    print("pre-gate", pre_metrics["score"], pre_metrics["score_HPC"])

    gated_hpc, fold_cfgs, cfg = nested_oof_gated_predictions(
        samples, ensemble_pred[HPC_TARGET].to_numpy(), folds, betas[HPC_TARGET]
    )
    gated = ensemble_pred.copy()
    gated[HPC_TARGET] = gated_hpc
    metrics = full_metrics(samples[TARGETS], gated[TARGETS])
    print("post-gate", metrics["score"], metrics["score_HPC"])

    gate_payload = {
        **config_to_dict(cfg),
        "fold_configs": fold_cfgs,
        "pre_gate_score": pre_metrics["score"],
        "pre_gate_score_HPC": pre_metrics["score_HPC"],
        "post_gate_score": metrics["score"],
        "post_gate_score_HPC": metrics["score_HPC"],
    }
    ensemble_sel[HPC_TARGET]["hpc_gate"] = gate_payload
    ensemble_sel[HPC_TARGET]["oof_score"] = metrics["score_HPC"]

    ensemble_pred.to_csv(reports / "oof_predictions_pre_gate.csv", index=False)
    gated.to_csv(reports / "oof_predictions.csv", index=False)
    (reports / "ensemble_pre_gate_oof_metrics.json").write_text(json.dumps(pre_metrics, indent=2))
    (reports / "ensemble_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "model_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "ensemble_weights.json").write_text(json.dumps(ensemble_sel, indent=2))
    (reports / "hpc_gate.json").write_text(json.dumps(gate_payload, indent=2))

    print("Retraining final models at original quantiles...")
    final_lgb, final_cb = {}, {}
    for target in TARGETS:
        q_lgb = ensemble_sel[target]["lgb_quantile"]
        q_cb = ensemble_sel[target]["cb_quantile"]
        final_lgb[target] = train_quantile_model(samples[feat_cols], samples[target], q_lgb)
        final_cb[target] = train_catboost_quantile_model(samples[feat_cols], samples[target], q_cb)
        final_lgb[target].booster_.save_model(str(models_dir / f"{target}.txt"))
        final_cb[target].save_model(str(models_dir / f"catboost_{target}.cbm"))

    gate_feats = available_gate_features(samples.columns)
    clf = fit_short_rul_classifier(samples[gate_feats], samples[HPC_TARGET].to_numpy())
    with open(models_dir / "hpc_short_rul_gate.pkl", "wb") as f:
        pickle.dump({"model": clf, "feature_cols": gate_feats, "config": config_to_dict(cfg)}, f)

    meta_path = models_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        {
            "feature_cols": feat_cols,
            "ensemble": ensemble_sel,
            "hpc_gate": config_to_dict(cfg),
            "oof_score": metrics["score"],
            "oof_score_pre_gate": pre_metrics["score"],
            "hpc_strategy": "q030_ensemble_plus_scale_gate",
            "note": "Low quantiles 0.15-0.25 trained but not selected; did not beat gated q0.30.",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    for kind, out_name in (("test", "submission.csv"), ("val", "validation_submission.csv")):
        infer = pd.read_pickle(processed / f"{kind}_windows.pkl")
        rows = pd.DataFrame({"file": infer["file"]})
        x_inf = infer.reindex(columns=feat_cols)
        for target in TARGETS:
            rows[target] = apply_ensemble(
                final_lgb[target].predict(x_inf),
                final_cb[target].predict(x_inf),
                ensemble_sel[target]["lgb_weight"],
                ensemble_sel[target]["offset"],
                target,
            )
        proba = predict_short_proba(clf, infer.reindex(columns=cfg.feature_cols))
        from src.hpc_gate import apply_hpc_gate

        rows[HPC_TARGET] = apply_hpc_gate(rows[HPC_TARGET].to_numpy(), proba, cfg)
        expected = infer["file"].tolist()
        write_submission(outputs / out_name, rows, expected)
        write_submission(root / out_name, rows, expected)
        print(f"wrote {out_name}")

    print("RESTORED overall", metrics["score"], "HPC", metrics["score_HPC"])


if __name__ == "__main__":
    main()
