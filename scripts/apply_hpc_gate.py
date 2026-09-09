"""Apply HPC short-RUL gate to cached OOF preds and refresh submissions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.data_loader import TARGETS, project_root
from src.ensemble import apply_ensemble
from src.hpc_gate import (
    HPC_TARGET,
    apply_hpc_gate,
    available_gate_features,
    config_to_dict,
    fit_short_rul_classifier,
    nested_oof_gated_predictions,
    predict_short_proba,
)
from src.scoring import full_metrics, target_betas
from src.split import load_or_create_folds
from src.submission import write_submission


def main() -> None:
    root = project_root()
    reports = root / "reports"
    models_dir = root / "models"
    processed = root / "data" / "processed"
    outputs = root / "outputs"

    samples = pd.read_pickle(processed / "train_windows.pkl")
    folds = load_or_create_folds(samples, reports / "engine_holdout_folds.csv")
    betas = target_betas(samples["Cycles_to_WW"], samples["Cycles_to_HPC_SV"], samples["Cycles_to_HPT_SV"])

    # Prefer explicit pre-gate file; otherwise treat current oof as pre-gate once.
    pre_path = reports / "oof_predictions_pre_gate.csv"
    cur_path = reports / "oof_predictions.csv"
    if pre_path.exists():
        pre_gate = pd.read_csv(pre_path)
    else:
        pre_gate = pd.read_csv(cur_path)
        pre_gate.to_csv(pre_path, index=False)

    assert len(pre_gate) == len(samples)
    assert (pre_gate["engine"].to_numpy() == samples["engine"].to_numpy()).all()

    pre_metrics = full_metrics(samples[TARGETS], pre_gate[TARGETS])
    print("pre-gate overall", pre_metrics["score"], "HPC", pre_metrics["score_HPC"])

    gated_hpc, fold_cfgs, cfg = nested_oof_gated_predictions(
        samples, pre_gate[HPC_TARGET].to_numpy(), folds, betas[HPC_TARGET]
    )
    gated = pre_gate.copy()
    gated[HPC_TARGET] = gated_hpc
    metrics = full_metrics(samples[TARGETS], gated[TARGETS])
    print("post-gate overall", metrics["score"], "HPC", metrics["score_HPC"])
    print("production gate", config_to_dict(cfg))

    gate_payload = {
        **config_to_dict(cfg),
        "fold_configs": fold_cfgs,
        "pre_gate_score": pre_metrics["score"],
        "pre_gate_score_HPC": pre_metrics["score_HPC"],
        "post_gate_score": metrics["score"],
        "post_gate_score_HPC": metrics["score_HPC"],
    }
    (reports / "hpc_gate.json").write_text(json.dumps(gate_payload, indent=2))
    (reports / "ensemble_pre_gate_oof_metrics.json").write_text(json.dumps(pre_metrics, indent=2))
    (reports / "ensemble_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "model_oof_metrics.json").write_text(json.dumps(metrics, indent=2))
    gated.to_csv(cur_path, index=False)
    pre_gate.to_csv(pre_path, index=False)

    weights_path = reports / "ensemble_weights.json"
    weights = json.loads(weights_path.read_text()) if weights_path.exists() else {}
    if HPC_TARGET not in weights:
        weights[HPC_TARGET] = {}
    weights[HPC_TARGET]["hpc_gate"] = gate_payload
    weights[HPC_TARGET]["oof_score"] = metrics["score_HPC"]
    weights_path.write_text(json.dumps(weights, indent=2))

    gate_feats = available_gate_features(samples.columns)
    clf = fit_short_rul_classifier(samples[gate_feats], samples[HPC_TARGET].to_numpy())
    with open(models_dir / "hpc_short_rul_gate.pkl", "wb") as f:
        pickle.dump({"model": clf, "feature_cols": gate_feats, "config": config_to_dict(cfg)}, f)

    meta_path = models_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["hpc_gate"] = config_to_dict(cfg)
    meta["oof_score_pre_gate"] = pre_metrics["score"]
    meta["oof_score"] = metrics["score"]
    if "ensemble" in meta and HPC_TARGET in meta["ensemble"]:
        meta["ensemble"][HPC_TARGET]["hpc_gate"] = gate_payload
        meta["ensemble"][HPC_TARGET]["oof_score"] = metrics["score_HPC"]
    meta_path.write_text(json.dumps(meta, indent=2))

    # Refresh submissions from frozen final models + gate.
    feat_cols = meta.get("feature_cols")
    ens = meta.get("ensemble")
    if not feat_cols or not ens:
        print("metadata incomplete; skipped submission refresh")
        return

    final_lgb = {}
    final_cb = {}
    for target in TARGETS:
        booster = lgb.Booster(model_file=str(models_dir / f"{target}.txt"))
        final_lgb[target] = booster
        cb_model = CatBoostRegressor()
        cb_model.load_model(str(models_dir / f"catboost_{target}.cbm"))
        final_cb[target] = cb_model

    for kind, out_name in (("test", "submission.csv"), ("val", "validation_submission.csv")):
        infer = pd.read_pickle(processed / f"{kind}_windows.pkl")
        rows = pd.DataFrame({"file": infer["file"]})
        x_inf = infer.reindex(columns=feat_cols)
        for target in TARGETS:
            pred_lgb = final_lgb[target].predict(x_inf)
            pred_cb = final_cb[target].predict(x_inf)
            rows[target] = apply_ensemble(
                pred_lgb,
                pred_cb,
                ens[target]["lgb_weight"],
                ens[target]["offset"],
                target,
            )
        gate_x = infer.reindex(columns=cfg.feature_cols)
        proba = predict_short_proba(clf, gate_x)
        rows[HPC_TARGET] = apply_hpc_gate(rows[HPC_TARGET].to_numpy(), proba, cfg)
        expected = infer["file"].tolist()
        write_submission(outputs / out_name, rows, expected)
        write_submission(root / out_name, rows, expected)
        print(f"wrote {out_name} n={len(rows)}")


if __name__ == "__main__":
    main()
