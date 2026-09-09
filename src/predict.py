"""End-to-end PHM remaining-cycle training and submission generation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.calibrate import apply_calibration, fit_additive_offset
from src.data_loader import TARGETS, load_split_files, load_training, project_root
from src.data_quality import write_profile_reports
from src.ensemble import apply_ensemble, fit_blend_and_offset
from src.features import ResidualModel, extract_window_sample, pivot_cycle_table, window_feature_row
from src.hpc_gate import (
    HPC_TARGET,
    apply_hpc_gate,
    available_gate_features,
    config_to_dict,
    fit_short_rul_classifier,
    nested_oof_gated_predictions,
    predict_short_proba,
)
from src.leakage import write_audit
from src.scoring import full_metrics, target_betas
from src.split import iter_fold_masks, load_or_create_folds
from src.submission import write_submission
from src.train import (
    QUANTILES,
    TARGET_QUANTILES,
    clip_pred,
    select_quantile_from_oof,
    train_catboost_quantile_model,
    train_quantile_model,
)

SEED = 42
WINDOW_POINTS = 151
WINDOW_STEP = 3


def _feature_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    feat_cols = [c for c in frame.columns if c not in ["engine", "end_cycle", "file"] + TARGETS]
    return frame, feat_cols


def build_train_samples(train_df: pd.DataFrame, residual_model: ResidualModel) -> pd.DataFrame:
    records = []
    for esn, g in train_df.groupby("ESN"):
        wide = pivot_cycle_table(g)
        cycles = wide.index.to_numpy()
        if len(cycles) < WINDOW_POINTS:
            continue
        print(f"engine {esn} cycles={len(cycles)}")
        n_win = 0
        for end_idx in range(WINDOW_POINTS - 1, len(cycles), WINDOW_STEP):
            start_idx = end_idx - (WINDOW_POINTS - 1)
            sl = wide.iloc[start_idx : end_idx + 1]
            end_cycle = int(cycles[end_idx])
            start_cycle = int(cycles[start_idx])
            window = g[(g["Cycles"] >= start_cycle) & (g["Cycles"] <= end_cycle)]
            feats = window_feature_row(sl)
            if residual_model.fitted:
                feats.update(residual_model.residuals_last_row(window))
            lab_rows = g[g["Cycles"] == end_cycle]
            feats["engine"] = int(esn)
            feats["end_cycle"] = end_cycle
            for t in TARGETS:
                feats[t] = float(lab_rows[t].iloc[0])
            records.append(feats)
            n_win += 1
            if n_win % 200 == 0:
                print(f"  {esn} windows {n_win}")
        print(f"engine {esn} done windows={n_win}")
    frame, _ = _feature_frame(records)
    return frame


def build_infer_samples(kind: str, residual_model: ResidualModel) -> pd.DataFrame:
    records = []
    for name, df in load_split_files(kind):
        feats = extract_window_sample(df, residual_model)
        feats["file"] = name
        feats["engine"] = int(df["ESN"].iloc[0])
        records.append(feats)
    frame, _ = _feature_frame(records)
    return frame


def median_baseline(y_train, n: int) -> np.ndarray:
    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    return dummy.predict(np.zeros((n, 1)))


def _empty_oof(n: int) -> dict:
    return {t: {q: np.zeros(n) for q in TARGET_QUANTILES[t]} for t in TARGETS}


def _ensure_oof_slots(oof: dict, n: int) -> list[tuple[str, float]]:
    """Make sure every target/quantile has an array; return missing (target, q) pairs."""
    missing: list[tuple[str, float]] = []
    for target in TARGETS:
        oof.setdefault(target, {})
        for q in TARGET_QUANTILES[target]:
            if q not in oof[target]:
                oof[target][q] = np.zeros(n)
                missing.append((target, float(q)))
    return missing


def _load_oof_cache(cache_path: Path, n: int) -> tuple[dict, set[int] | None]:
    """Return (oof, remaining-fold set). done_folds is None when the cache is complete."""
    if not cache_path.exists():
        return _empty_oof(n), set()
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "done_folds" in payload and "oof" in payload:
        done = set(int(f) for f in payload["done_folds"])
        print(f"Loading partial {cache_path.name} folds={sorted(done)}")
        return payload["oof"], done
    print(f"Loading cached OOF from {cache_path}")
    return payload, None


def _save_oof_cache(cache_path: Path, oof: dict, done_folds: set[int], complete: bool) -> None:
    payload = oof if complete else {"oof": oof, "done_folds": sorted(done_folds)}
    with cache_path.open("wb") as f:
        pickle.dump(payload, f)


def train_family_oof(family: str, samples: pd.DataFrame, feat_cols: list[str], folds, cache_path: Path) -> dict:
    oof, done_folds = _load_oof_cache(cache_path, len(samples))
    missing = _ensure_oof_slots(oof, len(samples))
    fold_ids = [fold for fold, *_ in iter_fold_masks(samples, folds)]

    def _train_jobs(jobs: list[tuple[str, float]], folds_to_run: list[int], label: str) -> set[int]:
        finished = set(done_folds or [])
        print(f"{label} ({family}) jobs={jobs} folds={folds_to_run}")
        for fold, held, tr_mask, va_mask in iter_fold_masks(samples, folds):
            if fold not in folds_to_run:
                continue
            x_tr, x_va = samples.loc[tr_mask, feat_cols], samples.loc[va_mask, feat_cols]
            print(
                f"  {family} fold {fold} holdout ESN {held} "
                f"n_train={tr_mask.sum()} n_valid={va_mask.sum()} jobs={len(jobs)}"
            )
            for target, q in jobs:
                y_tr = samples.loc[tr_mask, target]
                if family == "lightgbm":
                    model = train_quantile_model(x_tr, y_tr, q, x_va, samples.loc[va_mask, target])
                elif family == "catboost":
                    model = train_catboost_quantile_model(x_tr, y_tr, q, x_va, samples.loc[va_mask, target])
                else:
                    raise ValueError(family)
                oof[target][q][va_mask.to_numpy()] = model.predict(x_va)
            finished.add(fold)
            complete = set(fold_ids) <= finished
            _save_oof_cache(cache_path, oof, finished, complete)
            print(f"  saved {family} checkpoint folds={sorted(finished)}")
        return finished

    # Fill newly added quantiles on every fold (keeps old quantile OOF intact).
    if missing:
        done_folds = _train_jobs(missing, fold_ids, f"Extending OOF with {missing}")

    if done_folds is None:
        return oof

    remaining = [f for f in fold_ids if f not in done_folds]
    if remaining:
        jobs = [(t, q) for t in TARGETS for q in TARGET_QUANTILES[t]]
        done_folds = _train_jobs(jobs, remaining, "Engine-holdout quantile training")
    return oof


def family_selected_oof(oof: dict, samples: pd.DataFrame, betas: dict, family: str) -> tuple[dict, pd.DataFrame]:
    selected = {}
    frame = pd.DataFrame({"engine": samples["engine"].values})
    for target in TARGETS:
        y = samples[target].to_numpy()
        q, pred, qscore = select_quantile_from_oof(oof[target], y, target, betas)
        offset = fit_additive_offset(y, pred, target, betas[target])
        selected[target] = {
            "family": family,
            "quantile": q,
            "offset": offset,
            "oof_quantile_score": qscore,
        }
        frame[target] = apply_calibration(pred, offset, target)
        print(family, target, selected[target])
    return selected, frame


def run() -> None:
    root = project_root()
    reports = root / "reports"
    models_dir = root / "models"
    outputs = root / "outputs"
    processed = root / "data" / "processed"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    write_profile_reports(reports)
    write_audit(reports / "leakage_audit.csv")

    train_df = load_training()
    residual_model = ResidualModel().fit(train_df, max_cycle=200)

    cache_path = processed / "train_windows.pkl"
    if cache_path.exists():
        print(f"Loading cached training windows from {cache_path}")
        samples = pd.read_pickle(cache_path)
    else:
        print("Building training windows...")
        samples = build_train_samples(train_df, residual_model)
        samples.to_pickle(cache_path)
    print(f"train windows: {samples.shape}")

    feat_cols = [c for c in samples.columns if c not in ["engine", "end_cycle"] + TARGETS]
    folds = load_or_create_folds(samples, reports / "engine_holdout_folds.csv")
    engines = sorted(int(e) for e in samples["engine"].unique())
    betas = target_betas(samples["Cycles_to_WW"], samples["Cycles_to_HPC_SV"], samples["Cycles_to_HPT_SV"])

    oof_lgb = train_family_oof("lightgbm", samples, feat_cols, folds, processed / "lgb_oof.pkl")
    oof_cb = train_family_oof("catboost", samples, feat_cols, folds, processed / "cb_oof.pkl")

    selected_lgb, lgb_cal = family_selected_oof(oof_lgb, samples, betas, "lightgbm")
    selected_cb, cb_cal = family_selected_oof(oof_cb, samples, betas, "catboost")

    baseline_pred = pd.DataFrame({"engine": samples["engine"].values})
    ensemble_pred = pd.DataFrame({"engine": samples["engine"].values})
    ensemble_sel = {}
    lgb_raw = {}
    cb_raw = {}
    for target in TARGETS:
        y = samples[target].to_numpy()
        q_lgb = selected_lgb[target]["quantile"]
        q_cb = selected_cb[target]["quantile"]
        lgb_raw[target] = clip_pred(target, oof_lgb[target][q_lgb])
        cb_raw[target] = clip_pred(target, oof_cb[target][q_cb])
        fitted = fit_blend_and_offset(y, lgb_raw[target], cb_raw[target], target, betas[target])
        ensemble_sel[target] = {
            "lgb_quantile": q_lgb,
            "cb_quantile": q_cb,
            "lgb_weight": fitted["lgb_weight"],
            "cb_weight": fitted["cb_weight"],
            "offset": fitted["offset"],
            "oof_score": fitted["score"],
        }
        ensemble_pred[target] = fitted["prediction"]
        base = np.zeros(len(samples))
        for held in engines:
            va_mask = samples["engine"] == held
            tr_mask = ~va_mask
            base[va_mask.to_numpy()] = clip_pred(
                target, median_baseline(samples.loc[tr_mask, target], int(va_mask.sum()))
            )
        baseline_pred[target] = base
        print("ensemble", target, ensemble_sel[target])

    print("Fitting HPC short-RUL gate (nested LOEO)...")
    gated_hpc, fold_gate_cfgs, hpc_gate_cfg = nested_oof_gated_predictions(
        samples,
        ensemble_pred[HPC_TARGET].to_numpy(),
        folds,
        betas[HPC_TARGET],
    )
    ensemble_pre_gate = ensemble_pred.copy()
    ensemble_pred[HPC_TARGET] = gated_hpc
    ensemble_sel[HPC_TARGET]["hpc_gate"] = {
        **{k: v for k, v in config_to_dict(hpc_gate_cfg).items() if k != "feature_cols"},
        "feature_cols": hpc_gate_cfg.feature_cols,
        "fold_configs": fold_gate_cfgs,
    }
    print("hpc gate production config", ensemble_sel[HPC_TARGET]["hpc_gate"])

    metrics = {
        "baseline": full_metrics(samples[TARGETS], baseline_pred[TARGETS]),
        "lightgbm": full_metrics(samples[TARGETS], lgb_cal[TARGETS]),
        "catboost": full_metrics(samples[TARGETS], cb_cal[TARGETS]),
        "ensemble_pre_gate": full_metrics(samples[TARGETS], ensemble_pre_gate[TARGETS]),
        "ensemble": full_metrics(samples[TARGETS], ensemble_pred[TARGETS]),
    }
    (reports / "baseline_oof_metrics.json").write_text(json.dumps(metrics["baseline"], indent=2))
    (reports / "lgb_oof_metrics.json").write_text(json.dumps(metrics["lightgbm"], indent=2))
    (reports / "catboost_oof_metrics.json").write_text(json.dumps(metrics["catboost"], indent=2))
    (reports / "ensemble_pre_gate_oof_metrics.json").write_text(
        json.dumps(metrics["ensemble_pre_gate"], indent=2)
    )
    (reports / "ensemble_oof_metrics.json").write_text(json.dumps(metrics["ensemble"], indent=2))
    (reports / "model_oof_metrics.json").write_text(json.dumps(metrics["ensemble"], indent=2))
    (reports / "ensemble_weights.json").write_text(json.dumps(ensemble_sel, indent=2))
    (reports / "hpc_gate.json").write_text(json.dumps(ensemble_sel[HPC_TARGET]["hpc_gate"], indent=2))
    ensemble_pred.to_csv(reports / "oof_predictions.csv", index=False)
    ensemble_pre_gate.to_csv(reports / "oof_predictions_pre_gate.csv", index=False)
    lgb_cal.to_csv(reports / "oof_predictions_lgb.csv", index=False)
    cb_cal.to_csv(reports / "oof_predictions_catboost.csv", index=False)
    print("baseline score", metrics["baseline"]["score"])
    print("lightgbm oof score", metrics["lightgbm"]["score"])
    print("catboost oof score", metrics["catboost"]["score"])
    print("ensemble pre-gate oof score", metrics["ensemble_pre_gate"]["score"])
    print("ensemble oof score", metrics["ensemble"]["score"])

    print("Retraining selected LightGBM and CatBoost models on all training windows...")
    final_lgb = {}
    final_cb = {}
    for target in TARGETS:
        q_lgb = ensemble_sel[target]["lgb_quantile"]
        q_cb = ensemble_sel[target]["cb_quantile"]
        final_lgb[target] = train_quantile_model(samples[feat_cols], samples[target], q_lgb)
        final_cb[target] = train_catboost_quantile_model(samples[feat_cols], samples[target], q_cb)
        final_lgb[target].booster_.save_model(str(models_dir / f"{target}.txt"))
        final_cb[target].save_model(str(models_dir / f"catboost_{target}.cbm"))

    gate_feats = available_gate_features(samples.columns)
    hpc_gate_clf = fit_short_rul_classifier(samples[gate_feats], samples[HPC_TARGET].to_numpy())
    with open(models_dir / "hpc_short_rul_gate.pkl", "wb") as f:
        pickle.dump({"model": hpc_gate_clf, "feature_cols": gate_feats}, f)

    meta = {
        "seed": SEED,
        "window_points": WINDOW_POINTS,
        "window_step": WINDOW_STEP,
        "feature_cols": feat_cols,
        "selected_lightgbm": selected_lgb,
        "selected_catboost": selected_cb,
        "ensemble": ensemble_sel,
        "hpc_gate": config_to_dict(hpc_gate_cfg),
        "lgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
        "catboost": {"iterations": 250, "learning_rate": 0.05, "depth": 6},
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "catboost": __import__("catboost").__version__,
        },
        "oof_score": metrics["ensemble"]["score"],
        "oof_score_pre_gate": metrics["ensemble_pre_gate"]["score"],
        "baseline_score": metrics["baseline"]["score"],
        "lightgbm_oof_score": metrics["lightgbm"]["score"],
        "catboost_oof_score": metrics["catboost"]["score"],
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    for kind, out_name in (("test", "submission.csv"), ("val", "validation_submission.csv")):
        print(f"Predicting {kind}...")
        infer_path = processed / f"{kind}_windows.pkl"
        if infer_path.exists():
            infer = pd.read_pickle(infer_path)
        else:
            infer = build_infer_samples(kind, residual_model)
            infer.to_pickle(infer_path)
        rows = pd.DataFrame({"file": infer["file"]})
        x_inf = infer.reindex(columns=feat_cols)
        for target in TARGETS:
            pred_lgb = final_lgb[target].predict(x_inf)
            pred_cb = final_cb[target].predict(x_inf)
            rows[target] = apply_ensemble(
                pred_lgb,
                pred_cb,
                ensemble_sel[target]["lgb_weight"],
                ensemble_sel[target]["offset"],
                target,
            )
        gate_x = infer.reindex(columns=hpc_gate_cfg.feature_cols)
        proba = predict_short_proba(hpc_gate_clf, gate_x)
        rows[HPC_TARGET] = apply_hpc_gate(rows[HPC_TARGET].to_numpy(), proba, hpc_gate_cfg)
        expected = infer["file"].tolist()
        write_submission(outputs / out_name, rows, expected)
        write_submission(root / out_name, rows, expected)
        print(f"wrote {out_name} n={len(rows)}")


if __name__ == "__main__":
    run()
