"""End-to-end PHM remaining-cycle training and submission generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.calibrate import apply_calibration, fit_additive_offset
from src.data_loader import TARGETS, load_split_files, load_training, project_root
from src.data_quality import write_profile_reports
from src.features import ResidualModel, extract_window_sample, pivot_cycle_table, window_feature_row
from src.leakage import write_audit
from src.scoring import full_metrics, target_betas
from src.split import engine_holdout_folds
from src.submission import write_submission
from src.train import QUANTILES, clip_pred, select_quantile_from_oof, train_quantile_model

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


def run() -> None:
    root = project_root()
    reports = root / "reports"
    models_dir = root / "models"
    outputs = root / "outputs"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)

    write_profile_reports(reports)
    write_audit(reports / "leakage_audit.csv")

    train_df = load_training()
    residual_model = ResidualModel().fit(train_df, max_cycle=200)

    cache_path = root / "data" / "processed" / "train_windows.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f"Loading cached training windows from {cache_path}")
        samples = pd.read_pickle(cache_path)
    else:
        print("Building training windows...")
        samples = build_train_samples(train_df, residual_model)
        samples.to_pickle(cache_path)
    print(f"train windows: {samples.shape}")

    feat_cols = [c for c in samples.columns if c not in ["engine", "end_cycle"] + TARGETS]
    folds = engine_holdout_folds(samples, reports / "engine_holdout_folds.csv")

    oof = {t: {q: np.zeros(len(samples)) for q in QUANTILES} for t in TARGETS}
    engines = sorted(samples["engine"].unique())

    print("Engine-holdout LightGBM quantile training...")
    for fold, held in enumerate(engines):
        tr_mask = samples["engine"] != held
        va_mask = samples["engine"] == held
        x_tr, x_va = samples.loc[tr_mask, feat_cols], samples.loc[va_mask, feat_cols]
        print(f"  fold {fold} holdout ESN {held} n_train={tr_mask.sum()} n_valid={va_mask.sum()}")
        for target in TARGETS:
            y_tr = samples.loc[tr_mask, target]
            y_va = samples.loc[va_mask, target]
            for q in QUANTILES:
                model = train_quantile_model(x_tr, y_tr, q, x_va, y_va)
                oof[target][q][va_mask.to_numpy()] = model.predict(x_va)

    betas = target_betas(samples["Cycles_to_WW"], samples["Cycles_to_HPC_SV"], samples["Cycles_to_HPT_SV"])
    selected = {}
    oof_pred = pd.DataFrame({"engine": samples["engine"].values})
    baseline_pred = pd.DataFrame({"engine": samples["engine"].values})

    for target in TARGETS:
        y = samples[target].to_numpy()
        q, pred, qscore = select_quantile_from_oof(oof[target], y, target, betas)
        offset = fit_additive_offset(y, pred, target, betas[target])
        cal = apply_calibration(pred, offset, target)
        selected[target] = {"quantile": q, "offset": offset, "oof_quantile_score": qscore}
        oof_pred[target] = cal
        # baseline: median per training fold would be cleaner; global median is the simple baseline
        base = np.zeros(len(samples))
        for held in engines:
            va_mask = samples["engine"] == held
            tr_mask = ~va_mask
            base[va_mask.to_numpy()] = clip_pred(target, median_baseline(samples.loc[tr_mask, target], int(va_mask.sum())))
        baseline_pred[target] = base
        print(target, selected[target])

    base_metrics = full_metrics(samples[TARGETS], baseline_pred[TARGETS])
    model_metrics = full_metrics(samples[TARGETS], oof_pred[TARGETS])
    (reports / "baseline_oof_metrics.json").write_text(json.dumps(base_metrics, indent=2))
    (reports / "model_oof_metrics.json").write_text(json.dumps(model_metrics, indent=2))
    oof_pred.to_csv(reports / "oof_predictions.csv", index=False)
    print("baseline score", base_metrics["score"])
    print("model oof score", model_metrics["score"])

    print("Retraining selected models on all training windows...")
    final_models = {}
    for target in TARGETS:
        q = selected[target]["quantile"]
        model = train_quantile_model(samples[feat_cols], samples[target], q)
        final_models[target] = model
        model.booster_.save_model(str(models_dir / f"{target}.txt"))

    meta = {
        "seed": SEED,
        "window_points": WINDOW_POINTS,
        "window_step": WINDOW_STEP,
        "feature_cols": feat_cols,
        "selected": selected,
        "lgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "lightgbm": __import__("lightgbm").__version__,
        },
        "oof_score": model_metrics["score"],
        "baseline_score": base_metrics["score"],
    }
    (models_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    for kind, out_name in (("test", "submission.csv"), ("val", "validation_submission.csv")):
        print(f"Predicting {kind}...")
        infer = build_infer_samples(kind, residual_model)
        infer.to_pickle(root / "data" / "processed" / f"{kind}_windows.pkl")
        rows = pd.DataFrame({"file": infer["file"]})
        for target in TARGETS:
            x_inf = infer.reindex(columns=feat_cols)
            pred = final_models[target].predict(x_inf)
            rows[target] = apply_calibration(pred, selected[target]["offset"], target)
        expected = infer["file"].tolist()
        write_submission(outputs / out_name, rows, expected)
        write_submission(root / out_name, rows, expected)
        print(f"wrote {out_name} n={len(rows)}")


if __name__ == "__main__":
    run()
