"""OOF error analysis by target, engine, remaining-cycle buckets, late vs early."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.scoring import TARGETS, full_metrics, time_weighted_error


def analyze(y_true, y_pred, beta, alpha: float = 0.01) -> dict:
    err = y_pred - y_true
    twe = time_weighted_error(y_true, y_pred, alpha=alpha, beta=beta)
    late = err > 0
    early = err < 0
    return {
        "n": int(len(y_true)),
        "score": float(np.mean(twe)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_signed_error": float(np.mean(err)),
        "median_signed_error": float(np.median(err)),
        "late_rate": float(np.mean(late)),
        "early_rate": float(np.mean(early)),
        "late_mae": float(np.mean(np.abs(err[late]))) if late.any() else None,
        "early_mae": float(np.mean(np.abs(err[early]))) if early.any() else None,
        "late_score": float(np.mean(twe[late])) if late.any() else None,
        "early_score": float(np.mean(twe[early])) if early.any() else None,
        "late_mean_error": float(np.mean(err[late])) if late.any() else None,
        "early_mean_error": float(np.mean(err[early])) if early.any() else None,
        "p10_error": float(np.percentile(err, 10)),
        "p50_error": float(np.percentile(err, 50)),
        "p90_error": float(np.percentile(err, 90)),
        "true_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = pd.read_pickle(root / "data/processed/train_windows.pkl")
    pred = pd.read_csv(root / "reports/oof_predictions.csv")
    lgb = pd.read_csv(root / "reports/oof_predictions_lgb.csv")
    cb = pd.read_csv(root / "reports/oof_predictions_catboost.csv")

    assert len(samples) == len(pred)
    assert (samples["engine"].to_numpy() == pred["engine"].to_numpy()).all()

    alpha = 0.01
    betas = {
        "Cycles_to_WW": 1 / float(samples["Cycles_to_WW"].max()),
        "Cycles_to_HPC_SV": 2 / float(samples["Cycles_to_HPC_SV"].max()),
        "Cycles_to_HPT_SV": 2 / float(samples["Cycles_to_HPT_SV"].max()),
    }

    report: dict = {
        "n_windows": int(len(samples)),
        "overall": full_metrics(samples[TARGETS], pred[TARGETS]),
        "by_target": {},
        "by_engine": {},
        "by_bucket": {},
        "hpc_focus": {},
    }

    for t in TARGETS:
        report["by_target"][t] = analyze(samples[t].to_numpy(), pred[t].to_numpy(), betas[t], alpha)

    for eng in sorted(samples["engine"].unique()):
        m = samples["engine"] == eng
        eng_key = str(int(eng))
        report["by_engine"][eng_key] = {}
        for t in TARGETS:
            report["by_engine"][eng_key][t] = analyze(
                samples.loc[m, t].to_numpy(),
                pred.loc[m, t].to_numpy(),
                betas[t],
                alpha,
            )

    for t in TARGETS:
        y = samples[t].to_numpy()
        p = pred[t].to_numpy()
        qs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cuts = np.unique(np.quantile(y, qs))
        report["by_bucket"][t] = []
        if len(cuts) < 3:
            continue
        labels = [f"{int(cuts[i])}–{int(cuts[i + 1])}" for i in range(len(cuts) - 1)]
        cats = pd.cut(y, bins=cuts, include_lowest=True, labels=labels)
        for lab in cats.categories:
            m = cats == lab
            if m.sum() == 0:
                continue
            row = analyze(y[m], p[m], betas[t], alpha)
            row["bucket"] = str(lab)
            row["n_share"] = float(m.mean())
            row["score_contrib"] = row["score"] * row["n"] / len(y)
            report["by_bucket"][t].append(row)

    t = "Cycles_to_HPC_SV"
    y = samples[t].to_numpy()
    p = pred[t].to_numpy()
    pl = lgb[t].to_numpy()
    pc = cb[t].to_numpy()
    err = p - y
    late = err > 0
    early = err < 0
    twe_all = time_weighted_error(y, p, alpha, betas[t])

    hpc_bins = [0, 500, 1000, 2000, 4000, 6000, 8000, 12000]
    hpc_labels = ["0–500", "500–1k", "1k–2k", "2k–4k", "4k–6k", "6k–8k", "8k–12k"]
    hpc_cat = pd.cut(y, bins=hpc_bins, include_lowest=True, labels=hpc_labels)

    hpc_buckets = []
    for lab in hpc_labels:
        m = np.asarray(hpc_cat == lab)
        if m.sum() == 0:
            continue
        row = analyze(y[m], p[m], betas[t], alpha)
        row["bucket"] = lab
        row["n_share"] = float(m.mean())
        row["score_contrib"] = row["score"] * m.sum() / len(y)
        row["lgb_score"] = float(np.mean(time_weighted_error(y[m], pl[m], alpha, betas[t])))
        row["cb_score"] = float(np.mean(time_weighted_error(y[m], pc[m], alpha, betas[t])))
        hpc_buckets.append(row)

    eng_timing = []
    for eng in sorted(samples["engine"].unique()):
        m = (samples["engine"] == eng).to_numpy()
        e = p[m] - y[m]
        twe = time_weighted_error(y[m], p[m], alpha, betas[t])
        late_m = e > 0
        early_m = e < 0
        eng_timing.append(
            {
                "engine": int(eng),
                "n": int(m.sum()),
                "score": float(np.mean(twe)),
                "mae": float(np.mean(np.abs(e))),
                "mean_signed_error": float(np.mean(e)),
                "early_rate": float(np.mean(early_m)),
                "late_rate": float(np.mean(late_m)),
                "early_mae": float(np.mean(np.abs(e[early_m]))) if early_m.any() else None,
                "late_mae": float(np.mean(np.abs(e[late_m]))) if late_m.any() else None,
                "early_score": float(np.mean(twe[early_m])) if early_m.any() else None,
                "late_score": float(np.mean(twe[late_m])) if late_m.any() else None,
                "true_mean": float(np.mean(y[m])),
                "pred_mean": float(np.mean(p[m])),
                "true_max": float(np.max(y[m])),
                "true_min": float(np.min(y[m])),
            }
        )

    topk = int(max(25, 0.01 * len(y)))
    idx = np.argsort(-twe_all)[:topk]
    worst = pd.DataFrame(
        {
            "engine": samples["engine"].to_numpy()[idx],
            "end_cycle": samples["end_cycle"].to_numpy()[idx],
            "y_true": y[idx],
            "y_pred": p[idx],
            "error": p[idx] - y[idx],
            "twe": twe_all[idx],
        }
    )
    worst_summary = {
        "n": topk,
        "share_of_hpc_score_mass": float(twe_all[idx].sum() / twe_all.sum()),
        "early_frac": float(np.mean(worst["error"] < 0)),
        "late_frac": float(np.mean(worst["error"] > 0)),
        "mean_true": float(worst["y_true"].mean()),
        "mean_pred": float(worst["y_pred"].mean()),
        "mean_abs_error": float(worst["error"].abs().mean()),
        "by_engine": {int(k): int(v) for k, v in worst["engine"].value_counts().to_dict().items()},
        "true_bucket_mode": str(
            pd.cut(worst["y_true"], bins=hpc_bins, include_lowest=True, labels=hpc_labels).mode().iloc[0]
        ),
    }

    hist_edges = np.linspace(-6000, 4000, 11)
    hist_counts, _ = np.histogram(err, bins=hist_edges)
    hist = [
        {"lo": float(hist_edges[i]), "hi": float(hist_edges[i + 1]), "n": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    report["hpc_focus"] = {
        "overall": report["by_target"][t],
        "early_vs_late_score_mass": {
            "early_n": int(early.sum()),
            "late_n": int(late.sum()),
            "early_score_mass_share": float(twe_all[early].sum() / twe_all.sum()),
            "late_score_mass_share": float(twe_all[late].sum() / twe_all.sum()),
            "early_mean_score": float(np.mean(twe_all[early])),
            "late_mean_score": float(np.mean(twe_all[late])),
            "note": "late errors (pred > true) get 2x weight in official scorer",
        },
        "fixed_buckets": hpc_buckets,
        "by_engine": eng_timing,
        "worst_windows": worst_summary,
        "signed_error_hist": hist,
        "family_compare_overall": {
            "ensemble_score": float(np.mean(twe_all)),
            "lgb_score": float(np.mean(time_weighted_error(y, pl, alpha, betas[t]))),
            "cb_score": float(np.mean(time_weighted_error(y, pc, alpha, betas[t]))),
        },
    }

    engine_scores = []
    for eng, block in report["by_engine"].items():
        engine_scores.append(
            {
                "engine": int(eng),
                "HPT": block["Cycles_to_HPT_SV"]["score"],
                "HPC": block["Cycles_to_HPC_SV"]["score"],
                "WW": block["Cycles_to_WW"]["score"],
                "HPC_early_rate": block["Cycles_to_HPC_SV"]["early_rate"],
                "HPC_mse": block["Cycles_to_HPC_SV"]["mean_signed_error"],
            }
        )
    report["engine_score_table"] = engine_scores

    out_path = root / "reports" / "oof_error_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("wrote", out_path)

    print("\n=== OVERALL ===")
    print("ensemble", report["overall"]["score"])
    for tt in TARGETS:
        d = report["by_target"][tt]
        print(
            f"{tt}: score={d['score']:.2f} mae={d['mae']:.1f} "
            f"mse={d['mean_signed_error']:.1f} early={d['early_rate']:.1%} late={d['late_rate']:.1%}"
        )

    print("\n=== ENGINE x TARGET scores ===")
    for row in engine_scores:
        print(row)

    print("\n=== HPC buckets ===")
    for row in hpc_buckets:
        print(
            f"{row['bucket']}: n={row['n']} score={row['score']:.1f} contrib={row['score_contrib']:.1f} "
            f"early={row['early_rate']:.0%} mse={row['mean_signed_error']:.0f}"
        )

    print("\n=== HPC early/late mass ===")
    print(report["hpc_focus"]["early_vs_late_score_mass"])
    print("\n=== worst windows ===")
    print(report["hpc_focus"]["worst_windows"])


if __name__ == "__main__":
    main()
