---
name: phm-maintenance-prediction
description: Builds leakage-safe predictive-maintenance models for the PHM North America 2025 engine data challenge, including remaining-cycle prediction for HPT shop visits, HPC shop visits, and HPC water-wash events. Use when profiling the PHM challenge dataset, engineering temporal degradation features, designing time-aware validation, training LightGBM/CatBoost/quantile models, optimizing the official asymmetric score, performing error analysis, ensembling models, or generating challenge submissions.
metadata:
  version: "1.0.0"
  tags:
    - prognostics
    - predictive-maintenance
    - time-series
    - lightgbm
    - catboost
    - quantile-regression
    - phm
---

# PHM engine maintenance prediction

Use this skill to build an end-to-end solution for predicting:

- `Cycles_to_HPT_SV`
- `Cycles_to_HPC_SV`
- `Cycles_to_WW`

Treat the task as a longitudinal remaining-life / time-to-maintenance problem. Prioritize point-in-time correctness, temporal validation, domain-aware degradation features, and the official competition metric.

## Workflow

1. Profile the data before modeling.
2. Audit every feature for temporal leakage.
3. Define the prediction grain: snapshot, cycle, or required submission row.
4. Implement the official scoring function.
5. Create time-aware validation folds.
6. Establish simple baselines.
7. Build leakage-safe temporal and maintenance features.
8. Train one model per target.
9. Add quantile models and asymmetric calibration.
10. Run feature ablations and error analysis.
11. Build an out-of-fold ensemble.
12. Retrain the selected pipeline on all permitted training data.
13. Generate and validate the final submission.

Do not skip steps 1–5.

## Rules

- For prediction at cycle `t`, only use information available at or before `t`.
- Never use centered rolling windows.
- Never normalize using the full future history of an engine.
- Never use shuffled random train/test splits as the primary validation method.
- Rank models by the official competition score, not MAE alone.
- Train target-specific boosted-tree models before deep learning.
- Exclude any maintenance metadata whose timing semantics are uncertain until justified.

## Modeling order

```text
Data profiling
→ Leakage audit
→ Official scorer
→ Time-aware validation
→ Baselines
→ LightGBM
→ Temporal/degradation features
→ CatBoost/XGBoost
→ Quantile regression
→ Calibration
→ Feature ablation
→ Ensemble
→ Optional survival/sequence models
→ Final retraining
→ Submission
```

## Progressive disclosure

Read bundled references only when needed:

- `references/leakage-and-validation.md`
- `references/feature-engineering.md`
- `references/scoring.md`
- `references/project-structure.md`
