# Scoring

Use the exact challenge scoring implementation.

Track:
- total challenge score,
- WW score,
- HPC score,
- HPT score,
- MAE,
- RMSE,
- mean signed error,
- late prediction rate,
- early prediction rate.

Late predictions are more expensive than equally sized early predictions.

Use validation-only:
- quantile selection,
- additive offsets,
- multiplicative calibration,
- ensemble weighting.

Do not use hidden test outcomes for calibration or model selection.
