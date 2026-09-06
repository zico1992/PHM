# Leakage and validation

## Point-in-time rule

For every prediction at cycle `t`, all features must use data from `cycle <= t`.

Forbidden:
- future sensor values,
- centered rolling windows,
- full-history engine normalization,
- post-event metadata,
- target-derived features,
- preprocessing fitted on validation or test rows.

## Validation hierarchy

Use:
1. engine holdout,
2. forward temporal holdout,
3. rolling-origin validation.

Store fold assignments explicitly and reuse identical folds across model comparisons.

## Leakage audit

For each candidate feature record:
- feature name,
- feature family,
- whether available at prediction time,
- leakage risk,
- keep/exclude decision,
- reason.
