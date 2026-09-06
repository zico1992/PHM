# Feature engineering

## Sensor features

For each eligible sensor consider:
- current value,
- lag 1/2/5/10/25/50,
- rolling mean,
- rolling median,
- rolling std,
- rolling min/max/range,
- rolling slope,
- EWMA,
- first difference,
- delta from early-life baseline,
- relative delta from baseline.

Candidate rolling windows:
- 5,
- 10,
- 25,
- 50,
- 100 cycles.

## Snapshot or phase features

If snapshot IDs represent stable operating states:
- pivot sensor values by snapshot,
- compute cross-snapshot differences,
- compute phase-specific rolling trends.

## Maintenance-history features

If point-in-time valid:
- cycles since last WW,
- cycles since last HPC shop visit,
- cycles since last HPT shop visit,
- prior event counts,
- event-reset sensor features.

## Health indices

Optionally test:
- PCA,
- Mahalanobis distance,
- robust healthy-state distance,
- autoencoder reconstruction error.

Fit all transforms using training data only.
