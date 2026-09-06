# Project structure

Recommended implementation layout:

```text
project/
├── README.md
├── SKILL.md
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── leakage.py
│   ├── features.py
│   ├── split.py
│   ├── scoring.py
│   ├── train.py
│   ├── calibrate.py
│   ├── ensemble.py
│   ├── predict.py
│   └── submission.py
├── models/
├── reports/
├── outputs/
└── tests/
```

Track experiment metadata such as:
- data version,
- feature version,
- fold definition,
- model,
- hyperparameters,
- target,
- official score,
- calibration,
- random seed.
