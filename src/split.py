"""Engine-holdout folds reused across model comparisons."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def engine_holdout_folds(sample_frame: pd.DataFrame, out_path: Path | None = None) -> pd.DataFrame:
    engines = sorted(sample_frame["engine"].unique())
    rows = []
    for fold, held in enumerate(engines):
        for eng in engines:
            rows.append({"engine": int(eng), "fold": fold, "role": "valid" if eng == held else "train"})
    folds = pd.DataFrame(rows)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(out_path, index=False)
    return folds


def split_xy(frame: pd.DataFrame, feature_cols: list[str], target: str):
    x = frame[feature_cols]
    y = frame[target]
    return x, y
