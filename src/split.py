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


def load_or_create_folds(sample_frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Reuse identical stored engine-holdout folds across model families."""
    if path.exists():
        folds = pd.read_csv(path)
        expected = set(int(e) for e in sample_frame["engine"].unique())
        stored = set(int(e) for e in folds["engine"].unique())
        if expected != stored:
            raise ValueError(f"stored folds engines {stored} != sample engines {expected}")
        return folds
    return engine_holdout_folds(sample_frame, path)


def iter_fold_masks(sample_frame: pd.DataFrame, folds: pd.DataFrame):
    """Yield (fold_id, held_engine, train_mask, valid_mask) from stored folds."""
    for fold_id in sorted(folds["fold"].unique()):
        part = folds[folds["fold"] == fold_id]
        valid_engines = set(int(e) for e in part.loc[part["role"] == "valid", "engine"])
        if len(valid_engines) != 1:
            raise ValueError(f"fold {fold_id} must hold out exactly one engine")
        held = next(iter(valid_engines))
        va_mask = sample_frame["engine"] == held
        tr_mask = ~va_mask
        yield int(fold_id), held, tr_mask, va_mask


def split_xy(frame: pd.DataFrame, feature_cols: list[str], target: str):
    x = frame[feature_cols]
    y = frame[target]
    return x, y
