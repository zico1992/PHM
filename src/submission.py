"""Build submission CSVs with exact identifiers and column order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["file", "Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"]


def validate_submission(df: pd.DataFrame, expected_files: list[str]) -> None:
    if list(df.columns) != REQUIRED_COLS:
        raise ValueError(f"columns must be {REQUIRED_COLS}, got {list(df.columns)}")
    if len(df) != len(expected_files):
        raise ValueError(f"expected {len(expected_files)} rows, got {len(df)}")
    if df["file"].tolist() != expected_files:
        raise ValueError("file identifiers are missing, duplicated, or out of order")
    if df["file"].duplicated().any():
        raise ValueError("duplicate file identifiers")
    vals = df[REQUIRED_COLS[1:]]
    if vals.isna().any().any():
        raise ValueError("NaNs in predictions")
    if np.isinf(vals.to_numpy(dtype=float)).any():
        raise ValueError("infinities in predictions")
    if (vals.to_numpy(dtype=float) < 0).any():
        raise ValueError("negative cycle predictions")


def write_submission(path: Path, rows: pd.DataFrame, expected_files: list[str]) -> pd.DataFrame:
    out = rows[REQUIRED_COLS].copy()
    for c in REQUIRED_COLS[1:]:
        out[c] = np.clip(out[c].astype(float), 0, None)
    validate_submission(out, expected_files)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
