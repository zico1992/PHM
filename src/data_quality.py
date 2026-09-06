"""Data profiling for the PHM challenge tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loader import PRIMARY_SENSORS, TARGETS, load_split_files, load_training


def profile_training(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "engines": sorted(df["ESN"].unique().tolist()),
        "n_engines": int(df["ESN"].nunique()),
        "snapshots": sorted(df["Snapshot"].dropna().unique().tolist()),
        "cycle_min": int(df["Cycles"].min()),
        "cycle_max": int(df["Cycles"].max()),
        "n_cycles_per_engine": df.groupby("ESN")["Cycles"].nunique().to_dict(),
        "missing": df.isna().sum().to_dict(),
        "target_max": {t: float(df[t].max()) for t in TARGETS},
        "target_min": {t: float(df[t].min()) for t in TARGETS},
    }


def profile_file_split(kind: str) -> pd.DataFrame:
    rows = []
    for name, df in load_split_files(kind):
        rows.append(
            {
                "file": name,
                "esn": int(df["ESN"].iloc[0]),
                "n_rows": int(len(df)),
                "n_cycles": int(df["Cycles"].nunique()),
                "cycle_min": int(df["Cycles"].min()),
                "cycle_max": int(df["Cycles"].max()),
                "n_snapshots": int(df["Snapshot"].nunique()),
                "missing": int(df.isna().sum().sum()),
                "has_optional_p25": "Sensed_P25" in df.columns,
                "has_targets": all(t in df.columns for t in TARGETS),
                "primary_sensors": all(c in df.columns for c in PRIMARY_SENSORS),
            }
        )
    return pd.DataFrame(rows)


def write_profile_reports(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = load_training()
    train_profile = profile_training(train)
    pd.Series(train_profile).to_json(out_dir / "train_profile.json", indent=2)
    for kind in ("test", "val"):
        profile_file_split(kind).to_csv(out_dir / f"{kind}_file_profile.csv", index=False)
