"""Load PHM challenge train, validation, and test files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "training_data" / "training_data.csv"
TEST_DIR = ROOT / "test_data" / "test"
VAL_DIR = ROOT / "validation_data" / "val"

CYCLE_COL_TRAIN = "Cycles_Since_New"
CYCLE_COL_INFER = "Cycles"

PRIMARY_SENSORS = [
    "Sensed_Altitude",
    "Sensed_Mach",
    "Sensed_Pamb",
    "Sensed_Pt2",
    "Sensed_TAT",
    "Sensed_WFuel",
    "Sensed_VAFN",
    "Sensed_VBV",
    "Sensed_Fan_Speed",
    "Sensed_Core_Speed",
    "Sensed_T25",
    "Sensed_T3",
    "Sensed_Ps3",
    "Sensed_T45",
]

OPTIONAL_SENSORS = ["Sensed_P25", "Sensed_T5"]
TARGETS = ["Cycles_to_HPT_SV", "Cycles_to_HPC_SV", "Cycles_to_WW"]
UNCERTAIN_MAINT_META = ["Cumulative_WWs", "Cumulative_HPC_SVs", "Cumulative_HPT_SVs"]


def project_root() -> Path:
    return ROOT


def load_training() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    df = df.rename(columns={CYCLE_COL_TRAIN: "Cycles"})
    return df


def _sorted_split_files(directory: Path, prefix: str) -> list[Path]:
    files = list(directory.glob(f"{prefix}_*.csv"))
    return sorted(files, key=lambda p: int(p.stem.split("_")[1]))


def load_split_files(kind: str) -> list[tuple[str, pd.DataFrame]]:
    if kind == "test":
        paths = _sorted_split_files(TEST_DIR, "test")
    elif kind == "val":
        paths = _sorted_split_files(VAL_DIR, "val")
    else:
        raise ValueError(kind)
    out = []
    for path in paths:
        df = pd.read_csv(path)
        out.append((path.stem, df))
    return out
