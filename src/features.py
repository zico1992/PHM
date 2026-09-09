"""Leakage-safe window features for 1500-cycle prediction grains."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data_loader import PRIMARY_SENSORS, TARGETS

KEY_SENSORS = [
    "Sensed_T45",
    "Sensed_T3",
    "Sensed_T25",
    "Sensed_Ps3",
    "Sensed_Core_Speed",
    "Sensed_Fan_Speed",
    "Sensed_WFuel",
    "Sensed_VAFN",
]
OP_COLS = ["Sensed_Altitude", "Sensed_Mach", "Sensed_TAT"]
USE_SNAPSHOTS = [1, 2, 4, 6, 7, 8]
RESIDUAL_SENSORS = ["Sensed_T45", "Sensed_T3", "Sensed_Core_Speed", "Sensed_Ps3"]


def pivot_cycle_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Cycles", "Snapshot"] + [c for c in PRIMARY_SENSORS if c in df.columns]
    sub = df[keep].copy()
    wide = sub.pivot_table(index="Cycles", columns="Snapshot", values=[c for c in keep if c not in ("Cycles", "Snapshot")], aggfunc="mean")
    wide.columns = [f"{sensor}_s{int(snap)}" for sensor, snap in wide.columns]
    return wide.sort_index()


def _col(sensor: str, snap: int) -> str:
    return f"{sensor}_s{snap}"


class ResidualModel:
    """Operating-condition residual model fitted on training early-life data only."""

    def __init__(self):
        self.models: dict[tuple[str, int], LinearRegression] = {}
        self.fitted = False

    def fit(self, train_df: pd.DataFrame, max_cycle: int = 200) -> "ResidualModel":
        early = train_df[train_df["Cycles"] <= max_cycle]
        for sensor in RESIDUAL_SENSORS:
            for snap in USE_SNAPSHOTS:
                part = early[early["Snapshot"] == snap][OP_COLS + [sensor]].dropna()
                if len(part) < 20:
                    continue
                model = LinearRegression()
                model.fit(part[OP_COLS].values, part[sensor].values)
                self.models[(sensor, snap)] = model
        self.fitted = True
        return self

    def residuals_last_row(self, window_df: pd.DataFrame) -> dict[str, float]:
        last_cycle = window_df["Cycles"].max()
        last = window_df[window_df["Cycles"] == last_cycle]
        out = {}
        for sensor in RESIDUAL_SENSORS:
            for snap in USE_SNAPSHOTS:
                model = self.models.get((sensor, snap))
                row = last[last["Snapshot"] == snap]
                if model is None or row.empty:
                    out[f"resid_{sensor}_s{snap}"] = np.nan
                    continue
                x = row[OP_COLS].iloc[0].values.reshape(1, -1)
                if np.any(pd.isna(x)):
                    out[f"resid_{sensor}_s{snap}"] = np.nan
                    continue
                y = row[sensor].iloc[0]
                out[f"resid_{sensor}_s{snap}"] = float(y - model.predict(x)[0]) if pd.notna(y) else np.nan
        return out


def _nanmean(arr: np.ndarray) -> float:
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def _nanstd(arr: np.ndarray) -> float:
    return float(np.nanstd(arr)) if np.isfinite(arr).sum() > 1 else np.nan


def _series(wide: pd.DataFrame, name: str) -> np.ndarray:
    if name not in wide.columns:
        return np.full(len(wide), np.nan)
    return wide[name].to_numpy(dtype=float)


def window_feature_row(wide: pd.DataFrame) -> dict[str, float]:
    """Causal features from a window of cycle-indexed snapshot-pivoted sensors."""
    n = len(wide)
    feats: dict[str, float] = {
        "n_cycles": float(n),
        "span_cycles": float(wide.index.max() - wide.index.min()) if n else np.nan,
        "last_cycle_in_window": float(wide.index.max()) if n else np.nan,
    }
    if n == 0:
        return feats

    baseline_n = min(5, n)
    for sensor in KEY_SENSORS + OP_COLS:
        for snap in USE_SNAPSHOTS:
            name = _col(sensor, snap)
            s = _series(wide, name)
            last = s[-1]
            base = _nanmean(s[:baseline_n])
            feats[f"{name}_last"] = last
            feats[f"{name}_base"] = base
            feats[f"{name}_dlt"] = last - base if np.isfinite(last) and np.isfinite(base) else np.nan
            feats[f"{name}_rdlt"] = (
                (last - base) / base if np.isfinite(last) and np.isfinite(base) and abs(base) > 1e-6 else np.nan
            )
            feats[f"{name}_diff1"] = last - s[-2] if n >= 2 else np.nan
            for w in (10, 25):
                sl = s[-w:] if n else s
                feats[f"{name}_rmean{w}"] = _nanmean(sl)
                feats[f"{name}_rstd{w}"] = _nanstd(sl)
            if n >= 10:
                y = s[-25:] if n >= 25 else s
                x = np.arange(len(y), dtype=float)
                mask = np.isfinite(y)
                if mask.sum() >= 5:
                    feats[f"{name}_slope"] = float(np.polyfit(x[mask], y[mask], 1)[0])
                else:
                    feats[f"{name}_slope"] = np.nan
            else:
                feats[f"{name}_slope"] = np.nan

    for snap in USE_SNAPSHOTS:
        t45 = _series(wide, _col("Sensed_T45", snap))
        t3 = _series(wide, _col("Sensed_T3", snap))
        feats[f"t45_t3_s{snap}"] = t45[-1] - t3[-1] if n else np.nan
        core = _series(wide, _col("Sensed_Core_Speed", snap))
        fan = _series(wide, _col("Sensed_Fan_Speed", snap))
        feats[f"core_fan_s{snap}"] = core[-1] - fan[-1] if n else np.nan

    t45s4 = _series(wide, _col("Sensed_T45", 4))
    t45s1 = _series(wide, _col("Sensed_T45", 1))
    feats["t45_s4_minus_s1"] = t45s4[-1] - t45s1[-1] if n else np.nan

    # Approximate cycles since last T45 drop (possible WW) using snapshot 4 if present else 1.
    probe = t45s4 if np.isfinite(t45s4).any() else t45s1
    if np.isfinite(probe).sum() >= 8:
        d = np.diff(probe)
        drop_idx = np.where(np.isfinite(d) & (d < np.nanpercentile(d[np.isfinite(d)], 5)) & (d < -5))[0]
        if len(drop_idx):
            last_drop = int(drop_idx[-1] + 1)
            feats["cycles_since_t45_drop"] = float(n - 1 - last_drop)
        else:
            feats["cycles_since_t45_drop"] = float(n - 1)
    else:
        feats["cycles_since_t45_drop"] = np.nan

    # Event-proximity / degradation-spike composites (causal within-window).
    t45_dlt = feats.get("Sensed_T45_s4_dlt", np.nan)
    t45_slope = feats.get("Sensed_T45_s4_slope", np.nan)
    core_dlt = feats.get("Sensed_Core_Speed_s8_dlt", np.nan)
    t25_dlt = feats.get("Sensed_T25_s2_dlt", np.nan)
    feats["degrade_t45_heat_rise"] = float(t45_dlt) if np.isfinite(t45_dlt) else np.nan
    feats["degrade_t45_slope"] = float(t45_slope) if np.isfinite(t45_slope) else np.nan
    feats["degrade_core_drop"] = float(-core_dlt) if np.isfinite(core_dlt) else np.nan
    feats["degrade_t25_drop"] = float(-t25_dlt) if np.isfinite(t25_dlt) else np.nan
    spike_parts = [
        feats["degrade_t45_heat_rise"],
        feats["degrade_core_drop"],
        feats["degrade_t25_drop"],
    ]
    finite = [v for v in spike_parts if np.isfinite(v)]
    feats["degrade_spike_score"] = float(np.mean(finite)) if finite else np.nan
    return feats


def extract_window_sample(window_df: pd.DataFrame, residual_model: ResidualModel | None = None) -> dict:
    wide = pivot_cycle_table(window_df)
    feats = window_feature_row(wide)
    if residual_model is not None and residual_model.fitted:
        feats.update(residual_model.residuals_last_row(window_df))
        resid_t45 = [feats.get(f"resid_Sensed_T45_s{s}", np.nan) for s in (1, 4, 7)]
        finite = [v for v in resid_t45 if np.isfinite(v)]
        feats["degrade_resid_t45_mean"] = float(np.mean(finite)) if finite else np.nan
    return feats


def iter_train_windows(train_df: pd.DataFrame, window_points: int = 151, step: int = 2):
    """Yield (engine, end_cycle, window_df, labels) matching test 1500-cycle windows."""
    for esn, g in train_df.groupby("ESN"):
        cycles = np.sort(g["Cycles"].unique())
        if len(cycles) < window_points:
            continue
        for end_idx in range(window_points - 1, len(cycles), step):
            start_idx = end_idx - (window_points - 1)
            cyc = cycles[start_idx : end_idx + 1]
            w = g[g["Cycles"].isin(cyc)]
            end_cycle = int(cycles[end_idx])
            lab_rows = g[g["Cycles"] == end_cycle]
            labels = {t: float(lab_rows[t].iloc[0]) for t in TARGETS}
            yield int(esn), end_cycle, w, labels
