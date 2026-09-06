"""Point-in-time leakage audit for candidate features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

AUDIT_ROWS = [
    {
        "feature": "snapshot_pivoted_sensors",
        "family": "sensor",
        "available_at_t": True,
        "leakage_risk": "low",
        "decision": "keep",
        "reason": "Current-cycle snapshot values only.",
    },
    {
        "feature": "trailing_rolling_stats",
        "family": "temporal",
        "available_at_t": True,
        "leakage_risk": "low",
        "decision": "keep",
        "reason": "Right-aligned windows using cycle <= t within the observed window.",
    },
    {
        "feature": "window_early_baseline_delta",
        "family": "degradation",
        "available_at_t": True,
        "leakage_risk": "low",
        "decision": "keep",
        "reason": "Baseline uses only the first cycles of the current window, not future history.",
    },
    {
        "feature": "centered_rolling",
        "family": "temporal",
        "available_at_t": False,
        "leakage_risk": "high",
        "decision": "exclude",
        "reason": "Uses future cycles.",
    },
    {
        "feature": "full_engine_normalization",
        "family": "preprocessing",
        "available_at_t": False,
        "leakage_risk": "high",
        "decision": "exclude",
        "reason": "Would use complete future history of an engine.",
    },
    {
        "feature": "Cycles_Since_New_absolute",
        "family": "meta",
        "available_at_t": False,
        "leakage_risk": "high",
        "decision": "exclude",
        "reason": "Test/val files reset Cycles to a 0-1500 window; absolute age is not observed.",
    },
    {
        "feature": "Cumulative_WWs_HPC_HPT",
        "family": "maintenance_meta",
        "available_at_t": False,
        "leakage_risk": "high",
        "decision": "exclude",
        "reason": "Timing semantics are uncertain and columns are absent from test/val.",
    },
    {
        "feature": "Sensed_P25_T5",
        "family": "sensor",
        "available_at_t": False,
        "leakage_risk": "medium",
        "decision": "exclude",
        "reason": "Optional sensors are not present in the provided test/val files.",
    },
    {
        "feature": "target_derived_features",
        "family": "label",
        "available_at_t": False,
        "leakage_risk": "high",
        "decision": "exclude",
        "reason": "Would leak remaining-cycle labels.",
    },
    {
        "feature": "op_condition_residual_model",
        "family": "health_index",
        "available_at_t": True,
        "leakage_risk": "low",
        "decision": "keep",
        "reason": "Linear residual model fitted on training early-life rows only.",
    },
]


def write_audit(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(AUDIT_ROWS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
