import numpy as np
import pandas as pd

from src.hpc_gate import HpcGateConfig, apply_hpc_gate, apply_scale_gate, fit_gate_params


def test_scale_gate_only_masks_high_proba():
    pred = np.array([2000.0, 2000.0, 2000.0])
    proba = np.array([0.0, 0.5, 0.9])
    out = apply_scale_gate(pred, proba, threshold=0.4, scale_factor=0.5)
    assert out[0] == 2000.0
    assert out[1] == 1000.0
    assert out[2] == 1000.0


def test_fit_gate_prefers_pulling_down_late_near_event_misses():
    # Mostly good preds, plus late near-event bombs that a gate can fix.
    rng = np.random.default_rng(0)
    y = np.concatenate([np.full(80, 3000.0), np.full(20, 100.0)])
    pred = np.concatenate([y[:80] - 200.0, np.full(20, 2500.0)])
    proba = np.concatenate([np.full(80, 0.02), np.full(20, 0.8)])
    beta = 2 / float(y.max())
    cfg = fit_gate_params(y, pred, proba, beta)
    gated = apply_hpc_gate(pred, proba, cfg)
    assert gated[80:].mean() < pred[80:].mean()
    assert cfg.scale_factor < 1.0


def test_apply_hpc_gate_uses_post_offset():
    cfg = HpcGateConfig(
        feature_cols=["x"],
        proba_threshold=0.5,
        scale_factor=0.5,
        post_offset=-100.0,
    )
    pred = np.array([1000.0, 1000.0])
    proba = np.array([0.0, 1.0])
    out = apply_hpc_gate(pred, proba, cfg)
    assert np.isclose(out[0], 900.0)
    assert np.isclose(out[1], 400.0)
