import numpy as np

from src.ensemble import blend, fit_blend_and_offset
from src.scoring import score_target


def test_blend_endpoints():
    a = np.array([10.0, 20.0])
    b = np.array([0.0, 0.0])
    assert np.allclose(blend(a, b, 1.0), a)
    assert np.allclose(blend(a, b, 0.0), b)


def test_ensemble_picks_better_member_by_official_score():
    y = np.linspace(100, 400, 40)
    good = y - 20
    bad = y + 250
    beta = 1 / float(y.max())
    fitted = fit_blend_and_offset(y, good, bad, "Cycles_to_WW", beta, step=0.25)
    assert fitted["lgb_weight"] >= 0.75
    mixed_score = score_target(y, fitted["prediction"], 0.01, beta)
    bad_score = score_target(y, bad, 0.01, beta)
    assert mixed_score < bad_score
