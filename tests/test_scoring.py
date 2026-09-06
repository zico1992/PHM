from src.scoring import score_submitted_result, time_weighted_error
import numpy as np
import pandas as pd


def test_late_worse_than_early():
    y = np.array([100.0, 100.0])
    late = time_weighted_error(y, y + 10)
    early = time_weighted_error(y, y - 10)
    assert late.mean() > early.mean()


def test_submission_score_perfect_is_zero():
    df = pd.DataFrame(
        {
            "Cycles_to_WW": [100, 200],
            "Cycles_to_HPC_SV": [1000, 2000],
            "Cycles_to_HPT_SV": [500, 800],
        }
    )
    score, parts = score_submitted_result(df, df)
    assert score == 0
    assert parts["score_WW"] == 0
