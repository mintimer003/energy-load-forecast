"""Метрики: MASE=1 у сезонного наива, MAPE считается верно."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
from features import SEASONAL_LAG
from train import mape, mase


def test_mape_simple():
    assert np.isclose(mape([100, 200], [110, 180]), 10.0)


def test_mase_of_naive_is_one():
    rng = np.random.default_rng(1)
    hist = 1000 + rng.normal(0, 30, SEASONAL_LAG * 6)
    y_true = hist[SEASONAL_LAG:]
    y_naive = hist[:-SEASONAL_LAG]
    assert np.isclose(mase(y_true, y_naive, hist), 1.0, atol=1e-9)
