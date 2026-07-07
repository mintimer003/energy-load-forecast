"""Анти-утечка: признаки будущего часа не должны зависеть от значений позже (now - HORIZON)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
from features import HORIZON, LAGS, build_features


def _series(n_hours=24 * 40, seed=0):
    idx = pd.date_range("2018-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(seed)
    return pd.Series(1000 + rng.normal(0, 50, n_hours), index=idx)


def test_all_lags_at_least_horizon():
    assert min(LAGS) >= HORIZON


def test_features_do_not_see_last_horizon_hours():
    """Меняем последние HORIZON часов ряда — признаки последней строки меняться не должны."""
    s = _series()
    feats_before = build_features(s).iloc[-1].drop("load")
    s_changed = s.copy()
    s_changed.iloc[-HORIZON:] += 10_000            # «будущее» относительно момента прогноза
    feats_after = build_features(s_changed).iloc[-1].drop("load")
    pd.testing.assert_series_equal(feats_before, feats_after)


def test_lag_values_are_correct():
    s = _series()
    feats = build_features(s)
    for lag in LAGS:
        assert np.isclose(feats["lag_%d" % lag].iloc[-1], s.iloc[-1 - lag])
