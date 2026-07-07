"""Сквозной мини-тест: фичи строятся, квантильная модель обучается и упорядочена."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
from features import build_features


def test_quantile_train_and_order():
    from catboost import CatBoostRegressor

    idx = pd.date_range("2018-01-01", periods=24 * 30, freq="h")
    rng = np.random.default_rng(2)
    s = pd.Series(1000 + 100 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 20, len(idx)), index=idx)
    feats = build_features(s).dropna()
    X, y = feats.drop(columns=["load"]), feats["load"]

    preds = {}
    for q in (0.10, 0.50, 0.90):
        m = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=50,
                              random_seed=42, verbose=False)
        m.fit(X, y)
        preds[q] = m.predict(X.tail(24))
    assert (preds[0.10] <= preds[0.50] + 1e-6).mean() > 0.9
    assert (preds[0.50] <= preds[0.90] + 1e-6).mean() > 0.9
