"""Общие утилиты проекта: константы, логирование, метрики ошибки и бэктест
со скользящим началом. Импортируется остальными модулями."""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "AEP_MW"
HORIZON = 24            # прогноз на сутки вперёд
SEASONAL_LAG = 168      # недельная сезонность: тот же час, тот же день недели
FEATURES_CSV = Path("aep_features.csv")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mape(actual, forecast) -> float:
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100)


def mase(actual, forecast, y_train, m: int = SEASONAL_LAG) -> float:
    history = y_train.values
    scale = np.mean(np.abs(history[m:] - history[:-m]))   # ошибка сезонного наива на обучении
    return mae(actual, forecast) / scale


def rolling_origin(index, n_folds: int = 6, test_days: int = 30):
    test_size = test_days * 24
    n = len(index)
    folds = []
    for k in range(n_folds):
        test_end = n - k * test_size
        test_start = test_end - test_size
        if test_start - SEASONAL_LAG <= 0:
            break
        folds.append((index[:test_start], index[test_start:test_end]))
    return list(reversed(folds))


def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV, parse_dates=["Datetime"], index_col="Datetime")
