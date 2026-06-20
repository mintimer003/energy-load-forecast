"""Бейзлайн и обвязка для оценки.

Сезонный наивный прогноз (тот же час неделю назад) — эталон, который обязана
обойти любая модель. Считаем его на бэктесте со скользящим началом; MASE около
единицы подтверждает, что наив — это уровень безубыточности."""
from __future__ import annotations
import pandas as pd
from utils import (TARGET, SEASONAL_LAG, load_features, rolling_origin,
                   mae, rmse, mape, mase, get_logger)

log = get_logger(__name__)


def evaluate_seasonal_naive(features: pd.DataFrame) -> pd.DataFrame:
    y = features[TARGET]
    naive = y.shift(SEASONAL_LAG)
    records = []
    for fold, (train_idx, test_idx) in enumerate(rolling_origin(features.index), 1):
        actual, forecast = y.loc[test_idx], naive.loc[test_idx]
        records.append({
            "fold": fold,
            "MAE": mae(actual, forecast),
            "RMSE": rmse(actual, forecast),
            "MAPE": mape(actual, forecast),
            "MASE": mase(actual, forecast, y.loc[train_idx]),
        })
    return pd.DataFrame(records)


def main() -> None:
    scores = evaluate_seasonal_naive(load_features())
    for _, r in scores.iterrows():
        log.info("fold %d | MAE %.0f | MAPE %.2f%% | MASE %.2f", r.fold, r.MAE, r.MAPE, r.MASE)
    mean = scores.mean(numeric_only=True)
    log.info("baseline mean | MAE %.0f MW | MAPE %.2f%% | MASE %.3f", mean.MAE, mean.MAPE, mean.MASE)


if __name__ == "__main__":
    main()
