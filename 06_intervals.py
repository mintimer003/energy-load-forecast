"""Квантильный прогноз для интервалов неопределённости.

Одно число скрывает риск. Обучаем CatBoost на квантилях 10/50/90 и получаем
коридор P10–P90; верхняя граница P90 — резерв для расчёта мощности/давления.
Проверяем калибровку (~80% попаданий) и рисуем коридор против факта."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from utils import TARGET, load_features, get_logger

log = get_logger(__name__)
QUANTILES = (0.1, 0.5, 0.9)
TEST_DAYS = 30
FIGURE = Path("forecast_intervals.png")


def main() -> None:
    features = load_features()
    split = len(features) - TEST_DAYS * 24
    train_idx, test_idx = features.index[:split], features.index[split:]
    X_train, y_train = features.loc[train_idx].drop(columns=[TARGET]), features.loc[train_idx, TARGET]
    X_test, y_test = features.loc[test_idx].drop(columns=[TARGET]), features.loc[test_idx, TARGET]

    models = {q: CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=400,
                                   learning_rate=0.1, depth=6, random_seed=42, verbose=False)
                 .fit(X_train, y_train) for q in QUANTILES}
    forecast = {q: pd.Series(m.predict(X_test), index=test_idx) for q, m in models.items()}
    lower, median, upper = forecast[0.1], forecast[0.5], forecast[0.9]

    coverage = ((y_test >= lower) & (y_test <= upper)).mean() * 100
    log.info("P10-P90 coverage %.1f%% (target ~80) | mean width %.0f MW", coverage, (upper - lower).mean())

    window = test_idx[-168:]
    plt.figure(figsize=(13, 5))
    plt.fill_between(window, lower[window], upper[window], alpha=0.25, label="P10-P90")
    plt.plot(window, median[window], label="forecast P50")
    plt.plot(window, y_test[window], color="black", lw=1, label="actual")
    plt.title("Day-ahead forecast with prediction interval (last 7 days)")
    plt.ylabel("MW"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURE, dpi=100)
    log.info("saved %s", FIGURE)


if __name__ == "__main__":
    main()
