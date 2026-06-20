"""Построение признаков из чистого ряда.

Каждый лаг и скользящее окно сдвинуты минимум на горизонт прогноза (24 ч):
для цели на сутки вперёд на момент прогноза известны только данные старше 24 ч,
поэтому более короткие лаги дали бы утечку. Календарь — исключение, он известен
наперёд для любой метки времени."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from utils import TARGET, HORIZON, get_logger

log = get_logger(__name__)
CLEAN = Path("aep_clean.csv")
FEATURES = Path("aep_features.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET]
    index = df.index
    out = pd.DataFrame({TARGET: y}, index=index)

    out["hour"] = index.hour
    out["dayofweek"] = index.dayofweek
    out["month"] = index.month
    out["is_weekend"] = (index.dayofweek >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * index.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * index.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * index.dayofyear / 365.25)

    for lag in (24, 48, 168):
        out[f"lag_{lag}"] = y.shift(lag)

    recent = y.shift(HORIZON)                              # окно заканчивается за горизонт до t
    out["roll_mean_24"] = recent.rolling(24).mean()
    out["roll_mean_168"] = recent.rolling(168).mean()
    out["roll_std_24"] = recent.rolling(24).std()
    return out


def main() -> None:
    df = pd.read_csv(CLEAN, parse_dates=["Datetime"], index_col="Datetime")
    features = build_features(df).dropna()
    features.to_csv(FEATURES)
    log.info("features: %d rows, %d predictors", len(features), features.shape[1] - 1)


if __name__ == "__main__":
    main()
