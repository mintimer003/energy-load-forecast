"""Многоузловая модель: один CatBoost на несколько узлов сети.

Каждый регион (AEP, COMED, DAYTON, DEOK) — узел сети (аналог ГРПШ) со своим
профилем и масштабом. Обучаем ОДНУ модель с категориальным признаком `region`:
она ловит и общие закономерности, и специфику узла.

Признаки строятся ВНУТРИ каждого узла (лаги и окна не пересекают границу между
узлами). Метрики считаем по каждому узлу отдельно — узлы разного масштаба, общий
MAE смешал бы гигаватты с мегаваттами.

ЗАПУСК: python 07_multinode.py        (нужны *_hourly.csv узлов рядом)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from catboost import CatBoostRegressor
from utils import SEASONAL_LAG, HORIZON, mae, mape, get_logger

log = get_logger(__name__)

REGIONS = {"AEP": "AEP_MW", "COMED": "COMED_MW", "DAYTON": "DAYTON_MW", "DEOK": "DEOK_MW"}
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2004-01-01", end="2019-01-01")


def clean_series(path: Path, col: str) -> pd.Series:
    s = (pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime")
           .groupby("Datetime")[col].mean())
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s = s.fillna(s.shift(SEASONAL_LAG)).interpolate(method="time", limit_direction="both")
    s.name = "load"
    return s


def build_features(s: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"load": s})
    ix = s.index
    out["hour"] = ix.hour
    out["dayofweek"] = ix.dayofweek
    out["month"] = ix.month
    out["is_weekend"] = (ix.dayofweek >= 5).astype(int)
    out["is_holiday"] = ix.normalize().isin(HOLIDAYS).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * ix.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ix.hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * ix.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * ix.dayofyear / 365.25)
    for lag in (24, 48, 168):
        out[f"lag_{lag}"] = s.shift(lag)
    recent = s.shift(HORIZON)                          # окно заканчивается за горизонт до t
    out["roll_mean_24"] = recent.rolling(24).mean()
    out["roll_mean_168"] = recent.rolling(168).mean()
    out["roll_std_24"] = recent.rolling(24).std()
    return out


def load_all() -> tuple[pd.DataFrame, dict]:
    series, frames = {}, []
    for region, col in REGIONS.items():
        s = clean_series(Path(f"{region}_hourly.csv"), col)
        series[region] = s
        f = build_features(s)
        f["region"] = region                           # признак узла (категориальный)
        frames.append(f)
    data = pd.concat(frames).sort_index().dropna()
    return data, series


def time_folds(end, n_folds=6, test_days=30):
    H = pd.Timedelta(days=test_days)
    return [(end - (k + 1) * H, end - k * H) for k in range(n_folds)][::-1]


def main() -> None:
    data, series = load_all()
    log.info("nodes: %d | rows: %d | features: %d", data.region.nunique(), len(data), data.shape[1] - 2)

    feature_cols = [c for c in data.columns if c != "load"]    # включает категориальный region
    folds = time_folds(data.index.max() + pd.Timedelta(hours=1))

    stats = {r: {"b_mape": [], "m_mape": [], "b_mase": [], "m_mase": []} for r in REGIONS}
    for ts, te in folds:
        train = data[data.index < ts]
        test = data[(data.index >= ts) & (data.index < te)].copy()

        model = CatBoostRegressor(iterations=400, learning_rate=0.1, depth=6,
                                  cat_features=["region"], random_seed=42, verbose=False)
        model.fit(train[feature_cols], train["load"])
        test["pred"] = model.predict(test[feature_cols])

        for r in REGIONS:
            sub = test[test.region == r]
            if sub.empty:
                continue
            actual, base, pred = sub["load"], sub["lag_168"], sub["pred"]
            hist = series[r][series[r].index < ts].values
            denom = np.mean(np.abs(hist[SEASONAL_LAG:] - hist[:-SEASONAL_LAG]))
            stats[r]["b_mape"].append(mape(actual, base)); stats[r]["m_mape"].append(mape(actual, pred))
            stats[r]["b_mase"].append(mae(actual, base) / denom); stats[r]["m_mase"].append(mae(actual, pred) / denom)
        log.info("fold %s..%s trained", ts.date(), te.date())

    log.info("--- результаты по узлам (среднее по окнам): бейзлайн -> общая модель ---")
    for r in REGIONS:
        d = stats[r]
        log.info("  %-7s MAPE %5.2f%% -> %5.2f%% | MASE %.2f -> %.2f",
                 r, np.mean(d["b_mape"]), np.mean(d["m_mape"]), np.mean(d["b_mase"]), np.mean(d["m_mase"]))


if __name__ == "__main__":
    main()
