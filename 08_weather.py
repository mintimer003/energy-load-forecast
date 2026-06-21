"""Добавление погоды (температуры) как внешнего драйвера спроса.

Температуру тянем из открытого архива Open-Meteo (без файла и регистрации),
кусками по годам, и приклеиваем к ряду AEP по метке времени.

Важно про утечку: для day-ahead будущая температура ДОПУСТИМА — прогноз погоды
на завтра существует, поэтому температуру целевого часа брать можно (как
календарь). На истории используем фактическую температуру как замену прогноза.
Это отличие от лагов нагрузки: своего будущего спроса у нас нет, а погодный
прогноз есть.

Координаты — Колумбус, Огайо (зона AEP).

ЗАПУСК: python 08_weather.py    (нужен aep_features.csv рядом; нужен интернет)
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from utils import TARGET, rolling_origin, mape, mase, get_logger

log = get_logger(__name__)
LAT, LON, TZ = 39.9612, -82.9988, "America/New_York"      # Колумбус, Огайо (AEP)
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_temperature(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frames = []
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(f"{year}-01-01"))
        e = min(end, pd.Timestamp(f"{year}-12-31"))
        query = urllib.parse.urlencode({
            "latitude": LAT, "longitude": LON,
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m", "timezone": TZ,
        })
        with urllib.request.urlopen(f"{ARCHIVE}?{query}", timeout=60) as resp:
            block = json.load(resp)["hourly"]
        frames.append(pd.DataFrame({"Datetime": pd.to_datetime(block["time"]),
                                    "temp": block["temperature_2m"]}))
        log.info("temperature fetched: %d", year)
    return pd.concat(frames).drop_duplicates("Datetime").set_index("Datetime")["temp"]


def backtest(features: pd.DataFrame):
    rows, model = [], None
    for tr, te in rolling_origin(features.index):
        X_tr, y_tr = features.loc[tr].drop(columns=[TARGET]), features.loc[tr, TARGET]
        X_te, y_te = features.loc[te].drop(columns=[TARGET]), features.loc[te, TARGET]
        model = CatBoostRegressor(iterations=400, learning_rate=0.1, depth=6,
                                  random_seed=42, verbose=False)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"MAPE": mape(y_te, pred), "MASE": mase(y_te, pred, y_tr)})
    return pd.DataFrame(rows).mean(), model


def main() -> None:
    features = pd.read_csv("aep_features.csv", parse_dates=["Datetime"], index_col="Datetime")

    temp = fetch_temperature(features.index.min(), features.index.max())
    temp = temp.reindex(features.index).interpolate(method="time", limit_direction="both")

    base, _ = backtest(features)
    augmented = features.copy()
    augmented["temp"] = temp
    with_weather, model = backtest(augmented)

    log.info("без погоды: MAPE %.2f%% | MASE %.3f", base.MAPE, base.MASE)
    log.info("с погодой : MAPE %.2f%% | MASE %.3f", with_weather.MAPE, with_weather.MASE)
    importance = pd.Series(model.get_feature_importance(), index=model.feature_names_)
    log.info("важность температуры (вклад, %%): %.1f", importance.get("temp", float("nan")))

    augmented.to_csv("aep_features_weather.csv")
    log.info("сохранено: aep_features_weather.csv")


if __name__ == "__main__":
    main()
