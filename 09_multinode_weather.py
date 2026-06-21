"""Многоузловая модель С ПОГОДОЙ: одна модель на 4 зоны, у каждой своя температура.

У зон разный климат (Чикаго / Огайо / Цинциннати), поэтому температуру тянем по
СОБСТВЕННЫМ координатам каждой зоны (Open-Meteo, кусками по годам) и приклеиваем
к её ряду. Закачка кэшируется в weather_{ZONE}.csv — повторный запуск не качает
заново.

Сравниваем по каждому узлу: бейзлайн -> модель без погоды -> модель с погодой,
чтобы увидеть вклад температуры отдельно на каждой зоне.

Для day-ahead будущая температура допустима (прогноз погоды существует) — не
утечка, как календарь.

ЗАПУСК: python 09_multinode_weather.py   (нужны *_hourly.csv + интернет; долгий)
"""
from __future__ import annotations
import json
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from catboost import CatBoostRegressor
from utils import SEASONAL_LAG, HORIZON, mae, mape, get_logger

log = get_logger(__name__)

# зона: (колонка, широта, долгота, город)
REGIONS = {
    "AEP":    ("AEP_MW",    39.9612, -82.9988, "Columbus"),
    "COMED":  ("COMED_MW",  41.8781, -87.6298, "Chicago"),
    "DAYTON": ("DAYTON_MW", 39.7589, -84.1916, "Dayton"),
    "DEOK":   ("DEOK_MW",   39.1031, -84.5120, "Cincinnati"),
}
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2004-01-01", end="2019-01-01")
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/New_York"


def clean_series(path: Path, col: str) -> pd.Series:
    s = (pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime").groupby("Datetime")[col].mean())
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s = s.fillna(s.shift(SEASONAL_LAG)).interpolate(method="time", limit_direction="both")
    s.name = "load"
    return s


def fetch_temperature(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    frames = []
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(f"{year}-01-01")); e = min(end, pd.Timestamp(f"{year}-12-31"))
        query = urllib.parse.urlencode({"latitude": lat, "longitude": lon,
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m", "timezone": TZ})
        with urllib.request.urlopen(f"{ARCHIVE}?{query}", timeout=60) as resp:
            block = json.load(resp)["hourly"]
        frames.append(pd.DataFrame({"Datetime": pd.to_datetime(block["time"]), "temp": block["temperature_2m"]}))
    return pd.concat(frames).drop_duplicates("Datetime").set_index("Datetime")["temp"]


def temperature_for(region: str, lat: float, lon: float, index: pd.DatetimeIndex) -> pd.Series:
    cache = Path(f"weather_{region}.csv")
    if cache.exists():
        temp = pd.read_csv(cache, parse_dates=["Datetime"], index_col="Datetime")["temp"]
        log.info("%s: погода из кэша", region)
    else:
        temp = fetch_temperature(lat, lon, index.min(), index.max())
        temp.to_frame("temp").to_csv(cache)
        log.info("%s: погода скачана и сохранена", region)
    return temp.reindex(index).interpolate(method="time", limit_direction="both")


def build_features(s: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"load": s}); ix = s.index
    out["hour"]=ix.hour; out["dayofweek"]=ix.dayofweek; out["month"]=ix.month
    out["is_weekend"]=(ix.dayofweek>=5).astype(int); out["is_holiday"]=ix.normalize().isin(HOLIDAYS).astype(int)
    out["hour_sin"]=np.sin(2*np.pi*ix.hour/24); out["hour_cos"]=np.cos(2*np.pi*ix.hour/24)
    out["doy_sin"]=np.sin(2*np.pi*ix.dayofyear/365.25); out["doy_cos"]=np.cos(2*np.pi*ix.dayofyear/365.25)
    for L in (24, 48, 168): out[f"lag_{L}"]=s.shift(L)
    r = s.shift(HORIZON)
    out["roll_mean_24"]=r.rolling(24).mean(); out["roll_mean_168"]=r.rolling(168).mean(); out["roll_std_24"]=r.rolling(24).std()
    return out


def load_all() -> tuple[pd.DataFrame, dict]:
    series, frames = {}, []
    for region, (col, lat, lon, _) in REGIONS.items():
        s = clean_series(Path(f"{region}_hourly.csv"), col)
        f = build_features(s)
        f["temp"] = temperature_for(region, lat, lon, s.index)
        f["region"] = region
        series[region] = s; frames.append(f)
    return pd.concat(frames).sort_index().dropna(), series


def time_folds(end, n_folds=6, test_days=30):
    H = pd.Timedelta(days=test_days)
    return [(end - (k + 1) * H, end - k * H) for k in range(n_folds)][::-1]


def backtest(data, series, feature_cols):
    stats = {r: {"mape": [], "mase": []} for r in REGIONS}
    for ts, te in time_folds(data.index.max() + pd.Timedelta(hours=1)):
        train = data[data.index < ts]
        test = data[(data.index >= ts) & (data.index < te)].copy()
        model = CatBoostRegressor(iterations=400, learning_rate=0.1, depth=6,
                                  cat_features=["region"], random_seed=42, verbose=False)
        model.fit(train[feature_cols], train["load"])
        test["pred"] = model.predict(test[feature_cols])
        for r in REGIONS:
            sub = test[test.region == r]
            if sub.empty: continue
            hist = series[r][series[r].index < ts].values
            denom = np.mean(np.abs(hist[SEASONAL_LAG:] - hist[:-SEASONAL_LAG]))
            stats[r]["mape"].append(mape(sub["load"], sub["pred"]))
            stats[r]["mase"].append(mae(sub["load"], sub["pred"]) / denom)
    return {r: (np.mean(v["mape"]), np.mean(v["mase"])) for r, v in stats.items()}


def main() -> None:
    data, series = load_all()
    base_cols = [c for c in data.columns if c not in ("load", "temp")]   # без температуры
    full_cols = [c for c in data.columns if c != "load"]                 # с температурой
    log.info("nodes: %d | rows: %d", data.region.nunique(), len(data))

    log.info("обучение без погоды...")
    no_temp = backtest(data, series, base_cols)
    log.info("обучение с погодой...")
    with_temp = backtest(data, series, full_cols)

    log.info("--- MAPE по узлам: без погоды -> с погодой ---")
    for r in REGIONS:
        log.info("  %-7s %.2f%% -> %.2f%%   (MASE %.2f -> %.2f)",
                 r, no_temp[r][0], with_temp[r][0], no_temp[r][1], with_temp[r][1])


if __name__ == "__main__":
    main()
