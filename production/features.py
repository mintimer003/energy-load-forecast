"""features.py — признаки для продакшн-слоя (общие для train и predict).

Та же логика, что в исследовательской части (09): лаги >= горизонта (анти-утечка),
календарь, циклические, скользящие со сдвигом на горизонт, температура по
координатам зоны, категориальный признак region.
"""
from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

HORIZON = 24               # горизонт прогноза, часов
SEASONAL_LAG = 168         # "тот же час неделю назад"
LAGS = (24, 48, 168)

# зона: (колонка, широта, долгота)
REGIONS = {
    "AEP":    ("AEP_MW",    39.9612, -82.9988),
    "COMED":  ("COMED_MW",  41.8781, -87.6298),
    "DAYTON": ("DAYTON_MW", 39.7589, -84.1916),
    "DEOK":   ("DEOK_MW",   39.1031, -84.5120),
}
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2004-01-01", end="2030-01-01")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TZ = "America/New_York"

log = logging.getLogger(__name__)


def clean_series(path: Path, col: str) -> pd.Series:
    """Чистый непрерывный часовой ряд: сортировка, дедуп DST, сезонное заполнение."""
    s = (pd.read_csv(path, parse_dates=["Datetime"])
           .sort_values("Datetime").groupby("Datetime")[col].mean())
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s = s.fillna(s.shift(SEASONAL_LAG)).interpolate(method="time", limit_direction="both")
    s.name = "load"
    return s


def fetch_temperature(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Архивная температура Open-Meteo кусками по годам (индекс — локальное время зоны)."""
    frames = []
    for year in range(start.year, end.year + 1):
        s = max(start, pd.Timestamp(f"{year}-01-01"))
        e = min(end, pd.Timestamp(f"{year}-12-31"))
        query = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m", "timezone": TZ,
        })
        with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=60) as resp:
            block = json.load(resp)["hourly"]
        frames.append(pd.DataFrame({"Datetime": pd.to_datetime(block["time"]),
                                    "temp": block["temperature_2m"]}))
    return pd.concat(frames).drop_duplicates("Datetime").set_index("Datetime")["temp"]


def temperature_for(region: str, index: pd.DatetimeIndex, cache_dir: Path) -> pd.Series:
    """Температура зоны с дисковым кэшем weather_{REGION}.csv."""
    _, lat, lon = REGIONS[region]
    cache = cache_dir / f"weather_{region}.csv"
    if cache.exists():
        temp = pd.read_csv(cache, parse_dates=["Datetime"], index_col="Datetime")["temp"]
    else:
        temp = fetch_temperature(lat, lon, index.min(), index.max())
        temp.to_frame("temp").to_csv(cache)
        log.info("%s: температура скачана и закэширована", region)
    return temp.reindex(index).interpolate(method="time", limit_direction="both")


def build_features(s: pd.Series) -> pd.DataFrame:
    """Признаки одного узла. Все лаги/окна сдвинуты минимум на HORIZON (анти-утечка)."""
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
    for lag in LAGS:
        out[f"lag_{lag}"] = s.shift(lag)
    shifted = s.shift(HORIZON)
    out["roll_mean_24"] = shifted.rolling(24).mean()
    out["roll_mean_168"] = shifted.rolling(168).mean()
    out["roll_std_24"] = shifted.rolling(24).std()
    return out


def load_dataset(data_dir: Path, cache_dir: Path | None = None) -> pd.DataFrame:
    """Полный многоузловой датасет с температурой: index=Datetime, колонки=признаки+region."""
    cache_dir = cache_dir or data_dir
    frames = []
    for region, (col, _, _) in REGIONS.items():
        s = clean_series(data_dir / f"{region}_hourly.csv", col)
        f = build_features(s)
        f["temp"] = temperature_for(region, s.index, cache_dir)
        f["region"] = region
        frames.append(f)
    return pd.concat(frames).sort_index().dropna()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "load"]
