"""demo_prepare.py — данные для живой демки (Streamlit).

Обучает многоузловую модель на истории ДО 2018 (квантили P10/P50/P90),
прогнозирует 2018 по каждому узлу, складывает суммарный спрос сети и считает
требуемое давление на головном ГРП. Сохраняет demo_2018.csv для app.py.

Гидравлика ИЛЛЮСТРАТИВНАЯ (реальных газовых данных нет):
P_head = sqrt(P_min^2 + (P_max^2 - P_min^2) * q_norm^2) — давление следует за
прогнозным резервом сети (P90). Параметры P_min/P_max условные.

ЗАПУСК: python demo_prepare.py   (нужны *_hourly.csv узлов рядом)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from catboost import CatBoostRegressor
from utils import SEASONAL_LAG, HORIZON, get_logger

log = get_logger(__name__)
REGIONS = {"AEP": "AEP_MW", "COMED": "COMED_MW", "DAYTON": "DAYTON_MW", "DEOK": "DEOK_MW"}
HOLIDAYS = USFederalHolidayCalendar().holidays(start="2004-01-01", end="2019-01-01")
SPLIT = pd.Timestamp("2018-01-01")           # обучаем ДО 2018, проигрываем 2018
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
P_MIN, P_MAX = 0.10, 0.30          # МПа: среднее давление (по нормам 0,005-0,3); условные


def clean_series(path: Path, col: str) -> pd.Series:
    s = (pd.read_csv(path, parse_dates=["Datetime"]).sort_values("Datetime").groupby("Datetime")[col].mean())
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    s = s.fillna(s.shift(SEASONAL_LAG)).interpolate(method="time", limit_direction="both")
    s.name = "load"
    return s


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


def load_all() -> pd.DataFrame:
    frames = []
    for region, col in REGIONS.items():
        f = build_features(clean_series(Path(f"{region}_hourly.csv"), col))
        f["region"] = region
        frames.append(f)
    return pd.concat(frames).sort_index().dropna()


def main() -> None:
    data = load_all()
    data.index.name = "Datetime"
    feat_cols = [c for c in data.columns if c != "load"]
    train = data[data.index < SPLIT]
    sim = data[data.index >= SPLIT].copy()
    log.info("train rows: %d | sim(2018) rows: %d", len(train), len(sim))

    for name, q in QUANTILES.items():
        m = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", iterations=400, learning_rate=0.1,
                              depth=6, cat_features=["region"], random_seed=42, verbose=False)
        m.fit(train[feat_cols], train["load"])
        sim[name] = m.predict(sim[feat_cols])
        log.info("%s trained", name)

    # широкая таблица по узлам: {REGION}_load / _p10 / _p50 / _p90
    nodes = sim.reset_index().pivot_table(index="Datetime", columns="region", values=["load", "p10", "p50", "p90"])
    nodes.columns = [f"{region}_{value}" for value, region in nodes.columns]
    nodes = nodes.dropna()

    # агрегат сети + давление на головном ГРП
    agg = pd.DataFrame(index=nodes.index)
    agg["total_actual"] = sum(nodes[f"{r}_load"] for r in REGIONS)
    agg["total_p50"] = sum(nodes[f"{r}_p50"] for r in REGIONS)
    agg["total_p90"] = sum(nodes[f"{r}_p90"] for r in REGIONS)

    qmax = max(agg["total_actual"].max(), agg["total_p90"].max())
    qn = (agg["total_p90"] / qmax).clip(0, 1)
    agg["p_head_proactive"] = np.sqrt(P_MIN**2 + (P_MAX**2 - P_MIN**2) * qn**2)   # следует за прогнозом
    agg["p_head_fixed"] = P_MAX                                                    # «всегда высокое»

    out = nodes.join(agg)
    out.to_csv("demo_2018.csv")
    log.info("saved demo_2018.csv: %d часов, %d столбцов", len(out), out.shape[1])
    log.info("экономия давления (средняя): %.3f МПа", (agg["p_head_fixed"] - agg["p_head_proactive"]).mean())


if __name__ == "__main__":
    main()
