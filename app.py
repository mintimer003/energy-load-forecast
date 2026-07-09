"""app.py — живой дашборд сети (Streamlit).

Проигрывает 2018 год: 4 ГРПШ → головной ГРП. Показывает ежечасный расход на
узлах, суммарный спрос, давление на головном ГРП (проактивное vs фиксированное),
график прогноз/факт и иллюстративную экономию в рублях.

Топливо — demo_2018.csv (создаётся demo_prepare.py).

Спрос — прокси на электроданных; давление и экономика — упрощённые модели с
условными параметрами (см. константы ниже), подставляются значения конкретной сети.

ЗАПУСК: streamlit run app.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import streamlit as st

REGIONS = ["AEP", "COMED", "DAYTON", "DEOK"]
P_MIN, P_MAX = 0.10, 0.30                 # МПа: среднее давление (нормы 0,005-0,3); условные

# --- параметры экономической оценки (условные) ---
GAS_PRICE_RUB_PER_M3 = 7.0                # цена газа, ₽/м³
LEAK_AT_PMAX_M3_PER_H = 50.0             # утечка сети при максимальном давлении, м³/ч
# утечка принята линейной по давлению: leak(t) = LEAK_AT_PMAX * P(t)/P_MAX

st.set_page_config(page_title="Прогноз спроса сети", layout="wide")


@st.cache_data
def load_demo() -> pd.DataFrame:
    df = pd.read_csv("demo_2018.csv", parse_dates=["Datetime"], index_col="Datetime")
    for stat in ("p10", "p50", "p90"):
        df[f"total_{stat}"] = sum(df[f"{r}_{stat}"] for r in REGIONS)
    return df


def draw_network(row) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.3)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    gx, gy = 0.82, 0.5
    fill = max(0.0, min(1.0, (row["p_head_proactive"] - P_MIN) / (P_MAX - P_MIN)))
    ax.add_patch(Rectangle((gx - 0.08, gy - 0.13), 0.16, 0.26, fc="#16324f", ec="black"))
    ax.add_patch(Rectangle((gx - 0.08, gy - 0.13), 0.16, 0.26 * fill, fc="#e0773a", ec="none"))
    ax.text(gx, gy + 0.21, "Головной ГРП", ha="center", weight="bold")
    ax.text(gx, gy, f"{row['p_head_proactive']:.2f} МПа", ha="center", color="white", weight="bold")
    for reg, y in zip(REGIONS, np.linspace(0.12, 0.88, len(REGIONS))):
        ax.plot([0.22, gx - 0.08], [y, gy], color="#9aa7b2", lw=2, zorder=0)
        ax.add_patch(Rectangle((0.06, y - 0.055), 0.16, 0.11, fc="#eaf2f8", ec="#16324f"))
        ax.text(0.14, y + 0.013, reg, ha="center", weight="bold", fontsize=9)
        ax.text(0.14, y - 0.03, f"{row[reg + '_load']:,.0f} МВт", ha="center", fontsize=8)
    return fig


def draw_series(df, i, columns, title, fill=None, window=168) -> plt.Figure:
    lo, hi = max(0, i - window), min(len(df), i + 24)
    win = df.iloc[lo:hi]
    fig, ax = plt.subplots(figsize=(8, 3))
    if fill:
        ax.fill_between(win.index, win[fill[0]], win[fill[1]], alpha=0.22, color=fill[2], label=fill[3])
    for col, style in columns:
        ax.plot(win.index, win[col], **style)
    ax.axvline(df.index[i], color="red", lw=1)
    ax.set_title(title); ax.legend(fontsize=8); fig.autofmt_xdate()
    return fig


df = load_demo()

# экономия за весь период (иллюстративно): утечка линейна по давлению
leak_proactive = LEAK_AT_PMAX_M3_PER_H * df["p_head_proactive"] / P_MAX
saved_m3 = (LEAK_AT_PMAX_M3_PER_H - leak_proactive).sum()          # м³ за период (час × 1)
saved_rub = saved_m3 * GAS_PRICE_RUB_PER_M3

st.title("Сеть газораспределения — прогноз спроса и управление давлением")
st.caption("Демонстратор (симуляция 2018). Спрос — прокси на открытых данных по электропотреблению; "
           "давление и экономика — по упрощённым моделям с условными параметрами. Не измеренные газовые величины.")

i = st.slider("Момент времени (час 2018 года)", 0, len(df) - 1, 0)
row = df.iloc[i]
st.subheader(f"{df.index[i]:%Y-%m-%d  %H:%M}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Суммарный спрос сети", f"{row['total_actual']:,.0f} МВт")
c2.metric("Давление ГРП (проактивное)", f"{row['p_head_proactive']:.3f} МПа")
c3.metric("Давление (фиксированное)", f"{row['p_head_fixed']:.3f} МПа")
c4.metric("Δ давления сейчас", f"{row['p_head_fixed'] - row['p_head_proactive']:.3f} МПа")

st.pyplot(draw_network(row))

left, right = st.columns(2)
with left:
    st.pyplot(draw_series(df, i,
        [("total_p50", dict(label="прогноз")), ("total_actual", dict(color="black", lw=1, label="факт"))],
        "Суммарный спрос сети, МВт", fill=("total_p10", "total_p90", "#4a90d9", "P10–P90")))
with right:
    st.pyplot(draw_series(df, i,
        [("p_head_fixed", dict(ls="--", color="gray", label="фиксированное")),
         ("p_head_proactive", dict(color="green", label="проактивное"))],
        "Давление на головном ГРП, МПа", fill=("p_head_proactive", "p_head_fixed", "green", "экономия")))

st.subheader("Иллюстративная экономия за период симуляции (янв–авг 2018)")
e1, e2 = st.columns(2)
e1.metric("Сэкономлено газа (утечки)", f"{saved_m3:,.0f} м³")
e2.metric("Экономия", f"{saved_rub:,.0f} ₽")
st.caption(f"Допущения: утечка {LEAK_AT_PMAX_M3_PER_H:.0f} м³/ч при {P_MAX} МПа, "
           f"линейно по давлению; цена газа {GAS_PRICE_RUB_PER_M3:.1f} ₽/м³. Это инженерная прикидка порядка величины.")
