"""Загрузка, валидация и очистка исходного почасового ряда AEP.

Исходник не отсортирован и содержит артефакты перехода на летнее время:
дубли меток (осенний перевод) и пропуски (весенний). Строим непрерывную
часовую сетку и заполняем редкие пропуски значением того же часа неделю назад."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from utils import TARGET, SEASONAL_LAG, get_logger

log = get_logger(__name__)
RAW = Path("AEP_hourly.csv")
CLEAN = Path("aep_clean.csv")


def load_and_clean(raw_path: Path = RAW) -> pd.DataFrame:
    df = (pd.read_csv(raw_path, parse_dates=["Datetime"])
            .sort_values("Datetime")
            .set_index("Datetime"))

    df = df.groupby(level=0)[TARGET].mean().to_frame()        # усредняем дубли (осенний DST)

    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "Datetime"
    df["is_imputed"] = df[TARGET].isna().astype(int)

    df[TARGET] = df[TARGET].fillna(df[TARGET].shift(SEASONAL_LAG))   # тот же час неделю назад
    df[TARGET] = df[TARGET].interpolate(method="time", limit_direction="both")
    return df


def main() -> None:
    df = load_and_clean()
    df.to_csv(CLEAN)
    log.info("clean series: %d rows %s..%s, %d imputed",
             len(df), df.index.min().date(), df.index.max().date(), int(df.is_imputed.sum()))


if __name__ == "__main__":
    main()
