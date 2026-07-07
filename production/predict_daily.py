"""predict_daily.py — ежедневный batch-прогноз на сутки вперёд.

Продакшн-паттерн day-ahead: раз в сутки джоб строит прогноз на следующие 24 часа
по каждому узлу и пишет predictions/forecast_{дата}.csv (+ копия latest.csv,
которую раздаёт API). Модели НЕ переобучаются — загружаются из models/*.cbm.

Так как открытый датасет заканчивается в 2018, джоб работает в режиме симуляции:
"сейчас" = последняя метка данных (или --as-of). Температура будущих часов берётся
из архива как идеальный прогноз погоды; в реальной системе здесь был бы прогнозный
API погоды (для day-ahead это допустимый вход, как календарь).

ЗАПУСК:  python predict_daily.py --data-dir .. --models-dir models --out-dir predictions
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from features import HORIZON, REGIONS, build_features, clean_series, temperature_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_models(models_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((models_dir / "manifest.json").read_text())
    models = {}
    for q in manifest["quantiles"]:
        m = CatBoostRegressor()
        m.load_model(str(models_dir / f"model_p{int(q * 100)}.cbm"))
        models[q] = m
    return models, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(".."))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--as-of", default=None, help="метка 'сейчас' (по умолчанию конец данных)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    models, manifest = load_models(args.models_dir)
    cols = manifest["feature_columns"]

    rows = []
    for region, (col, _, _) in REGIONS.items():
        s = clean_series(args.data_dir / f"{region}_hourly.csv", col)
        now = pd.Timestamp(args.as_of) if args.as_of else s.index.max()
        target_index = pd.date_range(now + pd.Timedelta(hours=1), periods=HORIZON, freq="h")

        # ряд, продолженный пустыми будущими часами: лаги >= 24 заполнятся историей
        extended = s.reindex(s.index.union(target_index))
        feats = build_features(extended)
        feats["temp"] = temperature_for(region, extended.index, args.data_dir)
        feats["region"] = region
        block = feats.loc[target_index, cols]
        if block.isna().any().any():
            raise ValueError(f"{region}: NaN в признаках будущих часов — проверь глубину истории")

        out = pd.DataFrame({"Datetime": target_index, "region": region})
        for q, model in models.items():
            out[f"p{int(q * 100)}"] = model.predict(block)
        rows.append(out)
        log.info("%s: прогноз на %s .. %s", region, target_index[0], target_index[-1])

    forecast = pd.concat(rows, ignore_index=True)
    day = forecast["Datetime"].dt.date.iloc[0]
    forecast.to_csv(args.out_dir / f"forecast_{day}.csv", index=False)
    forecast.to_csv(args.out_dir / "latest.csv", index=False)
    log.info("записано: forecast_%s.csv и latest.csv (%d строк)", day, len(forecast))


if __name__ == "__main__":
    main()
