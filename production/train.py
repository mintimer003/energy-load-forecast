"""train.py — обучение продакшн-моделей + трекинг в MLflow.

Обучает три квантильные многоузловые модели CatBoost (P10/P50/P90) на всей
доступной истории, логирует параметры/метрики/важность признаков в MLflow и
сохраняет модели файлами model_p{10,50,90}.cbm + manifest.json (список фич,
граница данных). Перед финальным обучением быстрая проверка качества на
последнем 30-дневном отрезке (hold-out по времени, не случайный).

ЗАПУСК:  python train.py --data-dir .. --models-dir models
MLflow:  результаты в ./mlruns, смотреть: mlflow ui --backend-store-uri mlruns
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from catboost import CatBoostRegressor

from features import HORIZON, SEASONAL_LAG, load_dataset, feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

QUANTILES = (0.10, 0.50, 0.90)
PARAMS = dict(iterations=400, learning_rate=0.1, depth=6)   # дефолты: подбор (Optuna) их не превзошёл
HOLDOUT_DAYS = 30


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(y_true, y_pred, y_train_hist) -> float:
    naive_mae = np.mean(np.abs(y_train_hist[SEASONAL_LAG:] - y_train_hist[:-SEASONAL_LAG]))
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / naive_mae)


def fit_quantile(q: float, X: pd.DataFrame, y: pd.Series) -> CatBoostRegressor:
    model = CatBoostRegressor(loss_function=f"Quantile:alpha={q}", cat_features=["region"],
                              random_seed=42, verbose=False, **PARAMS)
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(".."))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--experiment", default="energy-load-forecast")
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.data_dir)
    cols = feature_columns(data)
    log.info("dataset: %d rows | %d features | %s .. %s",
             len(data), len(cols), data.index.min(), data.index.max())

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"train_{datetime.now(timezone.utc):%Y%m%d_%H%M}"):
        mlflow.log_params({**PARAMS, "quantiles": QUANTILES, "n_features": len(cols),
                           "rows": len(data), "data_end": str(data.index.max())})

        # -- контроль качества: hold-out по времени (последние 30 дней) --
        cutoff = data.index.max() - pd.Timedelta(days=HOLDOUT_DAYS)
        train_df, test_df = data[data.index <= cutoff], data[data.index > cutoff]
        p50_check = fit_quantile(0.50, train_df[cols], train_df["load"])
        pred = p50_check.predict(test_df[cols])
        holdout = {
            "holdout_mape": mape(test_df["load"], pred),
            "holdout_mase": mase(test_df["load"], pred, train_df["load"].values),
        }
        mlflow.log_metrics(holdout)
        log.info("hold-out(30d): MAPE %.2f%% | MASE %.3f", holdout["holdout_mape"], holdout["holdout_mase"])

        # -- важность признаков --
        importance = pd.Series(p50_check.get_feature_importance(), index=cols).sort_values(ascending=False)
        imp_path = args.models_dir / "feature_importance.csv"
        importance.to_csv(imp_path, header=["importance"])
        mlflow.log_artifact(str(imp_path))
        log.info("top-5 признаков: %s", ", ".join(f"{k}={v:.1f}%" for k, v in importance.head(5).items()))

        # -- финальное обучение на всей истории и сохранение --
        for q in QUANTILES:
            model = fit_quantile(q, data[cols], data["load"])
            path = args.models_dir / f"model_p{int(q * 100)}.cbm"
            model.save_model(str(path))
            mlflow.log_artifact(str(path))
            log.info("saved %s", path.name)

        manifest = {
            "feature_columns": cols,
            "quantiles": list(QUANTILES),
            "horizon_hours": HORIZON,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_end": str(data.index.max()),
            **holdout,
        }
        manifest_path = args.models_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(manifest_path))
        log.info("manifest записан; обучение завершено")


if __name__ == "__main__":
    main()
