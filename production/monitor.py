"""monitor.py — мониторинг качества, дрейфа и решение о фолбэке.

Три проверки:
  1) качество: сохранённые прогнозы против факта, MAPE/MASE/покрытие по узлам;
     MASE выше порога — модель хуже допуска;
  2) дрейф данных: сдвиг среднего нагрузки за последние 30 дней относительно
     обучающего распределения (границы — из manifest);
  3) решение: status.json с use_fallback (при деградации потребителю отдаётся
     сезонный наив — тот же час неделю назад) и retrain_recommended.

ЗАПУСК:  python monitor.py --data-dir .. --pred-dir predictions --models-dir models
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import REGIONS, SEASONAL_LAG, clean_series

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MASE_THRESHOLD = 0.9      # хуже этого — почти как наив, смысла в модели нет
DRIFT_SIGMAS = 3.0        # сдвиг среднего за 30 дней в std обучающего распределения


def evaluate_forecasts(pred_dir: Path, actuals: dict[str, pd.Series]) -> dict:
    """MAPE/MASE по всем сохранённым прогнозам, у которых уже есть факт."""
    files = sorted(pred_dir.glob("forecast_*.csv"))
    if not files:
        return {}
    frames = [pd.read_csv(f, parse_dates=["Datetime"]) for f in files]
    pred = pd.concat(frames, ignore_index=True).drop_duplicates(["Datetime", "region"])
    per_region = {}
    for region, s in actuals.items():
        sub = pred[pred["region"] == region].set_index("Datetime")
        joined = sub.join(s.rename("actual"), how="inner").dropna(subset=["actual"])
        if joined.empty:
            continue
        err = np.abs(joined["actual"] - joined["p50"])
        naive_mae = np.mean(np.abs(s.values[SEASONAL_LAG:] - s.values[:-SEASONAL_LAG]))
        per_region[region] = {
            "hours_scored": int(len(joined)),
            "mape": float(np.mean(err / joined["actual"]) * 100),
            "mase": float(err.mean() / naive_mae),
            "p10_p90_coverage": float(((joined["actual"] >= joined["p10"]) &
                                       (joined["actual"] <= joined["p90"])).mean()),
        }
    return per_region


def check_drift(actuals: dict[str, pd.Series], manifest: dict) -> dict:
    """Сдвиг среднего нагрузки за последние 30 дней относительно обучающего периода."""
    drift = {}
    train_end = pd.Timestamp(manifest.get("data_end"))
    for region, s in actuals.items():
        train_part = s[s.index <= train_end]
        recent = s[s.index > s.index.max() - pd.Timedelta(days=30)]
        if train_part.empty or recent.empty:
            continue
        shift_sigmas = abs(recent.mean() - train_part.mean()) / (train_part.std() + 1e-9)
        drift[region] = {"shift_sigmas": float(shift_sigmas),
                         "drifted": bool(shift_sigmas > DRIFT_SIGMAS)}
    return drift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(".."))
    parser.add_argument("--pred-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    manifest = json.loads((args.models_dir / "manifest.json").read_text())
    actuals = {r: clean_series(args.data_dir / f"{r}_hourly.csv", col)
               for r, (col, _, _) in REGIONS.items()}

    quality = evaluate_forecasts(args.pred_dir, actuals)
    drift = check_drift(actuals, manifest)

    worst_mase = max((v["mase"] for v in quality.values()), default=None)
    any_drift = any(v["drifted"] for v in drift.values())
    use_fallback = worst_mase is not None and worst_mase > MASE_THRESHOLD

    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "quality": quality,
        "worst_mase": worst_mase,
        "mase_threshold": MASE_THRESHOLD,
        "use_fallback": use_fallback,
        "fallback_strategy": "seasonal naive: тот же час неделю назад",
        "drift": drift,
        "retrain_recommended": bool(any_drift or use_fallback),
    }
    (args.pred_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))

    for region, v in quality.items():
        log.info("%s: MAPE %.2f%% | MASE %.3f | coverage %.0f%% (часов: %d)",
                 region, v["mape"], v["mase"], v["p10_p90_coverage"] * 100, v["hours_scored"])
    log.info("worst MASE: %s | fallback: %s | retrain: %s",
             f"{worst_mase:.3f}" if worst_mase else "n/a", use_fallback, status["retrain_recommended"])


if __name__ == "__main__":
    main()
