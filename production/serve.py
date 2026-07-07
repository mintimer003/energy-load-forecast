"""serve.py — API поверх batch-прогноза (FastAPI).

Для day-ahead задачи прогноз считается раз в сутки batch-джобом
(predict_daily.py), а сервис только раздаёт готовый результат — это дешевле и
надёжнее, чем инференс на каждый запрос. Логика чтения вынесена в чистые функции,
FastAPI — тонкая обвязка.

ЗАПУСК:  uvicorn serve:app --host 0.0.0.0 --port 8000
Эндпоинты:
  GET /health                 — статус сервиса и свежесть прогноза
  GET /forecast?region=AEP    — 24-часовой прогноз узла (P10/P50/P90)
  GET /forecast               — прогноз по всем узлам
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

PRED_DIR = Path(os.getenv("PREDICTIONS_DIR", "predictions"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

app = FastAPI(title="energy-load-forecast", version="1.0")


def read_latest() -> pd.DataFrame:
    path = PRED_DIR / "latest.csv"
    if not path.exists():
        raise FileNotFoundError("latest.csv не найден — сначала запусти predict_daily.py")
    return pd.read_csv(path, parse_dates=["Datetime"])


def forecast_payload(region: str | None = None) -> dict:
    df = read_latest()
    if region is not None:
        region = region.upper()
        if region not in set(df["region"]):
            raise KeyError(f"неизвестный узел: {region}")
        df = df[df["region"] == region]
    records = [
        {"datetime": row.Datetime.isoformat(), "region": row.region,
         "p10": round(float(row.p10), 1), "p50": round(float(row.p50), 1),
         "p90": round(float(row.p90), 1)}
        for row in df.itertuples()
    ]
    return {"horizon_hours": df["Datetime"].nunique(), "rows": records}


def health_payload() -> dict:
    manifest_path = MODELS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        latest = read_latest()
        forecast_start = latest["Datetime"].min().isoformat()
    except FileNotFoundError:
        forecast_start = None
    return {"status": "ok" if forecast_start else "no_forecast",
            "model_trained_at": manifest.get("trained_at_utc"),
            "holdout_mase": manifest.get("holdout_mase"),
            "forecast_start": forecast_start}


@app.get("/health")
def health():
    return health_payload()


@app.get("/forecast")
def forecast(region: str | None = None):
    try:
        return forecast_payload(region)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
