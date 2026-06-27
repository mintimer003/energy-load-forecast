"""10_tuning.py — подбор гиперпараметров CatBoost (Optuna) + важность признаков.

Optuna ищет гиперпараметры, оценивая каждую комбинацию на бэктесте со скользящим
началом. Число деревьев вручную НЕ задаём — его находит EARLY STOPPING на
ВРЕМЕННОЙ валидации (последний кусок обучающего окна, не случайный), чтобы
сохранить защиту от утечки. В конце: сравнение дефолт vs затюненный и важность
признаков затюненной модели.

ЗАПУСК: pip install optuna   затем   python 10_tuning.py   (нужен aep_features.csv)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from utils import TARGET, load_features, rolling_origin, mape, mase, get_logger

log = get_logger(__name__)
N_TRIALS = 30
TUNING_FOLDS = 3            # для скорости подбора; финал — на всех 6 окнах
VAL_FRACTION = 0.1         # последняя доля обучающего окна -> валидация early stopping
EARLY_STOPPING = 50
DEFAULT_PARAMS = dict(iterations=400, learning_rate=0.1, depth=6)


def fit_early_stopping(params, X_train, y_train, X_test):
    cut = int(len(X_train) * (1 - VAL_FRACTION))               # ВРЕМЕННОЙ срез, не случайный
    model = CatBoostRegressor(random_seed=42, verbose=False, **params)
    model.fit(X_train.iloc[:cut], y_train.iloc[:cut],
              eval_set=(X_train.iloc[cut:], y_train.iloc[cut:]),
              early_stopping_rounds=EARLY_STOPPING)
    return model, model.predict(X_test)


def backtest(features, params, n_folds=6, early_stop=True):
    scores, model = [], None
    for tr, te in rolling_origin(features.index, n_folds=n_folds):
        X_tr, y_tr = features.loc[tr].drop(columns=[TARGET]), features.loc[tr, TARGET]
        X_te, y_te = features.loc[te].drop(columns=[TARGET]), features.loc[te, TARGET]
        if early_stop:
            model, pred = fit_early_stopping(params, X_tr, y_tr, X_te)
        else:
            model = CatBoostRegressor(random_seed=42, verbose=False, **params).fit(X_tr, y_tr)
            pred = model.predict(X_te)
        scores.append({"MASE": mase(y_te, pred, y_tr), "MAPE": mape(y_te, pred)})
    return pd.DataFrame(scores).mean(), model


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    features = load_features()

    def objective(trial):
        params = {
            "iterations": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
        score, _ = backtest(features, params, n_folds=TUNING_FOLDS, early_stop=True)
        return score["MASE"]

    study = optuna.create_study(direction="minimize")
    log.info("подбор: %d проб на %d окнах (early stopping сам найдёт число деревьев)...", N_TRIALS, TUNING_FOLDS)
    study.optimize(objective, n_trials=N_TRIALS)
    best = {"iterations": 3000, **study.best_params}
    log.info("лучшие параметры: %s", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.best_params.items()})

    log.info("сравнение на всех 6 окнах: дефолт vs затюненный...")
    default_score, _ = backtest(features, DEFAULT_PARAMS, n_folds=6, early_stop=False)
    tuned_score, tuned_model = backtest(features, best, n_folds=6, early_stop=True)
    log.info("дефолт    : MASE %.3f | MAPE %.2f%%", default_score.MASE, default_score.MAPE)
    log.info("затюненный: MASE %.3f | MAPE %.2f%%", tuned_score.MASE, tuned_score.MAPE)
    log.info("улучшение MASE: %.1f%%", (default_score.MASE - tuned_score.MASE) / default_score.MASE * 100)

    importance = pd.Series(tuned_model.get_feature_importance(), index=tuned_model.feature_names_).sort_values(ascending=False)
    log.info("важность признаков (затюненная модель, %%):\n%s", importance.round(1).to_string())


if __name__ == "__main__":
    main()
