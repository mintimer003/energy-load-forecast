"""Обучение CatBoost на бэктесте и сравнение с сезонным наивом на одних и тех же
тестовых окнах."""
from __future__ import annotations
import pandas as pd
from catboost import CatBoostRegressor
from utils import TARGET, load_features, rolling_origin, mae, mape, mase, get_logger

log = get_logger(__name__)


def backtest(features: pd.DataFrame):
    records, model = [], None
    for fold, (train_idx, test_idx) in enumerate(rolling_origin(features.index), 1):
        X_train, y_train = features.loc[train_idx].drop(columns=[TARGET]), features.loc[train_idx, TARGET]
        X_test, y_test = features.loc[test_idx].drop(columns=[TARGET]), features.loc[test_idx, TARGET]

        model = CatBoostRegressor(iterations=400, learning_rate=0.1, depth=6,
                                  random_seed=42, verbose=False)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        baseline = X_test["lag_168"].values

        records.append({
            "fold": fold,
            "base_MAPE": mape(y_test, baseline), "model_MAPE": mape(y_test, prediction),
            "base_MASE": mase(y_test, baseline, y_train), "model_MASE": mase(y_test, prediction, y_train),
            "base_MAE": mae(y_test, baseline), "model_MAE": mae(y_test, prediction),
        })
        log.info("fold %d trained", fold)
    return pd.DataFrame(records), model


def main() -> None:
    scores, model = backtest(load_features())
    for _, r in scores.iterrows():
        log.info("fold %d | MAPE %.2f%% -> %.2f%% | MASE %.2f -> %.2f",
                 r.fold, r.base_MAPE, r.model_MAPE, r.base_MASE, r.model_MASE)
    mean = scores.mean(numeric_only=True)
    log.info("baseline | MAPE %.2f%% | MASE %.3f | MAE %.0f MW", mean.base_MAPE, mean.base_MASE, mean.base_MAE)
    log.info("catboost | MAPE %.2f%% | MASE %.3f | MAE %.0f MW", mean.model_MAPE, mean.model_MASE, mean.model_MAE)
    log.info("MAE improvement vs baseline: %.0f%%", (mean.base_MAE - mean.model_MAE) / mean.base_MAE * 100)

    importance = pd.Series(model.get_feature_importance(), index=model.feature_names_).sort_values(ascending=False)
    log.info("feature importance (%%):\n%s", importance.round(1).to_string())


if __name__ == "__main__":
    main()
