import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error

from .features import FEATURE_COLUMNS, TARGET_COLUMN, apply_inverter_categories

optuna.logging.set_verbosity(optuna.logging.WARNING)
CATEGORICAL_FEATURES = ["INVERTER_ID"]


def _matrices(df, categories):
    df = apply_inverter_categories(df, categories)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].to_numpy()
    return X, y


def _train_one(params, X_train, y_train, X_val, y_val, seed, num_boost_round=2000):
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=train_set, free_raw_data=False)
    full_params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        **params,
    }
    model = lgb.train(
        full_params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def tune_and_train(train_df, val_df, categories, seed, n_trials=40):
    X_train, y_train = _matrices(train_df, categories)
    X_val, y_val = _matrices(val_df, categories)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        model = _train_one(params, X_train, y_train, X_val, y_val, seed)
        pred = model.predict(X_val, num_iteration=model.best_iteration)
        return mean_absolute_error(y_val, pred)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = _train_one(study.best_params, X_train, y_train, X_val, y_val, seed)
    return best_model, study.best_params, study.best_value


def predict(model, df, categories, num_iteration=None):
    X, _ = _matrices(df, categories)
    if num_iteration is None:
        num_iteration = model.best_iteration if model.best_iteration else None
    return model.predict(X, num_iteration=num_iteration)


def load_model(path):
    return lgb.Booster(model_file=str(path))
