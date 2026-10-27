import json

import joblib
import pandas as pd
from django.conf import settings

from . import lgbm_model

_cache = {}


def load_clean_data():
    df = pd.read_parquet(settings.ARTIFACTS_DIR / "clean_data.parquet")
    df["INVERTER_ID"] = df["INVERTER_ID"].astype(str)
    return df


def load_artifacts():
    if _cache:
        return _cache
    artifacts_dir = settings.ARTIFACTS_DIR
    with open(artifacts_dir / "inverter_categories.json") as f:
        categories = json.load(f)
    with open(artifacts_dir / "inverter_peak_power.json") as f:
        peak_power = json.load(f)
    with open(artifacts_dir / "unseen_inverters.json") as f:
        unseen_ids = set(json.load(f))
    with open(artifacts_dir / "m1_best_params.json") as f:
        best_params = json.load(f)

    _cache.update({
        "categories": categories,
        "peak_power": peak_power,
        "unseen_ids": unseen_ids,
        "best_iteration": best_params.get("best_iteration"),
        "k_dict": joblib.load(artifacts_dir / "b1_k.joblib"),
        "gamma_dict": joblib.load(artifacts_dir / "b2_gamma.joblib"),
        "calib": joblib.load(artifacts_dir / "m1_calibration.joblib"),
        "m1_model": lgbm_model.load_model(artifacts_dir / "m1_lightgbm.txt"),
    })
    return _cache


def predict_m1(df):
    art = load_artifacts()
    return lgbm_model.predict(art["m1_model"], df, art["categories"], num_iteration=art["best_iteration"])


def artifacts_ready():
    required = [
        "clean_data.parquet", "inverter_categories.json", "b1_k.joblib",
        "b2_gamma.joblib", "m1_calibration.joblib", "m1_lightgbm.txt", "results.json",
    ]
    return all((settings.ARTIFACTS_DIR / name).exists() for name in required)
