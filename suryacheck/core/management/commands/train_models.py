import json
import time

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from core.ml import baselines, calibration, lgbm_model


def _load_clean_data():
    df = pd.read_parquet(settings.ARTIFACTS_DIR / "clean_data.parquet")
    df["INVERTER_ID"] = df["INVERTER_ID"].astype(str)
    return df


class Command(BaseCommand):
    help = "Train B0, B1, B2, M1 (LightGBM + Optuna) and the M1+cal calibration layer."

    def add_arguments(self, parser):
        parser.add_argument("--trials", type=int, default=40)

    def handle(self, *args, **options):
        settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df = _load_clean_data()
        with open(settings.ARTIFACTS_DIR / "inverter_categories.json") as f:
            categories = json.load(f)

        seen = df[~df["IS_UNSEEN"]]
        train = seen[seen["SPLIT"] == "train"].copy()
        val = seen[seen["SPLIT"] == "val"].copy()

        self.stdout.write(f"Training on {len(train)} rows, validating on {len(val)} rows.")

        self.stdout.write("Fitting B1 (irradiance rule)...")
        k_dict = baselines.fit_b1(train)

        self.stdout.write("Fitting B2 (rule + temperature)...")
        gamma_dict = baselines.fit_b2(train, k_dict)

        self.stdout.write(f"Tuning M1 (LightGBM, {options['trials']} Optuna trials)...")
        start = time.time()
        model, best_params, best_val_mae = lgbm_model.tune_and_train(
            train, val, categories, settings.RANDOM_SEED, n_trials=options["trials"]
        )
        self.stdout.write(f"M1 tuned in {time.time() - start:.1f}s, val MAE={best_val_mae:.3f}, params={best_params}")

        self.stdout.write("Fitting M1+cal per-inverter calibration...")
        m1_pred_train = lgbm_model.predict(model, train, categories)
        calib = calibration.fit_calibration(train, m1_pred_train)

        joblib.dump(k_dict, settings.ARTIFACTS_DIR / "b1_k.joblib")
        joblib.dump(gamma_dict, settings.ARTIFACTS_DIR / "b2_gamma.joblib")
        joblib.dump(calib, settings.ARTIFACTS_DIR / "m1_calibration.joblib")
        model.save_model(str(settings.ARTIFACTS_DIR / "m1_lightgbm.txt"), num_iteration=model.best_iteration)
        with open(settings.ARTIFACTS_DIR / "m1_best_params.json", "w") as f:
            json.dump(
                {"params": best_params, "val_mae": best_val_mae, "best_iteration": model.best_iteration},
                f, indent=2,
            )

        self.stdout.write(self.style.SUCCESS("train_models complete."))
