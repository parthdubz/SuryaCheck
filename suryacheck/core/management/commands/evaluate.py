import json
from datetime import datetime, timezone

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand

from core.ml import baselines, calibration, classify, detect, inject, lgbm_model, registry
from core.models import Alert, Inverter, ModelRun

MODEL_NAMES = ["b0", "b1", "b2", "m1", "m1_cal"]


def _guess_fault_type(ratios):
    if not ratios:
        return "unknown"
    min_ratio = min(ratios)
    if min_ratio < 0.15:
        return "outage"
    if (ratios[0] - ratios[-1]) > 0.25:
        return "step"
    return "soiling"


def _metrics(y_true, y_pred, inverter_ids, peak_power, default_peak):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    peaks = np.array([peak_power.get(i, default_peak) or default_peak for i in inverter_ids])
    peaks = np.clip(peaks, 1e-6, None)
    mae_pct_peak = float(np.mean(np.abs(err) / peaks) * 100)
    return {"mae": mae, "rmse": rmse, "mae_pct_of_peak": mae_pct_peak, "n": int(len(y_true))}


class Command(BaseCommand):
    help = "Evaluate baselines + M1/M1+cal, inject synthetic faults, tune detection, train cause classifier."

    def handle(self, *args, **options):
        df = registry.load_clean_data()
        art = registry.load_artifacts()
        categories, k_dict, gamma_dict, calib = art["categories"], art["k_dict"], art["gamma_dict"], art["calib"]
        peak_power = art["peak_power"]
        default_peak = float(np.mean(list(peak_power.values()))) if peak_power else 1.0

        seen = df[~df["IS_UNSEEN"]]
        train = seen[seen["SPLIT"] == "train"]
        test_seen = seen[seen["SPLIT"] == "test"]
        unseen_test = df[df["IS_UNSEEN"] & (df["SPLIT"] == "test")]

        persistence_lookup = baselines.build_persistence_lookup(df)

        def predict_all(subset_df, k, gamma, cal, model, num_iteration):
            preds = {
                "b0": baselines.predict_persistence(subset_df, persistence_lookup),
                "b1": baselines.predict_b1(subset_df, k),
                "b2": baselines.predict_b2(subset_df, k, gamma),
            }
            preds["m1"] = lgbm_model.predict(model, subset_df, categories, num_iteration=num_iteration)
            preds["m1_cal"] = calibration.predict_calibrated(subset_df, preds["m1"], cal)
            return preds

        self.stdout.write("Scoring standard test split...")
        results = {"model_comparison": {}}
        for scenario_name, subset in [("test", test_seen), ("unseen_inverters", unseen_test)]:
            preds = predict_all(subset, k_dict, gamma_dict, calib, art["m1_model"], art["best_iteration"])
            for name in MODEL_NAMES:
                results["model_comparison"].setdefault(name, {})[scenario_name] = _metrics(
                    subset["AC_POWER"], preds[name], subset["INVERTER_ID"], peak_power, default_peak
                )

        self.stdout.write("Training cross-plant diagnostic models (Plant 1 -> Plant 2)...")
        plant1_train = train[train["PLANT_ID"] == 4135001]
        plant1_val = seen[(seen["SPLIT"] == "val") & (seen["PLANT_ID"] == 4135001)]
        plant2_test = test_seen[test_seen["PLANT_ID"] == 4136001]

        cp_categories = sorted(plant1_train["INVERTER_ID"].unique())
        cp_k = baselines.fit_b1(plant1_train)
        cp_gamma = baselines.fit_b2(plant1_train, cp_k)
        cp_model, cp_params, cp_val_mae = lgbm_model.tune_and_train(
            plant1_train, plant1_val, cp_categories, settings.RANDOM_SEED, n_trials=15
        )
        cp_m1_pred_train = lgbm_model.predict(cp_model, plant1_train, cp_categories)
        cp_calib = calibration.fit_calibration(plant1_train, cp_m1_pred_train)

        cp_preds = predict_all(plant2_test, cp_k, cp_gamma, cp_calib, cp_model, cp_model.best_iteration)
        for name in MODEL_NAMES:
            results["model_comparison"][name]["cross_plant"] = _metrics(
                plant2_test["AC_POWER"], cp_preds[name], plant2_test["INVERTER_ID"], peak_power, default_peak
            )
        results["cross_plant_note"] = (
            "B1/B2/M1/M1+cal were refit on Plant 1 train only, then evaluated on Plant 2 test. "
            "B0 persistence has no training step so it is the same per-inverter lookup as the standard test scenario."
        )

        self.stdout.write("Injecting synthetic faults into validation period...")
        val_seen = seen[seen["SPLIT"] == "val"].copy()
        test_for_faults = test_seen.copy()

        val_faulted, val_fault_windows = self._inject_many(val_seen, settings.VAL_DAYS, settings.RANDOM_SEED)
        test_faulted, test_fault_windows = self._inject_many(test_for_faults, settings.TEST_DAYS, settings.RANDOM_SEED + 1)

        val_pred_m1cal = calibration.predict_calibrated(
            val_faulted, lgbm_model.predict(art["m1_model"], val_faulted, categories, art["best_iteration"]), calib
        )
        test_pred_m1cal = calibration.predict_calibrated(
            test_faulted, lgbm_model.predict(art["m1_model"], test_faulted, categories, art["best_iteration"]), calib
        )
        val_faulted["PRED"] = val_pred_m1cal
        test_faulted["PRED"] = test_pred_m1cal

        val_ratio = detect.daily_ratio_table(val_faulted)
        test_ratio = detect.daily_ratio_table(test_faulted)

        self.stdout.write("Tuning detection threshold (N consecutive days, ratio T) on validation faults...")
        best_n, best_t, val_detect_metrics = detect.tune_threshold(val_ratio, val_fault_windows)
        test_flagged = detect.flag_inverters(test_ratio, best_n, best_t)
        test_detect_metrics = detect.score_detection(test_flagged, test_fault_windows)

        results["detection"] = {
            "n_consecutive_days": best_n,
            "ratio_threshold": best_t,
            "validation": val_detect_metrics,
            "test": test_detect_metrics,
        }

        self.stdout.write("Ablation: re-running detection with a rule-based (B2) expected-power baseline...")
        results["ablation_rule_based_vs_ml"] = self._rule_based_ablation(
            val_faulted, test_faulted, val_fault_windows, test_fault_windows,
            k_dict, gamma_dict, best_n, best_t, val_detect_metrics, test_detect_metrics,
        )

        self.stdout.write("Training cause classifier on validation faults, testing on test faults...")
        val_normal = self._normal_windows(val_seen, val_fault_windows, settings.RANDOM_SEED)
        test_normal = self._normal_windows(test_for_faults, test_fault_windows, settings.RANDOM_SEED + 1)

        X_train, y_train = classify.build_training_set(val_ratio, val_faulted, val_fault_windows, val_normal)
        X_test, y_test = classify.build_training_set(test_ratio, test_faulted, test_fault_windows, test_normal)
        clf = classify.train_classifier(X_train, y_train, settings.RANDOM_SEED)
        results["cause_classifier"] = classify.evaluate_classifier(clf, X_test, y_test)

        results["generated_at"] = datetime.now(timezone.utc).isoformat()
        settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(settings.ARTIFACTS_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, allow_nan=False)

        ModelRun.objects.create(name="evaluate", metrics_json=results)

        self.stdout.write("Populating fleet dashboard with real (unfaulted) test-period alerts...")
        n_alerts = self._populate_initial_alerts(df, categories, calib, art, best_n, best_t, peak_power)
        self.stdout.write(f"Created/updated {n_alerts} open alerts from real test-period underperformance.")

        self.stdout.write(self.style.SUCCESS("evaluate complete. See artifacts/results.json"))

    def _populate_initial_alerts(self, df, categories, calib, art, n_days, threshold, peak_power):
        test_all = df[df["SPLIT"] == "test"].copy()
        pred_m1 = lgbm_model.predict(art["m1_model"], test_all, categories, art["best_iteration"])
        test_all["PRED"] = calibration.predict_calibrated(test_all, pred_m1, calib)

        ratio = detect.daily_ratio_table(test_all)
        flagged = detect.flag_inverters(ratio, n_days, threshold)

        n_alerts = 0
        for inverter_id, rows in flagged.groupby("INVERTER_ID"):
            flagged_rows = rows[rows["FLAGGED"]]
            if len(flagged_rows) == 0:
                continue
            inverter = Inverter.objects.filter(inverter_id=inverter_id).first()
            if inverter is None:
                continue

            deficit_kwh = max(float((flagged_rows["expected_energy"] - flagged_rows["actual_energy"]).sum()) * 0.25, 0.0)
            rupees_lost = deficit_kwh * settings.DEFAULT_TARIFF_PER_KWH
            fault_type = _guess_fault_type(flagged_rows["ratio"].tolist())
            start_day = int(flagged_rows["DAY_INDEX"].min())
            latest_ratio = float(flagged_rows["ratio"].iloc[-1])

            Alert.objects.update_or_create(
                inverter=inverter,
                status="open",
                defaults={
                    "fault_type": fault_type,
                    "likely_cause": fault_type,
                    "start_day": start_day,
                    "detected_day": start_day,
                    "ratio": latest_ratio,
                    "energy_deficit_kwh": deficit_kwh,
                    "rupees_lost": rupees_lost,
                },
            )
            n_alerts += 1
        return n_alerts

    def _rule_based_ablation(self, val_faulted, test_faulted, val_fault_windows, test_fault_windows,
                              k_dict, gamma_dict, ml_n, ml_t, ml_val_metrics, ml_test_metrics):
        val_rb = val_faulted.copy()
        val_rb["PRED"] = baselines.predict_b2(val_rb, k_dict, gamma_dict)
        test_rb = test_faulted.copy()
        test_rb["PRED"] = baselines.predict_b2(test_rb, k_dict, gamma_dict)

        val_ratio_rb = detect.daily_ratio_table(val_rb)
        test_ratio_rb = detect.daily_ratio_table(test_rb)

        rb_n, rb_t, rb_val_metrics = detect.tune_threshold(val_ratio_rb, val_fault_windows)
        rb_test_flagged = detect.flag_inverters(test_ratio_rb, rb_n, rb_t)
        rb_test_metrics = detect.score_detection(rb_test_flagged, test_fault_windows)

        return {
            "description": (
                "Detection re-run on the identical injected faults using B2 (irradiance rule + "
                "temperature derating) as the 'expected power' baseline, matching the physics-model "
                "approach of US20160190984A1 claim 3, instead of the tuned ML model (M1+cal). Each "
                "baseline gets its own independently tuned N-consecutive-days/ratio-threshold."
            ),
            "rule_based_b2": {
                "n_consecutive_days": rb_n,
                "ratio_threshold": rb_t,
                "validation": rb_val_metrics,
                "test": rb_test_metrics,
            },
            "ml_m1_cal": {
                "n_consecutive_days": ml_n,
                "ratio_threshold": ml_t,
                "validation": ml_val_metrics,
                "test": ml_test_metrics,
            },
        }

    def _inject_many(self, df, day_range, seed, n_per_type=5):
        rng = np.random.default_rng(seed)
        faults = inject.generate_synthetic_faults(df, day_range, n_per_type, seed)
        working = df.copy()
        windows = []
        for inverter_id, fault_type, start_day in faults:
            working, info = inject.inject_fault(working, inverter_id, fault_type, start_day, rng)
            if info is not None:
                info["end_day"] = min(day_range[1], start_day + 5)
                windows.append(info)
        return working, windows

    def _normal_windows(self, df, fault_windows, seed, n_normal=10):
        rng = np.random.default_rng(seed + 100)
        faulted_inverters = {f["inverter_id"] for f in fault_windows}
        candidates = [i for i in df["INVERTER_ID"].unique() if i not in faulted_inverters]
        if not candidates:
            return []
        chosen = rng.choice(candidates, size=min(n_normal, len(candidates)), replace=False)
        days = sorted(df["DAY_INDEX"].unique())
        start_day = days[0] if days else 1
        return [(inverter_id, start_day) for inverter_id in chosen]
