import json
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.conf import settings
from django.shortcuts import get_object_or_404, redirect, render

from core.ml import calibration, detect, features, inject, registry
from core.models import Alert, Feedback, Inverter, ModelRun, Plant


def _load_results():
    results_path = settings.ARTIFACTS_DIR / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def _tariff_from_request(request):
    raw_tariff = request.GET.get("tariff")
    if raw_tariff is None:
        return settings.DEFAULT_TARIFF_PER_KWH
    try:
        return float(raw_tariff)
    except ValueError:
        return settings.DEFAULT_TARIFF_PER_KWH


def _guess_fault_type(ratios):
    if not ratios:
        return "unknown"
    min_ratio = min(ratios)
    if min_ratio < 0.15:
        return "outage"
    if (ratios[0] - ratios[-1]) > 0.25:
        return "step"
    return "soiling"


def _save_upload_to_temp(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in uploaded_file.chunks():
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def _parse_date_time(series):
    try:
        return pd.to_datetime(series, dayfirst=True)
    except (ValueError, TypeError):
        return pd.to_datetime(series)


def _build_uploaded_dataframe(gen_path, weather_path, plant_id):
    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    gen_df["DATE_TIME"] = _parse_date_time(gen_df["DATE_TIME"])
    weather_df["DATE_TIME"] = _parse_date_time(weather_df["DATE_TIME"])

    gen_df = gen_df.drop_duplicates(subset=["DATE_TIME", "SOURCE_KEY"])
    weather_cols = ["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    weather_df = weather_df[[c for c in weather_cols if c in weather_df.columns]]
    weather_df = weather_df.drop_duplicates(subset=["DATE_TIME"])

    merged = gen_df.merge(weather_df, on="DATE_TIME", how="inner")
    merged["PLANT_ID"] = plant_id
    merged = features._add_derived_columns(merged)
    return merged


def _get_or_create_inverter(plant_id, source_key, inverter_id):
    plant, _ = Plant.objects.get_or_create(plant_id=str(plant_id), defaults={"name": str(plant_id)})
    inverter, _ = Inverter.objects.get_or_create(
        inverter_id=inverter_id,
        defaults={"plant": plant, "source_key": str(source_key)},
    )
    return inverter


def home(request):
    recent_runs = ModelRun.objects.order_by("-created_at")[:10]
    context = {"recent_runs": recent_runs, "artifacts_ready": registry.artifacts_ready()}
    return render(request, "core/home.html", context)


def run_batch(request):
    if request.method != "POST":
        return redirect("home")

    gen_file = request.FILES.get("generation_csv")
    weather_file = request.FILES.get("weather_csv")
    plant_id = request.POST.get("plant_id", "").strip()

    if not gen_file or not weather_file or not plant_id:
        return render(request, "core/run_error.html", {
            "error_message": "Please provide both CSV files and a plant id.",
        })

    if not registry.artifacts_ready():
        return render(request, "core/run_error.html", {
            "error_message": "The batch pipeline has not been trained yet. Run run.bat first.",
        })

    gen_path = None
    weather_path = None
    try:
        gen_path = _save_upload_to_temp(gen_file)
        weather_path = _save_upload_to_temp(weather_file)

        df = _build_uploaded_dataframe(gen_path, weather_path, plant_id)
        df = df[df["IRRADIATION"] > settings.IRRADIATION_DAYLIGHT_THRESHOLD].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No daylight rows found after merging the two CSVs. Check the DATE_TIME columns match.")

        m1_pred = registry.predict_m1(df)
        art = registry.load_artifacts()
        df["PRED"] = calibration.predict_calibrated(df, m1_pred, art["calib"])

        daily_ratio = detect.daily_ratio_table(df, pred_col="PRED")

        results = _load_results()
        n_days = results["detection"]["n_consecutive_days"] if results else 3
        threshold = results["detection"]["ratio_threshold"] if results else 0.7
        flagged = detect.flag_inverters(daily_ratio, n_days, threshold)

        inv_lookup = df.drop_duplicates("INVERTER_ID")[["INVERTER_ID", "PLANT_ID", "SOURCE_KEY"]]
        inv_lookup = inv_lookup.set_index("INVERTER_ID")

        n_alerts = 0
        for inverter_id, rows in flagged.groupby("INVERTER_ID"):
            flagged_rows = rows[rows["FLAGGED"]]
            if len(flagged_rows) == 0:
                continue

            deficit_kwh = float((flagged_rows["expected_energy"] - flagged_rows["actual_energy"]).sum()) * 0.25
            deficit_kwh = max(deficit_kwh, 0.0)
            rupees_lost = deficit_kwh * settings.DEFAULT_TARIFF_PER_KWH
            fault_type = _guess_fault_type(flagged_rows["ratio"].tolist())
            start_day = int(flagged_rows["DAY_INDEX"].min())
            detected_day = int(flagged_rows["DAY_INDEX"].min())
            latest_ratio = float(flagged_rows["ratio"].iloc[-1])

            if inverter_id not in inv_lookup.index:
                continue
            info_row = inv_lookup.loc[inverter_id]
            inverter = _get_or_create_inverter(info_row["PLANT_ID"], info_row["SOURCE_KEY"], inverter_id)

            Alert.objects.update_or_create(
                inverter=inverter,
                status="open",
                defaults={
                    "fault_type": fault_type,
                    "likely_cause": fault_type,
                    "start_day": start_day,
                    "detected_day": detected_day,
                    "ratio": latest_ratio,
                    "energy_deficit_kwh": deficit_kwh,
                    "rupees_lost": rupees_lost,
                },
            )
            n_alerts += 1

        ModelRun.objects.create(
            name="batch_upload",
            metrics_json={
                "plant_id": plant_id,
                "rows_processed": int(len(df)),
                "inverters_flagged": n_alerts,
            },
        )
    except Exception as exc:
        return render(request, "core/run_error.html", {"error_message": str(exc)})
    finally:
        for path in (gen_path, weather_path):
            if path and os.path.exists(path):
                os.remove(path)

    return redirect(f"/fleet/?ran=1")


def fleet_dashboard(request):
    tariff = _tariff_from_request(request)
    ran = request.GET.get("ran") == "1"

    alerts = list(Alert.objects.filter(status="open").select_related("inverter", "inverter__plant"))
    for alert in alerts:
        alert.display_rupees_lost = alert.energy_deficit_kwh * tariff
    alerts.sort(key=lambda a: a.display_rupees_lost, reverse=True)

    context = {
        "alerts": alerts,
        "tariff": tariff,
        "default_tariff": settings.DEFAULT_TARIFF_PER_KWH,
        "ran": ran,
        "artifacts_ready": registry.artifacts_ready(),
    }
    return render(request, "core/fleet.html", context)


def inverter_detail(request, inverter_id):
    if not registry.artifacts_ready():
        return render(request, "core/run_error.html", {
            "error_message": "The batch pipeline has not been trained yet. Run run.bat first.",
        })

    df = registry.load_clean_data()
    subset = df[df["INVERTER_ID"] == inverter_id]
    if len(subset) == 0:
        return render(request, "core/run_error.html", {
            "error_message": f"No data found for inverter {inverter_id}.",
        })

    test_df = subset[subset["SPLIT"] == "test"].sort_values("DATE_TIME")
    val_df = subset[subset["SPLIT"] == "val"]

    art = registry.load_artifacts()
    chart_html = None
    if len(test_df) > 0:
        pred_test = calibration.predict_calibrated(test_df, registry.predict_m1(test_df), art["calib"])

        band_lower = pred_test
        band_upper = pred_test
        if len(val_df) > 0:
            pred_val = calibration.predict_calibrated(val_df, registry.predict_m1(val_df), art["calib"])
            residuals = val_df["AC_POWER"].to_numpy() - pred_val
            low_pct, high_pct = np.percentile(residuals, [10, 90])
            band_lower = pred_test + low_pct
            band_upper = pred_test + high_pct

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df["DATE_TIME"], y=np.clip(band_upper, 0, None),
            line=dict(width=0), showlegend=False, name="band upper",
        ))
        fig.add_trace(go.Scatter(
            x=test_df["DATE_TIME"], y=np.clip(band_lower, 0, None),
            fill="tonexty", line=dict(width=0), fillcolor="rgba(255,165,0,0.2)",
            name="expected range (10th-90th pct of val residuals)",
        ))
        fig.add_trace(go.Scatter(x=test_df["DATE_TIME"], y=pred_test, name="expected (M1+cal)", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=test_df["DATE_TIME"], y=test_df["AC_POWER"], name="actual AC power", line=dict(color="blue")))
        fig.update_layout(
            title=f"{inverter_id} - test period (day {settings.TEST_DAYS[0]} to {settings.TEST_DAYS[1]})",
            xaxis_title="Time", yaxis_title="AC power (kW)", height=450,
        )
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id="chart")

    inverter = Inverter.objects.filter(inverter_id=inverter_id).select_related("plant").first()
    alerts = Alert.objects.filter(inverter__inverter_id=inverter_id).order_by("-created_at") if inverter else []

    context = {
        "inverter_id": inverter_id,
        "inverter": inverter,
        "chart_html": chart_html,
        "alerts": alerts,
    }
    return render(request, "core/inverter_detail.html", context)


def dismiss_alert(request, alert_id):
    if request.method != "POST":
        return redirect("home")

    alert = get_object_or_404(Alert, id=alert_id)
    reason = request.POST.get("reason", "")
    Feedback.objects.create(alert=alert, dismissed=True, reason=reason)
    alert.status = "dismissed"
    alert.save()
    return redirect("inverter_detail", inverter_id=alert.inverter.inverter_id)


def break_it_live(request):
    inverters = Inverter.objects.select_related("plant").order_by("inverter_id")

    if request.method != "POST":
        context = {
            "inverters": inverters,
            "fault_types": inject.FAULT_TYPES,
            "test_days": settings.TEST_DAYS,
        }
        return render(request, "core/break_it_live.html", context)

    if not registry.artifacts_ready():
        return render(request, "core/run_error.html", {
            "error_message": "The batch pipeline has not been trained yet. Run run.bat first.",
        })

    inverter_id = request.POST.get("inverter_id")
    fault_type = request.POST.get("fault_type")
    start_day = int(request.POST.get("start_day"))

    df = registry.load_clean_data()
    subset = df[(df["INVERTER_ID"] == inverter_id) & (df["SPLIT"] == "test")].copy()

    rng = np.random.default_rng(42)
    faulted_df, info = inject.inject_fault(subset, inverter_id, fault_type, start_day, rng)

    if info is None:
        context = {
            "inverters": inverters,
            "fault_types": inject.FAULT_TYPES,
            "test_days": settings.TEST_DAYS,
            "error_message": "Could not inject that fault - no rows found for that day.",
        }
        return render(request, "core/break_it_live.html", context)

    art = registry.load_artifacts()
    faulted_df["PRED"] = calibration.predict_calibrated(faulted_df, registry.predict_m1(faulted_df), art["calib"])

    daily_ratio = detect.daily_ratio_table(faulted_df, pred_col="PRED")
    results = _load_results()
    n_days = results["detection"]["n_consecutive_days"] if results else 3
    threshold = results["detection"]["ratio_threshold"] if results else 0.7
    flagged = detect.flag_inverters(daily_ratio, n_days, threshold)

    own_rows = flagged[(flagged["INVERTER_ID"] == inverter_id) & (flagged["DAY_INDEX"] >= start_day)]
    caught_rows = own_rows[own_rows["FLAGGED"]]
    caught = len(caught_rows) > 0
    detected_day = int(caught_rows["DAY_INDEX"].min()) if caught else None
    days_to_detect = (detected_day - start_day) if caught else None

    deficit_kwh = max(float((caught_rows["expected_energy"] - caught_rows["actual_energy"]).sum()) * 0.25, 0.0)
    rupees_lost = deficit_kwh * settings.DEFAULT_TARIFF_PER_KWH

    existing_losses = list(Alert.objects.filter(status="open").values_list("rupees_lost", flat=True))
    existing_losses.append(rupees_lost)
    existing_losses.sort(reverse=True)
    new_rank = existing_losses.index(rupees_lost) + 1

    context = {
        "inverters": inverters,
        "fault_types": inject.FAULT_TYPES,
        "test_days": settings.TEST_DAYS,
        "ran": True,
        "inverter_id": inverter_id,
        "fault_type": fault_type,
        "start_day": start_day,
        "caught": caught,
        "detected_day": detected_day,
        "days_to_detect": days_to_detect,
        "deficit_kwh": deficit_kwh,
        "rupees_lost": rupees_lost,
        "new_rank": new_rank,
        "fleet_size": len(existing_losses),
    }
    return render(request, "core/break_it_live.html", context)


def results_view(request):
    results = _load_results()
    cm_rows = None
    if results:
        labels = results["cause_classifier"]["labels"]
        matrix = results["cause_classifier"]["confusion_matrix"]
        cm_rows = list(zip(labels, matrix))
    context = {"results": results, "cm_rows": cm_rows}
    return render(request, "core/results.html", context)


def limits_view(request):
    return render(request, "core/limits.html")
