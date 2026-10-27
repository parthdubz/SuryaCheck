import numpy as np
import pandas as pd


def daily_ratio_table(df, pred_col="PRED", actual_col="AC_POWER"):
    grouped = df.groupby(["INVERTER_ID", "DAY_INDEX"]).agg(
        actual_energy=(actual_col, "sum"),
        expected_energy=(pred_col, "sum"),
    ).reset_index()
    grouped["expected_energy"] = grouped["expected_energy"].clip(lower=1e-6)
    grouped["ratio"] = grouped["actual_energy"] / grouped["expected_energy"]
    grouped["ratio"] = grouped["ratio"].clip(upper=2.0)
    return grouped.sort_values(["INVERTER_ID", "DAY_INDEX"]).reset_index(drop=True)


def flag_inverters(daily_ratio_df, n_days, threshold):
    df = daily_ratio_df.sort_values(["INVERTER_ID", "DAY_INDEX"]).copy()
    below = (df["ratio"] < threshold).astype(int)
    df["below"] = below
    df["FLAGGED"] = (
        df.groupby("INVERTER_ID")["below"]
        .rolling(window=n_days, min_periods=n_days)
        .sum()
        .reset_index(level=0, drop=True)
        == n_days
    )
    df["FLAGGED"] = df["FLAGGED"].fillna(False)
    return df


def score_detection(flagged_df, fault_windows):
    fault_days = set()
    for f in fault_windows:
        start = f["start_day"]
        end = f.get("end_day", start + 5)
        for d in range(start, end + 1):
            fault_days.add((f["inverter_id"], d))

    flagged_df = flagged_df.copy()
    flagged_df["IS_TRUE_FAULT_DAY"] = flagged_df.apply(
        lambda r: (r["INVERTER_ID"], r["DAY_INDEX"]) in fault_days, axis=1
    )

    tp = int(((flagged_df["FLAGGED"]) & (flagged_df["IS_TRUE_FAULT_DAY"])).sum())
    fp = int(((flagged_df["FLAGGED"]) & (~flagged_df["IS_TRUE_FAULT_DAY"])).sum())
    fn = int(((~flagged_df["FLAGGED"]) & (flagged_df["IS_TRUE_FAULT_DAY"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    n_inverter_days = len(flagged_df)
    false_alarms_per_100 = (fp / n_inverter_days) * 100 if n_inverter_days > 0 else 0.0

    days_to_detect = []
    for f in fault_windows:
        sub = flagged_df[(flagged_df["INVERTER_ID"] == f["inverter_id"]) & (flagged_df["DAY_INDEX"] >= f["start_day"])]
        hit = sub[sub["FLAGGED"]]
        if len(hit) > 0:
            days_to_detect.append(int(hit["DAY_INDEX"].min() - f["start_day"]))

    median_days = float(np.median(days_to_detect)) if days_to_detect else None
    return {
        "precision": precision,
        "recall": recall,
        "false_alarms_per_100_inverter_days": false_alarms_per_100,
        "median_days_to_detect": median_days,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def tune_threshold(daily_ratio_val, fault_windows_val, n_grid=(1, 2, 3, 4, 5), t_grid=None):
    if t_grid is None:
        t_grid = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    best = None
    for n_days in n_grid:
        for threshold in t_grid:
            flagged = flag_inverters(daily_ratio_val, n_days, threshold)
            metrics = score_detection(flagged, fault_windows_val)
            f1 = (
                2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
                if (metrics["precision"] + metrics["recall"]) > 0 else 0.0
            )
            candidate = (f1, -metrics["false_alarms_per_100_inverter_days"], n_days, threshold, metrics)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    _, _, best_n, best_t, best_metrics = best
    return best_n, best_t, best_metrics
