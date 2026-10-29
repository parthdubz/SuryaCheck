import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

FAULT_TO_LABEL = {"soiling": "dust", "step": "fault", "outage": "outage"}
LABELS = ["dust", "fault", "outage", "normal"]
WINDOW_LEN = 6
FEATURE_NAMES = ["ratio_slope", "max_single_day_drop", "frac_zero_output", "clipping_score"]


def _clipping_score(raw_rows):
    if len(raw_rows) < 2:
        return 0.0
    power = raw_rows["AC_POWER"].to_numpy()
    irradiation = raw_rows["IRRADIATION"].to_numpy()
    power_mean, irr_mean = power.mean(), irradiation.mean()
    if power_mean < 1e-6 or irr_mean < 1e-6:
        return 0.0
    power_cv = power.std() / power_mean
    irr_cv = irradiation.std() / irr_mean
    return float(irr_cv / (power_cv + 1e-3))


def window_features(daily_ratio_df, raw_df, inverter_id, start_day, window_len=WINDOW_LEN):
    days = list(range(start_day, start_day + window_len))
    ratio_rows = daily_ratio_df[
        (daily_ratio_df["INVERTER_ID"] == inverter_id) & (daily_ratio_df["DAY_INDEX"].isin(days))
    ].sort_values("DAY_INDEX")

    if len(ratio_rows) >= 2:
        slope = float(np.polyfit(ratio_rows["DAY_INDEX"], ratio_rows["ratio"], 1)[0])
        diffs = np.diff(ratio_rows["ratio"].to_numpy())
        max_drop = float(-diffs.min()) if len(diffs) > 0 else 0.0
    else:
        slope, max_drop = 0.0, 0.0

    raw_rows = raw_df[(raw_df["INVERTER_ID"] == inverter_id) & (raw_df["DAY_INDEX"].isin(days))]
    frac_zero = float((raw_rows["AC_POWER"] < 1e-6).mean()) if len(raw_rows) > 0 else 0.0
    clipping_score = _clipping_score(raw_rows)

    return [slope, max_drop, frac_zero, clipping_score]


def build_training_set(daily_ratio_df, raw_df, fault_windows, normal_windows, window_len=WINDOW_LEN):
    X, y = [], []
    for f in fault_windows:
        X.append(window_features(daily_ratio_df, raw_df, f["inverter_id"], f["start_day"], window_len))
        y.append(FAULT_TO_LABEL[f["type"]])
    for inverter_id, start_day in normal_windows:
        X.append(window_features(daily_ratio_df, raw_df, inverter_id, start_day, window_len))
        y.append("normal")
    return np.array(X), np.array(y)


def train_classifier(X, y, seed):
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    model.fit(X, y)
    return model


def evaluate_classifier(model, X, y):
    pred = model.predict(X)
    labels_present = sorted(set(y) | set(pred))
    cm = confusion_matrix(y, pred, labels=labels_present)
    accuracy = float((pred == y).mean()) if len(y) > 0 else None
    return {
        "labels": labels_present,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
    }
