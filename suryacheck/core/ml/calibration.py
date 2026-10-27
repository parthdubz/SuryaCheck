import numpy as np


def fit_calibration(train_df, m1_pred_train):
    df = train_df.copy()
    df["M1_PRED"] = m1_pred_train
    calib = {}
    for inverter_id, g in df.groupby("INVERTER_ID"):
        x = g["M1_PRED"].to_numpy()
        y = g["AC_POWER"].to_numpy()
        if len(x) < 10 or np.var(x) < 1e-6:
            calib[inverter_id] = (1.0, 0.0)
            continue
        slope, intercept = np.polyfit(x, y, 1)
        calib[inverter_id] = (float(slope), float(intercept))
    return calib


def predict_calibrated(df, m1_pred, calib):
    slopes = df["INVERTER_ID"].map(lambda i: calib.get(i, (1.0, 0.0))[0]).to_numpy()
    intercepts = df["INVERTER_ID"].map(lambda i: calib.get(i, (1.0, 0.0))[1]).to_numpy()
    corrected = slopes * m1_pred + intercepts
    return np.clip(corrected, 0, None)
