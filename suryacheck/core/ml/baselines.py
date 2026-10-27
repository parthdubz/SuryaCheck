import numpy as np
import pandas as pd


def _global_fallback(values_dict, default):
    if not values_dict:
        return default
    return float(np.mean(list(values_dict.values())))


def build_persistence_lookup(history_df):
    lookup = history_df.copy()
    lookup["TIME_STR"] = lookup["DATE_TIME"].dt.strftime("%H:%M")
    lookup = lookup[["INVERTER_ID", "DAY_INDEX", "TIME_STR", "AC_POWER"]]
    lookup = lookup.rename(columns={"DAY_INDEX": "PREV_DAY_INDEX", "AC_POWER": "PREV_AC_POWER"})
    return lookup.drop_duplicates(subset=["INVERTER_ID", "PREV_DAY_INDEX", "TIME_STR"])


def predict_persistence(df, lookup):
    df = df.copy()
    df["TIME_STR"] = df["DATE_TIME"].dt.strftime("%H:%M")
    df["PREV_DAY_INDEX"] = df["DAY_INDEX"] - 1
    merged = df.merge(lookup, on=["INVERTER_ID", "PREV_DAY_INDEX", "TIME_STR"], how="left")
    fallback = merged.groupby("INVERTER_ID")["AC_POWER"].transform("mean")
    pred = merged["PREV_AC_POWER"].fillna(fallback)
    return pred.to_numpy()


def fit_b1(train_df):
    k = {}
    for inverter_id, g in train_df.groupby("INVERTER_ID"):
        denom = float((g["IRRADIATION"] ** 2).sum())
        k[inverter_id] = float((g["AC_POWER"] * g["IRRADIATION"]).sum() / denom) if denom > 0 else 0.0
    return k


def predict_b1(df, k_dict):
    default_k = _global_fallback(k_dict, 0.0)
    k_series = df["INVERTER_ID"].map(k_dict).fillna(default_k)
    return (k_series.to_numpy() * df["IRRADIATION"].to_numpy())


def fit_b2(train_df, k_dict):
    gamma = {}
    default_k = _global_fallback(k_dict, 0.0)
    for inverter_id, g in train_df.groupby("INVERTER_ID"):
        k = k_dict.get(inverter_id, default_k)
        b1 = k * g["IRRADIATION"].to_numpy()
        valid = b1 > 1e-6
        if valid.sum() < 5:
            gamma[inverter_id] = 0.0
            continue
        x = (g["MODULE_TEMPERATURE"].to_numpy()[valid] - 25.0)
        y = 1.0 - (g["AC_POWER"].to_numpy()[valid] / b1[valid])
        denom = float((x ** 2).sum())
        gamma[inverter_id] = float((x * y).sum() / denom) if denom > 0 else 0.0
    return gamma


def predict_b2(df, k_dict, gamma_dict):
    b1 = predict_b1(df, k_dict)
    default_gamma = _global_fallback(gamma_dict, 0.0)
    gamma_series = df["INVERTER_ID"].map(gamma_dict).fillna(default_gamma).to_numpy()
    correction = 1.0 - gamma_series * (df["MODULE_TEMPERATURE"].to_numpy() - 25.0)
    return b1 * correction
