import numpy as np
import pandas as pd

LEAKAGE_COLUMNS = ["DAILY_YIELD", "TOTAL_YIELD", "DC_POWER"]
FEATURE_COLUMNS = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "TOD_SIN",
    "TOD_COS",
    "DAY_INDEX",
    "INVERTER_ID",
]
TARGET_COLUMN = "AC_POWER"

PLANT_FILES = {
    4135001: {
        "generation": "Plant_1_Generation_Data.csv",
        "weather": "Plant_1_Weather_Sensor_Data.csv",
        "date_format": "%d-%m-%Y %H:%M",
    },
    4136001: {
        "generation": "Plant_2_Generation_Data.csv",
        "weather": "Plant_2_Weather_Sensor_Data.csv",
        "date_format": "%Y-%m-%d %H:%M:%S",
    },
}
WEATHER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def assert_no_leakage(feature_columns):
    leaking = set(feature_columns) & set(LEAKAGE_COLUMNS)
    assert not leaking, f"Leakage columns present in features: {leaking}"


def _load_plant(data_dir, plant_id, cfg):
    gen = pd.read_csv(data_dir / cfg["generation"])
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], format=cfg["date_format"])
    gen = gen.drop_duplicates(subset=["DATE_TIME", "SOURCE_KEY"])

    weather = pd.read_csv(data_dir / cfg["weather"])
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], format=WEATHER_DATE_FORMAT)
    weather = weather.drop_duplicates(subset=["DATE_TIME", "PLANT_ID"])
    weather = weather[["DATE_TIME", "PLANT_ID", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    merged = gen.merge(weather, on=["DATE_TIME", "PLANT_ID"], how="inner")
    return merged


def _add_derived_columns(df):
    df = df.copy()
    df["INVERTER_ID"] = df["PLANT_ID"].astype(str) + "_" + df["SOURCE_KEY"].astype(str)

    min_date = df.groupby("PLANT_ID")["DATE_TIME"].transform(lambda s: s.dt.normalize().min())
    df["DAY_INDEX"] = (df["DATE_TIME"].dt.normalize() - min_date).dt.days + 1

    minutes_of_day = df["DATE_TIME"].dt.hour * 60 + df["DATE_TIME"].dt.minute
    angle = 2 * np.pi * minutes_of_day / (24 * 60)
    df["TOD_SIN"] = np.sin(angle)
    df["TOD_COS"] = np.cos(angle)
    return df


def build_dataset(data_dir, daylight_threshold):
    frames = [_load_plant(data_dir, plant_id, cfg) for plant_id, cfg in PLANT_FILES.items()]
    df = pd.concat(frames, ignore_index=True)
    df = _add_derived_columns(df)
    df = df.sort_values(["PLANT_ID", "SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)

    daylight = df[df["IRRADIATION"] > daylight_threshold].reset_index(drop=True)

    assert_no_leakage(FEATURE_COLUMNS)
    keep_cols = [
        "DATE_TIME", "PLANT_ID", "SOURCE_KEY", "INVERTER_ID", "DAY_INDEX",
        TARGET_COLUMN, "DC_POWER", "DAILY_YIELD", "TOTAL_YIELD",
    ] + [c for c in FEATURE_COLUMNS if c not in ("INVERTER_ID", "DAY_INDEX")]
    return daylight[keep_cols]


def time_split(df, train_days, val_days, test_days):
    train = df[df["DAY_INDEX"].between(*train_days)].copy()
    val = df[df["DAY_INDEX"].between(*val_days)].copy()
    test = df[df["DAY_INDEX"].between(*test_days)].copy()
    return train, val, test


def unseen_inverter_split(df, fraction, seed):
    inverters = sorted(df["INVERTER_ID"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(inverters)
    n_unseen = max(1, int(len(inverters) * fraction))
    unseen = set(inverters[:n_unseen])
    seen_mask = ~df["INVERTER_ID"].isin(unseen)
    return df[seen_mask].copy(), df[~seen_mask].copy(), unseen


def fit_inverter_categories(train_df):
    return sorted(train_df["INVERTER_ID"].unique())


def apply_inverter_categories(df, categories):
    df = df.copy()
    df["INVERTER_ID"] = pd.Categorical(df["INVERTER_ID"], categories=categories)
    return df


def inverter_peak_power(train_df):
    return train_df.groupby("INVERTER_ID")[TARGET_COLUMN].quantile(0.99).to_dict()
