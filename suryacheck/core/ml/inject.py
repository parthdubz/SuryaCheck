import numpy as np

FAULT_TYPES = ["soiling", "step", "outage"]


def _daylight_rows(df, inverter_id, day_index):
    mask = (df["INVERTER_ID"] == inverter_id) & (df["DAY_INDEX"] == day_index)
    return df[mask]


def inject_fault(df, inverter_id, fault_type, start_day, rng):
    df = df.copy()
    df["IS_FAULT"] = False
    inv_mask = df["INVERTER_ID"] == inverter_id

    if fault_type == "soiling":
        day_mask = inv_mask & (df["DAY_INDEX"] >= start_day)
        days_since = df.loc[day_mask, "DAY_INDEX"] - start_day
        multiplier = np.clip(1.0 - 0.004 * days_since, 0.0, 1.0)
        df.loc[day_mask, "AC_POWER"] = df.loc[day_mask, "AC_POWER"] * multiplier
        df.loc[day_mask, "IS_FAULT"] = True
        info = {"type": fault_type, "inverter_id": inverter_id, "start_day": int(start_day),
                "fault_start_time": None, "fault_end_time": None}

    elif fault_type == "step":
        candidates = _daylight_rows(df, inverter_id, start_day)
        if len(candidates) == 0:
            return df, None
        start_row = candidates.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        fault_start_time = start_row["DATE_TIME"]
        day_mask = inv_mask & (df["DATE_TIME"] >= fault_start_time)
        df.loc[day_mask, "AC_POWER"] = df.loc[day_mask, "AC_POWER"] * 0.7
        df.loc[day_mask, "IS_FAULT"] = True
        info = {"type": fault_type, "inverter_id": inverter_id, "start_day": int(start_day),
                "fault_start_time": fault_start_time, "fault_end_time": None}

    elif fault_type == "outage":
        candidates = _daylight_rows(df, inverter_id, start_day)
        if len(candidates) == 0:
            return df, None
        start_row = candidates.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        duration_hours = int(rng.integers(2, 7))
        fault_start_time = start_row["DATE_TIME"]
        fault_end_time = fault_start_time + np.timedelta64(duration_hours, "h")
        day_mask = inv_mask & (df["DATE_TIME"] >= fault_start_time) & (df["DATE_TIME"] < fault_end_time)
        df.loc[day_mask, "AC_POWER"] = 0.0
        df.loc[day_mask, "IS_FAULT"] = True
        info = {"type": fault_type, "inverter_id": inverter_id, "start_day": int(start_day),
                "fault_start_time": fault_start_time, "fault_end_time": fault_end_time}

    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

    return df, info


def generate_synthetic_faults(df, day_range, n_per_type, seed):
    rng = np.random.default_rng(seed)
    inverters = sorted(df["INVERTER_ID"].unique())
    faults = []
    for fault_type in FAULT_TYPES:
        chosen_inverters = rng.choice(inverters, size=min(n_per_type, len(inverters)), replace=False)
        for inverter_id in chosen_inverters:
            start_day = int(rng.integers(day_range[0], day_range[1] + 1))
            faults.append((inverter_id, fault_type, start_day))
    return faults
