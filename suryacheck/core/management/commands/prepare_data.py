import json

from django.conf import settings
from django.core.management.base import BaseCommand

from core.ml import features
from core.models import Inverter, Plant, Reading

PLANT_NAMES = {"4135001": "Plant 1", "4136001": "Plant 2"}


class Command(BaseCommand):
    help = "Load CSVs, build features, split data, save to Parquet and SQLite."

    def handle(self, *args, **options):
        self.stdout.write("Building dataset from CSVs...")
        df = features.build_dataset(settings.DATA_DIR, settings.IRRADIATION_DAYLIGHT_THRESHOLD)

        seen_df, unseen_df, unseen_ids = features.unseen_inverter_split(
            df, settings.UNSEEN_INVERTER_FRACTION, settings.RANDOM_SEED
        )
        df["IS_UNSEEN"] = df["INVERTER_ID"].isin(unseen_ids)

        train, val, test = features.time_split(seen_df, settings.TRAIN_DAYS, settings.VAL_DAYS, settings.TEST_DAYS)
        df["SPLIT"] = None
        df.loc[df["DAY_INDEX"].between(*settings.TRAIN_DAYS), "SPLIT"] = "train"
        df.loc[df["DAY_INDEX"].between(*settings.VAL_DAYS), "SPLIT"] = "val"
        df.loc[df["DAY_INDEX"].between(*settings.TEST_DAYS), "SPLIT"] = "test"

        categories = features.fit_inverter_categories(train)
        peak_power = features.inverter_peak_power(train)

        settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(settings.ARTIFACTS_DIR / "clean_data.parquet", index=False)
        with open(settings.ARTIFACTS_DIR / "inverter_categories.json", "w") as f:
            json.dump(categories, f)
        with open(settings.ARTIFACTS_DIR / "inverter_peak_power.json", "w") as f:
            json.dump(peak_power, f)
        with open(settings.ARTIFACTS_DIR / "unseen_inverters.json", "w") as f:
            json.dump(sorted(unseen_ids), f)

        self.stdout.write(
            f"Rows: total={len(df)} train={len(train)} val={len(val)} test={len(test)} "
            f"unseen_inverters={len(unseen_ids)} ({len(unseen_df)} rows)"
        )

        self._load_sqlite(df, peak_power, unseen_ids)
        self.stdout.write(self.style.SUCCESS("prepare_data complete."))

    def _load_sqlite(self, df, peak_power, unseen_ids):
        Reading.objects.all().delete()
        Inverter.objects.all().delete()
        Plant.objects.all().delete()

        plants = {}
        for plant_id in sorted(df["PLANT_ID"].astype(str).unique()):
            plants[plant_id] = Plant.objects.create(plant_id=plant_id, name=PLANT_NAMES.get(plant_id, plant_id))

        inverters = {}
        inv_info = df.drop_duplicates("INVERTER_ID")[["INVERTER_ID", "PLANT_ID", "SOURCE_KEY"]]
        for _, row in inv_info.iterrows():
            inverter_id = row["INVERTER_ID"]
            inverters[inverter_id] = Inverter.objects.create(
                plant=plants[str(row["PLANT_ID"])],
                source_key=row["SOURCE_KEY"],
                inverter_id=inverter_id,
                peak_power_kw=peak_power.get(inverter_id),
                is_unseen_in_training=inverter_id in set(unseen_ids),
            )

        batch = []
        for _, row in df.iterrows():
            batch.append(
                Reading(
                    inverter=inverters[row["INVERTER_ID"]],
                    timestamp=row["DATE_TIME"],
                    day_index=int(row["DAY_INDEX"]),
                    ac_power=float(row["AC_POWER"]),
                    dc_power=float(row["DC_POWER"]),
                    irradiation=float(row["IRRADIATION"]),
                    module_temperature=float(row["MODULE_TEMPERATURE"]),
                    ambient_temperature=float(row["AMBIENT_TEMPERATURE"]),
                )
            )
            if len(batch) >= 5000:
                Reading.objects.bulk_create(batch)
                batch = []
        if batch:
            Reading.objects.bulk_create(batch)
