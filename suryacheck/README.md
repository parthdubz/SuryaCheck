# SuryaCheck

SuryaCheck looks at a fleet of rooftop-solar inverters and ranks them by how many
rupees each one is likely losing due to underperformance, together with a guess at
the likely cause (dust/soiling, a sudden fault, or an outage). The idea is to give
an installer or O&M team a prioritized list of which sites/inverters to visit first,
instead of checking every inverter on a fixed schedule.

This is a student prototype built for a course project (FAI CA2), trained on the
public Kaggle "Solar Power Generation" dataset (two plants, 34 days of 15-minute
data). It is not a production monitoring tool — see "Known limitations" below.

## 1. Setup

Requirements:
- Windows 10/11
- Python 3.13 on PATH (this was built and tested against Python 3.13.1; any recent
  Python 3.10+ should work since no GPU-specific or OS-specific packages are used)
- No GPU needed - everything runs on CPU.

To run it:
1. Double-click `run.bat` (or run it from a terminal: `run.bat`).
2. **First run**: it will
   - create a virtual environment in `.venv`,
   - install everything in `requirements.txt`,
   - apply Django migrations (`manage.py migrate`),
   - since `artifacts/results.json` does not exist yet, run the full pipeline:
     `prepare_data` -> `train_models` -> `evaluate` (train_models takes a few
     minutes on a laptop CPU because it runs an Optuna hyperparameter search),
   - then start the Django dev server and open `http://127.0.0.1:8000/` in your
     browser.
3. **Later runs**: `.venv` and the installed packages already exist, and
   `artifacts/results.json` already exists, so `run.bat` skips straight to
   `migrate` and then starts the server and opens the browser. This is fast
   (a few seconds).
4. The terminal window stays open and shows the Django dev server log. Press
   `Ctrl+C` in that window to stop the server.

If you want to force a complete retrain (new data, changed code, or you just want
fresh numbers), use `retrain.bat` instead - see section 4.

## 2. Where to put the data

Put the four Kaggle CSV files directly inside the `data/` folder (no
subfolders). The pipeline reads these exact filenames (confirmed against
`PLANT_FILES` in `core/ml/features.py`):

```
data/Plant_1_Generation_Data.csv
data/Plant_1_Weather_Sensor_Data.csv
data/Plant_2_Generation_Data.csv
data/Plant_2_Weather_Sensor_Data.csv
```

These are already present in this project's `data/` folder, and the filenames
match exactly what the code expects - no renaming needed.

## 3. What each command does

These are Django management commands (`python manage.py <command>`), and they
must be run in this order because each one depends on artifacts written by the
previous one:

- **`prepare_data`** - reads the 4 CSVs from `data/`, merges generation and
  weather readings, builds time-of-day and day-index features, and splits the
  data into train/val/test and seen/unseen-inverter groups. Writes
  `artifacts/clean_data.parquet` plus JSON sidecars (`inverter_categories.json`,
  `unseen_inverters.json`, `inverter_peak_power.json`) and loads the data into
  SQLite via Django models.
- **`train_models`** - fits the two simple baselines (B1, B2), then tunes and
  trains the main LightGBM model (M1) with an Optuna search (40 trials by
  default), then fits a calibration layer on top of M1 (M1+cal). Saves
  `artifacts/b1_k.joblib`, `artifacts/b2_gamma.joblib`, `artifacts/m1_lightgbm.txt`,
  `artifacts/m1_best_params.json`, `artifacts/m1_calibration.joblib`. This is the
  slow step (a few minutes on CPU).
- **`evaluate`** - scores all five models (B0 persistence baseline, B1, B2, M1,
  M1+cal) on the test split, on inverters the model never saw during training,
  and on a cross-plant scenario (train on Plant 1, test on Plant 2). It then
  injects synthetic faults into held-out days, tunes a fault-detection threshold,
  trains a cause classifier, and writes everything to `artifacts/results.json`.

**`run.bat` vs `retrain.bat`**:
- `run.bat` only runs `prepare_data` / `train_models` / `evaluate` the *first*
  time (i.e. only if `artifacts/results.json` is missing). On every later run it
  just migrates and starts the server, so you don't accidentally wait several
  minutes every time you want to open the app.
- `retrain.bat` always deletes the existing artifacts (`results.json`,
  `clean_data.parquet`, and all the model/joblib/json artifacts) and reruns
  `prepare_data` -> `train_models` -> `evaluate` from scratch, regardless of
  whether artifacts already exist, then starts the server. Use this after
  changing the data, the feature code, or the model code, or whenever you want
  a clean set of numbers. It requires `.venv` to already exist (i.e. you ran
  `run.bat` at least once before); if it doesn't, it tells you to run `run.bat`
  first and exits.

## 4. Reading `artifacts/results.json`

This file is produced by `evaluate` (see `core/management/commands/evaluate.py`)
and has three top-level sections plus a timestamp. We have not run the pipeline
ourselves while writing this document, so no specific numbers are given here -
run `evaluate` (via `run.bat` or `retrain.bat`) and open the file to see your
own results.

### `model_comparison`
A dictionary keyed by model name - `b0`, `b1`, `b2`, `m1`, `m1_cal` - where `b0`
is the naive persistence baseline, `b1`/`b2` are simple physics-informed
baselines, `m1` is the tuned LightGBM model, and `m1_cal` is M1 with a
calibration layer on top. Each model's entry has up to four scenario keys:

- `test` - standard held-out test days, inverters the model has seen before.
- `unseen_inverters` - test days, but for inverters that were excluded from
  training entirely (tests generalization to a "new" inverter).
- `cross_plant` - B1/B2/M1/M1+cal refit on Plant 1 training data only, then
  evaluated on Plant 2 test data (B0 has no training step so it reuses the same
  per-inverter lookup as in `test`). Tuned with a lighter Optuna search (15
  trials instead of 40) purely to keep runtime down - see limitations.
- (for `b0`/`b1`/`b2`/`m1`/`m1_cal`, all four scenarios appear except where
  noted)

Each scenario's value is a small dict:
```
{ "mae": <float>, "rmse": <float>, "mae_pct_of_peak": <float>, "n": <int> }
```
`mae` and `rmse` are in the same units as `AC_POWER` (kW). `mae_pct_of_peak`
expresses the MAE as a percentage of that inverter's estimated peak AC power
(its 99th-percentile output during training), which is more interpretable than
raw kW when comparing inverters of different sizes. `n` is the number of
rows scored.

There is also a `cross_plant_note` string explaining the cross-plant setup in
plain English.

### `detection`
Describes the fault-detection step: whether an inverter's recent output ratio
(actual energy / model-expected energy) has stayed low for long enough to flag
it.
```
{
  "n_consecutive_days": <int>,      // tuned "N" - how many consecutive low-ratio days trigger a flag
  "ratio_threshold": <float>,       // tuned ratio cutoff below which a day counts as "low"
  "validation": { ... },            // metrics on the validation split (used to pick N and the threshold)
  "test": { ... }                   // metrics on the held-out test split (unseen during tuning)
}
```
Both `validation` and `test` have the same shape:
```
{
  "precision": <float>,
  "recall": <float>,
  "false_alarms_per_100_inverter_days": <float>,
  "median_days_to_detect": <float or null>,
  "true_positives": <int>,
  "false_positives": <int>,
  "false_negatives": <int>
}
```
`median_days_to_detect` is how many days after a fault actually started it
typically takes the system to flag it (0 = caught on the first day).

### `cause_classifier`
Metrics for the Random Forest that guesses the likely fault type given a
window of ratio behaviour:
```
{
  "labels": [<string>, ...],         // class labels present, e.g. "dust", "fault", "outage", "normal"
  "confusion_matrix": [[int, ...], ...],  // rows = true label, columns = predicted label, in the order of "labels"
  "accuracy": <float or null>
}
```

### `generated_at`
ISO-8601 UTC timestamp of when `evaluate` was last run.

## 5. Leakage rules (why some columns are off-limits)

`core/ml/features.py` defines `LEAKAGE_COLUMNS = ["DAILY_YIELD", "TOTAL_YIELD", "DC_POWER"]`
and these are never allowed into the model's feature list (`FEATURE_COLUMNS`).
This is enforced in code with an assertion - `assert_no_leakage()` - not just a
comment or convention, so the pipeline will fail loudly (`AssertionError`) if
someone accidentally adds one of these columns to `FEATURE_COLUMNS` later.

Why these three are excluded:
- **`DAILY_YIELD`** and **`TOTAL_YIELD`** are cumulative energy counters that
  are derived from the same generation process we are trying to predict (they
  are running totals of the inverter's own output). Using them as a feature
  would mean the model is partly "predicting" the target from a transformation
  of itself, which inflates accuracy in a way that would not hold up on new,
  truly unseen data.
- **`DC_POWER`** looks like a reasonable physical signal, but in this dataset
  Plant 1's `DC_POWER` values are roughly 10x Plant 2's for the same AC output.
  That is a metering/unit inconsistency between the two plants' data, not a
  real physical relationship, so letting the model use `DC_POWER` would let it
  learn "which plant is this" rather than "how much power should this inverter
  be producing," which would not generalize.

The model's real feature set is `IRRADIATION`, `MODULE_TEMPERATURE`,
`AMBIENT_TEMPERATURE`, `TOD_SIN`, `TOD_COS`, `DAY_INDEX`, and `INVERTER_ID`.

## 6. 3-minute demo walkthrough (for judges)

1. **Home / Fleet dashboard** - open `http://127.0.0.1:8000/`. Show the ranked
   list of inverters, sorted by estimated rupees lost to underperformance. No
   data upload needed - it's already using the Kaggle data baked into
   `artifacts/`.
2. **Inverter detail** - click into one of the flagged inverters near the top
   of the list. Show its chart: actual output vs. the model's expected output,
   with a confidence band around the prediction.
3. **"Break it live"** - go to the fault-injection page, pick a currently
   healthy inverter, and inject a synthetic fault (e.g. a sudden output drop or
   gradual soiling). Show that after the injection it gets caught by the
   detection logic and climbs the ranked list.
4. **Results page** - open the results/metrics page and show the actual numbers
   from `artifacts/results.json` (model comparison, detection precision/recall,
   cause-classifier accuracy) rendered in the UI.
5. **Limits page** - briefly point at the limitations page so judges see that
   the team is aware of the dataset's constraints (see section 7 below) rather
   than overselling the prototype.

## 7. Known limitations

- Only 34 days of data, from only 2 plants. This is a small, short window and
  results should not be read as "this is how solar inverters behave in
  general."
- Each plant has only one weather sensor, not one per inverter. All inverters
  at a given plant share the same irradiation/temperature reading. This limits
  how finely a detected issue can be attributed to something hyper-local like
  partial shading on a single inverter's panels vs. a panel-level hardware
  issue - the model cannot distinguish "this inverter's own panels are dirty"
  from "something plant-wide changed," because its only weather input is
  shared across the whole plant.
- All faults used for training/evaluating detection and the cause classifier
  are synthetically injected by the pipeline (`core/ml/inject.py`). There is no
  real fault ground truth in the original Kaggle dataset, so detection/
  classification numbers reflect performance on synthetic faults, not
  confirmed real-world failures.
- The Kaggle dataset's listed license is "© Original Authors" - this project
  (and its outputs) should be treated as academic/coursework use only, not as
  a basis for a commercial product or deployment decision.
- The `cross_plant` scenario in `model_comparison` uses a lighter Optuna search
  (15 trials) instead of the 40 trials used for the main M1 model, purely to
  keep total runtime within a reasonable budget for a student project. Its
  numbers are therefore not directly comparable in rigor to the main model's.
- Training is CPU-only; no GPU is used or required anywhere in this project.

## 8. Final checklist

Use this to confirm the whole pipeline actually worked, end to end:

| Step | Command / action | What proves it worked |
|---|---|---|
| Migrate | `python manage.py migrate` | Terminal shows `Applying ... OK` lines for each migration, no errors, and `db.sqlite3` exists in the project root. |
| Prepare data | `python manage.py prepare_data` | Command exits without error; `artifacts/clean_data.parquet` and the sidecar JSON files (`inverter_categories.json`, `unseen_inverters.json`, `inverter_peak_power.json`) now exist. |
| Train models | `python manage.py train_models` | Command exits without error (after the Optuna trials finish); `artifacts/b1_k.joblib`, `artifacts/b2_gamma.joblib`, `artifacts/m1_lightgbm.txt`, `artifacts/m1_best_params.json`, `artifacts/m1_calibration.joblib` now exist. |
| Evaluate | `python manage.py evaluate` | Command prints `evaluate complete. See artifacts/results.json` and that file now exists and is valid JSON with `model_comparison`, `detection`, and `cause_classifier` keys. |
| Start server | `python manage.py runserver 127.0.0.1:8000` (done automatically by `run.bat`/`retrain.bat`) | Terminal shows `Starting development server at http://127.0.0.1:8000/` with no traceback. |
| Open Home / Fleet dashboard | Browser navigates to `http://127.0.0.1:8000/` | Page loads and shows a ranked list of inverters (not a Django error page). |
| Open inverter detail page | Click an inverter from the fleet list | A chart renders showing actual vs. expected output with a confidence band for that specific inverter. |
| Open "Break it live" page | Navigate to the fault-injection page | The page loads, lets you pick an inverter and a fault type, and running it visibly changes that inverter's status/rank. |
| Open Results page | Navigate to the results/metrics page | The page shows numbers matching what's in `artifacts/results.json` (model comparison table, detection precision/recall, classifier accuracy). |
| Open Limits page | Navigate to the limitations page | The page loads and lists the known limitations (matching section 7 above). |
