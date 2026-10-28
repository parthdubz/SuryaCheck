# SuryaCheck — the solution, explained

This document exists so every teammate can explain, from first principles, what
SuryaCheck does and why each technical choice was made. Every number in this
document is read directly from `artifacts/results.json` and
`artifacts/m1_best_params.json` after running the real pipeline — nothing here is
invented or estimated. If you re-run `train_models`/`evaluate`, your own numbers
will differ slightly (LightGBM and the fault injection use fixed seeds, but small
floating-point and library-version differences are normal) — that's expected and
fine to say out loud to judges.

---

## 1. The one-paragraph pitch

Rooftop-solar installers can't check every inverter on every site every day.
SuryaCheck watches each inverter's actual power output against how much it
*should* be producing given the sunlight and temperature it's seeing right now,
and turns the gap into three things an installer actually acts on: a ranked list
of which inverters are losing the most money, a plain-language guess at *why*
(dust / step fault / outage), and an estimate in ₹ of what that's costing this
week.

---

## 2. The problem, and why it's harder than it sounds

- We have **34 days** of 15-minute readings from **two real solar plants** in
  India (Kaggle dataset, Ani Kannal). That's not much history to learn "normal"
  behaviour from, and it's also not much time to see faults happen naturally.
- **No fault ever actually happened in this data that we know of.** Nobody
  labelled "this inverter was dusty on this day." So we cannot train a model to
  recognise real faults directly — there's nothing to learn *from*. Section 9
  explains how we work around this.
- **Each plant has exactly one weather sensor**, shared by every inverter at
  that plant. So if Inverter A is sitting in shade and Inverter B isn't, our
  only irradiation reading comes from one sensor that can't see that
  difference. This is a real limitation, not a bug — it's called out in the
  app's Limits page too.
- Two **data bugs** sit inside the raw CSVs themselves (see Section 3) that, if
  missed, would silently wreck every model downstream.

---

## 3. The data, and the two real bugs in it

Four CSVs, loaded by [`core/ml/features.py`](../core/ml/features.py):

| File | Columns |
|---|---|
| `Plant_1_Generation_Data.csv`, `Plant_2_Generation_Data.csv` | `DATE_TIME, PLANT_ID, SOURCE_KEY` (inverter id), `DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD` |
| `Plant_1_Weather_Sensor_Data.csv`, `Plant_2_Weather_Sensor_Data.csv` | `DATE_TIME, PLANT_ID, SOURCE_KEY` (sensor id), `AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, IRRADIATION` |

**Bug 1 — the two plants write `DATE_TIME` in different formats.** Plant 1's
generation file writes `15-05-2020 00:00` (day-month-year). Plant 2 writes
`2020-05-15 00:00:00` (year-month-day). If you parse both the same way, roughly
half the timestamps silently become the wrong date. `features.py`'s
`PLANT_FILES` dict stores a separate `date_format` string per plant so each
file is parsed correctly before anything else happens.

**Bug 2 — `DC_POWER` is not a real physical signal.** Plant 1's `DC_POWER`
values run roughly 10× higher than Plant 2's, for the same AC output — a
metering/unit artifact of how each plant's loggers record it, not a real
difference in the panels. A model fed this column would partly learn "which
plant is this row from," which is not what we want. **We never use it as a
feature at all** (see Section 5).

---

## 4. Supervised learning, from scratch

Everything in Sections 7–11 is a form of **supervised learning**: you show the
computer many examples of `(inputs) → (correct answer)`, and it learns a
function that predicts the answer from the inputs alone, so that later it can
guess the answer for new inputs it's never seen.

- **Feature** — one input number the model is allowed to look at (e.g. today's
  sunlight reading, `IRRADIATION`).
- **Label / target** — the correct answer the model is trying to predict. Here
  it's always `AC_POWER`, the inverter's actual AC power output in kW.
- **Regression** — supervised learning where the label is a number on a
  continuous scale (kW of power), as opposed to **classification**, where the
  label is one of a fixed set of categories (Section 11 uses this — "dust" vs
  "fault" vs "outage" vs "normal").
- **Training set** — the examples the model is allowed to learn from.
- **Validation set** — examples held back from training, used only to choose
  between candidate models/settings (Section 8's tuning uses this).
- **Test set** — examples touched by nobody until the very end, used once to
  report how good the final model actually is.
- **Overfitting** — when a model memorises quirks of the training examples
  instead of learning the real underlying pattern, so it looks great on
  training data but does badly on new data. This is *why* validation and test
  sets must be kept separate from training.
- **Generalization** — how well a model's learned pattern holds up on data it
  wasn't trained on. Sections 6 and 7 specifically stress-test this with data
  the model has never seen in three different ways.

---

## 5. The leakage rule — and why it's enforced, not just agreed on

**Data leakage** is when a feature you feed the model secretly contains
information that only exists *because* the label already happened — so the
model looks accurate during testing but is actually cheating, and will fail in
the real world where that information isn't available yet.

Three columns in the raw data leak the answer:

- `DAILY_YIELD`, `TOTAL_YIELD` — these are *cumulative energy counters*, built
  by continuously adding up the exact `AC_POWER` values we're trying to
  predict. Using them as a feature is close to handing the model the answer.
- `DC_POWER` — leaks *which plant* the row is from (Bug 2 above), which isn't
  a real causal signal.

`core/ml/features.py` hardcodes these three in `LEAKAGE_COLUMNS`, and
`assert_no_leakage()` runs on every call to `build_dataset()`:

```python
def assert_no_leakage(feature_columns):
    leaking = set(feature_columns) & set(LEAKAGE_COLUMNS)
    assert not leaking, f"Leakage columns present in features: {leaking}"
```

This is a Python `assert` — if anyone ever adds one of those three columns to
`FEATURE_COLUMNS`, the program **crashes immediately** rather than silently
producing a model that looks good and lies. That's the difference between "we
promise not to leak" and "the code physically won't let us."

The seven features actually used:
`IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`, `TOD_SIN`, `TOD_COS`
(sine/cosine of time-of-day — this encodes "6pm and 6am are both near sunrise/
sunset" in a way a single number like "18:00" can't), `DAY_INDEX` (which day
of the 34 this is), and `INVERTER_ID` (which physical inverter, as a
category).

---

## 6. The three splits — and why each one exists

| Split | Days | Purpose |
|---|---|---|
| Train | 1–22 | What every model learns from |
| Validation | 23–28 | What Optuna tunes LightGBM against, and what the fault-detection threshold is tuned against |
| Test | 29–34 | Touched once, to report final numbers |

On top of the day-based split, two more held-out groups test **generalization**
specifically:

- **Unseen inverters** — 20% of all inverters (8 of 44) are removed from
  training *entirely*, for *every* day. We then ask: can the model predict a
  physical inverter it has literally never seen a single reading from? This
  matters because a real installer's fleet grows — new inverters get added
  after a model is trained.
- **Cross-plant** — B1/B2/M1/M1+cal are **refit from scratch using only Plant
  1's training data**, then tested on Plant 2's test data. This checks whether
  the model learned something about solar physics in general, or just
  memorised Plant 1's specific quirks.

---

## 7. The five models, in increasing complexity

All five predict the same thing — `AC_POWER` right now — using the same seven
features, so their errors are directly comparable. Reported in **kW** (lower is
better) and as **% of that inverter's peak power** (so a small inverter's
error doesn't look artificially huge next to a big one's).

| Model | What it is | Test MAE (kW) | Test MAE (% of peak) | Unseen-inverter MAE (% of peak) | Cross-plant MAE (% of peak) |
|---|---|---|---|---|---|
| **B0** Persistence | "same as this time yesterday" | 193.56 | 15.35% | 15.71% | 16.98% |
| **B1** Irradiance rule | `AC = k × IRRADIATION`, k fit per inverter | 106.26 | 8.50% | 8.41% | 11.11% |
| **B2** Rule + temperature | B1 × (1 − γ·(temp − 25°C)) | 102.69 | 8.21% | 7.75% | 11.10% |
| **M1** LightGBM | tuned gradient-boosted trees | **80.58** | **6.42%** | **5.28%** | **10.08%** |
| **M1+cal** | M1 + per-inverter correction | 84.84 | 6.76% | 5.28% | 10.08% |

*(numbers pulled directly from `artifacts/results.json`, `model_comparison`)*

### B0 — Persistence baseline
Not a learned model at all — it has no parameters. For a given inverter and
time of day, it just predicts "whatever that same inverter produced at that
same time yesterday" (`core/ml/baselines.py::predict_persistence`). It exists
so every other model has to prove it's better than doing nothing clever. It's
the worst performer here (193.56 kW MAE) — which is the expected and correct
outcome: it ignores today's actual weather entirely.

### B1 — Irradiance rule
A **linear regression through the origin**, fit *per inverter*: solar power
should scale roughly linearly with sunlight, so `AC_POWER ≈ k × IRRADIATION`.
`k` is fit by least squares (`fit_b1` in `baselines.py`) separately for each
inverter, because different inverters/panel strings have different capacities.
This is a real, physics-motivated model — and it already beats B0 by 45%.

### B2 — Rule + temperature derating
Solar panels lose efficiency as they heat up. B2 multiplies B1's prediction by
a correction term `(1 − γ·(MODULE_TEMPERATURE − 25°C))`, with `γ` (gamma) fit
per inverter the same way. This is still **linear regression**, just with one
more physically-motivated input. It improves on B1 slightly (102.69 vs 106.26
kW) and, notably, generalizes *best to unseen inverters* (7.75%) of all five
models — a physics-based rule doesn't need to have "seen" a specific inverter
before to work on it.

### M1 — LightGBM (gradient-boosted decision trees)
This is the actual machine-learning model, and the one that has to justify
calling this project "ML" at all.

- A **decision tree** is a model that asks a sequence of yes/no questions
  about the features ("is IRRADIATION > 400?", then "is MODULE_TEMPERATURE >
  45?"...) and lands on a predicted number at the end of the path.
- A single tree is usually a weak predictor. **Gradient boosting** builds many
  small trees *one after another*, where each new tree is trained specifically
  to correct the mistakes (residual errors) of all the trees built before it.
  The final prediction is the sum of all trees' contributions.
- **LightGBM** is a fast, well-engineered implementation of gradient-boosted
  trees that also handles `INVERTER_ID` as a true categorical feature — the
  model can split on "is this one of these specific inverters?" directly,
  instead of needing it converted to a number first.
- We trained with `objective: regression_l1`, meaning the trees are optimized
  to minimize **MAE** directly (the L1/absolute-error loss), rather than RMSE.

M1 is the best model on every single scenario in the table — test, unseen
inverters, *and* cross-plant. That's the evidence this project needed: a
tuned ML model beats hand-written physics rules, even on data it generalizes
to.

#### How M1 was tuned — Optuna and hyperparameters
A **hyperparameter** is a setting you choose *before* training that controls
*how* the model learns (as opposed to a parameter, which the model learns
*from* the data) — e.g. how many leaves a tree is allowed to have.

**Optuna** is a hyperparameter-search library. We ran it for **40 trials**
(`train_models` command, `tune_and_train` in `core/ml/lgbm_model.py`): each
trial proposes a set of hyperparameters, trains a LightGBM model on the
training set with those settings, and scores it by **validation MAE**. Optuna
uses a **TPE (Tree-structured Parzen Estimator) sampler** — a Bayesian-style
strategy that looks at which hyperparameter regions scored well so far and
proposes new trials biased toward those regions, instead of guessing blindly
(grid search) or purely randomly.

**Early stopping**: during each trial's training, we keep checking validation
MAE after every added tree, and stop adding more trees as soon as 50
consecutive rounds fail to improve it — this prevents overfitting to the
training set within a single trial.

The actual best hyperparameters found (`artifacts/m1_best_params.json`):

| Hyperparameter | Value | What it controls |
|---|---|---|
| `learning_rate` | 0.0184 | how much each new tree is allowed to correct the previous total |
| `num_leaves` | 91 | how complex/flexible each individual tree can be |
| `min_child_samples` | 8 | smallest group of data points a tree is allowed to split on (prevents trees memorising single rows) |
| `feature_fraction` | 0.959 | fraction of features randomly considered per tree (adds robustness) |
| `bagging_fraction` | 0.994 | fraction of training rows randomly used per tree |
| `lambda_l2` | 4.52 | L2 regularization strength — penalizes overly extreme tree outputs |
| best_iteration | 479 | number of trees actually kept, decided by early stopping |

### M1+cal — per-inverter calibration
**Calibration** here means a small, final linear correction applied *on top
of* M1's output, fit separately per inverter:
`corrected = slope × M1_prediction + intercept` (`core/ml/calibration.py`).
The idea: M1 is trained globally across all inverters, so it might be
systematically a little high or low for one particular inverter's quirks;
calibration nudges it back using that inverter's own training-day residuals.

Interestingly, on this dataset M1+cal is *not* better than raw M1 on test
(84.84 vs 80.58 kW) — a real, honestly-reported result, not a bug. On unseen
inverters and cross-plant it's **identical** to M1 (66.69 kW / 125.58 kW in
both rows), because calibration has no training history for an inverter it's
never seen, so it correctly falls back to "no correction" (`slope=1,
intercept=0`) for those. This is a good thing to be able to explain to judges:
it shows the team understands *why* their own numbers look the way they do,
rather than just reporting whichever number is lowest.

---

## 8. (Covered inside Section 7 — LightGBM + Optuna tuning details are above, to keep each model's explanation in one place.)

---

## 9. Synthetic faults — because there's no real ground truth

Since no real fault is labelled anywhere in this dataset (Section 2), we can't
train or test fault *detection* directly. Instead, `core/ml/inject.py`
deliberately injects three fake-but-realistic fault patterns into the
validation and test periods only (never into training), with a fixed random
seed so results are reproducible:

- **Soiling ramp** (dust build-up): `output × (1 − 0.004 × days_since_start)`
  — a slow, worsening decline.
- **Step fault**: `output × 0.7` starting from a randomly chosen daylight hour
  onward — a sudden, persistent drop (e.g. a damaged string).
- **Outage**: `output = 0` for a random 2–6 daylight hours — total,
  temporary loss (e.g. tripped breaker).

This is **synthetic data** — invented, not observed — and that's stated
plainly everywhere it matters (the app's Results and Limits pages, and here).
It lets us *measure* detection and classification performance against a known
answer, which we otherwise couldn't do at all. It does **not** prove the
system catches every kind of real-world fault; it proves the system catches
*these three specific, clearly-defined patterns*.

---

## 10. The detection rule — and its real, tuned numbers

For each inverter-day, we compute a **daily ratio**:

```
ratio = (actual energy that day) / (expected energy that day, per M1+cal)
```

(`core/ml/detect.py::daily_ratio_table`). An inverter is **flagged** if its
ratio stays below a threshold `T` for `N` consecutive days
(`flag_inverters`).

`N` and `T` are **tuned on validation faults only** — a small grid search over
`N ∈ {1,2,3,4,5}` and `T ∈ {0.50, 0.55, ..., 0.95}`, picking whichever
combination gets the best **F1 score** (defined below), with ties broken by
fewer false alarms (`tune_threshold`).

- **Precision** = (flags that were real faults) ÷ (all flags raised). High
  precision = few false alarms.
- **Recall** = (real faults that got flagged) ÷ (all real faults that
  existed). High recall = few faults missed.
- **F1 score** = the harmonic mean of precision and recall — a single number
  that penalizes a system for being extreme in either direction.
- **False alarms per 100 inverter-days** = a normalized false-positive rate,
  so it's comparable across datasets of different sizes.

**Actual tuned result**: `N = 1` consecutive day, `T = 0.90` ratio threshold.

| | Validation | Test |
|---|---|---|
| Precision | 0.395 | 0.409 |
| Recall | 0.472 | 0.429 |
| False alarms / 100 inverter-days | 12.04 | 12.04 |
| Median days to detect | 0.0 | 0.0 |
| True positives | 17 | 18 |
| False positives | 26 | 26 |
| False negatives | 19 | 24 |

Read honestly: this catches less than half of injected faults, and roughly 4
in 10 flags are false alarms. That's a real, modest result for a 34-day
dataset with a tight time budget — not something to oversell to judges, but
something the team can explain clearly: *why* detection is hard here (short
history, no per-inverter weather, aggressive threshold chosen by the F1
tradeoff) is a legitimate answer to "why isn't this higher?"

---

## 11. The cause classifier — Random Forest

Once something is flagged, we want a guess at *why*. This is
**classification**, not regression: the output is one of four categories
(`"dust"`, `"fault"`, `"outage"`, `"normal"`), not a number.

**Features** computed per flagged window (`core/ml/classify.py::window_features`):
slope of the daily ratio over the window (a steady decline looks like dust),
the single biggest day-to-day drop in ratio (a sudden drop looks like a step
fault), and the fraction of daylight time-slots with zero output (a long zero
streak looks like an outage).

A **Random Forest** is an ensemble of many decision trees (Section 7
explained a single tree), each trained on a random subset of the training
examples and a random subset of features, with the final prediction being
whichever class most trees voted for ("majority vote"). Randomizing what each
tree sees makes the ensemble much less likely to overfit than any single tree
would be.

**Actual result** — trained on validation faults, tested on test-period faults,
accuracy **0.48** (48%):

Confusion matrix (rows = true label, columns = predicted; order:
dust, fault, normal, outage):

| true \\ predicted | dust | fault | normal | outage |
|---|---|---|---|---|
| **dust** | 0 | 4 | 1 | 0 |
| **fault** | 0 | 1 | 4 | 0 |
| **normal** | 0 | 1 | 9 | 0 |
| **outage** | 0 | 1 | 2 | **2** |

This is a genuinely weak classifier — it never once correctly identifies
"dust," and "normal" soaks up a lot of the confusion. Reported honestly, not
spun: with ~15 training examples per class, there simply isn't enough data for
a 4-class classifier to learn reliable patterns yet. The *process* (train on
validation faults, test on held-out test faults, report the real confusion
matrix) is what's defensible here — not the current accuracy number.

---

## 12. Django app architecture

```
Browser ──► Django views (core/views.py) ──► core/ml/* (same code as evaluate.py)
                     │
                     ├─ Home           – upload CSVs, kick off a batch run
                     ├─ Fleet Dashboard – Alerts ranked by ₹ lost (editable tariff)
                     ├─ Inverter Detail – Plotly chart: actual vs expected ± confidence band
                     ├─ Break It Live   – inject a fault into real data, watch it get caught
                     ├─ Results         – renders artifacts/results.json directly
                     └─ Limits          – the honest caveats, in one place
                     │
                     ▼
            SQLite (core/models.py): Plant, Inverter, Reading,
            Prediction, Alert, Feedback, ModelRun
```

Everything the web pages show is computed by the *same* functions in
`core/ml/` that `prepare_data` / `train_models` / `evaluate` used — there's no
separate "demo-only" code path that could silently diverge from what was
actually measured.

---

## 13. Glossary (alphabetical)

- **Calibration** — a post-hoc correction applied to a model's output to fix
  systematic bias, fit per-group (here, per inverter) on that group's own
  data.
- **Classification** — supervised learning where the label is one of a fixed
  set of categories.
- **Confusion matrix** — a table showing, for each true class, how many
  examples were predicted into each class (correct predictions sit on the
  diagonal).
- **Data leakage** — when a feature secretly contains information derived
  from the very label being predicted, making a model look accurate while
  actually being uninformative in the real world.
- **Decision tree** — a model that makes a prediction by following a sequence
  of yes/no questions about the input features.
- **Early stopping** — halting training once validation performance stops
  improving, to avoid overfitting.
- **Ensemble** — a model built from many smaller models combined (e.g. many
  decision trees), usually more accurate and more stable than any one of
  them alone.
- **F1 score** — the harmonic mean of precision and recall.
- **False alarm / false positive** — a flag raised when nothing was actually
  wrong.
- **Feature** — an input value a model is allowed to use when making a
  prediction.
- **Generalization** — how well a model performs on data it wasn't trained
  on.
- **Gradient boosting** — building an ensemble of trees sequentially, where
  each new tree is trained to correct the errors of all previous trees
  combined.
- **Hyperparameter** — a setting chosen before training that controls how
  learning happens (vs. a parameter, which is learned from data).
- **Label / target** — the correct answer a supervised model is trained to
  predict.
- **LightGBM** — a fast library implementing gradient-boosted decision trees.
- **MAE (Mean Absolute Error)** — the average of |predicted − actual| across
  all examples; in the same units as the thing being predicted (kW here).
- **Overfitting** — learning the training data's noise/quirks instead of the
  real underlying pattern, so performance looks good on training data but
  worse on new data.
- **Precision** — of everything flagged, the fraction that was actually a
  real fault.
- **Random Forest** — an ensemble of decision trees, each trained on a random
  subset of data/features, predicting by majority vote (classification) or
  averaging (regression).
- **Recall** — of all real faults that existed, the fraction that got flagged.
- **Regression** — supervised learning where the label is a continuous
  number.
- **Regularization** — a penalty added during training that discourages a
  model from fitting the training data *too* exactly, to reduce overfitting
  (e.g. `lambda_l2` in Section 7).
- **RMSE (Root Mean Squared Error)** — like MAE, but squares errors before
  averaging (then square-roots the result), so large individual errors are
  penalized more heavily than many small ones.
- **Supervised learning** — training a model on many `(input → correct
  answer)` examples so it can predict the answer for new inputs.
- **Synthetic data** — data deliberately generated/simulated rather than
  observed in the real world; used here because no real fault labels exist.
- **Test set** — data touched only once, at the very end, to report final
  performance.
- **TPE (Tree-structured Parzen Estimator)** — the search strategy Optuna
  used to propose promising hyperparameter combinations based on earlier
  trials' results.
- **Training set** — the data a model is allowed to learn from.
- **Validation set** — data held out from training, used to choose between
  models/settings before touching the test set.

---

## 14. Reproduce every number in this document

```bash
cd suryacheck
python manage.py prepare_data   # rebuilds artifacts/clean_data.parquet + splits
python manage.py train_models   # ~15 min: fits B0/B1/B2, tunes+trains M1, fits M1+cal
python manage.py evaluate       # injects faults, scores all 5 models, tunes
                                 # detection, trains the cause classifier,
                                 # writes artifacts/results.json
```

Then open `artifacts/results.json` and `artifacts/m1_best_params.json`
yourself — every table in this document is copied verbatim from those two
files, generated at `2026-10-25T19:05:24Z` in this run. Re-running will
produce slightly different numbers (fixed seeds control most randomness, but
small floating-point/library differences are normal) — if a judge asks "are
these numbers real," the honest answer is "yes, run `evaluate` yourself and
compare."
