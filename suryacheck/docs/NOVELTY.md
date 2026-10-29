# Prior art and novelty

This document exists so the team can answer, specifically and honestly, "isn't
this just [existing patent/product] X?" — because the honest answer is: at the
core mechanism level, no, this idea is not new, and pretending otherwise to a
judge who knows the space will cost credibility. What follows is where
SuryaCheck genuinely extends known prior art, and where it doesn't.

## The closest prior art we found

**US Patent Application 20160190984A1**, "System and method for performance
analysis and classification of losses for solar power systems" (AlsoEnergy
Inc., filed 2014-12-29, published 2016-06-30, status: *Abandoned* — it was
never granted). [Full text on Google Patents](https://patents.google.com/patent/US20160190984A1).

It describes a system that:
- Builds an **expected power output** for each PV string/inverter from a
  **physics model**: sun position, measured irradiance, module temperature,
  array characteristics, electrical-loss factors (claim 3).
- Compares that expected output to actual output to flag an underperforming
  string/inverter (claim 1).
- Classifies the reason using **hardcoded signal-shape rules** — snow cover,
  soiling, or inverter clipping. Clipping specifically is defined (claim 9) as
  "actual power output ... substantially constant ... during a period of time
  when irradiance ... is variable."
- Reports the amount of power lost to a user (claim 13).

This is, point for point, the same core idea as SuryaCheck's "expected vs.
actual power → flag → classify → quantify loss" pipeline. **That core
mechanism is not novel** — it's prior art from 2014, and almost certainly
prior art elsewhere too (commercial platforms like PowerFactors, Raptor Maps,
and Sandia's statistical fault-diagnosis work all do variants of the same
thing — see the project's earlier literature comparison).

## What SuryaCheck actually extends, specifically against this patent

| Gap in the patent's 20 claims | What SuryaCheck does instead |
|---|---|
| "Expected output" is always a fixed physics formula — no learned model anywhere in the claims | M1 (LightGBM, Optuna-tuned) is trained and benchmarked *against* the patent's style of physics model (our B1/B2 baselines are effectively a re-derivation of claim 3) — and measurably beats it. See the ablation below. |
| Cause classification is hardcoded if-then signal rules | Our cause classifier is a trained Random Forest on engineered features, not fixed rules — and it also includes the patent's own clipping rule as one of its input features (Section below), making it a strict superset of the patent's rule set. |
| No validation methodology is described anywhere — the patent claims a mechanism, never claims or shows it works | SuryaCheck injects synthetic faults and reports real, measured precision/recall/false-alarm numbers (`artifacts/results.json`) — including an honest report when those numbers are weak. |
| No generalization testing of any kind | SuryaCheck explicitly tests on inverters never seen in training and on a different plant than it was trained on (`unseen_inverters`, `cross_plant` in `results.json`). |
| The "report module" is a one-way report to a user — no feedback mechanism | SuryaCheck's Alert → Feedback (dismiss + reason) flow is a human-in-the-loop label-collection mechanism: every installer dismissal is potential ground truth for a future retrain, which directly addresses the "no real fault labels exist" limitation this project (and the patent) both share. See `core/management/commands/export_feedback_labels.py`. |
| Loss is reported per-string/inverter; no fleet-wide prioritization | SuryaCheck ranks the entire fleet by estimated rupees lost, with an editable tariff, as an actionable worklist rather than a static report. |

## Honest bottom line

Don't claim "nobody has done solar performance monitoring with expected-vs-
actual comparison" — this patent proves that's false, and it's an easy claim
for a judge to puncture. The defensible claim is narrower: **we replaced a
2014-era hardcoded physics/rule-based detection layer with a validated,
measured ML layer, and we added a feedback mechanism and generalization
testing that the prior art doesn't have.** That's smaller than "we invented
this," but it's true, specific, and survives scrutiny.
