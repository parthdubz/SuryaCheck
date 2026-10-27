# SuryaCheck

A first-year B.Tech prototype: tells rooftop-solar installers which inverters are
producing less than expected, the likely cause (dust / fault / outage), and the
estimated rupees lost, as a ranked visit list.

The actual project lives in [`suryacheck/`](suryacheck/) — a Django app plus an
ML pipeline (leakage-checked features, B0/B1/B2 baselines, a LightGBM model
tuned with Optuna, per-inverter calibration, synthetic fault injection,
detection tuning, and a cause classifier).

## Quick start

```bash
cd suryacheck
run.bat
```

First run installs dependencies and trains everything from scratch (roughly
15 minutes on a laptop CPU — mostly the Optuna hyperparameter search). Later
runs skip straight to starting the server, since the trained artifacts are
cached under `suryacheck/artifacts/` (not committed to this repo).

See [`suryacheck/README.md`](suryacheck/README.md) for full setup details,
what each management command does, how to read `results.json`, a demo
walkthrough, and known limitations.

## License note

The Kaggle dataset bundled in `suryacheck/data/` is © Original Authors,
academic use only. This repo is for the course project it was built for.
