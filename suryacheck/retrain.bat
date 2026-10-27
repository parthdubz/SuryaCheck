@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  SuryaCheck - retrain.bat
echo  Forces a full retrain: prepare_data -> train_models -> evaluate
echo ============================================================

rem --- Make sure the venv already exists (run.bat should have made it) ---
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: .venv was not found. Run run.bat first to set up the environment,
    echo        then run retrain.bat again.
    pause
    exit /b 1
)

echo [1/5] Activating virtual environment ...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [2/5] Removing old artifacts so the pipeline rebuilds everything from scratch ...
if exist "artifacts\results.json" del /f /q "artifacts\results.json"
if exist "artifacts\clean_data.parquet" del /f /q "artifacts\clean_data.parquet"
if exist "artifacts\inverter_categories.json" del /f /q "artifacts\inverter_categories.json"
if exist "artifacts\b1_k.joblib" del /f /q "artifacts\b1_k.joblib"
if exist "artifacts\b2_gamma.joblib" del /f /q "artifacts\b2_gamma.joblib"
if exist "artifacts\m1_calibration.joblib" del /f /q "artifacts\m1_calibration.joblib"
if exist "artifacts\m1_lightgbm.txt" del /f /q "artifacts\m1_lightgbm.txt"
if exist "artifacts\m1_best_params.json" del /f /q "artifacts\m1_best_params.json"
if exist "artifacts\unseen_inverters.json" del /f /q "artifacts\unseen_inverters.json"
if exist "artifacts\inverter_peak_power.json" del /f /q "artifacts\inverter_peak_power.json"
echo        Old artifacts removed (if any existed).

echo [3/5] Running: python manage.py prepare_data
python manage.py prepare_data
if errorlevel 1 (
    echo.
    echo ERROR: "python manage.py prepare_data" failed. See the output above.
    pause
    exit /b 1
)

echo [4/5] Running: python manage.py train_models  (Optuna tuning + LightGBM training, a few minutes on CPU)
python manage.py train_models
if errorlevel 1 (
    echo.
    echo ERROR: "python manage.py train_models" failed. See the output above.
    pause
    exit /b 1
)

echo        Running: python manage.py evaluate
python manage.py evaluate
if errorlevel 1 (
    echo.
    echo ERROR: "python manage.py evaluate" failed. See the output above.
    pause
    exit /b 1
)

echo        Retrain complete. artifacts\results.json has been rebuilt.

echo [5/5] Starting Django dev server at http://127.0.0.1:8000/ ...
echo        Press Ctrl+C in this window to stop the server when you are done.
echo.

echo        Opening browser in 3...
timeout /t 1 /nobreak >nul
echo        Opening browser in 2...
timeout /t 1 /nobreak >nul
echo        Opening browser in 1...
timeout /t 1 /nobreak >nul
start http://127.0.0.1:8000/
echo        Loading SuryaCheck ...
python manage.py runserver 127.0.0.1:8000

echo.
echo Server stopped.
pause
