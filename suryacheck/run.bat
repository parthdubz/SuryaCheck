@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  SuryaCheck - run.bat
echo  Working directory: %cd%
echo ============================================================

rem --- Step 1: virtual environment -----------------------------------
if exist ".venv\Scripts\python.exe" (
    echo [1/6] Virtual environment already exists. Skipping creation.
) else (
    echo [1/6] Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

rem --- Step 2: activate -------------------------------------------------
echo [2/6] Activating virtual environment ...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

rem --- Step 3: install dependencies --------------------------------------
echo [3/6] Installing/checking dependencies from requirements.txt ...
echo        This can take a few minutes on first run. Progress is shown below.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. See the output above for details.
    pause
    exit /b 1
)

rem --- Step 4: migrate ---------------------------------------------------
echo [4/6] Applying Django database migrations ...
python manage.py migrate
if errorlevel 1 (
    echo.
    echo ERROR: "python manage.py migrate" failed. See the output above.
    pause
    exit /b 1
)

rem --- Step 5: run ML pipeline only if results.json is missing -----------
if exist "artifacts\results.json" (
    echo [5/6] artifacts\results.json already exists. Skipping prepare_data/train_models/evaluate.
    echo        (Use retrain.bat if you want to force a full retrain.)
) else (
    echo [5/6] No artifacts\results.json found. Running the ML pipeline for the first time.
    echo        This runs prepare_data, then train_models (a few minutes on CPU), then evaluate.

    echo        -- Running: python manage.py prepare_data
    python manage.py prepare_data
    if errorlevel 1 (
        echo.
        echo ERROR: "python manage.py prepare_data" failed. See the output above.
        pause
        exit /b 1
    )

    echo        -- Running: python manage.py train_models
    python manage.py train_models
    if errorlevel 1 (
        echo.
        echo ERROR: "python manage.py train_models" failed. See the output above.
        pause
        exit /b 1
    )

    echo        -- Running: python manage.py evaluate
    python manage.py evaluate
    if errorlevel 1 (
        echo.
        echo ERROR: "python manage.py evaluate" failed. See the output above.
        pause
        exit /b 1
    )

    echo        Pipeline complete. artifacts\results.json has been created.
)

rem --- Step 6: start server and open browser -----------------------------
echo [6/6] Starting Django dev server at http://127.0.0.1:8000/ ...
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
