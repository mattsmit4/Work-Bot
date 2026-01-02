@echo off
REM ST-Bot Quick Setup for Windows
REM Run this script to set up ST-Bot automatically

echo ================================
echo ST-Bot Windows Setup
echo ================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo Done!
    echo.
) else (
    echo [1/4] Virtual environment already exists
    echo.
)

REM Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Done!
echo.

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Done!
echo.

REM Run migration script
echo [4/4] Running migration script...
python migrate_from_old.py
echo.

echo ================================
echo Setup Complete!
echo ================================
echo.
echo Next steps:
echo 1. Run tests: pytest tests\test_sample.py -v
echo 2. Start app: streamlit run app.py
echo.
echo Virtual environment is ACTIVE (you should see (venv) in prompt)
echo.
pause
