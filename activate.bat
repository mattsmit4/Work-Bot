@echo off
REM Quick activation script for ST-Bot virtual environment
REM Double-click this file to activate the virtual environment

echo Activating ST-Bot virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo.
    echo ERROR: Could not activate virtual environment
    echo Make sure you've run setup_windows.bat first
    pause
    exit /b 1
)

echo.
echo Virtual environment activated!
echo You should see (venv) at the start of your prompt
echo.
echo Quick commands:
echo   - Run tests:  pytest tests\test_sample.py -v
echo   - Start app:  streamlit run app.py
echo   - Deactivate: deactivate
echo.

REM Keep the window open in the activated environment
cmd /k
