@echo off
title Launch Sadanga Gangsa Rhythm Analysis & Generation System
echo ======================================================================
echo   Sadanga Gangsa Rhythm Analysis & Generation System - Quick Launcher
echo   Starting local application at http://localhost:8501...
echo ======================================================================
echo.

cd /d "%~dp0"

:: Check if virtual environment exists
if exist ".venv\Scripts\streamlit.exe" (
    start http://localhost:8501
    ".venv\Scripts\streamlit.exe" run thesis_system\app.py --server.headless false
) else if exist "thesis_system\.venv\Scripts\streamlit.exe" (
    start http://localhost:8501
    "thesis_system\.venv\Scripts\streamlit.exe" run thesis_system\app.py --server.headless false
) else (
    echo [Notice] Virtual environment not found at .venv. Attempting system python...
    start http://localhost:8501
    streamlit run thesis_system\app.py --server.headless false
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application stopped.
    pause
)
