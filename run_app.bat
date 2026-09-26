@echo off
title Sadanga Gangsa Rhythm Analysis and Generation System
echo ======================================================================
echo   Sadanga Gangsa Rhythm Analysis and Generation System
echo   Starting local research application in your web browser...
echo ======================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    start http://localhost:8501
    ".venv\Scripts\streamlit.exe" run thesis_system\app.py --server.headless false
) else if exist "thesis_system\.venv\Scripts\streamlit.exe" (
    start http://localhost:8501
    "thesis_system\.venv\Scripts\streamlit.exe" run thesis_system\app.py --server.headless false
) else (
    echo [Notice] Virtual environment not detected at .venv. Attempting global streamlit...
    start http://localhost:8501
    streamlit run thesis_system\app.py --server.headless false
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application stopped with an error. Check the terminal messages above.
    pause
)
