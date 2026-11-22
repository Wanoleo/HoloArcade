@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [HoloArcade] .venv not found. Run: python -m venv .venv ^&^& .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
