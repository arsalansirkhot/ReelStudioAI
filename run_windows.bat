@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 from https://www.python.org/downloads/
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo Could not create Python 3.11 environment. Ensure Python 3.11 is installed.
    pause
    exit /b 1
  )
)

echo Installing or updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo Starting Reel Studio AI...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
