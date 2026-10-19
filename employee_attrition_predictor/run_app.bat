@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Please follow the README setup instructions first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" main.py
if errorlevel 1 pause
