@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run install.bat first.
    pause
    exit /b 1
)

set "PYTHONUNBUFFERED=1"
set "KMP_DUPLICATE_LIB_OK=TRUE"
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo The application exited with an error. See logs\app.log
    pause
    exit /b 1
)
exit /b 0
