@echo off
setlocal
chcp 65001 >nul
title Industrial AI Quality Inspection System

echo ======================================================================
echo   Industrial AI Quality Inspection and Automated Sorting System
echo ======================================================================
echo.

REM 1. Check if Python is installed
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in system PATH.
    echo Please install Python and enable "Add Python to PATH".
    pause
    exit /b 1
)

REM 2. Check if .venv exists, if not create it
if not exist ".venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment in .venv folder...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully.
) else (
    echo [1/3] Virtual environment .venv already exists.
)

REM 3. Activate virtual environment
echo [2/3] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM 4. Check and install dependencies
if exist "requirements.txt" (
    echo [3/3] Checking dependencies from requirements.txt...
    python -m pip install -r requirements.txt --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo [WARNING] Some dependencies had warnings during verification.
    ) else (
        echo [OK] All dependencies are satisfied and verified.
    )
)

echo.
echo ======================================================================
echo   Launching main application app.py...
echo ======================================================================
echo.

python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error.
    pause
)
