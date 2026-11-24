@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%" || exit /b 1

set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" -c "import psutil, pywinauto, pyautogui, PIL" >nul 2>nul
if errorlevel 1 (
    echo Missing dependencies. Run: python -m pip install -r requirements.txt -r requirements-desktop.txt
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Usage: start-agent.bat "Describe the authorized task"
    pause
    exit /b 2
)

"%PYTHON%" -m agent run --dashboard --approval interactive %*
exit /b %errorlevel%
