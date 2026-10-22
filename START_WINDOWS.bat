@echo off
title Yoruba Speech Recognition - starting
cd /d "%~dp0"
echo.
echo ============================================================
echo   Yoruba Speech Recognition - starting the application
echo ============================================================
echo.

echo [1/4] Checking that Docker Desktop is running...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo   Docker Desktop is NOT running.
    echo   Open Docker Desktop from the Start menu, wait until it shows
    echo   "Engine running" ^(green^), then double-click this file again.
    echo.
    pause
    exit /b 1
)

echo [2/4] Checking the trained models...
if not exist "models\V4\model.safetensors" (
    echo.
    echo   The folder "models" is missing or incomplete.
    echo   It must contain V1, V2, V3 and V4, each with model.safetensors.
    echo   See HOW_TO_RUN.txt, section "Problems".
    echo.
    pause
    exit /b 1
)

echo [3/4] Building and starting the application.
echo       The FIRST time this downloads about 1.5 GB and takes 5-15 minutes.
echo       Later starts take a few seconds.
echo.
docker compose up -d --build
if errorlevel 1 (
    echo.
    echo   Docker could not start the application. See HOW_TO_RUN.txt, section "Problems".
    echo.
    pause
    exit /b 1
)

echo.
echo [4/4] Waiting for the application to be ready...
set /a tries=0
:wait
set /a tries+=1
timeout /t 5 /nobreak >nul
curl -s -o nul http://localhost:7870/ && goto ready
if %tries% lss 60 goto wait
echo.
echo   The application did not answer after 5 minutes.
echo   Run this command to see what happened:  docker compose logs
echo.
pause
exit /b 1

:ready
echo.
echo ============================================================
echo   READY!  Opening http://localhost:7870 in your browser.
echo   (The first transcription takes about 15 seconds while the
echo    model loads; after that each clip takes a few seconds.)
echo   To stop the application later, double-click STOP_WINDOWS.bat
echo ============================================================
start "" http://localhost:7870
echo.
pause
