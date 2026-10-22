@echo off
title Yoruba Speech Recognition - stopping
cd /d "%~dp0"
echo Stopping the Yoruba Speech Recognition application...
docker compose down
echo.
echo Stopped. Double-click START_WINDOWS.bat to start it again.
echo.
pause
