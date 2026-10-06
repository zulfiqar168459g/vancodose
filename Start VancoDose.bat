@echo off
rem VancoDose launcher for Windows. Double-click to start.
rem Installs anything missing (Python, packages, model) and opens VancoDose in your browser.
setlocal
title VancoDose
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-vancodose.ps1" %*
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="0" goto :end
if "%CODE%"=="130" goto :end
echo.
echo VancoDose could not start (code %CODE%). The messages above explain why;
echo technical details are in "%~dp0logs\launcher.log".
if not defined CI pause
:end
endlocal & exit /b %CODE%
