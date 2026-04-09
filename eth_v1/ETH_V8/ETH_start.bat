@echo off
chcp 65001 >nul
title ETH Trading Bot V8 - Auto Restart

:loop
echo ============================================================
echo   [%date% %time%] ETH Bot Starting...
echo ============================================================
python eth_main_v8.py
echo.
echo   [%date% %time%] Bot exited! Restarting in 30 seconds...
echo   Press Ctrl+C to stop auto-restart.
echo.
timeout /t 30
goto loop
