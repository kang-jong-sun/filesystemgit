@echo off
chcp 65001 >nul
title SOL Trading Bot V1 - V12:Mass 75:25 + Skip2@4loss + 12.5% Compound

:loop
echo ============================================================
echo   [%date% %time%] SOL Bot V1 Starting...
echo   Strategy: V12:Mass 75:25 Mutex + Skip2@4loss
echo   Compound: 12.5%% of balance ^| Leverage: 10x
echo ============================================================
D:\filesystem\futures\sol_env\Scripts\python.exe sol_main_v1.py
echo.
echo   [%date% %time%] Bot exited! Restarting in 30 seconds...
echo   Press Ctrl+C to stop auto-restart.
echo.
timeout /t 30
goto loop
