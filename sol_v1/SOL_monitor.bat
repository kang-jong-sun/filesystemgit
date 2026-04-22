@echo off
chcp 65001 >nul
title SOL V1 Monitor - Real-time Dashboard

:loop
echo ============================================================
echo   SOL V1 콘솔 모니터링 시작...
echo   봇과 별개 창에서 실행 (봇이 먼저 돌고 있어야 함)
echo   Ctrl+C 로 종료
echo ============================================================
D:\filesystem\futures\sol_env\Scripts\python.exe sol_monitor.py
echo.
echo   Monitor exited! Restarting in 5 seconds...
timeout /t 5
goto loop
