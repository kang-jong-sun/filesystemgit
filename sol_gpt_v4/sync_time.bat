@echo off
echo 시스템 시간 동기화 중...
w32tm /resync
net time /set /y
echo.
echo 시간 동기화 완료!
echo.
pause