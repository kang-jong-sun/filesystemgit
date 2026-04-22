@echo off
REM SOL V1 AWS 자동 배포 스크립트
REM 로컬 수정 → Git push → AWS 전송 → 재시작

title SOL V1 AWS Deploy
color 0B
echo ============================================================
echo   SOL V1 AWS Deploy (로컬 ^> Git ^> AWS)
echo ============================================================
echo.

REM 1. Git status 확인
echo [1/5] Git 상태 확인...
cd /d D:\filesystem\futures
git status --short sol_v1/
echo.

set /p COMMIT_MSG="커밋 메시지 입력 (빈 값이면 자동): "
if "%COMMIT_MSG%"=="" set COMMIT_MSG=SOL V1 update

REM 2. Git commit + push
echo.
echo [2/5] Git commit + push...
git add sol_v1/
git commit -m "%COMMIT_MSG%"
git push origin main
if errorlevel 1 (
    echo Git push 실패! 중단.
    pause
    exit /b 1
)

REM 3. 수정된 파이썬 파일만 AWS로 전송 (git archive 이용)
echo.
echo [3/5] AWS로 전송...
git archive --format=tar HEAD sol_v1 ^| ssh -i "C:\Users\대표\Downloads\eth-bot-key.pem" -o StrictHostKeyChecking=no ubuntu@18.183.150.105 "cd /home/ubuntu && tar -xf - --overwrite --exclude='sol_v1/.env' --exclude='sol_v1/logs' --exclude='sol_v1/state' --exclude='sol_v1/cache' --exclude='sol_v1/*.db'"

REM 4. AWS에서 문법 검증
echo.
echo [4/5] AWS 문법 검증...
ssh -i "C:\Users\대표\Downloads\eth-bot-key.pem" -o StrictHostKeyChecking=no ubuntu@18.183.150.105 "cd /home/ubuntu/sol_v1 && source sol_env/bin/activate && python -m py_compile sol_*.py && echo 'SYNTAX OK'"
if errorlevel 1 (
    echo 문법 오류 발견! 봇 재시작 안 함.
    pause
    exit /b 1
)

REM 5. 서비스 재시작
echo.
echo [5/5] sol-bot.service 재시작...
ssh -i "C:\Users\대표\Downloads\eth-bot-key.pem" -o StrictHostKeyChecking=no ubuntu@18.183.150.105 "sudo systemctl restart sol-bot.service && sleep 3 && sudo systemctl is-active sol-bot.service"

echo.
echo ============================================================
echo   배포 완료!
echo   - 대시보드: http://18.183.150.105:8081
echo   - 로그 확인: ssh ... && tail -f logs/sol_trading_*.log
echo ============================================================
pause
