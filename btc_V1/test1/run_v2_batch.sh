#!/bin/bash
# V2 백테스트 배치 실행기 - 50건씩 처리 후 체크포인트에서 이어하기
cd "D:/filesystem/futures/btc_V1/test1"

TOTAL=772
BATCH=50
DONE=0

while true; do
    echo "=== Batch run (done so far: ~$DONE) ==="
    python -u ai_backtest_v2.py --batch $BATCH 2>&1 | tail -5

    # 체크포인트 확인
    if [ ! -f "v2_checkpoint.pkl" ]; then
        echo "=== COMPLETE (no checkpoint = finished) ==="
        break
    fi

    DONE=$((DONE + BATCH))
    if [ $DONE -ge $TOTAL ]; then
        echo "=== All batches done ==="
        break
    fi

    echo "--- sleeping 2s before next batch ---"
    sleep 2
done

echo "=== DONE ==="
