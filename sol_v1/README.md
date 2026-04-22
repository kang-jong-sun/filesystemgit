# SOL/USDT Futures Trading Bot V1

**전략**: V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% Compound
**백테스트 성과**: +175,456% / MDD 40.6% / PF 2.07 / WR 30.2% / Calmar 4,323 (51개월)

---

## 🎯 전략 개요

### Strategy A: V12.1 추세추종 (75% 할당)
- **Entry**: EMA(9) / SMA(400) 15m cross
- **Filters**: ADX≥22 rising + RSI 30-65 + LR±0.5 + Skip Same + ADX rising 6
- **Monitor**: 24 bars, Delay: 5 bars
- **Pyramiding**: 3-Leg (Leg2 +3%, Leg3 +7.5%, margin × 0.5)
- **Confidence Sigmoid**: k=5, base 0.5, range 1.85 → multiplier 0.5~2.35
- **Session Split**: US (UTC 16-24) aggressive (SL 4.0/TSL 11)
- **Margin Scaling**: Bull 1.5× / Bear 0.5× (BTC 1H EMA200)
- **ATR Defense**: ATR(14) > 1.5 × ATR(50) → margin flat

### Strategy B: Mass Index Reversal (25% 할당)
- **Signal**: Mass Index 25-bar sum > 27 (bulge) → < 26.5 (reversal)
- **Direction**: 반대 방향 진입 (counter-trend)
- **Params**: SL 3% / TA 5% / TSL 8%

### Mutex
- 동시 진입 금지 (한쪽 활성 시 타쪽 무시)
- 자본 pool 공유

### Skip2@4loss
- 4연속 손실 발생 시 → 다음 2거래 자동 스킵
- 5/5 Fold 일관 개선 검증

### 12.5% Compound
- 매 거래 마진 = balance × 12.5%
- V12 slot: balance × 9.375%
- Mass slot: balance × 3.125%
- ⚠ **공격적 설정** (10%는 balanced, 15%는 aggressive)

---

## 📋 PDF 제약 (백테스트 기준)

| 항목 | 값 |
|---|---|
| Init Capital | $5,000 |
| Leverage | 10x |
| Margin Mode | Isolated |
| Fee | 0.05% (taker) |
| Slippage | 0.05% |
| Daily Loss Limit | -2.5% |

---

## 🚀 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. .env 설정
`.env` 파일에 필수 키 확인:
```
BINANCE_API_KEY="..."
BINANCE_API_SECRET="..."
TELEGRAM_BOT_TOKEN="..."
TELEGRAM_CHAT_ID="..."
BINANCE_WS_FUTURES_URL='wss://fstream.binance.com/ws'
```

### 3. Binance Futures 사전 설정
- **Isolated margin** 선택
- **Leverage 10x** 설정 (봇이 자동 설정도 시도)
- **SOL/USDT 선물** 활성화
- **One-Way 모드** 권장 (Hedge 모드도 지원)
- 최소 잔액: $100+ (추천 $500+)

### 4. 실행
```bash
# 직접 실행
python sol_main_v1.py

# 자동 재시작 (권장)
SOL_start.bat
```

---

## 📁 파일 구조

| 파일 | 역할 |
|---|---|
| `sol_main_v1.py` | 메인 async runner (진입점) |
| `sol_core_v1.py` | V12.1 + Mass + Mutex + Skip 로직 |
| `sol_data_v1.py` | SOL 15m + BTC 1h 데이터 + 지표 |
| `sol_executor_v1.py` | Binance 주문 실행 + DB |
| `sol_telegram_v1.py` | 텔레그램 알림 + 명령 |
| `.env` | API 키 (gitignore 필수) |
| `requirements.txt` | Python 패키지 |
| `SOL_start.bat` | Windows 자동 재시작 |
| `state/sol_state.json` | 봇 상태 (재시작 후 복원) |
| `sol_trading_bot.db` | SQLite (entries/trades/daily_stats) |
| `logs/` | 로그 파일 (일별, 1년 자동 삭제) |

---

## 📱 텔레그램 명령

| 명령 | 설명 |
|---|---|
| `/status` | 상태 리포트 (잔액/MDD/PF/WR/Skip 상태) |
| `/balance` | 잔액 조회 |
| `/close` | 현재 포지션 수동 청산 |
| `/stop` | 봇 종료 (포지션 유지) |
| `/help` | 명령 목록 |

---

## 🔔 알림 이벤트

- 🚀 봇 시작 / 🛑 봇 종료
- 📥 ENTRY (V12/MASS, conf score, margin scaling)
- 📤 EXIT (SL/TSL/REV/FC, PnL, ROI)
- ⤴ Pyramid Leg2/3 추가
- ⚠ Skip2@4loss TRIGGER
- 📊 3시간마다 상태 리포트
- ❗ 에러 발생

---

## ⚠️ 중요 주의사항

### 1. 12.5% 복리는 공격적 설정
- **MDD 40.6%** 백테스트 (최악 -40%)
- 실전에서는 슬리피지/스프레드로 더 악화 가능 (MDD 50%+ 예상)
- **보수적 대안**: `sol_core_v1.py`의 `COMPOUND_PCT = 0.10` 수정

### 2. PC와 AWS 동시 실행 금지
- 중복 진입 위험
- 반드시 하나만 실행

### 3. 실시간 모니터링 필수
- 초기 1주일: 텔레그램 확인 매 수 시간마다
- 매일 PnL, Consec losses 체크

### 4. 첫 배포는 소액
- 권장: $100 × 5 계정 (2주 Paper mode)
- 안정 확인 후 Full $5,000

### 5. Skip2@4loss 상태 유지
- 봇 재시작 시 consec_losses, skip_remaining 자동 복원
- State/sol_state.json 삭제 금지

### 6. 실전 Fold 5 리스크
- 2025 H2 같은 choppy regime 재발 시 손실 클러스터
- Mass Index가 자동 방어 but 완전 면역 아님
- MDD 초과 시 수동 중단 고려

---

## 🧪 Paper Trade 검증 절차 (1개월)

1. **Testnet 계정 생성** (https://testnet.binancefuture.com)
2. `.env`의 BINANCE_API_URL을 testnet으로 변경 (코드 수정 필요)
3. Faucet에서 테스트 USDT 받기
4. `python sol_main_v1.py` 실행
5. 1개월간 트레이드 기록 확인:
   - Trade count 예상: 월 5~10건
   - Mutex 작동 (V12/Mass 동시 진입 없음)
   - Skip2@4loss 트리거 여부 (예상: 월 1~2회)
   - 슬리피지 vs 백테스트 추정

---

## 🎓 파라미터 조정 가이드

### 보수적 (Conservative)
```python
# sol_core_v1.py
COMPOUND_PCT = 0.075    # 12.5% → 7.5%
SL_PCT_BASE = 3.0       # 3.8% → 3.0% (tighter)
DAILY_LOSS_LIMIT = -0.015  # -2.5% → -1.5%
```
**예상**: Return -60%, MDD -40%

### 균형 (Balanced, 권장)
```python
COMPOUND_PCT = 0.10     # 백테스트 검증 sweet spot
# 나머지 기본
```
**예상**: +60,948% / MDD 33.9% / Calmar 1,797

### 공격 (Aggressive, 현재 설정)
```python
COMPOUND_PCT = 0.125    # 현재
```
**예상**: +175,456% / MDD 40.6% / Calmar 4,323

### 극공격 (⚠ 비권고)
```python
COMPOUND_PCT = 0.15
```
**예상**: +434,576% / MDD 46.6% (계좌 포기 위험)

---

## 📊 모니터링 지표 (매일 체크)

### Healthy
- Balance 증가 추세
- Daily PnL > -2.5%
- Consec losses ≤ 3
- PF > 1.5

### Warning
- Balance 감소 7일 연속
- Daily PnL -2.5% 도달 (일일 손실 정지)
- Consec losses = 3 (4th loss 예상)
- PF < 1.2

### Critical
- MDD > 30% (초과 시 수동 검토)
- Skip trigger 월 3+ 회 (regime shift 의심)
- WS 연결 불안정
- Exchange 오류 빈발

---

## 🔧 트러블슈팅

### 봇 시작 시 오류
1. `.env` 키 확인
2. Binance API IP 화이트리스트 확인
3. `state/sol_state.json` 파일 권한 확인
4. `pip install -r requirements.txt` 재실행

### 포지션 동기화 문제
- `sol_main_v1.py`가 자동 감지 및 동기화 시도
- 수동 진입 감지 시 텔레그램 알림
- state 파일 삭제 후 재시작 가능 (거래 통계 리셋됨)

### WebSocket 재연결
- 자동 재연결 구현됨 (5초 대기 후 재시도)
- 연속 오류 발생 시 `BINANCE_WS_FUTURES_URL` 확인

---

## 📞 지원

- 로그: `logs/sol_trading_YYYYMMDD.log`
- 거래 기록: `logs/sol_trades.txt`
- DB: `sol_trading_bot.db` (SQLite)
- 상태: `state/sol_state.json`

---

## 📜 출처

- Orchestra v2 R1~R32 (17라운드 자율주행)
- V32.2: V12:Mass 75:25 Mutex 발견
- V33: Skip2@4loss Free Lunch 발견
- V33 Compound: 12.5% sweet spot 결정

**생성일**: 2026-04-20
**버전**: V1.0
**최종 검증**: 2026-04 (51개월 백테스트)
