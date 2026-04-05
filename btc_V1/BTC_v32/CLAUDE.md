# ETH/USDT 선물 백테스트 — GPU/CUDA 가속 규칙

### 환경
- **GPU**: NVIDIA GeForce RTX 4060 Ti 8GB
- **CUDA Toolkit**: 12.6 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`)
- **Numba**: 0.64.0 (JIT + CUDA 지원)
- **CuPy**: 14.0.1
- **Python**: 3.12

### 백테스트 최적화 시 필수 규칙
1. **대량 파라미터 최적화** (1,000개 이상 조합)는 반드시 **Numba CUDA GPU**로 실행할 것
2. **단건 백테스트**는 **Numba @njit** (CPU JIT)로 실행할 것 (0.001초/건)
3. **실시간 트레이딩 봇**은 기존 Python 그대로
4. GPU 사용 시 **배치 크기 1,500** (8GB VRAM 기준), 초과 시 자동 분할

### 검증된 CUDA 백테스트 엔진 (이 디렉토리)
- `eth_v7_3_numba.py` — Numba JIT 버전 (CPU, 460배 가속, 검증 완료)
- `eth_v7_3_cuda.py` — CUDA GPU 버전 (417 jobs/s, 검증 완료)
- 기존 원본: `eth_v7_3_gpt_collab.py` (수정 안 함)

### 새 백테스트/최적화 스크립트 작성 시
1. `backtest()` 함수는 **Numba @njit** 데코레이터 적용
2. dict.get() 대신 **개별 float 인자**로 전달
3. 문자열/Boolean → **숫자 코드**로 변환 (예: dual_mode='BOTH' → 1.0)
4. 대량 최적화는 `numba.cuda` 커널로 작성, `run_cuda_batch()` 패턴 사용
5. 결과 검증: 기존 Python 버전과 **1:1 비교** 후 사용

### 기존 백테스트 엔진 구조
| 엔진 파일 | 주요 함수 | 사용 스크립트 |
|-----------|----------|-------------|
| `fast_backtest.py` | `run_fast()`, `fast_bt()` | `run_fast_opt.py`, `run_fixed1000_opt.py` |
| `eth_v6_backtest.py` | `v6_backtest()`, `compute_all()` | `eth_v7_fixed1000.py`, `eth_v7_optimize.py` |
| `eth_v7_backtest.py` | `backtest()` | `eth_v7_3_gpt_collab.py` |
| `backtest_engine.py` | 지표 유틸리티 | 여러 스크립트 |

### 성능 비교 (검증 완료)
| 방식 | 1건 속도 | 64,800건 | CPU 부하 |
|------|---------|---------|---------|
| 원본 Python | 0.461초 | ~50분 | 96% |
| Numba JIT | 0.001초 | ~7초 | 96% |
| CUDA GPU | 0.002초/건 | ~2.5분 | ~5% |




# BTC/USDT Futures v32.2 - EMA(100)/EMA(600) Tight-SL Trend System

## 개요
EMA(100)/EMA(600) 크로스오버 기반 BTC/USDT 선물 자동매매 시스템.
6개 독립 엔진으로 교차검증 완료 ($5,000 -> $24,073,329, 70거래, PF 5.8).

## 파일 구조
```
btc_main_v32.py         # 메인 실행 파일 (환경설정, 메인루프, 종료처리)
btc_core_v32.py         # 핵심 트레이딩 로직 (신호판단, SL/TSL/REV 상태머신)
btc_data_v32.py         # 데이터 수집 (Binance API, EMA/ADX/RSI 계산, WebSocket)
btc_executor_v32.py     # 주문 실행 (시장가주문, SL관리, DB/TXT 기록)
btc_monitor_v32.py      # 실시간 모니터링 (30초 간격, 화면 클리어 방식)
btc_telegram_v32.py     # 텔레그램 알림 (비동기 전송, 시작/종료는 즉시 await)
btc_ai_analyzer_v32.py  # AI 분석 (미사용 - main에서 제거됨)
requirements.txt        # 패키지 목록
.env                    # API 키 (직접 생성)
```

## 전략 파라미터
```
SYMBOL            = BTC/USDT
TIMEFRAME         = 30분봉
INITIAL_CAPITAL   = $5,000
FEE_RATE          = 0.04% (진입/청산 각각)

# MA 설정
FAST_MA           = EMA(100)   # pandas ewm(span=100, adjust=False)
SLOW_MA           = EMA(600)   # pandas ewm(span=600, adjust=False)

# 필터
ADX_PERIOD        = 20         # ewm(alpha=1/20) Wilder 방식
ADX_MIN           = 30.0
ADX_RISE_BARS     = 6          # 현재 > 6봉 전
RSI_PERIOD        = 10         # ewm(alpha=1/10)
RSI_RANGE         = 40 ~ 80
EMA_GAP_MIN       = 0.2%       # |Fast-Slow|/Slow
MONITOR_WINDOW    = 24봉       # 크로스 후 진입 허용 기간
SKIP_SAME_DIR     = True       # 동일방향 재진입 스킵

# 리스크
SL_PCT            = 3.0%       # 가격 기준
TA_PCT            = 12.0%      # TSL 활성화 기준 (고가/저가)
TSL_PCT           = 9.0%       # 고점 대비 트레일링 (종가 기준)
MARGIN_PCT        = 35%        # 잔액 대비
LEVERAGE          = 10x
MARGIN_MODE       = ISOLATED
DAILY_LOSS_LIMIT  = -20%
```

## SL/TSL 우선순위 (핵심)
```
TSL 미활성 (ton=False):
  -> SL만 작동 (저가/고가 기준, SL가에 청산)
  -> SL 발동 시 TSL/REV 체크 안함

TSL 활성 (ton=True):
  -> SL 비활성! (SL 체크 건너뜀)
  -> TSL만 작동 (종가 기준)
  -> TSL 미발동 시 REV 체크
```

## 가격 기준
| 항목 | 기준 |
|------|------|
| 진입 | 종가 (close) |
| SL 발동 | 저가/고가 (intrabar) |
| SL 청산가 | SL가 (slp) |
| TA 활성화 | 고가/저가 (intrabar) |
| TSL 고점추적 | 고가/저가 (TSL 활성 후에만) |
| TSL 청산 | 종가 (close) |
| REV 판단 | 종가 (close) |

## SL 주문 방식
바이낸스가 STOP_MARKET을 Algo Order API로 이전하여,
**봇 자체 모니터링 방식**으로 SL 관리:
- 30초마다 가격 체크
- core._check_exit()에서 SL/TSL/REV 조건 판단
- 조건 충족 시 시장가 청산 실행

## 포지션 동기화
30초마다 거래소 포지션과 봇 상태를 동기화:
- 사용자 수동 진입 -> 자동 감지, SL/TSL/REV 관리 시작
- 사용자 수동 청산 -> 자동 감지, 실제 청산가 조회 후 기록
- 방향 전환 -> 기존 청산 + 새 포지션 등록

## 거래 기록
- **DB**: btc_trading_bot.db (entries/trades/daily_stats 테이블)
- **TXT**: logs/btc_trades.txt (누적 로그)
- **source 구분**: BOT (봇 자동) / USER (사용자 수동)
- 진입/청산 모두 기록

## 상태 저장 (재시작 시 복원)
`state/position_state.json`에 매 30초마다 저장:
- TSL 활성 여부, SL가, 고점/저점 추적값
- 진입 시각, Peak ROI
- 거래 통계 (SL/TSL/REV 카운트, 승패)
- 재시작 시 자동 복원 -> TSL 상태 유지

## 헤지모드 지원
시작 시 바이낸스 계정의 포지션 모드 자동 감지:
- One-Way: reduceOnly 파라미터 사용
- Hedge: positionSide (LONG/SHORT) 파라미터 사용

## 모니터링
30초 간격 화면 클리어 + 전체 재출력:
- BTC 현재가, EMA(100), EMA(600), ADX, RSI
- 바이낸스 지갑 잔액, 가용 잔액, 미실현 손익, 총 자산
- 포지션 정보 (방향, 진입가, ROI, SL가, TSL 상태)
- 거래 통계 (SL/TSL/REV, 승률, PF, MDD)
- 최근 거래 내역 (BOT/USER 구분 표시)

## 텔레그램 알림
- 시작/종료: async 즉시 전송 (await)
- 진입/청산: 비동기 큐 전송
- TSL 활성화, TSL 업데이트
- 수동 포지션 감지
- 에러 알림
- 종료 시 큐 flush 후 세션 닫기

## 종료 처리
- Ctrl+C -> 5초 내 강제 종료 (타임아웃)
- 모니터 먼저 중지 (화면 클리어 방지)
- 텔레그램 종료 알림 전송 (5초 타임아웃)
- WebSocket -> Exchange -> DB 순서로 정리
- ccxt/asyncio 경고 메시지 억제
- 열린 포지션 경고 표시

## .env 설정
```env
# Binance API (필수)
BINANCE_API_KEY="your_key"
BINANCE_API_SECRET="your_secret"

# 텔레그램 (필수)
TELEGRAM_BOT_TOKEN="your_token"
TELEGRAM_CHAT_ID="your_chat_id"

# WebSocket URL
BINANCE_WS_FUTURES_URL='wss://fstream.binance.com/ws'
```

## 실행
```bash
pip install -r requirements.txt
python btc_main_v32.py
```

## 검증 수치 (6엔진 교차검증 완료)
| 항목 | 값 |
|------|-----|
| 최종 잔액 | $24,073,329 |
| 수익률 | +481,367% |
| PF | 5.8 |
| MDD | 43.5% |
| 거래 | 70건 (SL:30 TSL:17 REV:23) |
| 승률 | 47.1% (33W/37L) |
| 6엔진 차이 | $0.000000 |
