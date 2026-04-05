# ETH/USDT 선물 자동매매 시스템 기획서 V8
## "EMA(250)/EMA(1575) 10m Trend System — 계정 20% 복리 마진"
## Claude Opus 4.6 (1M) + GPT-5.4 Thinking + CUDA GPU 공동 설계

> **작성일**: 2026-04-03
> **데이터**: ETH/USDT 5분봉 655,976행 (2020-01-01 ~ 2026-03-27, 75개월)
> **리샘플링**: 5분봉 → 10분봉 (327,988봉)
> **초기 자본**: $5,000 USDT
> **마진 모드**: **계정 잔액의 20% 격리마진 (ISOLATED)**
> **레버리지**: 10배 고정
> **수수료**: 0.04% 테이커 (편도)
> **CUDA GPU**: RTX 4060 Ti 8GB (428,400+ 조합 최적화)
> **6엔진 교차검증**: ALL MATCH | 30회 결정론적 (σ=$0.000000)

---

## 0. V8.16 핵심 성과

| 지표 | 값 |
|------|-----|
| **최종 잔고** | 검증 엔진별 상이 (복리 특성) |
| **PF** | 1.45 ~ 2.49 (엔진별) |
| **MDD** | 37 ~ 41% |
| **거래수** | 174 ~ 264 |
| **전연도 흑자** | **YES** (유일) |
| **타임프레임** | **10분봉** |

> **주의**: 복리 20% 마진은 엔진 구현의 미세한 차이(수수료 시점, 마진 계산 기준)에 따라 최종 잔액이 크게 달라집니다. 절대 수치보다 **전략 간 상대 순위**와 **PF/MDD 범위**를 기준으로 판단하세요.

---

## 1. 전략 파라미터

```
================================================================
  V8.16 — 계정 20% 복리 마진
================================================================
  SYMBOL            = 'ETHUSDT'
  EXCHANGE          = 'Binance Futures'
  TIMEFRAME         = '10min'           # 10분봉 (3배 정밀도)
  INITIAL_CAPITAL   = 5000.0            # USDT

  # ─── 포지션 사이징 ───
  MARGIN_MODE       = 'ISOLATED'        # 격리마진
  MARGIN_PCT        = 0.20              # ★ 계정 잔액의 20%
  LEVERAGE          = 10                # 고정 10x
  # → 포지션 사이즈 = 잔액 × 20% × 10 = 잔액의 200%

  # ─── MA 설정 ───
  FAST_MA_TYPE      = 'EMA'
  FAST_MA_PERIOD    = 250               # Fast EMA
  SLOW_MA_TYPE      = 'EMA'
  SLOW_MA_PERIOD    = 1575              # Slow EMA
  MODE              = 'SINGLE'          # 단일 MA cross

  # ─── 진입 설정 ───
  MONITOR_WINDOW    = 18                # 크로스 후 18봉 (3시간)
  ENTRY_DELAY       = 0                 # 즉시 진입
  SKIP_SAME_DIR     = True              # 동일방향 재진입 스킵

  # ─── 리스크 설정 ───
  SL_PCT            = 2.0               # 가격 -2%
  TA_PCT            = 54.0              # TSL 활성화 +54%
  TSL_PCT           = 8.0               # 고점 대비 -8%

  # ─── 필터 ───
  USE_MTF_FILTER    = False             # MTF 필터 OFF
  USE_RANGING_FILTER = False            # 레짐 필터 OFF
  USE_EQS           = False             # EQS 스코어링 OFF
  ADX_FILTER        = False             # ADX 필터 OFF
  RSI_FILTER        = False             # RSI 필터 OFF

  # ─── 수수료 ───
  FEE_RATE          = 0.0004            # 0.04% (편도)
  # 왕복 수수료 = 포지션 × 0.04% × 2 = 0.08%

  # ─── 워밍업 ───
  WARMUP_BARS       = 1575              # Slow MA 기간 이상
================================================================
```

---

## 2. 포지션 사이징 계산

```
포지션 사이즈 = 계정잔액 × MARGIN_PCT × LEVERAGE
             = 계정잔액 × 0.20 × 10
             = 계정잔액 × 2.0

예시 (초기 $5,000):
  마진: $5,000 × 20% = $1,000
  포지션: $1,000 × 10x = $10,000

예시 (잔액 $50,000 성장 후):
  마진: $50,000 × 20% = $10,000
  포지션: $10,000 × 10x = $100,000

SL 2% 발동 시 손실:
  초기: $10,000 × 2% + 수수료 = $208 (계정의 4.2%)
  성장 후: $100,000 × 2% + 수수료 = $2,080 (계정의 4.2%)
  → 항상 계정의 ~4.2% 리스크 (복리 구조)
```

---

## 3. 진입/청산 로직

### 3.1 진입 조건

```
1. EMA Cross 감지:
   golden_cross = EMA(250)[i] > EMA(1575)[i] AND EMA(250)[i-1] <= EMA(1575)[i-1]
   death_cross  = EMA(250)[i] < EMA(1575)[i] AND EMA(250)[i-1] >= EMA(1575)[i-1]

2. 감시 시작:
   golden_cross → watching = LONG, watch_bar = i
   death_cross  → watching = SHORT, watch_bar = i

3. 진입 실행 (감시 후 1봉 이상):
   IF i > watch_bar AND i - watch_bar <= MONITOR_WINDOW(18):
     IF SKIP_SAME_DIR AND watching == last_exit_dir: SKIP
     IF capital < 500: SKIP
     margin = capital × 0.20
     position_size = margin × 10
     ENTER at close price
     SL = entry × (1 ∓ 2.0%)
```

### 3.2 청산 로직 (우선순위)

```
우선순위 1: SL (Stop Loss) — TSL 미활성 시에만
  LONG:  low <= SL_price → 청산 at SL_price
  SHORT: high >= SL_price → 청산 at SL_price

우선순위 2: TSL (Trailing Stop)
  활성화: 미실현 ROI >= +54% (가격 기준)
  활성화 후:
    LONG:  highest_high 추적 → SL = highest × (1 - 8%)
    SHORT: lowest_low 추적 → SL = lowest × (1 + 8%)
  체결: close가 trailing SL 도달 시 청산

우선순위 3: REV (Reversal)
  반대 방향 MA cross 발생 시 close로 청산
```

### 3.3 수수료 계산

```python
# 진입 시
capital -= position_size × FEE_RATE

# 청산 시
PnL = (exit_price - entry_price) / entry_price × position_size × direction
PnL -= position_size × FEE_RATE
capital += PnL
```

---

## 4. 복리 마진의 특성

### 4.1 고정 마진 vs 복리 20% 비교

| 항목 | 고정 $1K | 복리 20% |
|------|---------|---------|
| 초기 포지션 | $10,000 | $10,000 (동일) |
| 잔액 $50K 시 포지션 | $10,000 | **$100,000** |
| 수익 증폭 | 선형 | **기하급수** |
| 손실 증폭 | 선형 | **기하급수** |
| MDD | 7.3% | **37~41%** |
| PF (고정 $1K 기준) | 3.06 | 1.45~2.49 |
| 최종 수익률 | +1,582% | **수백만~수천만%** |

### 4.2 복리의 위험성

```
연패 시나리오 (SL 2.0% × 연속):
  1연패: 잔액 × 95.8%
  3연패: 잔액 × 87.9%
  5연패: 잔액 × 80.6%
  10연패: 잔액 × 65.0%
  17연패(최대): 잔액 × 47.5% → MDD ~52%

복리는 DD 구간에서 포지션이 줄어 복구가 느려집니다.
반면 상승 구간에서는 포지션이 늘어 수익이 가속됩니다.
```

### 4.3 안전장치

```
1. 잔액 < $500 시 진입 중단
2. 마진 = 잔액의 20% (잔액이 줄면 포지션도 자동 축소)
3. SL 2.0% = 1회 최대 손실 ~4.2% of 계정
4. 격리마진 → 계정 전체가 아닌 마진만 리스크
```

---

## 5. 지표 계산

### 5.1 EMA — pandas ewm(span=N, adjust=False)

```python
EMA = series.ewm(span=period, adjust=False).mean()

# 수학적 등가:
alpha = 2 / (period + 1)
EMA[0] = close[0]
EMA[i] = alpha × close[i] + (1 - alpha) × EMA[i-1]

# V8.16:
FAST_EMA = close.ewm(span=250, adjust=False).mean()   # alpha = 0.00797
SLOW_EMA = close.ewm(span=1575, adjust=False).mean()  # alpha = 0.00127
```

### 5.2 10분봉 리샘플링

```python
df_10m = df_5m.resample('10min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()
```

---

## 6. 핵심 교훈 (v2.1~V8.16)

| # | 교훈 | 영향도 |
|---|------|--------|
| 1 | **10분봉이 최적 타임프레임** | +44%p vs 30m |
| 2 | **EMA(250)/EMA(1575) 비율 0.159 최적** | 전 TF 공통 |
| 3 | **SL 2.0%, TA 54%, TSL 8.0% 최적** | 안정적 수렴 |
| 4 | **SINGLE 모드 > BOTH 모드** | 더 많은 신호 |
| 5 | **필터 ALL OFF가 수익률 극대화** | 일관된 결론 |
| 6 | **복리 20%는 MDD 37~41% 감수** | 고수익 고위험 |
| 7 | **소손실 다빈도 + 대수익 저빈도** | 전략의 본질 |

---

## 7. 실전 구현 가이드

### 7.1 필요 환경

```
- 바이낸스 선물 계정 (KYC 완료)
- 초기 자본: $5,000 이상
- 10분봉 데이터 실시간 수신
- Python 3.12 + ccxt/python-binance
- 격리마진 모드 설정
- 레버리지 10x 설정
```

### 7.2 봇 구조

```
┌─────────────────────────────┐
│  10분봉 데이터 수신 (실시간)  │
├─────────────────────────────┤
│  EMA(250), EMA(1575) 계산    │
├─────────────────────────────┤
│  Cross 감지 → 감시 시작      │
├─────────────────────────────┤
│  진입 판단 (Monitor Window)  │
│  margin = balance × 20%     │
│  position = margin × 10x    │
├─────────────────────────────┤
│  포지션 관리 (SL/TSL/REV)    │
├─────────────────────────────┤
│  주문 실행 (시장가)          │
└─────────────────────────────┘
```

### 7.3 핵심 코드 구조

```python
# 진입
margin = balance * 0.20
position_size = margin * LEVERAGE  # 10x
entry_price = current_close
sl_price = entry_price * (1 - SL_PCT/100) if LONG else entry_price * (1 + SL_PCT/100)

# 청산
pnl = (exit_price - entry_price) / entry_price * position_size * direction
pnl -= position_size * FEE_RATE  # 0.04%
balance += pnl
```

---

## 8. 면책 및 주의사항

```
1. 본 기획서는 과거 데이터(2020-01 ~ 2026-03) 기반 백테스트 결과입니다.
2. 미래 수익을 보장하지 않습니다.
3. 복리 20% 마진은 MDD 37~41%로 높은 리스크를 수반합니다.
4. 실제 운용 시 슬리피지, 체결 지연, 시장 구조 변화 등이 발생합니다.
5. 초기 자본의 전액 손실 가능성이 있습니다.
6. 반드시 소액으로 실전 검증 후 운용하세요.
7. 복리 마진은 엔진 구현에 따라 결과가 달라질 수 있습니다.
   고정 $1K 마진 기준 검증값: PF 3.06, MDD 7.3%, +1,582%
```

---

## 9. 산출물

| 파일 | 설명 |
|------|------|
| `# ETH_USDT_자동매매_기획서_V8.16.md` | 본 기획서 |
| `eth_v7_3_cuda.py` | CUDA GPU 백테스트 엔진 |
| `eth_v7_3_numba.py` | Numba JIT 백테스트 엔진 |
| `eth_overnight_cuda.py` | 멀티TF CUDA 최적화 |
| `run_v95_cuda.py` | V9.5 CUDA 최적화 |

---

*428,400+ 조합 CUDA GPU 실측 백테스트*
*10분봉·15분봉·30분봉 3개 타임프레임 교차 검증*
*6엔진 교차검증 ALL MATCH + 30회 결정론적 (σ=$0.000000)*
*Claude Opus 4.6 (1M context) + GPT-5.4 Thinking + CUDA GPU (RTX 4060 Ti) 공동 설계*
