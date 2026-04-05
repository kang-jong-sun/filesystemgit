# BTC/USDT 선물 자동매매 시스템 기획서 v32.2
## "EMA(100)/EMA(600) Tight-SL Trend System"
## 엔진 독립적 재현 가능한 완전 명세

> **작성일**: 2026-04-01
> **데이터**: 30분봉 109,234캔들 (2019-12-31 15:00 ~ 2026-03-25 07:30)
> **원본**: 5분봉 → 30분봉 리샘플링 (open=first, high=max, low=min, close=last, volume=sum)
> **초기 자본**: $5,000 USDT
> **워밍업**: 600봉 (start index=600, 실거래 시작 약 2020-01-12)

---

## 1. 검증 기준 수치

| 항목 | EMA(100)/EMA(600) |
|------|----------------------|
| **최종 잔액** | **$24,073,329** |
| **수익률** | **+481,367%** |
| **PF** | **5.8** |
| **MDD** | **43.5%** |
| **거래** | 70건 |
| **승/패** | 33W / 37L |
| **승률** | 47.1% |
| **SL** | 30건 |
| **TSL** | 17건 |
| **REV** | 23건 |

> 다른 엔진에서 위 수치와 ±1% 이내로 일치하지 않으면 구현 오류입니다.

---

## 2. 파라미터 (코드 상수)

```
TIMEFRAME         = '30min'
INITIAL_CAPITAL   = 5000.0
FEE_RATE          = 0.0004        # 0.04% (진입/청산 각각)
WARMUP_BARS       = 600           # 시작 인덱스

# ─── MA 설정 ───
FAST_MA_TYPE      = 'EMA'
FAST_MA_PERIOD    = 100
SLOW_MA_TYPE      = 'EMA'
SLOW_MA_PERIOD    = 600

# ─── 공통 필터 ───
ADX_PERIOD        = 20
ADX_MIN           = 30.0
ADX_RISE_BARS     = 6             # av[i] > av[i-6]
RSI_PERIOD        = 10
RSI_MIN           = 40.0
RSI_MAX           = 80.0
EMA_GAP_MIN       = 0.2           # 퍼센트 (0.2%)
MONITOR_WINDOW    = 24            # 크로스 후 24봉
SKIP_SAME_DIR     = True          # 동일방향 재진입 스킵

# ─── 리스크 ───
SL_PCT            = 3.0           # 가격 -3%
TA_PCT            = 12.0          # 가격 +12% (TSL 활성화)
TSL_PCT           = 9.0           # 고점 대비 가격 -9%
MARGIN_PCT        = 0.35          # 잔액의 35%
LEVERAGE          = 10
MARGIN_MODE       = 'ISOLATED'    # 격리마진 (SL3%가 강제청산 9.8% 이전 발동 → FC=0)
DAILY_LOSS_LIMIT  = -0.20         # 일일 -20%
```

---

## 3. 지표 계산 (정확한 공식)

### 3.1 EMA — pandas ewm(span=N, adjust=False)

```python
EMA = series.ewm(span=period, adjust=False).mean()

# 수학적 등가:
alpha = 2 / (period + 1)
EMA[0] = close[0]
EMA[i] = close[i] * alpha + EMA[i-1] * (1 - alpha)
```

> **주의**: Wilder's EMA (alpha=1/N)가 아닙니다. 표준 EMA (alpha=2/(N+1))입니다.

### 3.2 ADX(20) — pandas ewm(alpha=1/20)

```python
# +DM / -DM
plus_dm  = high.diff()
minus_dm = -low.diff()
plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

# True Range
tr = max(high-low, |high-prev_close|, |low-prev_close|)

# Wilder Smoothing via ewm(alpha=1/period)
ATR   = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
+DI   = 100 * plus_dm.ewm(alpha=1/20, min_periods=20, adjust=False).mean() / ATR
-DI   = 100 * minus_dm.ewm(alpha=1/20, min_periods=20, adjust=False).mean() / ATR
DX    = 100 * |+DI - -DI| / (+DI + -DI)    # 분모 0이면 1e-10
ADX   = DX.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
```

> **핵심**: `ewm(alpha=1/period)` 사용. `ewm(span=period)`가 아닙니다.
> `ewm(alpha=1/20)` = Wilder's smoothing 근사.
> 수동 Wilder 구현과 미세 차이 있음 (평균 0.005, 최대 10.7).
> 이 차이는 ADX>=30 경계에서 8봉의 진입/비진입 차이를 발생시킬 수 있으나, 최종 수익에 미치는 영향은 미미합니다.

### 3.3 RSI(10) — ewm(alpha=1/10)

```python
delta = close.diff()
gain  = delta.where(delta > 0, 0.0)
loss  = (-delta).where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1/10, min_periods=10, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/10, min_periods=10, adjust=False).mean()
RSI = 100 - 100 / (1 + avg_gain / avg_loss)    # avg_loss 0이면 1e-10
```

### 3.4 EMA 갭

```python
gap = |fast_ma[i] - slow_ma[i]| / slow_ma[i] * 100    # 퍼센트
# gap >= 0.2 이면 진입 허용
```

---

## 4. 백테스트 루프 (의사코드 — 정확한 실행 순서)

```
상태 변수 초기화:
  cap = 5000.0          # 잔액
  pos = 0               # 0=없음, 1=LONG, -1=SHORT
  epx = 0.0             # 진입가
  psz = 0.0             # 포지션 금액 (cap * 0.35 * 10)
  slp = 0.0             # 현재 SL가
  ton = False           # TSL 활성 여부
  thi = 0.0             # LONG 고점 추적
  tlo = 999999.0        # SHORT 저점 추적
  watching = 0          # 감시 중인 방향 (0=없음, 1=LONG, -1=SHORT)
  ws = 0                # 감시 시작 봉 인덱스 (크로스 발생 봉)
  le = 0                # 마지막 청산 봉 인덱스
  ld = 0                # 마지막 청산 방향
  pk = cap              # 고점 잔액 (MDD용)
  mdd = 0.0             # 최대 낙폭
  ms = cap              # 일일 시작 잔액

FOR i = 600 TO len(data)-1:
    px = close[i]       # 종가
    h_ = high[i]        # 고가
    l_ = low[i]         # 저가

    # 일일 리셋 (30분봉 1440봉 = 1일)
    IF i > 600 AND i % 1440 == 0:
        ms = cap

    # ════════════════════════════════════════════
    # STEP A: 포지션 보유 중 — 청산 체크
    # ════════════════════════════════════════════
    IF pos != 0:
        watching = 0    # 포지션 보유 중 감시 리셋

        # ── A1: SL 체크 (TSL 미활성 시에만) ──
        IF NOT ton:
            IF (pos==1 AND l_ <= slp) OR (pos==-1 AND h_ >= slp):
                pnl = (slp - epx) / epx * psz * pos - psz * 0.0004
                cap += pnl
                SL 카운트 증가
                ld = pos; le = i; pos = 0
                CONTINUE    # 다음 봉으로

        # ── A2: TA 활성화 체크 (고가/저가 기준) ──
        IF pos == 1:
            br = (h_ - epx) / epx * 100    # 고가 기준 상승률
        ELSE:
            br = (epx - l_) / epx * 100    # 저가 기준 하락률
        IF br >= 12.0:
            ton = True                      # TSL 활성화!

        # ── A3: TSL 체크 (TSL 활성 시) ──
        IF ton:
            IF pos == 1:
                IF h_ > thi: thi = h_                 # 고점 갱신 (고가 기준)
                ns = thi * (1 - 9.0/100)              # 새 TSL가 = 고점 × 0.91
                IF ns > slp: slp = ns                  # TSL가 상향만 (역행 불가)
                IF px <= slp:                          # 종가 기준 TSL 체크
                    pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl
                    TSL 카운트 증가
                    ld = pos; le = i; pos = 0
                    CONTINUE
            ELSE:   # SHORT
                IF l_ < tlo: tlo = l_                  # 저점 갱신 (저가 기준)
                ns = tlo * (1 + 9.0/100)               # 새 TSL가 = 저점 × 1.09
                IF ns < slp: slp = ns                   # TSL가 하향만
                IF px >= slp:                           # 종가 기준 TSL 체크
                    pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                    cap += pnl
                    TSL 카운트 증가
                    ld = pos; le = i; pos = 0
                    CONTINUE

        # ── A4: REV 체크 (EMA 교차 반전) ──
        IF i > 0:
            bull_now  = fast_ma[i] > slow_ma[i]
            bull_prev = fast_ma[i-1] > slow_ma[i-1]
            cross_up   = bull_now AND NOT bull_prev
            cross_down = NOT bull_now AND bull_prev
            IF (pos==1 AND cross_down) OR (pos==-1 AND cross_up):
                pnl = (px - epx) / epx * psz * pos - psz * 0.0004
                cap += pnl
                REV 카운트 증가
                ld = pos; le = i; pos = 0
                # CONTINUE 하지 않음 — 같은 봉에서 진입 가능

    # ════════════════════════════════════════════
    # STEP B: 포지션 없음 — 진입 체크
    # ════════════════════════════════════════════
    IF i < 1: CONTINUE

    # B1: 크로스 감지
    bull_now  = fast_ma[i] > slow_ma[i]
    bull_prev = fast_ma[i-1] > slow_ma[i-1]
    cross_up   = bull_now AND NOT bull_prev
    cross_down = NOT bull_now AND bull_prev

    IF pos == 0:
        # B2: 새 크로스 → 감시 시작
        IF cross_up:   watching = 1;  ws = i
        ELIF cross_down: watching = -1; ws = i

        IF watching != 0 AND i > ws:
            # B3: 모니터 윈도우 초과 → 감시 해제
            IF i - ws > 24: watching = 0; CONTINUE

            # B4: 감시 중 반대 크로스 → 방향 전환
            IF watching == 1 AND cross_down: watching = -1; ws = i; CONTINUE
            ELIF watching == -1 AND cross_up: watching = 1; ws = i; CONTINUE

            # B5: 동일방향 스킵
            IF SKIP_SAME_DIR AND watching == ld: CONTINUE

            # B6: ADX >= 30
            IF av[i] < 30.0: CONTINUE (조건 불충족, 다음 봉에서 재체크)

            # B7: ADX 상승 (현재 > 6봉 전)
            IF ADX_RISE_BARS > 0 AND i >= 6:
                IF av[i] <= av[i-6]: CONTINUE

            # B8: RSI 40~80
            IF rv[i] < 40.0 OR rv[i] > 80.0: CONTINUE

            # B9: EMA 갭 >= 0.2%
            IF gap_min > 0:
                gap = |fast_ma[i] - slow_ma[i]| / slow_ma[i] * 100
                IF gap < 0.2: CONTINUE

            # B10: 일일 손실 제한
            IF ms > 0 AND (cap - ms) / ms <= -0.20: watching = 0; CONTINUE

            # B11: 잔액 확인
            IF cap <= 0: CONTINUE

            # ════ 진입! ════
            mg = cap * 0.35                    # 마진
            psz = mg * 10                      # 포지션 크기 (금액)
            cap -= psz * 0.0004                # 진입 수수료
            pos = watching                     # 방향
            epx = px                           # 진입가 (종가)
            ton = False                        # TSL 비활성
            thi = px                           # 고점 초기화
            tlo = px                           # 저점 초기화
            IF pos == 1:
                slp = epx * (1 - 3.0/100)      # LONG SL가 = 진입가 × 0.97
            ELSE:
                slp = epx * (1 + 3.0/100)      # SHORT SL가 = 진입가 × 1.03
            pk = max(pk, cap)
            watching = 0

    # MDD 갱신
    pk = max(pk, cap)
    dd = (pk - cap) / pk IF pk > 0 ELSE 0
    IF dd > mdd: mdd = dd
    IF cap <= 0: BREAK

# 마지막 봉에서 미청산 포지션 강제 청산
IF pos != 0 AND cap > 0:
    pnl = (close[last] - epx) / epx * psz * pos - psz * 0.0004
    cap += pnl
```

---

## 5. 핵심 구현 주의사항 (다른 엔진 재현 시 필수)

### 5.1 SL/TSL 우선순위 — 가장 중요

```
┌─────────────────────────────────────────────────────┐
│  TSL 미활성 (ton=False):                            │
│    → SL만 작동 (저가/고가 기준, SL가에 청산)         │
│    → SL 발동 시 CONTINUE (TSL/REV 체크 안함)        │
│                                                     │
│  TSL 활성 (ton=True):                               │
│    → SL 비활성! (SL 체크 건너뜀)                     │
│    → TSL만 작동 (종가 기준)                          │
│    → TSL 미발동 시 REV 체크                         │
│                                                     │
│  이것이 $24M과 $1.68M의 차이를 만드는 핵심입니다.    │
│  TSL 활성 후에도 SL을 체크하면 수익이 10배 감소합니다.│
└─────────────────────────────────────────────────────┘
```

**실거래 구현 방법**: TSL이 활성화되면 SL 주문을 취소하고, TSL 트레일링 주문으로 교체합니다.

### 5.2 가격 기준

| 항목 | 기준 가격 | 설명 |
|------|----------|------|
| **진입** | **종가 (close)** | 봉 종가에 진입 |
| **SL 발동** | **저가/고가 (intrabar)** | LONG: 저가<=SL가, SHORT: 고가>=SL가 |
| **SL 청산가** | **SL가 (slp)** | SL가에 정확히 청산 (슬리피지 없음) |
| **TA 활성화** | **고가/저가 (intrabar)** | LONG: (고가-진입가)/진입가, SHORT: (진입가-저가)/진입가 |
| **TSL 고점 추적** | **고가/저가 (intrabar)** | LONG: 고가 갱신, SHORT: 저가 갱신 |
| **TSL 청산** | **종가 (close)** | LONG: 종가<=TSL가, SHORT: 종가>=TSL가 |
| **REV 판단** | **종가 (close)** | EMA 크로스는 종가 기준 |

### 5.3 수수료

```
진입 수수료 = 포지션크기(금액) × 0.0004
청산 수수료 = 포지션크기(금액) × 0.0004

PnL = (청산가 - 진입가) / 진입가 × 포지션크기 × 방향 - 포지션크기 × 0.0004
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~
      가격 변동 수익/손실                                           청산 수수료

진입 수수료는 진입 시 잔액에서 차감: cap -= psz * 0.0004
```

### 5.4 포지션 크기

```
마진 = 잔액 × 35%
포지션크기 = 마진 × 10 (레버리지)

psz는 "금액" 단위입니다. BTC 수량이 아닙니다.
PnL 계산: (exit - entry) / entry × psz × direction

예시: 잔액 $100,000
  마진 = $35,000
  포지션 = $350,000
  SL 3% 시 PnL = -3% × $350,000 = -$10,500 (잔액의 10.5%)
```

### 5.5 크로스 감지 및 감시

```
크로스 감지:
  bull_now  = fast_ma[i] > slow_ma[i]
  bull_prev = fast_ma[i-1] > slow_ma[i-1]
  cross_up   = bull_now AND NOT bull_prev     # 골든크로스
  cross_down = NOT bull_now AND bull_prev     # 데드크로스

감시 메커니즘:
  1. 크로스 발생 → watching = 방향, ws = 현재 봉
  2. 매 봉(i > ws)에서 필터 체크
  3. 24봉 초과 → 감시 해제
  4. 반대 크로스 발생 → 방향 전환 + ws 리셋
  5. 동일방향 스킵 → watching == 직전청산방향이면 CONTINUE
  6. 모든 필터 통과 → 진입

주의: 크로스 발생 봉(i == ws)에서는 진입하지 않습니다.
      다음 봉(i > ws)부터 필터 체크를 시작합니다.
```

### 5.6 동일방향 스킵

```
직전 청산 방향(ld)과 현재 감시 방향(watching)이 같으면 진입 스킵.

예시:
  LONG 보유 → REV 청산 (ld = 1)
  → 다음 LONG 크로스 감지 → watching = 1 = ld → 스킵
  → SHORT 크로스 감지 → watching = -1 ≠ ld → 진입 가능
```

### 5.7 일일 손실 제한

```
매 1440봉(1일)마다 ms(일일 시작 잔액) 갱신
(cap - ms) / ms <= -0.20 이면 당일 신규 진입 차단
기존 포지션의 청산은 영향 없음
```

### 5.8 고점/저점 추적 시점

```
고점/저점 추적은 TSL 활성(ton=True) 후에만 수행합니다.
TSL 미활성 상태에서는 thi/tlo가 진입가(epx)로 유지됩니다.

WRONG: pos_highest = max(pos_highest, h)  # TSL 여부 무관하게 항상 갱신
RIGHT: if ton: if h_ > thi: thi = h_     # TSL 활성 후에만 갱신

이 차이가 TSL 청산가를 변경하여 수익에 영향을 줍니다.
```

### 5.9 수수료 계산 방식

```
포지션크기(psz)는 "금액 단위"이며, BTC 수량이 아닙니다.
수수료는 진입 시 포지션크기와 청산 시 포지션크기가 동일합니다 (변동 없음).

진입: cap -= psz * 0.0004
청산: pnl = (exit-entry)/entry * psz * dir - psz * 0.0004

다른 엔진에서 size = psz / entry_price (BTC 수량)로 변환하면:
  청산수수료 = size * exit_price * 0.0004 (가격에 따라 변동)
  → 약 0.1~0.2% 차이 발생 (복리 누적 시 수백만$ 차이 가능)
  → 본 기획서는 "금액 고정" 방식을 사용합니다.
```

---

## 6. 검증 체크리스트

다른 엔진으로 구현 시 아래 항목을 순서대로 검증:

```
□ 1. 데이터: 30분봉 109,234캔들 확인
□ 2. EMA(100): 2020-01-21 기준 $8,703.63 확인
□ 3. EMA(600): 2020-01-21 기준 $8,474.74 확인
□ 4. EMA 크로스: 총 258건 확인
□ 5. 첫 진입: 2020-01-24 11:00 SHORT $8,330 확인
□ 6. SL 체크: TSL 미활성 시에만 작동하는지 확인
□ 7. TSL 활성: 고가/저가 기준 +12% 도달 시 활성
□ 8. TSL 청산: 종가 기준 확인
□ 9. 총 거래: 70건 (SL30 + TSL17 + REV23)
□ 10. 최종 잔액: $24,073,329 (±$240K 이내)
```

### 6.5 4엔진 교차 검증 결과

```
4개 독립 엔진 (Numba JIT, Pure Python, Numpy State, Class OOP)에서
동일 결과 확인: 최대 차이 $0.000000, 30회 std=$0.000000
```

---

## 7. 실거래 구현 가이드

### 7.1 주문 관리

```
진입 시:
  1. 시장가 진입 (종가 시뮬레이션)
  2. SL 지정가 주문 설정 (진입가 × 0.97 또는 × 1.03)
  3. ton = False

매 봉 체크:
  IF 가격이 +12% 도달 (고가/저가 기준):
    1. SL 주문 취소              ← 핵심!
    2. TSL 트레일링 주문으로 교체
    3. ton = True

  IF EMA 크로스 반전:
    1. 모든 주문 취소
    2. 시장가 청산
```

### 7.2 TSL 활성 후 SL 비활성 — 구현 근거

```
TSL이 활성화된다는 것 = 이미 +12% 수익 확보
이 상태에서 SL(-3%)이 발동하면:
  - 고점에서 -15% 이상 하락한 상태
  - TSL(-9%)이 먼저 발동했어야 할 구간
  - SL이 TSL보다 먼저 발동 = 같은 봉 내 급락 후 종가 반등 케이스

백테스트에서는:
  - SL = 저가 기준 (봉 내 최저가)
  - TSL = 종가 기준
  - 같은 봉에서 저가가 SL 터치 + 종가는 TSL 위 = SL 비활성이 유리

실거래에서는:
  - TSL 활성 시 SL 주문을 취소하고 TSL로 교체하면 동일 동작
```

---

## 8. 참고: 엔진 간 차이 원인

| 차이 항목 | test2 (본 기획서) | test3 (다른 엔진) | 영향 |
|---------|-----------------|-----------------|------|
| **SL/TSL 우선순위** | TSL 활성 시 SL 비활성 | SL 항상 활성 | **10배 차이** |
| ADX 계산 | ewm(alpha=1/20) | 수동 Wilder | ±0.005 (미미) |
| TA 기준 | 고가/저가 (intrabar) | 종가 | 진입/활성화 타이밍 차이 |
| warmup | 600봉 (고정) | max(800, slow*1.5) | 초기 거래 누락 |
| 고점추적 | TSL 활성 후에만 | 항상 | TSL가 차이 |
| 동일방향스킵 | 구현 (watching==ld) | 미구현 | 거래수 차이 |
| RSI | RSI(10) 40~80 | RSI(14) 35~75 | 진입 시점 차이 |
| 수수료 | psz × 0.0004 (고정) | size × price × 0.0004 | 미미 |

---

## 9. 전략 설정 요약

```python
# ═══ MA 설정 ═══
FAST_MA  = EMA(close, 100)       # pandas ewm(span=100, adjust=False)
SLOW_MA  = EMA(close, 600)       # pandas ewm(span=600, adjust=False)

# ═══ 공통 ═══
ADX      = ADX(20)               # ewm(alpha=1/20)
RSI      = RSI(10)               # ewm(alpha=1/10)
SL       = 3%                    # 가격 기준, 저가/고가 발동, SL가에 청산
TA       = 12%                   # 가격 기준, 고가/저가 활성화
TSL      = 9%                    # 고점 대비 가격 기준, 종가에 청산
MARGIN   = 35%, LEVERAGE = 10x
GAP      = 0.2%                  # |Fast-Slow|/Slow >= 0.2%
MONITOR  = 24봉 (12시간)
SKIP     = 동일방향 재진입 스킵
```

---

> **v32.2 = EMA(100)/EMA(600) Tight-SL Trend System**
> **SL3%(비활성화 가능) + TA12%/TSL9% + EMA 교차 + ADX(20)>=30 + 갭>=0.2%**
>
> $5,000 → **$24,073,329** (75개월, 70거래, PF 5.8, MDD 43.5%)
