# BTC/USDT 선물 자동매매 시스템 기획서 v22.0_2 (Engine B)

> 원본: v22.0 기획서에서 분리

---

## 공통 설정

| 항목 | 설정 |
|------|------|
| 거래소 | Binance Futures |
| 심볼 | BTC/USDT (Perpetual) |
| 초기 자본 (전체) | $5,000 |
| 운영 | 24시간 자동 거래 |
| 수수료 | 0.04% × 2 = 0.08% |
| 마진 모드 | 격리마진 (ISOLATED) |
| 데이터 | BTC/USDT 5분봉 75개월 → 15m/30m 리샘플링 |
| 최적화 | 201,600+ 조합 4단계 × 2엔진 (Numba JIT) |
| 검증 | 30회 반복 PASS (Deterministic, std=0.00) |

---

## Engine B: 고빈도 코어

### 파라미터

| 항목 | 설정 |
|------|------|
| 타임프레임 | **30분봉** |
| Fast MA | **WMA(3)** |
| Slow MA | EMA(200) |
| ADX | **(20) >= 35** |
| RSI | 30 ~ 60 |
| SL | -8% (안전장치, 75개월 0회) |
| TSL 활성화 | +3% |
| TSL 폭 | -2% (고점 대비) |
| 마진 | **1000 USDT 고정** |
| 레버리지 | 10배 |
| 보호 | 없음 |
| 딜레이 | 0 |
| 동일방향 | 재진입 스킵 |
| 자본 배분 | 30% ($900) |

### 성과

| 항목 | 값 |
|------|-----|
| 초기 자본 | $5,000 (전체의 30%) |
| **최종 잔액** | **재계산 필요** |
| **수익률** | **재계산 필요** |
| **PF** | **12.9** |
| **MDD** | **14.5%** |
| 거래 | **45회** (75개월, 월 0.6회) |
| 승률 | 53.3% (24W / 21L) |
| SL | 0 \| TSL | 26 \| REV | 19 \| FL | 0 |

### 특징
- **WMA(3)**: 기존 EMA(3) 대비 PF 4.7배 향상 (v16.0 발견)
- **TSL +3%/-2%**: 초타이트 트레일링으로 수익 확보
- **SL 0회**: TSL/REV가 SL 전에 처리
- **45거래**: Engine A (10)의 4.5배로 복리 효과 보완

### 연도별 성과

> ⚠️ 연도별 수익률 표 삭제됨 (포지션 크기 변경으로 재계산 필요)


### 청산 로직

| 순위 | 트리거 | 방식 | 75개월 실적 |
|------|--------|------|-----------|
| 1 | SL -8% | 서버 Stop Market | **0회** |
| 2 | TSL +3%/-2% | 봇 30분봉 종가 | **26회 (58%)** |
| 3 | 역신호 | 봇 30분봉 | **19회 (42%)** |

### 리스크

- **MDD 14.5%**: 단독 운용 시 이 수준까지 하락 가능.
- **RSI 30~60**: 상한 60으로 일부 기회 제한.

---

## JSON 설정 (Engine B)

```python
# ═══ Engine B: 30m 고빈도 코어 ═══
B_TIMEFRAME = '30m'
B_MA_FAST_TYPE = 'WMA'
B_MA_FAST_PERIOD = 3
B_MA_SLOW_TYPE = 'EMA'
B_MA_SLOW_PERIOD = 200
B_ADX_PERIOD = 20
B_ADX_MIN = 35
B_RSI_MIN = 30
B_RSI_MAX = 60
B_SL_PCT = 0.08
B_TRAIL_ACTIVATE = 0.03
B_TRAIL_PCT = 0.02
B_LEVERAGE = 10
B_MARGIN = FIXED           # 고정 1000 USDT
B_CAPITAL_RATIO = 0.30  # 전체의 30%
B_SKIP_SAME_DIR = True  # 동일방향 재진입 스킵

# ═══ 공통 ═══
FEE_RATE = 0.0004
MARGIN_MODE = 'ISOLATED'
```

---

## 지표 구현

### WMA(3) - Engine B Fast MA

```python
def calc_wma(close, period):
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = weights.sum()
    result = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        result[i] = np.dot(close[i - period + 1:i + 1], weights) / weight_sum
    return result
```

### ADX (Wilder's Smoothing)

```python
# rolling().mean() 사용 금지!
adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
```

---

## 권장사항 (Engine B)

1. **첫 6개월**: M25% 10x로 보수적 시작
2. **동일방향 재진입 스킵 반드시 구현**
3. **격리마진 필수**
4. **동시 포지션**: A와 B가 반대 방향 가능 (헤지 효과)
5. **자본 재배분**: 6개월마다 A:B 비율 재조정 검토

---

## 듀얼 합산 참고 (v22.0 전체)

| 항목 | Engine A | Engine B | **듀얼 합산** |
|------|----------|----------|--------------|
| 초기 자본 | $5,000 | $900 | **$5,000** |
| **최종 잔액** | $183,133 | $19,406 | **$202,539** |
| **수익률** | +8,620% | +2,056% | **+6,651%** |
| **PF** | ∞ | 12.9 | **12.9+** |
| **MDD** | 0.3% | 14.5% | **~4.3%** |
| **거래** | 10 | 45 | **55** |
| **SL** | 0 | 0 | **0** |
| **FL** | 0 | 0 | **0** |

---

**문서 버전:** v22.0_2 (Engine B)
**원본 버전:** v22.0
**작성일:** 2026-03-28
**데이터:** BTC/USDT 5분봉 75개월 → 30m 리샘플링
**수수료:** 0.04% x 2 = 0.08%
**핵심:** B(30%): 안정적 복리 축적 → 빈도 보완
