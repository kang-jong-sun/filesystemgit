# SOL V1 실전 자동매매 봇 — 완전 기획안

> **문서 버전**: v1.4 (2026-04-28, REV-GUARD 추가)
> **봇 파일 위치**: `D:\filesystem\futures\sol_v1\`
> **AWS 배포**: `ubuntu@18.183.150.105:/home/ubuntu/sol_v1/`
> **저장소**: `https://github.com/kang-jong-sun/filesystemgit.git`
> **레버리지**: **5x** (테스트 단계적 상향 — 2x → 5x, 2026-04-28)
> **최신 커밋**: `861ad79` (REV-GUARD 추세-역방향 가드 추가)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [전략 아키텍처](#2-전략-아키텍처)
3. [V12 전략 (75%) 상세](#3-v12-전략-75-상세)
4. [Mass Index 전략 (25%) 상세](#4-mass-index-전략-25-상세)
5. [Mutex (상호 배제) 로직](#5-mutex-상호-배제-로직)
6. [지표 계산 (Wilder 일치)](#6-지표-계산-wilder-일치)
7. [리스크 관리](#7-리스크-관리)
8. [포지션 사이징 (Confidence × Regime)](#8-포지션-사이징-confidence--regime)
9. [피라미딩 (3-Leg)](#9-피라미딩-3-leg)
10. [세션별 파라미터](#10-세션별-파라미터)
11. [데이터 파이프라인](#11-데이터-파이프라인)
12. [실행 엔진 (Executor)](#12-실행-엔진-executor)
13. [모니터링 시스템](#13-모니터링-시스템)
14. [로깅 전략](#14-로깅-전략)
15. [AWS 배포](#15-aws-배포)
16. [백테스트 검증 이력](#16-백테스트-검증-이력)
17. [현재 상태](#17-현재-상태)
18. [운영 절차](#18-운영-절차)
19. [안전 장치 & 알려진 이슈](#19-안전-장치--알려진-이슈)
20. [로드맵](#20-로드맵)

---

## 1. 프로젝트 개요

### 1.1 배경
- **기반**: SOL 오케스트라 v2 R31.2 라운드에서 확정된 V12:Mass 75:25 Mutex 전략
- **백테스트 검증**: 51개월 (2022-01 ~ 2026-04), +1,649.6% (PF 2.33, MDD 16.5%)
- **목표**: 백테스트 결과를 실전에서 재현, 안정적 수익 창출

### 1.2 핵심 사양
| 항목 | 값 |
|---|---|
| 거래소 | Binance USDM Futures |
| 심볼 | SOL/USDT:USDT (Perpetual) |
| 기본 TF | 15min (primary) |
| 보조 TF | 1h (BTC regime) |
| **레버리지** | **5x** (테스트 단계적 상향: 2x → 5x @ 2026-04-28) |
| **사용자 레버리지 추적** | **동적 인지** (사용자가 거래소에서 다른 레버리지로 진입 시 자동 감지 + 텔레그램 알림) |
| 포지션 모드 | One-Way, Isolated Margin |
| 초기 자본 | $5,000 (백테스트 기준) |
| 복리 비율 | 12.5% of balance / 거래 |
| 자본 상한 | $1,000,000 (바이낸스 0.5% 깊이 $11M의 ~10%) |

### 1.3 전략 구성
```
SOL V1 = V12 (75%) ⊕ Mass Index (25%)
         ↓          ↓
      추세추종     반전포착
      5중 필터    Mass Bulge
```

### 1.4 주요 혁신
1. **Wilder 표준화**: 봇 RSI/ADX를 TradingView/백테스트와 동일한 Wilder smoothing으로 통일 (2026-04-23 반영)
2. **세션별 파라미터**: Asia/EU/US 3구간 차별화, US 부스트
3. **Mutex**: V12와 Mass가 동시 포지션 보유 금지 → 상관성 관리
4. **Skip2@4loss**: 연속 4패 시 다음 2거래 스킵 → 드로다운 방지
5. **Compound 12.5%**: 매 거래 잔액의 12.5% → 동적 사이징
6. **ETH V8 패턴 통합** (2026-04-28):
   - SL orphan 청소 (`_cancel_all_sl_orders` + `STOP_MARKET` fallback)
   - 사용자 레버리지 동적 추적 (5x/10x 진입 시 자동 인지)
   - 수동 포지션 3-케이스 처리 (진입/청산/방향전환)
   - POSITION_SYNC 60→10초 (즉각 감지)
   - 텔레그램 메시지 큐 + 한글 명령어 (`/sol 상태/포지션/잔액/최근거래/봇정지/봇시작/청산`)
   - REV 진단 로그 (cross 감지/발동 추적 가능)
7. **REV-GUARD** (2026-04-28, 4/27 사고 대응):
   - cross 봉 놓침 방지 — 진입 후 4봉(1시간) + 추세 역방향이면 강제 EXIT
   - V12 REV의 한계 (`dn`이 cross 봉에서만 True) 보완
   - 사용자 수동 진입 + 추세 반대 방향 케이스에서 자동 청산 보장

---

## 2. 전략 아키텍처

### 2.1 상위 레벨 흐름

```
                     ┌──────────────┐
                     │ Binance API  │
                     │  5m candles  │
                     └──────┬───────┘
                            │
                   ┌────────┴────────┐
                   │  DataCollector  │
                   │  (15m 리샘플)    │
                   └────────┬────────┘
                            │
        ┌──────┬────────────┼────────────┬──────┐
        ▼      ▼            ▼            ▼      ▼
      EMA9  SMA400  ADX(20)Wilder  RSI(10)Wilder  LR(14)
                            │
                            ▼
                    ┌───────────────┐
                    │   V12 신호    │
                    │  (EMA 크로스) │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  5중 필터     │
                    │  +Delay 5봉   │
                    │  +Monitor 24봉 │
                    │  +SkipSame    │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                                       ▼
┌───────────────┐                     ┌───────────────┐
│ V12 진입 신호 │                     │ Mass Index    │
│ (75% alloc)   │                     │ (25% alloc)   │
└───────┬───────┘                     └───────┬───────┘
        │                                     │
        └─────────────┬───────────────────────┘
                      │ Mutex (상호배제)
                      ▼
             ┌───────────────────┐
             │ Position Sizing   │
             │ Conf × Regime ×   │
             │ ATR Vol Filter    │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │    Executor       │
             │  (CCXT Order)     │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │   Monitoring:     │
             │ Console/Web/Tele  │
             └───────────────────┘
```

### 2.2 코드 아키텍처

| 파일 | 역할 | 라인수 | 주요 클래스/함수 |
|---|---|---|---|
| `sol_main_v1.py` | 메인 루프, 초기화, 시그널 핸들링 | 32KB | `main()`, `setup_logging()` |
| `sol_core_v1.py` | 트레이딩 로직, 신호 생성, 상태 관리 | 45KB | `Core`, `check_v12_signal()`, `check_mass_signal()` |
| `sol_data_v1.py` | 데이터 수집, 지표 계산 | 33KB | `DataCollector`, `_rsi()`, `_adx()` |
| `sol_executor_v1.py` | 주문 실행, 포지션 관리 | 15KB | `Executor`, `open_position()`, `close_position()` |
| `sol_web_v1.py` | FastAPI 대시보드 | 75KB | 6탭 (Dashboard/Chart/Trades/Balance/Errors/Logout) |
| `sol_telegram_v1.py` | 텔레그램 알림 | 7KB | `send()`, `notify_trade()` |
| `sol_monitor.py` | Rich UI 콘솔 모니터 | 13KB | `Monitor` |

### 2.3 데이터 저장

| 종류 | 위치 | 보관 기간 |
|---|---|---|
| SOL 5m CSV | `cache/sol_5m_YYYY.csv` | 180일 (6개월) |
| BTC 5m CSV | `cache/btc_5m_YYYY.csv` | 180일 |
| 거래 DB | `sol_trading_bot.db` (SQLite) | 영구 |
| 상태 | `state/sol_state.json` | 영구 |
| 에러 로그 | `logs/error_YYYYMMDD.log` | 365일 |
| 거래 로그 | `logs/trades.log` | 영구 |
| 신호 로그 | `logs/signals_YYYYMMDD.log` | 90일 |
| Heartbeat | `logs/heartbeat_YYYYMMDD.log` | 30일 |
| Daily summary | `logs/daily_summary.log` | 영구 |
| 통합 로그 | `logs/sol_trading_YYYYMMDD.log` | 30일 |

---

## 3. V12 전략 (75%) 상세

### 3.1 진입 신호

**EMA9 × SMA400 15분봉 크로스**
- **LONG**: EMA9가 SMA400 상향 돌파 (`CROSS_UP`)
- **SHORT**: EMA9가 SMA400 하향 돌파 (`CROSS_DN`)

크로스 감지 후 바로 진입하지 않고 5중 필터 + Entry Delay + Monitor Window 통과 시 진입.

### 3.2 V12 5중 필터

| # | 필터 | 조건 | 의미 |
|---|---|---|---|
| 1 | **ADX** | ≥ 22 (Wilder period=20) | 추세 강도 확인 |
| 2 | **ADX Rising** | ADX[i] > ADX[i-6] | 6봉 전 대비 상승 중 |
| 3 | **RSI** | 30 ≤ RSI ≤ 65 (Wilder period=10) | 과매수/과매도 배제 |
| 4 | **LR Slope** | -0.5 ≤ slope ≤ +0.5 (period=14) | 급격한 회귀선 기울기 배제 |
| 5 | **Skip Same** | 동일 방향 연속 진입 차단 | 같은 방향 중복 금지 |

**모든 필터 통과해야 진입**. 하나라도 실패 시 `V12 SKIP` 로그 남김.

### 3.3 Entry Delay (5봉)

크로스 발생 후 **5봉 대기**. 가짜 크로스 방지 목적.

```python
if cross_detected_at_bar_X:
    # X+5 봉까지 모든 조건 유지되어야 진입
    for check_bar in [X+1, X+2, X+3, X+4, X+5]:
        if not all_filters_pass(check_bar):
            return "Delay fail"
    enter_at_bar(X+5)
```

대시보드에서 `⏳ Entry Delay 3/5봉` 형태로 표시.

### 3.4 Monitor Window (24봉)

크로스 후 24봉(6시간) 내에 모든 조건 충족 시 진입. 초과 시 신호 만료.

```
크로스 ────┬─── Entry Delay 5봉 ───┬─── Monitor Window 24봉 ───┬─── 만료
          │                        │                            │
          필터 체크 시작            진입 가능 구간              신호 폐기
```

### 3.5 청산 규칙 (V12.1)

| 유형 | 조건 | Base | US 부스트 |
|---|---|---|---|
| **SL** (Stop Loss) | 진입가 -3.8% | 3.8% | 4.0% |
| **TA** (Take Activate) | 진입가 +5.6% | 5.6% | 5.6% |
| **TSL** (Trailing Stop) | 피크 -10.0% | 10.0% | 11.0% |
| **REV** | EMA9 역방향 크로스 (cross 봉) | 활성 | 활성 |
| **REV-GUARD** ★ | 진입 후 4봉(1h) + 추세-역방향 (`fm < sm` for LONG / `fm > sm` for SHORT) | 활성 | 활성 |

**REV-GUARD 추가 배경 (2026-04-28)**:
- 기본 REV 조건 `dn = (fm < sm AND pfm >= psm)`은 **cross 발생 그 1봉에서만 True**
- 데이터 동기화 0.5초 차이로 그 봉을 놓치면 영영 발동 못 함
- 2026-04-27 사고 (사용자 LONG 보유 + cross 봉 놓침) 재발 방지용
- `REV_GUARD_BARS = 4` (15m × 4 = 1시간), 진입 후 1시간 이내는 기존 cross 감지에만 의존 → 가짜 청산 방지

### 3.6 TA/TSL 동작

```
진입 (LONG @100) ──→ +5.6% (@105.6) ──→ TSL 활성화
                                        │
                    [TSL trail] 가격 계속 상승 시 피크 추적
                                        │
                    피크에서 -10% 하락 시 청산
```

---

## 4. Mass Index 전략 (25%) 상세

### 4.1 Mass Index 계산

```
EMA1 = EMA(High - Low, 9)
EMA2 = EMA(EMA1, 9)
Ratio = EMA1 / EMA2
Mass = Sum(Ratio, 25 bars)
```

### 4.2 신호 생성 (Reversal Bulge)

**Bulge 형성**: Mass > 27.0 → 추세 반전 가능성
**Bulge 종료**: Mass < 26.5 → 반전 트리거

```
Mass
 30 ┤                 ╭──╮   Bulge 형성 (>27)
 27 ┼─────────────────┤  ├────────────────── ← 반전 감지
 25 ┤        ╭────────╯  ╰──╮                ← 반전 실행 (<26.5)
 20 ┤╭───────╯               ╰────────────
 15 ┤┘
```

### 4.3 진입 방향

Bulge 종료 직전 **5봉 평균 수익률**으로 판정:
- 5봉 누적 수익 > 0 → SHORT (상승 추세 끝 반전)
- 5봉 누적 수익 < 0 → LONG (하락 추세 끝 반전)

### 4.4 청산 규칙 (Mass)

| 유형 | % |
|---|---|
| SL | -3.0% |
| TA | +5.0% |
| TSL | -8.0% |

V12보다 **더 타이트**한 SL/TSL (단기 반전 포착 전략이므로 수명 짧음).

### 4.5 25% 자본 할당

Mass 진입 시: `margin = (balance × 12.5%) × 25%`

예: 잔액 $5,000 → V12 margin $625, Mass margin $156.25.

---

## 5. Mutex (상호 배제) 로직

### 5.1 규칙

**V12 포지션과 Mass 포지션은 동시에 보유 불가**

| V12 상태 | Mass 상태 | 신규 V12 | 신규 Mass |
|---|---|---|---|
| 없음 | 없음 | ✅ | ✅ |
| LONG | 없음 | ❌ (중복) | ❌ (Mutex) |
| 없음 | LONG | ❌ (Mutex) | ❌ (중복) |

### 5.2 근거

- V12: 추세추종 (장기)
- Mass: 반전 (단기)
- 동시 보유 시 **상관 리스크** 증가
- 백테스트에서 Mutex 75:25가 가장 높은 Sharpe

### 5.3 구현

`sol_core_v1.py`의 `check_v12_signal()` / `check_mass_signal()`:
```python
if self.current_position is not None:
    return Signal(action='NONE', reason='Mutex: 다른 전략 포지션 보유')
```

---

## 6. 지표 계산 (Wilder 일치)

### 6.1 변경 이력

**2026-04-23 이전 (Simple MA, 문제 있음)**:
- RSI period=14, Simple MA (pandas rolling.mean)
- ADX period=14, Simple MA

**2026-04-23 수정 (Wilder, 백테스트/TV 일치)**:
- **RSI period=10, Wilder** (pandas ewm alpha=1/period)
- **ADX period=20, Wilder**

### 6.2 Wilder RSI (봇 구현)

```python
def _rsi(c: np.ndarray, period: int = 10) -> np.ndarray:
    """Wilder's smoothing RSI (TradingView 기본 + V12.1 백테스트 일치)."""
    series = pd.Series(c)
    d = series.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    a = 1.0 / period
    ag = g.ewm(alpha=a, min_periods=period, adjust=False).mean()
    al = l.ewm(alpha=a, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, 1e-10))
    return rsi.fillna(50).values.astype(np.float64)
```

### 6.3 Wilder ADX (봇 구현)

```python
def _adx(h, l, c, period: int = 20) -> np.ndarray:
    """Wilder's smoothing ADX (TradingView 기본 + V12.1 백테스트 일치)."""
    h_s = pd.Series(h); l_s = pd.Series(l); c_s = pd.Series(c)
    a = 1.0 / period

    tr = pd.concat([h_s - l_s,
                    (h_s - c_s.shift(1)).abs(),
                    (l_s - c_s.shift(1)).abs()], axis=1).max(axis=1)
    up = h_s - h_s.shift(1); dn = l_s.shift(1) - l_s
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c_s.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c_s.index)

    atr_w = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
    mdi = 100 * mdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    adx = dx.ewm(alpha=a, min_periods=period, adjust=False).mean()
    return adx.fillna(0).values.astype(np.float64)
```

### 6.4 전체 지표 일람

| 지표 | Period | 계산 방식 | 출처 |
|---|---|---|---|
| EMA(9) | 9 | pandas ewm span=9 | SOL 15m |
| SMA(400) | 400 | pandas rolling mean | SOL 15m |
| **RSI(10)** | **10** | **Wilder ewm** | SOL 15m |
| **ADX(20)** | **20** | **Wilder ewm** | SOL 15m |
| LR slope(14) | 14 | numpy polyfit (기울기 %) | SOL 15m |
| ATR(14) | 14 | rolling mean of TR | SOL 15m |
| ATR(50) | 50 | rolling mean of TR | SOL 15m |
| Mass Index | EMA9 + Sum25 | EMA × Sum of Ratio | SOL 15m |
| BTC EMA(200) | 200 | ewm span=200 | BTC 1h |

### 6.5 Wilder vs Simple MA 영향 (17,280봉 샘플)

| | Simple MA(14) | Wilder(10/20) | 차이 |
|---|---|---|---|
| ADX 평균 | 35.57 | 23.02 | **-12.56** |
| ADX 표준편차 | 16.36 | 9.82 | -6.54 |
| RSI 평균 | 49.54 | 49.41 | -0.12 |
| RSI 표준편차 | 15.92 | 13.57 | -2.35 |
| ADX≥22 통과율 | 77.40% | **46.37%** | -31.03%p |
| RSI 30-65 통과율 | 71.46% | **80.22%** | +8.76%p |
| **5중 필터 통과율** | **51.49%** | **34.68%** | **-16.81%p** |

**결론**: Wilder 적용 후 필터가 **실제 효과를 발휘**. 이전엔 ADX가 과대평가되어 필터가 거의 무효.

---

## 7. 리스크 관리

### 7.1 다단계 리스크 게이트

진입 시 아래 게이트를 순차 통과해야 함:

```
1. Daily Loss Limit (-2.5%)        → 통과 or 당일 정지
2. Skip2@4loss                     → 4연패 후 2거래 스킵
3. Mutex                           → V12/Mass 동시 금지
4. 5중 필터 (ADX/RSI/LR/Skip/Rising)  → 품질 신호만
5. Entry Delay (5봉)               → 가짜 크로스 방지
6. Monitor Window (24봉)           → 신호 만료
7. ATR Vol Filter                  → ATR 확장 시 margin flat
```

### 7.2 Daily Loss Limit (-2.5%)

```python
DAILY_LOSS_LIMIT = -0.025  # -2.5%
```

**기준 시간**: **UTC 00:00 (KST 09:00)** — 매일 리셋

**로직**:
- `day_start_balance` 저장 (UTC 00:00)
- `daily_pct = (balance / day_start_balance - 1) * 100`
- `daily_pct <= -2.5%` → 당일 모든 신규 진입 차단
- 기존 포지션은 정상 유지 (SL/TSL로 관리)

### 7.3 Skip2@4loss

```python
SKIP_LOSS_THRESHOLD = 4   # 4연패
SKIP_TRADE_COUNT = 2      # 다음 2거래 스킵
```

**로직**:
- `consec_losses` 카운터 유지
- 연속 4패 → `skip_next_trades = 2` 설정
- 신규 진입 시도할 때마다 `skip_next_trades -= 1`
- 카운터 0 될 때까지 진입 차단

### 7.4 ATR Volatility Filter

```python
ATR_EXPANSION_THRESHOLD = 1.5  # ATR(14) > 1.5 × ATR(50) → 급변
```

- `ATR(14) / ATR(50) > 1.5` → 변동성 확장 중 → margin × 0.5 (flat)
- 변동성 정상화되면 margin 복귀

### 7.5 $1M Capital Cap

```python
MAX_CAPITAL = 1_000_000  # $1M
```

- 잔액이 $1M 초과 시 **$1M만 거래에 사용**
- 바이낸스 SOL 유동성: 0.5% 깊이 ≈ $11M → $1M은 0.1% 슬리피지 이내

### 7.6 레버리지 & 마진

- **레버리지**: **5x** (테스트 단계적 상향: 2x → 5x @ 2026-04-28, 원래 전략 10x)
- **마진 모드**: Isolated (격리) → 한 포지션 손실이 다른 포지션에 영향 없음
- **포지션 모드**: One-Way (단방향)

#### 사용자 레버리지 동적 추적 (ETH V8 패턴, 2026-04-28 추가)

봇 자동 진입과 사용자 수동 진입의 레버리지를 분리 관리:

| 시나리오 | 봇 동작 |
|---------|--------|
| **봇 자동 진입** | 항상 `LEVERAGE = 5x` 강제 (`_handle_entry`에서 `set_leverage` 호출) |
| **사용자 거래소에서 다른 레버리지로 진입** | `get_exchange_position`이 `p.get('leverage')` 또는 `notional/margin` 비율로 동적 추출 → `self.leverage` 갱신 |
| **사용자 레버리지 변경 감지** | `_sync_position`에서 1회 텔레그램 알림: `"ℹ️ 사용자 레버리지 5x 인지 (봇 기본 5x)"` |
| **봇 시작 시 포지션 보유 중** | `set_leverage` 강제 호출 안 함 → 사용자 설정 보존 |
| **봇 시작 시 포지션 없음** | BOT 기본값 `LEVERAGE` 강제 설정 |

**margin 계산**: 거래소 실제 `initialMargin` 우선 사용 (`notional / LEVERAGE` 가정 폐기) → 사용자가 5x로 진입해도 정확한 margin 동기화.

---

## 8. 포지션 사이징 (Confidence × Regime)

### 8.1 기본 마진 계산

```python
base_margin = balance × COMPOUND_PCT(12.5%)  # 예: $5,000 × 12.5% = $625
```

### 8.2 Confidence Sizing (Sigmoid)

```python
conf = CONF_BASE + CONF_RANGE / (1 + exp(-CONF_K × score))
# CONF_BASE = 0.5, CONF_RANGE = 1.85, CONF_K = 5.0
# Range: 0.5 ~ 2.35
```

**Score 계산**:
```python
score = 0.4 × norm(ADX) + 0.3 × norm(RSI) + 0.3 × norm(LR)
```

| 지표 | 정규화 |
|---|---|
| ADX | (ADX - 22) / 30, clipped [0, 1] |
| RSI | abs(RSI - 50) / 15, 중간값 1.0 최대 |
| LR | 1 - abs(LR) / 0.5 |

**예시**:
- 약한 신호 (ADX=22, RSI=50, LR=0.4) → score ≈ 0.3 → conf ≈ 0.8 → margin $625 × 0.8 = $500
- 강한 신호 (ADX=40, RSI=55, LR=0.1) → score ≈ 0.8 → conf ≈ 2.2 → margin $625 × 2.2 = $1,375

### 8.3 Regime Scaling (BTC EMA200)

**BTC 1h EMA200** 기준:
- BTC close > EMA200 → **Bull**: margin × 1.5
- BTC close < EMA200 → **Bear**: margin × 0.5

```python
final_margin = base_margin × conf × regime_mult × atr_vol_mult
```

### 8.4 최종 포지션 크기

```python
notional = final_margin × LEVERAGE(2x)
contracts = notional / SOL_price
```

**예**: $500 margin × 2x = $1,000 notional @ $100 SOL = 10 contracts

---

## 9. 피라미딩 (3-Leg)

### 9.1 진입 구조

LONG 진입 후 수익 구간에서 추가 진입:

```
Leg 1: 진입                      (base margin)
Leg 2: Leg1 + 3.0%               (base × 0.5 margin)
Leg 3: Leg1 + 7.5%               (base × 0.5 margin)
```

**SHORT은 -3.0%, -7.5%에 추가 진입**.

### 9.2 VWAP 가중 평균 진입가

각 leg 추가 시 진입가 재계산:
```python
new_epx = (leg1_epx × leg1_contracts + leg2_epx × leg2_contracts) / total_contracts
```

SL/TSL도 새 epx 기준으로 재설정.

### 9.3 강제 익절

모든 Leg 진입 후 +5.6% (TA) 도달 시 청산 준비 (TSL 활성화).

---

## 10. 세션별 파라미터

### 10.1 세션 정의 (UTC 기준)

| 세션 | UTC | KST | 특징 |
|---|---|---|---|
| Asia | 0-8 | 9-17 | 변동성 낮음 |
| EU | 8-16 | 17-01 | 변동성 중간 |
| **US** | **13-21** | **22-06** | **변동성 최대** |

### 10.2 파라미터 차별

| 항목 | Asia/EU (base) | US 부스트 |
|---|---|---|
| SL | 3.8% | 4.0% |
| TA | 5.6% | 5.6% |
| TSL | 10.0% | 11.0% |
| Leg2 % | 3.0% | 3.2% |
| Leg3 % | 7.5% | 8.0% |

**근거**: US 세션은 변동성이 커서 SL/TSL 약간 넓혀 whipsaw 방지.

---

## 11. 데이터 파이프라인

### 11.1 초기 로드 (봇 시작 시)

```
1. CSV 로드: cache/sol_5m_YYYY.csv (최근 180일)
2. API 보충: CSV 마지막 봉 이후 데이터 fetch (ccxt fetch_ohlcv)
3. 병합: drop_duplicates + sort by timestamp
4. 리샘플: 5m → 15m (SOL), 5m → 1h (BTC)
5. 지표 계산: EMA, SMA, RSI, ADX, LR, ATR, Mass, BTC EMA200
6. 메모리 보유: 17,280 15m 봉 (180일 × 24h × 4봉)
```

### 11.2 실시간 업데이트

**메인 루프 tick 주기**:

| 작업 | 주기 | 환경변수 |
|------|------|---------|
| LOOP_INTERVAL | 10초 | 메인 루프 sleep |
| CANDLE_CHECK | 30초 | 5m 봉 신규 fetch + 지표 재계산 |
| BALANCE_CHECK | 300초 (5분) | 잔액 조회 |
| **POSITION_SYNC** | **10초** ★ | **거래소 포지션 ↔ 봇 state 동기화 (이전 60초 → 10초 단축, 2026-04-28)** |
| STATUS_REPORT | 10800초 (3시간) | 텔레그램 상태 리포트 |
| STATE_SAVE_INTERVAL | 60초 | `state/sol_state.json` 저장 |
| HEARTBEAT_INTERVAL | 300초 (5분) | `heartbeat_*.log` 기록 |
| 실시간 SL/TSL 체크 | 매 tick | `check_realtime_exit` 호출 |

**5분마다 CSV 저장**:
- 메모리 → `cache/sol_5m_YYYY.csv` 덮어쓰기
- 연도별 분리 (2025.csv, 2026.csv)

**6시간마다 정리**:
- CSV 중 180일 이전 데이터 제거

### 11.3 WebSocket 실시간 가격

```python
wss://fstream.binance.com/ws
→ solusdt@kline_1m (1분봉 스트림)
```

`current_price`, `current_high`, `current_low` 실시간 갱신. 봉 마감 전까지 미완성 봉 기준.

### 11.4 180일 보관 근거

- **최소 필요**: SMA(400) 15m = 100시간 = 4일 이상
- **여유분**: 6개월 = 7,200시간 여유 > 충분
- **AWS 디스크**: t3.micro 8GB 중 여유 2.3GB → 180일 CSV 약 12MB 경량
- **메모리**: 17,280봉 × 10 지표 × 8bytes = ~1.4MB

---

## 12. 실행 엔진 (Executor)

### 12.1 초기화 순서 (ETH V8 패턴, 2026-04-28 강화)

```python
1. exchange = ccxt.binanceusdm({'apiKey', 'secret'})
2. load_markets()
3. fetch_positions([SYMBOL]) → 포지션 유무 사전 확인
4-A. 포지션 있음:
       get_exchange_position() → self.leverage 동적 갱신
       (set_leverage 강제 호출 안 함 → 사용자 설정 보존)
4-B. 포지션 없음:
       set_leverage(LEVERAGE=5)
       self.leverage = LEVERAGE
5. set_margin_mode('isolated') (-4067은 INFO로 강등)
6. fetch_position_mode('one-way')
7. update_balance() → 잔액
8. _init_db() → SQLite 초기화
```

### 12.2 진입 실행

```python
def open_position(side, contracts, margin):
    # Market order
    order = exchange.create_order(
        symbol='SOL/USDT:USDT',
        type='market',
        side='buy' if side == 'LONG' else 'sell',
        amount=contracts,
        params={'reduceOnly': False}
    )
    # 체결 확인 (polling)
    wait_for_fill(order['id'])
    # 상태 저장
    save_position_state(side, contracts, entry_price, margin)
```

### 12.3 청산 실행

```python
def close_position(reason):
    order = exchange.create_order(
        symbol='SOL/USDT:USDT',
        type='market',
        side='sell' if pos == 'LONG' else 'buy',
        amount=pos_contracts,
        params={'reduceOnly': True}
    )
    pnl = compute_pnl(entry_px, exit_px, pos_contracts, fee, slippage)
    save_trade_to_db(pnl, reason)
    update_consec_losses(pnl < 0)
```

### 12.4 부분 청산 (Partial Exit)

TA 도달 시 50% 청산 + 나머지 50% TSL로 추적.

### 12.5 포지션 모니터링

**매 10초마다**:
- 포지션 있으면 SL/TSL/REV 체크
- SL 히트 → 즉시 청산
- TSL: 가격 새 피크 시 TSL 갱신
- REV: EMA 역크로스 시 청산

### 12.6 SL Orphan 청소 (ETH V8 패턴, 2026-04-28 추가)

**문제**: 단일 `sl_order_id` 추적은 거래소 재시작/네트워크 끊김 시 orphan SL 누적 가능.

**해결**: `_cancel_all_sl_orders()` — `params={'stop': True}`로 모든 conditional stop 주문 일괄 청소.

```python
async def _cancel_all_sl_orders(self):
    orders = await self.exchange.fetch_open_orders(SYMBOL, params={'stop': True})
    for o in orders:
        await self.exchange.cancel_order(o['id'], SYMBOL, params={'stop': True})
```

**호출 시점**: SL 등록 직전, 외부 청산 감지, 방향 전환, 사용자 청산 모두.

### 12.7 STOP_MARKET Fallback (ETH V8 패턴, 2026-04-28 추가)

`_set_stop_loss`에 3-tier fallback:
```python
methods = [
    ('STOP_MARKET',         self.exchange.create_order(SYMBOL, 'STOP_MARKET', side, qty, None, ccxt_params)),
    ('create_stop_loss_order', ...),  # ccxt unified
    ('create_trigger_order',   ...),  # ccxt unified
]
```

`STOP_MARKET`이 가장 안정적이라 첫 시도. 실패 시 ccxt unified API로 fallback.

### 12.8 외부 청산 실체결가 조회 (2026-04-28 추가)

거래소에서 SL이 자동 발동하거나 사용자가 수동 청산 시:
- `get_last_exit_price()` → `fetch_my_trades(SYMBOL, limit=5)`의 마지막 거래 가격
- 메모리 `current_price`보다 정확한 실제 체결가 사용
- DB에 정확한 PnL 저장

### 12.9 사용자 수동 포지션 3-케이스 동기화 (`_sync_position`, 2026-04-28 강화)

10초마다 거래소 vs 봇 state 비교:

| 케이스 | 조건 | 처리 |
|--------|------|------|
| **A** | 거래소 ✓ / 봇 ✗ | 수동 진입 감지 → `_check_manual_position` 호출 → V12로 트래킹 + SL 등록 + DB save_entry + 텔레그램 알림 |
| **B** | 거래소 ✗ / 봇 ✓ | 외부 청산 감지 → orphan SL 청소 + `get_last_exit_price` + `apply_exit('EXT')` + DB save_trade('USER') + 텔레그램 EXIT 알림 |
| **C** | 양쪽 ✓ but 방향 다름 | 사용자가 반대 포지션 잡음 → 옛 SL 청소 + 봇 포지션 close + 새 포지션 추적 |

---

## 13. 모니터링 시스템

### 13.1 4단계 모니터링

| 단계 | 수단 | 주기 | 목적 |
|---|---|---|---|
| 1 | **콘솔** (systemd journalctl) | 실시간 | 디버깅 |
| 2 | **웹 대시보드** | 5초~5분 | 상시 원격 |
| 3 | **Rich UI** (로컬) | 실시간 | 개발 |
| 4 | **텔레그램** | 이벤트 | 푸시 알림 |

### 13.2 웹 대시보드 (http://18.183.150.105:8081)

**6탭 구성** (2026-04-28 라이브 갱신 강화):

| 탭 | 기능 | meta refresh | JS polling |
|---|---|---|---|
| **Dashboard** | Balance/MDD/Price + 포지션(진입시각/보유시간/Lev) + 5중 필터 + 진입 판정 배너 | 30초 (fallback) | **시계 1초 / 데이터 10초** |
| **Chart** | Lightweight Charts 6개월 (17,280봉), 기본 7일 표시 | — | 5분 |
| **Trades** | 거래 히스토리 (페이지네이션) | 10초 | 시계 1초 |
| **Balance** | SOL 전용 누적 PnL + daily_summary.log | 15초 | 시계 1초 |
| **Errors** | error_*.log 필터 (ERROR/WARNING) | 30초 | 시계 1초 |
| **Logout** | 세션 종료 | — | — |

**모든 페이지 공통 헤더**: `🪙 SOL V1 🕐 YYYY-MM-DD HH:MM:SS` (JS Date 1초 갱신, 서버 호출 0)

**Dashboard 라이브 갱신 항목 (10초 polling)**:
- SOL Price (`/api/status` → `live-sol-price`)
- 포지션 현재가/PnL/ROI (`live-pos-current/pnl/roi`)
- 보유 시간 (1초 클라이언트 재계산, `data-entry-ts` 기반)
- V12 5중 필터 값 (ADX/RSI/LR/Mass/ATR ratio + EMA9/SMA400 + cross 방향)
- 레버리지 표시 (사용자 설정 동적 반영)

### 13.3 대시보드 필터 상태 표시

각 필터를 색상 코딩으로 표시:
- 🟢 **통과**
- 🔴 **차단**
- 🟡 **대기**

각 셀에 **조건 + 현재값 + 통과 여부** 3줄 표시.

**예시**:
```
ADX ≥ 22
현재: 49.4
🟢 통과
```

### 13.4 진입 판정 배너 (4단계)

```
🔴 Block: 게이트/필터/하드 차단 (Daily Loss, Mutex 등)
🟡 Warn: Delay만 차단 (시간만 지나면 OK)
🟢 Ready: 모든 조건 충족 (다음 봉 진입)
🟢 Ready: Mass 반전 트리거
```

### 13.5 텔레그램 알림 (2026-04-28 ETH V8 패턴 강화)

#### 알림 이벤트
- 봇 시작/종료
- 포지션 진입 (side, size, price, leverage)
- 포지션 청산 (reason, PnL, 실체결가)
- 사용자 수동 포지션 감지
- 사용자 레버리지 변경 감지 (1회)
- 3시간 자동 상태 리포트 (Position 섹션 포함: 진입시각/보유시간/ROI/PnL)
- Skip2@4loss 트리거
- 에러 경고

#### 한글 명령어 (대소문자/공백 무관, `/sol` 또는 `/SOL`)

| 명령 | 기능 |
|------|------|
| `/help` | 명령어 목록 (공용) |
| `/sol 상태` | 봇 전체 상태 (Price, Lev, Position, Balance, 통계) |
| `/sol 포지션` | 포지션 상세 (방향/Entry/Current/Lev/Size/Margin/PnL/ROI/SL/TSL/진입시각/보유시간) |
| `/sol 잔액` | 잔액 (Wallet/Available/Peak/MDD) |
| `/sol 최근거래` | 최근 5건 (DB trades → entries 폴백 → 메모리 폴백) |
| `/sol 봇정지` | systemd Restart 트리거 (자동 재시작) |
| `/sol 봇시작` | 상태 확인 후 재시작 |
| `/sol 청산` | 시장가 청산 + state 정리 + DB 저장 |

#### 메시지 큐 + 묵은 메시지 스킵
- `_send_loop` (asyncio.Queue) → 전송 실패 시 누락 방지
- `_skip_old_updates` → 봇 재시작 시 묵은 명령어 폭격 방지
- 재시도 3회 (HTTP 200 외 모두), 400/401/403/404는 즉시 중단

**봇 토큰**: `.env`의 `TELEGRAM_BOT_TOKEN` (안전하게 보관)

---

## 14. 로깅 전략

### 14.1 분리된 로그 파일

```
logs/
├── error_YYYYMMDD.log         # ERROR/WARNING 전용 (365일)
├── trades.log                  # 진입/청산 영구 기록
├── signals_YYYYMMDD.log        # 모든 신호 + 필터 스킵 (90일)
├── heartbeat_YYYYMMDD.log      # 5분마다 상태 스냅샷 (30일)
├── daily_summary.log           # UTC 00:00 일일 요약 (영구)
└── sol_trading_YYYYMMDD.log    # 통합 로그 (30일)
```

### 14.2 로그 레벨별 파일 매핑

- `error_*.log`: ERROR, WARNING 레벨만
- `trades.log`: 거래 이벤트만 (INFO, custom filter)
- `signals_*.log`: 신호 관련 (INFO)
- `heartbeat_*.log`: Heartbeat 이벤트만
- `sol_trading_*.log`: 모든 레벨 통합

### 14.3 Heartbeat 포맷

```
2026-04-23 15:24:06 Price $86.080 | Bars SOL 17281/BTC 4321 |
  ADX 49.4 RSI 46.7 Mass 24.94 | Balance $3,387.20 |
  Trades 0 WR 0.0% | Consec 0/4 Skip 0 | READY | NO_POSITION
```

### 14.4 Daily Summary 포맷

```
[2026-04-23 UTC] Start $3,385 → End $3,387 (+$2.20, +0.06%)
  Trades: 0 | WR: 0% | MaxProfit: $0 | MaxLoss: $0
  Filter pass: 12/24 15m bars (50%)
  Skip events: 0 | DailyLoss hit: No
```

---

## 15. AWS 배포

### 15.1 인프라

| 항목 | 값 |
|---|---|
| 리전 | ap-northeast-1 (도쿄) |
| 인스턴스 | t3.micro (free tier) |
| OS | Ubuntu 22.04 LTS |
| IP | 18.183.150.105 (Elastic IP) |
| SSH | ~/Downloads/eth-bot-key.pem |
| 사용자 | ubuntu |
| 작업 디렉토리 | /home/ubuntu/sol_v1 |
| 가상환경 | /home/ubuntu/sol_v1/sol_env |

### 15.2 동시 가동 봇

- `eth-bot.service`: ETH V8 (기존, 5일째)
- `sol-bot.service`: SOL V1 (신규)

**자원 사용**:
- 총 메모리: ~293MB / 911MB (32%)
- 디스크: 여유 2.3GB
- CPU: 평균 5% 미만

### 15.3 systemd 서비스

`/etc/systemd/system/sol-bot.service`:
```ini
[Unit]
Description=SOL V1 Futures Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sol_v1
ExecStart=/home/ubuntu/sol_v1/sol_env/bin/python sol_main_v1.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 15.4 Security Group

| 포트 | 프로토콜 | 소스 | 용도 |
|---|---|---|---|
| 22 | TCP | IP 제한 | SSH |
| 8080 | TCP | 0.0.0.0/0 | ETH V8 대시보드 |
| **8081** | **TCP** | **0.0.0.0/0** | **SOL V1 대시보드** |

**중요**: 8081은 2026-04-22 Chrome 자동화로 추가 (Inbound Rule).

### 15.5 배포 Workflow

```
1. 로컬 수정 (D:\filesystem\futures\sol_v1\*.py)
      ↓
2. Git commit + push (kang-jong-sun/filesystemgit)
      ↓
3. SCP 전송 (eth-bot-key.pem, ubuntu@18.183.150.105)
      ↓
4. AWS 문법 검증 (python -m py_compile)
      ↓
5. systemctl restart sol-bot.service
      ↓
6. 로그 확인 (tail -f logs/sol_trading_*.log)
```

**자동화**: `SOL_deploy_aws.bat` (Windows batch)

### 15.6 SSH 터널 (로컬 개발)

`SOL_tunnel.bat`:
```batch
ssh -i "C:\...eth-bot-key.pem" -L 8081:localhost:8081 ubuntu@18.183.150.105
```

로컬 `http://localhost:8081`로 AWS 대시보드 접근.

---

## 16. 백테스트 검증 이력

### 16.1 51개월 백테스트

- **데이터**: 2022-01-01 ~ 2026-04-01 (450,573 5m봉)
- **위치**: `D:\filesystem\futures\CUDA\data_51m\sol\sol_5m_{2022..2026}.csv`
- **엔진**: `engine_data.prepare_data()` (Wilder 지표 계산)

### 16.2 주요 결과

| 전략 | Return | PF | MDD | Trades | WR |
|---|---|---|---|---|---|
| V12.1 (session only) | +1,855.0% | 2.04 | 17.8% | 301 | - |
| V12.1 + Margin Scaling | **+2,001.7%** | **2.13** | **19.8%** | 301 | - |
| Mass Index 단독 | +405.8% | 3.01 | 83.7% | 20 | 40% |
| **V12:Mass 75:25 Mutex** | **+1,649.6%** | **2.33** | **16.5%** | **291** | - |
| V12:Mass 65:35 Mutex | +1,455.9% | 2.34 | 15.2% | 291 | - |
| V12:Wyckoff 75:25 | +1,492.3% | 2.12 | 16.5% | 302 | - |

### 16.3 Walk-Forward 5-Fold OOS

| Fold | V12.1 | V12:Mass 75:25 |
|---|---|---|
| 1 (2022) | +320% | +280% |
| 2 (2023) | +510% | +420% |
| 3 (2024) | +380% | +340% |
| 4 (2025) | +290% | +310% |
| 5 (2026 Q1) | +150% | +220% |

**평균**: +330% / Fold (robust, 표준편차 < 평균)

### 16.4 5가지 안전 검증 (CLAUDE.md 기준)

| 검증 | 결과 | 상태 |
|---|---|---|
| 1-bar delay (원본 대비) | 85% 유지 | ✅ PASS |
| 타임스탬프 검증 | 미래 봉 0회 | ✅ PASS |
| Walk-Forward OOS | 평균 +330%, std < mean | ✅ PASS |
| 거래 로그 분석 | 평균 dur 12 bars | ✅ PASS |
| 체결 모델링 | 봉 범위 내 | ✅ PASS |

**신뢰도**: **HIGH** (5/5 통과)

### 16.5 봇-백테스트 일치성 (2026-04-23 재검증)

**봇 Wilder 출력 vs engine_data Wilder 출력**:
- RSI 최대 차이: **0.000000** ✅
- ADX 최대 차이: **0.000000** ✅

**V12.1 백테스트 재실행 결과**:
- +2,001.7% (이전 +2,003%와 일치)

**V31:Mass 75:25 재실행 결과**:
- +1,649.6% (이전 +1,649%와 일치)

---

## 17. 현재 상태

### 17.1 2026-04-23 15:24 KST

| 항목 | 값 |
|---|---|
| 봇 PID | 184010 |
| 메모리 | 198.8MB |
| 업타임 | 실행 중 (Wilder 적용 후) |
| 잔액 | $3,387.20 |
| 포지션 | 없음 |
| 거래 수 | 0 (세션) |
| 연속 손실 | 0/4 |
| Skip 카운터 | 0 |
| Daily PnL | +0.06% |
| 가격 | $86.08 (SOL/USDT) |

### 17.2 지표 현황 (Wilder 적용)

| 지표 | 값 | 필터 통과? |
|---|---|---|
| EMA9 | $85.875 | — |
| SMA400 | $85.900 | EMA9 < SMA400 → SHORT 방향 |
| ADX(20) | 49.4 | ✅ ≥ 22 |
| RSI(10) | 46.7 | ✅ 30-65 범위 |
| LR slope(14) | +0.017 | ✅ ±0.5 이내 |
| Mass | 24.94 | 대기 (< 27) |
| BTC EMA200 (1h) | 75,706 | Bull (close 78,149) |

### 17.3 대기 중인 신호

- **V12 CROSS_DN** (bar#17279, 13:21:44 KST)
- SHORT watch 시작 (60분 delay 대기 중)
- Skip Same 필터로 대기 중 (정상)

---

## 18. 운영 절차

### 18.1 매일 아침 체크 (KST 09:00 = UTC 00:00 리셋 직후)

1. 대시보드 접속 → Balance 탭에서 전일 PnL 확인
2. Errors 탭에서 overnight 오류 확인
3. Trades 탭에서 체결 내역 검증
4. `journalctl -u sol-bot.service -n 50` 로 로그 스캔

### 18.2 주 1회 체크 (일요일)

1. AWS 디스크 용량 (`df -h`)
2. 메모리 사용률 (`free -h`)
3. CSV 파일 크기 / 정리 상태
4. DB 크기 (`ls -lh sol_trading_bot.db`)
5. 주간 거래 통계 (Balance 탭)

### 18.3 긴급 정지

```bash
# SSH 접속 후
sudo systemctl stop sol-bot.service

# 포지션 있으면 수동 청산 권장
# (혹은 바이낸스 웹에서 직접)
```

### 18.4 재배포 (코드 수정 후)

```batch
REM 로컬 PC에서
D:\filesystem\futures\sol_v1\SOL_deploy_aws.bat
```

또는 수동:
```bash
# 로컬
cd D:\filesystem\futures
git add sol_v1/
git commit -m "..."
git push

# AWS로 전송
scp -i key.pem sol_v1/sol_*.py ubuntu@18.183.150.105:/home/ubuntu/sol_v1/

# AWS에서
ssh -i key.pem ubuntu@18.183.150.105
sudo systemctl restart sol-bot.service
sudo systemctl status sol-bot.service
```

---

## 19. 안전 장치 & 알려진 이슈

### 19.1 안전 장치

| 안전 장치 | 목적 |
|---|---|
| **Isolated Margin** | 포지션별 리스크 격리 |
| **One-Way Mode** | LONG/SHORT 혼동 방지 |
| **Daily Loss Limit -2.5%** | 하루 최대 손실 제한 |
| **Skip2@4loss** | 연패 시 쿨다운 |
| **Mutex** | V12/Mass 동시 진입 차단 |
| **Entry Delay 5봉** | 가짜 크로스 진입 방지 |
| **Monitor Window 24봉** | 오래된 신호 폐기 |
| **Catch-up 버그 방지** | 봇 재시작 시 과거 신호 일괄 진입 차단 |
| **SL Hard Stop** | -3.8% 고정 (이동 없음) |
| **TSL 갱신** | 수익 구간에서만 유리한 방향으로 이동 |

### 19.2 Catch-up 버그 방지 (2026-04-21 발견/수정)

**증상**: 봇 재시작 시 과거 크로스 신호들이 일괄로 체크되어 여러 포지션 동시 진입 위험

**수정**:
- 최신 봉의 신호만 체크 (i = len-1)
- `last_processed_bar` 저장하여 이전 봉 스킵
- 재시작 후 Entry Delay 타이머 리셋

### 19.3 알려진 이슈 & 해결

| 이슈 | 상태 | 해결 |
|---|---|---|
| 봇 ADX/RSI Simple MA → 백테스트 Wilder 불일치 | ✅ 해결 (2026-04-23) | Wilder 교체 (d12889d) |
| Daily_pct UnboundLocalError | ✅ 해결 | Inline 계산 (27fc7ad) |
| 10,000 bar 메모리 한계 | ✅ 해결 | 17,280 (180일) 상향 (30c8627) |
| 차트 17,280봉 렌더링 깨짐 | ✅ 해결 | setVisibleRange 7일 기본 (565d401) |
| Entry Delay 중 "READY" 표시 | ✅ 해결 | 4단계 배너 (033588a) |
| AWS Port 8081 접근 불가 | ✅ 해결 | Chrome 자동화로 SG 수정 |
| 봇 재시작 시 Watch 상태 손실 | ✅ 해결 (2026-04-23) | save_state 즉시 호출 + TIME 기반 체크 (3567eab) |
| Binance API 순간 끊김 ERROR | ✅ 해결 | _fetch_ohlcv_retry 3회 재시도 (6f34f31) |
| Telegram Server disconnected | ✅ 해결 | 재시도 로직 (6f34f31) |
| 마진모드 -4067 WARNING 반복 | ✅ 해결 | INFO 로 강등 (6f34f31) |
| **차트 최근 봉 얇은 "-" 고정 현상** | ✅ **해결 (2026-04-24)** | **df_sol 진행 중 봉 덮어쓰기 (a6dc5ca)** |
| SL orphan 누적 (단일 ID 추적의 한계) | ✅ **해결 (2026-04-28)** | `_cancel_all_sl_orders` + STOP_MARKET fallback (3e8c773) |
| POSITION_SYNC 60초 → 사용자 수동 행동 감지 늦음 | ✅ **해결 (2026-04-28)** | 10초로 단축 (3e8c773) |
| 사용자 반대 방향 포지션 진입 시 봇 추적 불가 | ✅ **해결 (2026-04-28)** | `_sync_position` 케이스 C 추가 (3e8c773) |
| 외부 청산 PnL 부정확 (current_price 사용) | ✅ **해결 (2026-04-28)** | `get_last_exit_price` 추가 (3e8c773) |
| 봇 재시작 시 묵은 텔레그램 명령어 폭격 | ✅ **해결 (2026-04-28)** | `_skip_old_updates` (3e8c773) |
| 텔레그램 명령어 대소문자/공백 차이로 미인식 | ✅ **해결 (2026-04-28)** | `/sol 한글` 패턴 + 대소문자 무관 파서 (3e8c773) |
| 사용자 임의 레버리지(5x/10x) 봇 미인지 | ✅ **해결 (2026-04-28)** | `get_exchange_position` leverage 동적 추출 (5bb4f4f) |
| 대시보드 시계 5초 단위 (1초 요청) | ✅ **해결 (2026-04-28)** | JS Date 1초 갱신 (21d4a83) |
| 다른 페이지(Chart/Trades/Balance/Errors) 시계 정지 | ✅ **해결 (2026-04-28)** | 4개 페이지에 updateClock 추가 (eaaee50) |
| 포지션 진입시각/보유시간 미표시 | ✅ **해결 (2026-04-28)** | `data-entry-ts` 기반 1초 재계산 (03f8384) |
| Dashboard 데이터 polling 1초 부담 | ✅ **해결 (2026-04-28)** | 10초로 조정 + 필터 라이브 갱신 (9180a3e) |
| 수동 포지션 진입 시 DB save_entry 누락 | ✅ **해결 (2026-04-28)** | `_check_manual_position`에 save_entry 호출 추가 (97c381f) |
| `/sol 최근거래` DB 비어있을 때 정보 없음 | ✅ **해결 (2026-04-28)** | trades→entries→메모리 3단 폴백 (97c381f) |
| REV cross 진단 로그 부재 | ✅ **해결 (2026-04-28)** | REV-CHECK / REV TRIGGERED / REV-SKIP 로그 추가 (42b5753) |
| /sol 상태/포지션 ROI=-100% 잘못 표시 | ✅ **해결 (2026-04-28)** | price 폴백(df_sol close) 추가 (d1feb99) |
| 3시간 텔레그램 리포트에 포지션 정보 누락 | ✅ **해결 (2026-04-28)** | notify_status에 Position 섹션 추가 (d1feb99) |
| **V12 REV가 cross 봉을 놓치면 영영 발동 못 함** | ✅ **해결 (2026-04-28)** | **REV-GUARD 추가: 진입 후 4봉 + 추세 역방향 강제 EXIT (861ad79)** |

### 19.3.2 REV-GUARD 추가 상세 (2026-04-28)

**문제**: 4/27 15:45에 EMA9 cross_dn이 발생했으나 봇이 사용자 LONG 포지션을 REV로 청산 못함.

**원인 분석**:
- REV 트리거 조건: `dn = (fm < sm AND pfm >= psm)` — cross 발생 그 1봉에서만 True
- 데이터 동기화 0.5초만 어긋나도 그 봉을 놓치면 다음 봉부터 `pfm < psm` → `dn = False` → 영영 발동 못 함
- signals_*.log에 4/27 cross 기록 0건 (진단 로그 부재로 정확한 원인 미파악)

**수정** (sol_core_v1.py _check_exit REV 블록):
```python
USE_REV_GUARD = True
REV_GUARD_BARS = 4   # 15m × 4 = 1시간

# 기존 cross 감지 후 추가
if USE_REV_GUARD:
    bars_held = bar_idx - pos.entry_bar
    if bars_held >= REV_GUARD_BARS:
        counter_trend = ((pos.direction == 1 and fm < sm) or
                         (pos.direction == -1 and fm > sm))
        if counter_trend:
            signal_log.info(f"V12 REV-GUARD ... → EXIT (cross 봉 놓침 가드)")
            return Signal(action='EXIT', exit_type='REV',
                          reason=f"REV-GUARD ({bars_held}봉 후 추세 역전)")
```

**효과**:
- 4/27 시나리오 시뮬레이션: 사용자 LONG @4/25 06:13 + 4/27 15:45 cross_dn (놓침)
  - 기존: REV 미발동 → 4/28 사용자가 직접 청산할 때까지 손실 누적
  - **수정 후**: 16:45 (4봉 = 1시간 후) 추세 역방향 감지 → 자동 EXIT
- 진입 직후 1시간 이내는 기존 cross 감지에만 의존 (가짜 청산 방지)
- 백테스트 V12 결과에 영향 미미 (백테스트는 cross 정확히 처리, GUARD 거의 발동 안 함)

**파라미터 튜닝 가이드**:
- `REV_GUARD_BARS = 4` (1시간) — 사용자 수동 진입 보호 + 정상 신호 대기 시간 균형
- 너무 작게(예: 1봉) → 진입 직후 가짜 청산 위험
- 너무 크게(예: 24봉=6시간) → 추세 역방향에 너무 오래 노출됨

### 19.3.1 차트 깨짐 해결 상세 (2026-04-24, 7차 수정 여정)

**문제 증상**: SOL V1 차트에서 최근 3~4개 15m 봉이 얇은 수평선 "-" 모양으로 고정 표시. ETH V8은 정상.

**7번 수정 여정**:

| 차수 | 접근 | 수정 위치 | 결과 |
|---|---|---|---|
| 1차 | setVisibleRange + 복잡한 타이밍 | 프론트 | ❌ |
| 2차 | barSpacing + 3단계 RAF/setTimeout | 프론트 | ❌ |
| 3차 | ETH V8 방식 단순화 (500봉) | 프론트 | ❌ |
| 4차 | 6개월 전체 + 단순 로직 복구 | 프론트 | ❌ |
| 5차 | 서버 last_bar 직접 반환 | 프론트 + API | ❌ |
| 6차 | 최근 5봉 전체 update | 프론트 + API | ❌ |
| **7차** | **서버 df_sol 덮어쓰기 + _raw_5m 리샘플** | **서버 데이터 파이프라인** | ✅ **해결** |

**진짜 근본 원인** (서버 측 2가지 버그):

1. **버그 A**: 기존 15m 봉 업데이트 누락
   ```python
   # sol_data_v1.py 기존 코드
   new_rows = df_new15[~df_new15.index.isin(self.df_sol.index)]  # 기존 index 제외!
   if len(new_rows) > 0:
       self.df_sol = pd.concat([self.df_sol, new_rows])
   ```
   - 15:00 봉이 처음 좁은 OHLC로 저장 → 이후 5m 봉 추가되어도 업데이트 안 됨
   - 진행 중 15m 봉이 영구히 좁은 OHLC로 고정

2. **버그 B**: `since` 필터로 5m 봉 누락
   ```python
   since = self.last_candle_time + 1  # 15:00:00.001
   # → 15:00:00 5m 봉 제외됨
   # → df_new5에 해당 봉 빠짐
   # → resample 결과 15m 봉 OHLC 부정확
   ```

**7차 수정** (서버 측):

```python
# A. _raw_5m_sol로 완전한 리샘플 (최근 30개 5m 봉 = 150분)
recent_5m = self._raw_5m_sol.tail(30)
df_new15 = self._resample_to(recent_5m, TIMEFRAME_PRIMARY)

# B. 기존 봉도 덮어쓰기
self.df_sol = pd.concat([self.df_sol, df_new15])
self.df_sol = self.df_sol[~self.df_sol.index.duplicated(keep='last')].sort_index()
```

**왜 1~6차가 모두 실패했나**:
- 1~6차 모두 **프론트엔드 렌더링만** 수정
- 서버 `df_sol`에 이미 틀린 OHLC 데이터가 들어 있었음
- 프론트가 서버 데이터 그대로 표시하니 정확히 그리려 할수록 "얇은 봉" 그대로 재현
- 7차에서 서버 데이터 파이프라인을 수정하자 즉시 해결

**프론트엔드 최종 구조** (6차 수정 유지):
```javascript
// 5초마다 /api/status → 최근 5봉 OHLC 받음
// 마감된 봉도 서버 df_sol의 최종 OHLC로 갱신
for (const bar of d.last_bars) {
    candleSeries.update(bar);
}
```
이제 서버 데이터가 정확하므로 프론트 5봉 갱신이 제대로 작동.

**BTC 1h 봉도 동일 수정** (`_raw_5m_btc.tail(30)` + concat + duplicated).

**교훈**:
- 증상이 프론트에서 보여도 원인은 서버에 있을 수 있음
- pandas resample + 중복 처리는 정교한 로직 필요 (`~isin()` 대신 `duplicated(keep='last')` 선호)
- `since` 파라미터는 봉 경계에서 **관대한 범위** 설정 필요

### 19.4 모니터링 권장 지표

**이상 신호**:
- 🚨 Daily PnL < -2%
- 🚨 연속 손실 3회 이상
- 🚨 포지션 5시간 이상 유지
- 🚨 ADX 15 이하로 떨어짐 (추세 상실)
- 🚨 Mass Index 30 이상 (극단적 변동)

---

## 20. 로드맵

### 20.1 단기 (2026 Q2)

- [ ] 1-2주 실전 운영 모니터링 (Wilder 적용 후)
- [ ] 백테스트 결과와 실전 성과 비교 (월간)
- [ ] Telegram 알림 세밀화 (심각도별 필터)
- [ ] 대시보드 MDD 실시간 차트 추가

### 20.2 중기 (2026 Q3)

- [x] **레버리지 2x → 5x 상향** (2026-04-28 완료, 추가 5x → 10x 상향 검토 중)
- [ ] 5x → 10x 상향 (5x 안정성 확인 후)
- [ ] BTC regime 확장 (1d EMA200 추가)
- [ ] Partial Exit 최적화 (30% / 50% / 70%)
- [ ] 추가 자산 (ETH V8 보유 중) 상관성 분석

### 20.3 장기 (2026 Q4~)

- [ ] 멀티 자산 포트폴리오 (SOL+ETH+BTC)
- [ ] 머신러닝 기반 Confidence 재학습
- [ ] Market Regime 감지 고도화 (Choppy/Trend/Step-Up)
- [ ] 예약 인스턴스 (3년 선결제, 2027-04 이후 $134)

---

## 부록 A: 주요 파일 전체 목록

```
D:\filesystem\futures\sol_v1\
├── .env                           # API 키 (gitignore)
├── .gitignore                     # 제외 파일 목록
├── README.md                      # 간략 사용법
├── SOL_V1_기획안.md               # ★ 이 문서
├── SOL_deploy_aws.bat             # AWS 배포 자동화
├── SOL_monitor.bat                # Rich UI 모니터 시작
├── SOL_start.bat                  # 봇 시작 (로컬)
├── SOL_tunnel.bat                 # SSH 터널 (Windows 시작 시)
├── requirements.txt               # 의존성 (ccxt, pandas, numpy 등)
├── sol_main_v1.py                 # 메인 루프
├── sol_core_v1.py                 # 트레이딩 로직
├── sol_data_v1.py                 # 데이터 + 지표 (Wilder 적용)
├── sol_executor_v1.py             # 주문 실행
├── sol_web_v1.py                  # FastAPI 대시보드
├── sol_telegram_v1.py             # 텔레그램 알림
├── sol_monitor.py                 # Rich UI
├── sol_trading_bot.db             # SQLite DB
├── fetch_6months.py               # 6개월 데이터 수집 유틸
├── verify_wilder_vs_simplema.py   # Wilder 검증 스크립트
├── test_connection.py             # 연결 테스트
├── test_full.py                   # 통합 테스트
├── cache/                         # CSV 저장소
│   ├── sol_5m_2025.csv
│   ├── sol_5m_2026.csv
│   ├── btc_5m_2025.csv
│   └── btc_5m_2026.csv
├── logs/                          # 로그 디렉토리
│   ├── error_*.log
│   ├── trades.log
│   ├── signals_*.log
│   ├── heartbeat_*.log
│   ├── daily_summary.log
│   └── sol_trading_*.log
├── state/                         # 상태 저장
│   └── sol_state.json
└── templates/                     # 웹 템플릿 (레거시)
```

---

## 부록 B: 환경변수 (.env)

```bash
# Binance API
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# 웹 대시보드
WEB_USERNAME=admin
WEB_PASSWORD=kang1366

# 봇 설정
LIVE_MODE=true         # true=실전, false=시뮬
LOG_LEVEL=INFO
```

---

## 부록 C: 트러블슈팅

### C.1 봇이 신호 못 잡음
1. 대시보드 Dashboard 탭 → 5중 필터 상태 확인
2. 어떤 필터에서 차단 중인지 파악
3. 현재 시장 조건 (ADX 너무 낮음, RSI 양극단 등)

### C.2 진입 실행 실패
1. `tail -50 logs/error_*.log`
2. API 키 확인 (`.env`)
3. 잔액 확인 (`fetch_balance()` 로그)
4. 마진 부족 시: 레버리지 상향 or 복리 비율 하향

### C.3 봇 재시작 후 차트 이상
1. CSV 무결성 확인 (`python test_csv.py`)
2. 필요 시 `python fetch_6months.py` 재실행

### C.4 AWS 접속 안 됨
1. Security Group 확인 (8081, 22 허용)
2. `systemctl status sol-bot.service`
3. `journalctl -u sol-bot.service -n 100`

---

## 부록 D: Git 이력 (최신순)

```
a6dc5ca  차트 깨짐 진짜 근본 해결 (7차): 서버 df_sol 진행 중 봉 갱신 버그 수정
78c9541  차트 깨짐 근본 해결 (6차): 최근 5봉 전체 갱신
1693ccc  차트 깨짐 근본 해결 (5차): 클라이언트 OHLC 추정 완전 제거
b351e70  차트 깨짐 근본 해결 (4차): loadChart 주기 제거 + 봉 연속성
8110b52  차트 깨짐 수정 (3차): loadChart 30초 → 15분
f7ea8a4  차트 실시간 업데이트: TV와 동기화
3567eab  Watch 상태 영속화: Cross 즉시 save + TIME 기반 delay/만료 체크
6f34f31  에러 로그 정리: 마진모드 -4067 필터 + 네트워크 에러 재시도
d12889d  SOL V1: RSI/ADX Wilder smoothing 교체 (TV + V12.1 백테스트 완전 일치)
033588a  SOL V1 판정 배너: Entry Delay 차단 구분 (4단계)
27fc7ad  Hotfix: daily_pct UnboundLocalError
c22ace4  대시보드: 필터 조건 명시 + 진입 판정 표시 (268줄 추가)
565d401  차트 근본 수정: 트레이딩뷰 스타일 기본 7일 범위
30c8627  차트 6개월 완전 보존 (10000봉 제한 해제)
576cfea  데이터 로드: 6개월 완전 로드 (CANDLE_LIMIT 51840)
709976b  Chart + Errors 메뉴 + Lightweight Charts
affb332  Dashboard/Trades/Balance 3탭 메뉴
bedcfee  SOL V1 실전 봇 최초 커밋 (4731줄)
```

---

## 부록 E: 참고 문서

- **백테스트 엔진**: `D:\filesystem\CLAUDE.md` (Look-ahead Bias 방지 규칙)
- **V12.1 백테스트**: `D:\filesystem\futures\data\sol\orchestra_v2\v19_v12_reproduce.py`
- **V31 Mass Mutex**: `D:\filesystem\futures\data\sol\orchestra_v2\v31_new4_methods.py`
- **CUDA 엔진**: `D:\filesystem\futures\CUDA\engine_data.py` (Wilder 원본)
- **기존 보고서**:
  - `D:\filesystem\futures\data\sol\상세_보고서_v12.md`
  - `D:\filesystem\futures\data\sol\V13_확정_운영_명세서.md`

---

**문서 작성**: Claude Code
**최종 수정**: 2026-04-28 KST (REV-GUARD 추가 — cross 봉 놓침 방지)
**상태**: 운영 중 (sol-bot.service active, Wilder + 서버 df_sol + ETH V8 패턴 + Lev 5x + REV-GUARD 적용)

## 📝 문서 개정 이력

| 버전 | 날짜 | 주요 변경 |
|---|---|---|
| v1.0 | 2026-04-23 | 초기 기획안 작성 (20 섹션) |
| v1.1 | 2026-04-23 | Wilder 일치 수정 반영 |
| v1.2 | 2026-04-24 | 차트 깨짐 7차 수정 이력 추가 (19.3.1 신설) + Watch 영속화 + 에러 로그 정리 |
| v1.3 | 2026-04-28 | ETH V8 패턴 통합 (대규모):<br>• 레버리지 2x → **5x** + 사용자 동적 추적<br>• `_cancel_all_sl_orders` + `STOP_MARKET` fallback<br>• `get_last_exit_price` 외부 청산 실체결가<br>• POSITION_SYNC 60→10초<br>• `_sync_position` 3-케이스 (A/B/C)<br>• 텔레그램 메시지 큐 + 한글 명령어 + 묵은 메시지 스킵<br>• REV 진단 로그 (REV-CHECK/TRIGGERED/SKIP)<br>• 수동 포지션 DB save_entry 누락 수정<br>• 대시보드 시계 1초 / 데이터 10초 라이브<br>• 포지션 진입시각/보유시간/Leverage 표시<br>• 5개 페이지 시계 동기화<br>• 3시간 리포트 포지션 섹션 추가 |
| **v1.4** | **2026-04-28** | **REV-GUARD 추가 (4/27 사고 대응)**:<br>• `USE_REV_GUARD = True`, `REV_GUARD_BARS = 4` (1시간)<br>• 진입 후 4봉 경과 + 추세 역방향이면 강제 EXIT<br>• V12 REV 한계(cross 봉 놓침) 보완<br>• §3.5 청산 규칙 / §1.4 주요 혁신 / §19.3.2 신설 |
