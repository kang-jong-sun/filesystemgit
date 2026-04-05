# ALT/USDT 선물 자동매매 봇

## 개요
바이낸스 선물 거래소에서 ALT/USDT 페어를 자동으로 거래하는 트레이딩 봇입니다.
HMA, DEMA, VWMA, WMA, EMA 및 MACD 지표를 조합하여 진입 신호를 생성합니다.

## 주요 기능
- 5분봉 기반 실시간 트레이딩
- 다중 지표 각도 분석을 통한 정확한 진입
- 트레일링 스톱로스를 통한 리스크 관리
- 포지션 상태 자동 저장 및 복구
- 실시간 모니터링 대시보드

## 설치 방법

### 1. 필수 요구사항
- Python 3.8 이상
- 바이낸스 API 키 (선물 거래 권한 필요)

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정
`.env.example` 파일을 복사하여 `.env` 파일을 생성하고 API 키 입력:

```bash
cp .env.example .env
```

`.env` 파일 수정:
```
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_secret_key
```

## 실행 방법

```bash
python alt_main.py
```

## 트레이딩 전략

### 지표 계산 체인
1. HMA(2) ← HLC3
2. DEMA(10) ← HMA
3. VWMA(1) ← DEMA
4. WMA(10) ← VWMA
5. EMA(10) ← VWMA
6. MACD(5,800,1) ← VWMA

### 진입 조건
**매수 (LONG)**
- 혼합각도 273°~357° (정배열)
- MACD 신호선 각도 273°~357°

**매도 (SHORT)**
- 혼합각도 3°~87° (역배열)
- MACD 신호선 각도 3°~87°

## 리스크 관리
- 기본 손절: -5%
- 동적 트레일링 스톱 (10%~15% 수익 구간)
- 포지션 크기: 계정의 5%
- 레버리지: 2배

## 파일 구조
```
ALTUSDT_FUTURES/
├── alt_main.py         # 메인 실행 파일
├── alt_core.py         # 트레이딩 로직
├── alt_indicators.py   # 지표 계산
├── alt_data.py         # 데이터 수집
├── alt_executor.py     # 주문 실행
├── alt_monitor.py      # 모니터링
├── .env.example        # 환경변수 예제
├── requirements.txt    # 의존성 목록
└── README.md          # 문서
```

## 주의사항
- 실제 자금으로 거래하기 전에 테스트넷에서 충분히 테스트하세요
- API 키는 절대 공유하지 마세요
- 시장 상황에 따라 손실이 발생할 수 있습니다

## 로그 및 데이터
- 거래 로그: `logs/` 디렉토리
- 거래 기록: `alt_trading_bot.db` SQLite 데이터베이스