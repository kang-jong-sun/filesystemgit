# 🚀 SOL Futures 자동매매 시스템 v4.0

## 📋 개요
Quadruple EMA Touch 전략을 사용하는 SOL/USDT 선물 자동매매 시스템입니다.
4개의 EMA (EMA10, EMA21-시가/고가/저가)를 활용하여 EMA10이 EMA21의 고가/저가 기준선에 터치할 때 진입하는 전략으로, 기본적으로 순수 기술적 지표로 작동하며, 선택적으로 AI 분석 기능을 활용할 수 있습니다.

## 🔴 주요 특징

### 전략
- **메인 지표**: Quadruple EMA Touch Strategy
  - EMA10_Open: 시가(Open) 기준
  - EMA21_Open: 시가(Open) 기준
  - EMA21_HIGH: 고가(High) 기준
  - EMA21_LOW: 저가(Low) 기준
- **타임프레임**: 30분봉 (메인), 1m/3m/5m/15m (보조)

### 📈 매수(Long) 포지션 진입 조건
**조건 1 (근접 터치):**
- EMA10_open > EMA21_open (정배열)
- EMA21_high 0.2% 이내 근접
- 3분(180초) 이상 유지
- 1m/3m/5m/15m 중 2개 이상 정배열 3분 이상 유지

**조건 2 (상향 돌파):**
- EMA10_open > EMA21_open (정배열)
- EMA10_open > EMA21_high (돌파)
- 3분(180초) 이상 유지
- 1m/3m/5m/15m 중 2개 이상 정배열 3분 이상 유지

### 📉 매도(Short) 포지션 진입 조건
**조건 1 (근접 터치):**
- EMA10_open < EMA21_open (역배열)
- EMA21_low 0.2% 이내 근접
- 3분(180초) 이상 유지
- 1m/3m/5m/15m 중 2개 이상 역배열 3분 이상 유지

**조건 2 (하향 돌파):**
- EMA10_open < EMA21_open (역배열)
- EMA10_open < EMA21_low (돌파)
- 3분(180초) 이상 유지
- 1m/3m/5m/15m 중 2개 이상 역배열 3분 이상 유지

### 포지션 관리
- **마진 모드**: 격리마진 (ISOLATED)
- **포지션 크기**: 계정 잔액의 20%
- **레버리지**: 10배
- **보유 시간**: 무제한

### 포지션 청산 조건
1. **손절**: ROI -30% 도달시
2. **역신호**: 반대 진입 신호 발생시 즉시 청산 후 포지션 전환
3. **긴급청산**:
   - 롱 포지션: 1m/3m/5m 모두 120분간 연속 역배열시
   - 숏 포지션: 1m/3m/5m 모두 120분간 연속 정배열시

### 리스크 관리
- **손절**: -20%
- **익절**: 없음 (역신호까지 무제한 보유)
- **트레일링 스톱**: ROI 30%에서 손익분기점 이동, 최고점에서 15% 트레일링
- **최대 보유시간**: 무제한

### 특수 기능
- **수동 포지션 추적**: 사용자가 수동으로 진입한 포지션도 자동 관리
- **텔레그램 알림**: 모든 주요 이벤트 실시간 알림
- **다중 타임프레임 분석**: 5개 타임프레임 동시 모니터링
- **AI 분석**: GPT-5를 통한 선택적 시장 분석 (OPENAI_API_KEY 필요)

## 📁 파일 구조
```
sol_main_v4.py         # 메인 실행 파일 (환경 설정 및 시스템 초기화)
sol_core_v4.py         # 핵심 트레이딩 로직 (거래 설정 및 전략 관리)
sol_data_v4.py         # 데이터 수집 및 EMA 계산 (Binance API 연동)
sol_executor_v4.py     # 주문 실행 및 포지션 관리 (거래 통계 추적)
sol_monitor_v4.py      # 실시간 모니터링 (30초 간격 상태 표시)
sol_telegram_v4.py     # 텔레그램 알림 (비동기 메시지 전송)
sol_ai_analyzer_v4.py  # AI 시장 분석 (선택적 GPT-4o 분석)

# 기타 파일
requirements.txt       # 필수 패키지 목록
setup_env.py          # 환경 설정 도우미
.env                  # API 키 및 환경 변수 (직접 생성 필요)

# 로그 및 데이터
logs/                 # 거래 로그 파일
monitoring_logs/      # 모니터링 로그 파일
sol_trading_bot.db    # SQLite 거래 데이터베이스
```

## 🔧 설치 및 설정

### 1. 환경 변수 설정
`.env` 파일 생성:
```env
# Binance API (필수)
BINANCE_API_KEY="your_binance_api_key"
BINANCE_API_SECRET="your_binance_api_secret"

# 텔레그램 봇 (필수)
TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
TELEGRAM_CHAT_ID="your_telegram_chat_id"

# OpenAI API (선택적 - AI 분석 사용시)
OPENAI_API_KEY="your_openai_api_key"

# 기타 API (선택적)
NEWSDATA_IO__KEY="your_newsdata_key"
COINGECKO_API_KEY="your_coingecko_key"

# WebSocket URL
BINANCE_WS_FUTURES_URL='wss://fstream.binance.com/ws'
```

### 2. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

또는 개별 설치:
```bash
pip install ccxt>=4.0.0 pandas>=2.0.0 numpy>=1.24.0 python-dotenv>=1.0.0 aiohttp>=3.8.0
pip install openai>=1.0.0 rich>=13.0.0 aiosqlite>=0.19.0 websockets>=11.0
```

### 3. 실행
```bash
python sol_main_v4.py
```

## 📊 모니터링 화면
30초마다 업데이트되는 정보:
- 현재가 및 Quadruple EMA 값
- EMA 상태 (정배열/역배열)
- 포지션 정보 (방향, 수량, ROI, 보유시간)
- SL 레벨 및 트레일링 정보
- 부분청산 여부
- 수동 포지션 추적 상태
- 거래 통계

## 📱 텔레그램 알림
- 프로그램 시작/종료
- 포지션 진입/청산 (크로스 타입 포함)
- 부분청산 실행
- 트레일링 스톱 업데이트
- 최대 보유시간 도달
- 수동 포지션 추적 시작
- 시스템 오류

## ⚠️ 주의사항
1. **실거래 모드**: 실제 자금이 거래됩니다
2. **충분한 테스트**: 소액으로 충분히 테스트 후 사용
3. **시장 리스크**: 급격한 시장 변동시 손실 가능
4. **API 권한**: Binance API에 선물거래 권한 필요
5. **네트워크**: 안정적인 인터넷 연결 필수

## 🛑 안전한 종료
- Ctrl+C를 눌러 안전하게 종료
- 열린 포지션이 있을 경우 경고 메시지 표시
- 모든 컴포넌트가 순차적으로 정리됨

## 📈 수동 포지션 추적
프로그램 시작시 이미 열려있는 포지션을 자동으로 감지하여:
- SL 자동 설정
- 트레일링 시스템 적용
- 부분청산 관리
- 최대 보유시간 관리

## 🔍 로그 파일
- 위치: `logs/sol_trading_YYYYMMDD.log`
- 모든 거래 활동 기록
- 오류 추적 및 디버깅 정보

## 📝 데이터베이스
- 파일: `sol_trading_bot.db`
- 거래 기록 저장
- EMA 크로스 기록
- 통계 분석용 데이터

## 🚀 버전 정보
- Version: 4.0
- Strategy: Quadruple EMA Touch
- AI: 선택적 사용 (GPT-4o)
- Last Updated: 2025-08-01

## 🔄 시스템 아키텍처

### 비동기 처리 구조
- **메인 루프**: asyncio 기반 이벤트 루프
- **데이터 수집**: 5분봉 캔들 데이터 실시간 수집
- **신호 감지**: EMA 크로스오버 실시간 모니터링
- **주문 실행**: 비동기 Binance Futures API 호출
- **모니터링**: 30초 간격 상태 업데이트
- **알림 전송**: 비동기 텔레그램 메시지 전송

### 데이터 흐름
1. **DataCollector**: Binance API → 캔들 데이터 → Quadruple EMA 계산
2. **Core**: EMA 터치 감지 → 거래 신호 생성
3. **Executor**: 신호 수신 → 주문 실행 → 포지션 관리
4. **Monitor**: 상태 수집 → 화면 표시 → 로그 기록
5. **Telegram**: 이벤트 수신 → 메시지 포맷팅 → 알림 전송

### AI 분석 모듈 (선택적)
- **모델**: GPT-4o
- **캐싱**: 60초 캐시로 API 호출 최적화
- **분석 내용**: 
  - EMA 상태 및 트렌드 분석
  - 시장 상황 평가
  - 리스크 레벨 판단

## 💡 주요 설정값
- **메인 타임프레임**: 30분봉 (모든 진입/청산 신호)
- **보조 타임프레임**: 1m, 3m, 5m, 15m (신호 필터링용)
- **터치 임계값**: 0.2% (변경: 기존 0.35%)
- **홀드 타임**: 1분(60초) 이상 유지 필수
- **멀티 TF 조건**: 
  - 매수: 모든 경우 2개 이상 정배열
  - 매도: 모든 경우 2개 이상 역배열
- **긴급청산**: 1m/3m/5m 모두 120분간 반대 배열시
- **포지션 크기**: 계좌 잔액의 20%
- **레버리지**: 10배
- **손절**: -20%
- **최대 보유시간**: 무제한
- **트레일링 스톱**: ROI 15%에서 손익분기점 이동, 최고점에서 30% 트레일링

## 🛡️ 안전 기능
- **격리마진**: 포지션별 독립적인 마진 관리
- **최대 손실 제한**: 15% 손절선
- **수동 포지션 추적**: 사용자 수동 거래도 자동 관리
- **안전한 종료**: Ctrl+C로 모든 컴포넌트 순차 정리