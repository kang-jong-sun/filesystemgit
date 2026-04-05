#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SOL Futures 자동매매 시스템 v4.0 - Simple Core
Quadruple EMA Touch 기반 단순 거래 엔진 (AI 제거)
"""

import os
import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import sqlite3
from pathlib import Path
import pandas as pd

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# AI Analyzer 임포트
try:
    from sol_ai_analyzer_v4 import AIAnalyzer
except ImportError:
    AIAnalyzer = None

# === 🔴 Simple 설정 클래스 (v4.0) ===
@dataclass
class TradingConfig:
    """SOL 거래 시스템 설정 v4.0 - Simple Version"""
    
    # API 키
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))  # alias
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))  # alias
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    
    # 거래 설정
    symbol: str = "SOL/USDT"
    margin_mode: str = "ISOLATED"
    timeframe: str = "30m"  # 30분봉 기준 (백테스트 최적 설정)
    
    # 🔴 포지션 설정 (v4.md)
    position_size: float = 20.0  # 계정 잔액의 20%
    leverage: int = 10  # 10배 레버리지
    min_position_size: float = 10.0  # 최소 포지션 크기 (USDT)
    
    # 🔴 ROI 기반 손절/익절 설정
    base_stop_loss: float = 20.0   # 기본 손절 20%
    
    # 🔴 트레일링 스톱 설정
    # 30% 수익시 진입가로 이동, 이후 최고점에서 30% 트레일링
    trailing_stop_configs: Dict[float, float] = field(default_factory=lambda: {
        30.0: 0.0   # 30% 수익시 진입가로 SL 이동 (손익분기점)
    })
    trailing_percent: float = 30.0  # 최고점에서 30% 트레일링
    
    
    # 🔴 최대 보유 시간
    max_holding_hours: int = 0  # 0 = 무제한

    # 🔴 변동성/필터 & 쿨다운 설정
    atr_min_percent: float = 0.5   # 최소 ATR%
    atr_max_percent: float = 6.0   # 최대 ATR%
    require_htf_alignment: bool = True  # 상위 타임프레임 정렬 요구
    htf_timeframe: str = "1h"  # 상위 타임프레임 기준
    cooldown_seconds: int = 300  # 청산 후 재진입 쿨다운(초)
    min_signal_interval_seconds: int = 120  # 동일 방향 신호 최소 간격(초)
    
    # 모니터링
    monitoring_interval: int = 30  # 30초 (화면 출력)
    check_interval: int = 5  # 5초 (시장 체크 및 거래)

# === 로깅 설정 ===
def setup_logging(config: TradingConfig) -> Tuple[logging.Logger, sqlite3.Connection]:
    """로깅 및 데이터베이스 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 (날짜별)
    log_file = log_dir / f"sol_trading_{datetime.now().strftime('%Y%m%d')}.log"
    
    # 로거 설정
    logger = logging.getLogger("SOLTrading")
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 데이터베이스 설정
    db = sqlite3.connect('sol_trading_bot.db')
    cursor = db.cursor()
    
    # 거래 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            symbol TEXT,
            side TEXT,
            quantity REAL,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_percent REAL,
            holding_time_minutes REAL,
            close_reason TEXT,
            order_id TEXT,
            partial_closed INTEGER DEFAULT 0,
            breakeven_moved INTEGER DEFAULT 0,
            highest_price REAL,
            lowest_price REAL,
            trailing_sl_price REAL
        )
    ''')
    
    # EMA 터치 기록 테이블 (Quadruple EMA Touch 전략)
    # 기존 데이터베이스 호환성을 위해 테이블명은 유지
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ema_crosses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            cross_type TEXT,
            ema10 REAL,
            ema21 REAL,
            ema34 REAL,
            price REAL,
            action_taken TEXT
        )
    ''')
    
    # 기존 테이블에 새로운 컬럼 추가 (없을 경우)
    columns_to_add = [
        ("partial_closed", "INTEGER DEFAULT 0"),
        ("breakeven_moved", "INTEGER DEFAULT 0"),
        ("highest_price", "REAL"),
        ("lowest_price", "REAL"),
        ("trailing_sl_price", "REAL")
    ]
    
    for column_name, column_type in columns_to_add:
        try:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {column_type}")
            db.commit()
            logger.info(f"✅ trades 테이블에 {column_name} 컬럼 추가")
        except:
            # 이미 컬럼이 존재하는 경우 무시
            pass
    
    db.commit()
    
    logger.info("✅ 데이터베이스 초기화 완료")
    
    return logger, db

# === 🔴 Trading Bot ===
class TradingBot:
    """Quadruple EMA Touch 기반 자동매매 봇"""
    
    def __init__(self, config: TradingConfig, logger: logging.Logger, db: sqlite3.Connection):
        self.config = config
        self.logger = logger
        self.db = db
        
        # 컴포넌트
        self.data_collector = None
        self.order_executor = None
        self.ai_analyzer = None  # AI 분석기 추가
        
        # 상태
        self.is_running = False
        self.active_positions = {}
        self.last_signal = None
        self.last_cross_time = None
        
        # 트레일링 업데이트 추적
        self.last_trailing_update = {
            'sl_price': None,
            'trigger_roi': None,
            'sl_update_time': None,
            'highest_price': None,  # 최고가 추적
            'lowest_price': None,   # 최저가 추적 (숏용)
            'breakeven_moved': False  # 손익분기점 이동 여부
        }
        
        # EMA 상태
        self.current_ema_state = None  # 'bullish', 'bearish'
        self.previous_ema_state = None
        
        # 터치 상태 추적 (1분 유지 조건용)
        self.touch_state = {
            'long': {
                'is_touching': False,
                'start_time': None,
                'confirmed': False,
                'touch_type': None  # 'near' (0.2% 이내) or 'above' (위에 위치)
            },
            'short': {
                'is_touching': False,
                'start_time': None,
                'confirmed': False,
                'touch_type': None  # 'near' (0.2% 이내) or 'below' (아래 위치)
            }
        }
        self.state_initialized = False  # 첫 상태 초기화 확인
        
        # 긴급청산 추적
        self.emergency_exit_state = {
            'long': {
                'checking': False,
                'start_time': None,
                'aligned_count': 0
            },
            'short': {
                'checking': False,
                'start_time': None,
                'aligned_count': 0
            }
        }

        # 재진입 쿨다운 및 필터 상태
        self.last_exit_time: Optional[datetime] = None
        self.cooldown_seconds: int = self.config.cooldown_seconds
        self.last_signal_time: Optional[datetime] = None
        self.min_signal_interval_seconds: int = self.config.min_signal_interval_seconds
        
        # 1m/3m/5m 동시 정렬 추적
        self.tf_alignment_state = {
            'bullish': {  # 모두 정배열
                'is_aligned': False,
                'start_time': None
            },
            'bearish': {  # 모두 역배열
                'is_aligned': False,
                'start_time': None
            }
        }
        
        # 2개 이상 타임프레임 정렬 추적 (포지션 진입용)
        self.multi_tf_alignment_state = {
            'bullish': {  # 2개 이상 정배열
                'is_aligned': False,
                'start_time': None,
                'count': 0
            },
            'bearish': {  # 2개 이상 역배열
                'is_aligned': False,
                'start_time': None,
                'count': 0
            }
        }
        
        # AI 분석기 초기화
        self.ai_analyzer = None
        if AIAnalyzer and os.getenv("OPENAI_API_KEY"):
            try:
                self.ai_analyzer = AIAnalyzer(config, logger)
                self.logger.info("🤖 AI Analyzer 활성화")
            except Exception as e:
                self.logger.warning(f"AI Analyzer 초기화 실패: {e}")
                self.logger.info("🤖 Pure EMA Strategy 모드로 실행")
        else:
            self.logger.info("🤖 Pure EMA Strategy (AI 없음)")
        
        self.logger.info("🚀 Simple SOL 선물 자동매매 시스템 v4.0 초기화 완료")
        self.logger.info("📊 전략: Quadruple EMA Touch")
        self.logger.info(f"💰 포지션: 계정의 {config.position_size}%, {config.leverage}x 레버리지")
    
    def check_and_reset_stale_touch(self):
        """오래된 터치 상태 자동 리셋 (30분 이상)"""
        current_time = datetime.now()
        
        # 롱 터치 체크
        if self.touch_state['long']['is_touching'] and self.touch_state['long']['start_time']:
            duration = (current_time - self.touch_state['long']['start_time']).total_seconds()
            if duration > 1800:  # 30분
                self.logger.warning(f"⚠️ 롱 터치 상태 자동 리셋 (30분 초과: {duration/60:.1f}분)")
                self.touch_state['long']['is_touching'] = False
                self.touch_state['long']['start_time'] = None
                self.touch_state['long']['confirmed'] = False
                self.touch_state['long']['touch_type'] = None
        
        # 숏 터치 체크
        if self.touch_state['short']['is_touching'] and self.touch_state['short']['start_time']:
            duration = (current_time - self.touch_state['short']['start_time']).total_seconds()
            if duration > 1800:  # 30분
                self.logger.warning(f"⚠️ 숏 터치 상태 자동 리셋 (30분 초과: {duration/60:.1f}분)")
                self.touch_state['short']['is_touching'] = False
                self.touch_state['short']['start_time'] = None
                self.touch_state['short']['confirmed'] = False
                self.touch_state['short']['touch_type'] = None
    
    async def run(self):
        """메인 실행 루프"""
        self.is_running = True
        self.logger.info("🚀 메인 실행 루프 시작")
        
        last_check_time = 0
        last_position_check = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 5초마다 시장 체크 (감지 및 거래)
                if current_time - last_check_time >= self.config.check_interval:
                    await self.check_market_and_trade()
                    last_check_time = current_time
                
                # 10초마다 포지션 체크
                if current_time - last_position_check >= 10:
                    await self.check_positions()
                    last_position_check = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"메인 루프 오류: {e}")
                self.logger.error(traceback.format_exc())
                await asyncio.sleep(5)
    
    async def check_market_and_trade(self):
        """시장 체크 및 거래 실행"""
        try:
            # 오래된 터치 상태 자동 리셋 (30분 이상)
            self.check_and_reset_stale_touch()
            
            # 수동 포지션 변경 감지 (5초마다)
            await self.detect_manual_position_changes()
            
            # 데이터 수집
            market_data = await self.data_collector.collect_market_data()
            if not market_data:
                return
            
            # EMA 계산
            df = market_data.get('5m')
            if df is None or df.empty:
                return
            
            ema_data = self.data_collector.get_current_ema()
            if not ema_data:
                return
            # 변동성 필터 (ATR%)
            atr_percent = float(ema_data.get('atr_percent', 0.0))
            # 설정 기반 필터: atr_min_percent ~ atr_max_percent 범위만 거래
            is_volatility_ok = self.config.atr_min_percent <= atr_percent <= self.config.atr_max_percent
            if not is_volatility_ok:
                self.logger.info(
                    f"⛔ 변동성 필터 미통과: ATR%={atr_percent:.2f}% (허용 {self.config.atr_min_percent:.2f}~{self.config.atr_max_percent:.2f})"
                )
                # 포지션 없는 경우에만 신호 무시; 포지션 보유시에는 관리만 수행
                if not self.order_executor.has_position():
                    # 그래도 상태 추적은 계속
                    await self.track_tf_alignment()
                    await self.manage_positions(market_data)
                    return
            
            # 현재 EMA 상태 확인
            self.update_ema_state(ema_data)
            
            # 매번 EMA 상태 로그 (디버깅용)
            ema10 = ema_data['ema10']
            ema21 = ema_data['ema21']
            ema21_high = ema_data.get('ema21_high', 0)
            ema21_low = ema_data.get('ema21_low', 0)
            
            # 각 EMA와의 차이 계산 (각 EMA 기준)
            diff_21 = abs(ema10 - ema21) / ema21 * 100 if ema21 > 0 else 0
            diff_21_high = abs(ema10 - ema21_high) / ema21_high * 100 if ema21_high > 0 else 0
            diff_21_low = abs(ema10 - ema21_low) / ema21_low * 100 if ema21_low > 0 else 0
            
            # 위치 관계 표시
            if ema10 > ema21:
                self.logger.info(f"[EMA 체크] EMA10: ${ema10:.2f} (현재) | EMA21H: ${ema21_high:.2f} ↓{diff_21_high:.2f}% | EMA21: ${ema21:.2f} ↓{diff_21:.2f}% | EMA21L: ${ema21_low:.2f} ↓{diff_21_low:.2f}% | {self.current_ema_state}")
            else:
                self.logger.info(f"[EMA 체크] EMA21H: ${ema21_high:.2f} ↑{diff_21_high:.2f}% | EMA21: ${ema21:.2f} ↑{diff_21:.2f}% | EMA21L: ${ema21_low:.2f} ↑{diff_21_low:.2f}% | EMA10: ${ema10:.2f} (현재) | {self.current_ema_state}")
            
            # 디버깅 로그 추가
            if self.previous_ema_state != self.current_ema_state:
                self.logger.info(f"📊 EMA 상태 변경: {self.previous_ema_state} → {self.current_ema_state}")
                if ema10 > ema21:
                    self.logger.info(f"   EMA10(${ema10:.2f})이 EMA21(${ema21:.2f})보다 {diff_21:.3f}% 위에 위치")
                else:
                    self.logger.info(f"   EMA10(${ema10:.2f})이 EMA21(${ema21:.2f})보다 {diff_21:.3f}% 아래에 위치")
            
            # 터치 기반 진입/청산 조건: EMA10이 EMA21(고가/저가)에 터치
            trade_signal = await self.check_ema_touch_signal(ema_data)
            
            # 디버깅: 터치 상태 로그
            if self.touch_state['long']['is_touching']:
                duration = (datetime.now() - self.touch_state['long']['start_time']).total_seconds()
                self.logger.debug(f"[디버그] 롱 터치 상태 - 유지시간: {duration:.0f}초, confirmed: {self.touch_state['long']['confirmed']}")
            if self.touch_state['short']['is_touching']:
                duration = (datetime.now() - self.touch_state['short']['start_time']).total_seconds()
                self.logger.debug(f"[디버그] 숏 터치 상태 - 유지시간: {duration:.0f}초, confirmed: {self.touch_state['short']['confirmed']}")
            
            if trade_signal:
                # 재진입 쿨다운 체크 (직전 청산 이후 일정 시간 대기)
                if self.last_exit_time:
                    elapsed = (datetime.now() - self.last_exit_time).total_seconds()
                    if elapsed < self.cooldown_seconds:
                        self.logger.info(f"⏳ 재진입 쿨다운 대기 중: {elapsed:.0f}/{self.cooldown_seconds}s")
                        trade_signal = None
                # 동일 방향 신호 최소 간격
                if trade_signal and self.last_signal_time:
                    elapsed_sig = (datetime.now() - self.last_signal_time).total_seconds()
                    if elapsed_sig < self.min_signal_interval_seconds:
                        self.logger.info(f"⏳ 동일 방향 신호 최소 간격 대기: {elapsed_sig:.0f}/{self.min_signal_interval_seconds}s")
                        trade_signal = None
                self.logger.info(f"🔔 [터치 감지] {trade_signal} 신호!")
                if trade_signal == 'LONG_SIGNAL':
                    self.logger.info(f"   EMA10(${ema10:.2f})이 EMA21(고가)(${ema21_high:.2f})에 {diff_21_high:.3f}% 근접 (터치)")
                elif trade_signal == 'SHORT_SIGNAL':
                    self.logger.info(f"   EMA10(${ema10:.2f})이 EMA21(저가)(${ema21_low:.2f})에 {diff_21_low:.3f}% 근접 (터치)")
                
                # 현재 포지션 상태 로그
                if self.order_executor.has_position():
                    pos = self.order_executor.current_position
                    self.logger.info(f"   현재 포지션: {pos['side']} {pos['quantity']:.3f} SOL")
                
                # AI 검증 (중요한 신호일 때만)
                if self.ai_analyzer:
                    try:
                        self.logger.info("🤖 AI 신호 검증 중...")
                        # trade_signal을 전달하여 AI 검증
                        ai_valid, ai_reason = await self.ai_analyzer.validate_trade_signal(
                            trade_signal, ema_data, market_data
                        )
                        self.logger.info(f"🤖 AI 검증 결과: {'✅ 승인' if ai_valid else '❌ 거부'} - {ai_reason}")
                        
                        if not ai_valid:
                            self.logger.warning(f"AI가 신호를 거부했습니다: {ai_reason}")
                            return
                    except Exception as e:
                        self.logger.error(f"AI 검증 오류: {str(e)}, 기본 신호 진행")
                
                if trade_signal:
                    await self.handle_trade_signal(trade_signal, ema_data, market_data)
            
            # 역신호 체크: 제거 (터치 전략에서는 역신호 처리 불필요)
            
            # 1m/3m/5m 정렬 상태 추적 (포지션 유무와 관계없이)
            await self.track_tf_alignment()
            
            # 기존 포지션 관리
            await self.manage_positions(market_data)
            
        except Exception as e:
            self.logger.error(f"시장 체크 오류: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def update_ema_state(self, ema_data: Dict):
        """EMA 상태 업데이트"""
        # 이전 상태 저장 (첫 실행이 아닐 때만)
        if self.state_initialized:
            self.previous_ema_state = self.current_ema_state
        
        # EMA 비교 (단순 비교)
        if ema_data['ema10'] > ema_data['ema21']:
            self.current_ema_state = 'bullish'
        else:
            self.current_ema_state = 'bearish'
        
        # 첫 실행 시 이전 상태도 현재 상태로 설정
        if not self.state_initialized:
            self.previous_ema_state = self.current_ema_state
            self.state_initialized = True
            self.logger.info(f"📊 초기 EMA 상태: {self.current_ema_state}")
    
    async def check_multi_tf_alignment(self, required_type='bullish', min_required=2):
        """다중 타임프레임 정렬 체크 (3분 이상 유지 확인)
        - 2개 이상 타임프레임이 3분 이상 동일 방향 유지시 True
        """
        try:
            # 정렬 상태 확인
            alignment_state = self.multi_tf_alignment_state.get(required_type)
            if not alignment_state or not alignment_state['is_aligned']:
                self.logger.debug(f"❌ {required_type} 정렬 없음 또는 2개 미만")
                return False
            
            # 3분 이상 유지 확인
            current_time = datetime.now()
            duration = (current_time - alignment_state['start_time']).total_seconds()
            
            if duration >= 180:  # 3분 이상
                self.logger.info(f"✅ {required_type} 정렬 3분 이상 유지: {alignment_state['count']}개 TF, {duration/60:.1f}분")
                return True
            else:
                self.logger.debug(f"⏱️ {required_type} 정렬 대기 중: {alignment_state['count']}개 TF, {duration:.0f}초/180초")
                return False
            
        except Exception as e:
            self.logger.error(f"다중 타임프레임 정렬 체크 오류: {e}")
            return False
    
    async def check_ema_touch_signal(self, ema_data: Dict) -> Optional[str]:
        """개선된 EMA 터치 신호 감지 - v4 최신 전략
        
        매수(롱) 조건:
        1. EMA10 > EMA21(시가) + EMA21(고가) 0.2% 이내 터치 + 1분 유지 + 1/3/5/15분 중 2개 이상 정배열
        2. EMA10 > EMA21(시가) + EMA10 > EMA21(고가) + 1분 유지 + 1/3/5/15분 중 2개 이상 정배열
        
        매도(숏) 조건:
        1. EMA10 < EMA21(시가) + EMA21(저가) 0.2% 이내 터치 + 1분 유지 + 1/3/5/15분 중 2개 이상 역배열
        2. EMA10 < EMA21(시가) + EMA10 < EMA21(저가) + 1분 유지 + 1/3/5/15분 중 2개 이상 역배열
        """
        try:
            ema10 = ema_data.get('ema10', 0)
            ema21 = ema_data.get('ema21', 0)  # 시가 기준
            ema21_high = ema_data.get('ema21_high', 0)
            ema21_low = ema_data.get('ema21_low', 0)
            atr_percent = float(ema_data.get('atr_percent', 0.0))
            
            if not (ema10 and ema21 and ema21_high and ema21_low):
                return None
            
            # 현재 포지션 확인
            has_position = self.order_executor and self.order_executor.has_position()
            current_position = self.order_executor.current_position if self.order_executor else None
            
            # 터치 범위 설정
            touch_threshold = 0.002  # 0.2%
            
            # EMA 배열 상태 확인
            is_bullish = ema10 > ema21  # 정배열
            is_bearish = ema10 < ema21  # 역배열
            
            # 상위 타임프레임 정렬 필터 (옵션)
            if self.config.require_htf_alignment:
                multi_tf_ema = self.data_collector.get_multi_tf_ema()
                htf = self.config.htf_timeframe
                htf_data = multi_tf_ema.get(htf, {}) if multi_tf_ema else {}
                if htf_data and htf_data.get('ema10') and htf_data.get('ema21'):
                    htf_bullish = htf_data['ema10'] > htf_data['ema21']
                    htf_bearish = htf_data['ema10'] < htf_data['ema21']
                else:
                    htf_bullish = False
                    htf_bearish = False
            else:
                htf_bullish = True
                htf_bearish = True

            # 변동성 필터 (신호 레벨에서도 2차 확인)
            if not (self.config.atr_min_percent <= atr_percent <= self.config.atr_max_percent):
                return None

            current_time = datetime.now()
            
            # 정배열 상태에서 EMA21(고가)에 터치 체크
            high_diff = abs(ema10 - ema21_high) / ema21_high
            is_high_touching = is_bullish and high_diff <= touch_threshold
            
            # === 롱 포지션 체크 (정배열) ===
            if is_bullish and (not self.config.require_htf_alignment or htf_bullish):
                # 조건 체크
                high_diff = abs(ema10 - ema21_high) / ema21_high
                is_near_high = high_diff <= touch_threshold  # 0.2% 이내 터치
                is_above_high = ema10 > ema21_high  # EMA21_HIGH 위에 위치
                
                # 두 조건 중 하나라도 만족
                is_high_touching = is_near_high or is_above_high
                
                # 롱 터치 상태 관리
                if is_high_touching:
                    touch_type = 'above' if is_above_high else 'near'
                    
                    if not self.touch_state['long']['is_touching']:
                        # 새로운 터치 시작
                        self.touch_state['long']['is_touching'] = True
                        self.touch_state['long']['start_time'] = current_time
                        self.touch_state['long']['touch_type'] = touch_type
                        
                        if touch_type == 'near':
                            self.logger.info(f"⏱️ 롱 터치 시작 (근접): EMA10(${ema10:.2f}) → EMA21_HIGH(${ema21_high:.2f}) 차이: {high_diff*100:.3f}%")
                        else:
                            self.logger.info(f"⏱️ 롱 터치 시작 (상위): EMA10(${ema10:.2f}) > EMA21_HIGH(${ema21_high:.2f})")
                    else:
                        # 터치 유지 중 - 시간 체크
                        duration = (current_time - self.touch_state['long']['start_time']).total_seconds()
                        if duration >= 180:
                            # 3분 이상 유지됨 - 다중 타임프레임 정렬 체크 (매수는 2개 이상 필요)
                            multi_tf_aligned = await self.check_multi_tf_alignment('bullish', min_required=2)
                            
                            if multi_tf_aligned:
                                # 모든 조건 만족
                                # 이미 롱 포지션이 있으면 신호 무시
                                if has_position and current_position and current_position['side'] in ['buy', 'long']:
                                    self.logger.debug(f"이미 롱 포지션 보유 중 - 롱 신호 무시")
                                    return None
                                
                                # 이미 확정된 신호면 다시 발송하지 않음
                                if self.touch_state['long']['confirmed']:
                                    self.logger.debug(f"이미 확정된 롱 신호 - 추가 신호 무시")
                                    return None
                                
                                # 신호 확정
                                self.touch_state['long']['confirmed'] = True
                                touch_type_msg = "0.2% 이내 터치" if self.touch_state['long']['touch_type'] == 'near' else "EMA21_HIGH 위에 위치"
                                self.logger.info(f"✅ 롱 신호 확정!")
                                self.logger.info(f"   1️⃣ EMA10 > EMA21_OPEN ✓")
                                self.logger.info(f"   2️⃣ {touch_type_msg} ✓")
                                self.logger.info(f"   3️⃣ 3분 이상 유지 ({duration:.0f}초) ✓")
                                self.logger.info(f"   4️⃣ 하위 TF 2개 이상 정배열 3분 이상 유지 ✓")
                                return 'LONG_SIGNAL'
                            else:
                                self.logger.debug(f"롱 조건 미충족: 하위 TF 정렬 부족 (2개 이상 필요)")
                        elif duration < 180:
                            self.logger.debug(f"롱 터치 유지 중: {duration:.0f}초/180초")
            else:
                # 터치 해제
                if self.touch_state['long']['is_touching']:
                    duration = (current_time - self.touch_state['long']['start_time']).total_seconds()
                    self.logger.info(f"❌ 롱 터치 해제 (유지시간: {duration:.0f}초)")
                self.touch_state['long']['is_touching'] = False
                self.touch_state['long']['start_time'] = None
                self.touch_state['long']['confirmed'] = False
            
            # === 숏 포지션 체크 (역배열) ===
            if is_bearish and (not self.config.require_htf_alignment or htf_bearish):
                # 조건 체크
                low_diff = abs(ema10 - ema21_low) / ema21_low
                is_near_low = low_diff <= touch_threshold  # 0.2% 이내 터치
                is_below_low = ema10 < ema21_low  # EMA21_LOW 아래에 위치
                
                # 두 조건 중 하나라도 만족
                is_low_touching = is_near_low or is_below_low
                
                # 숏 터치 상태 관리
                if is_low_touching:
                    touch_type = 'below' if is_below_low else 'near'
                    
                    if not self.touch_state['short']['is_touching']:
                        # 새로운 터치 시작
                        self.touch_state['short']['is_touching'] = True
                        self.touch_state['short']['start_time'] = current_time
                        self.touch_state['short']['touch_type'] = touch_type
                        
                        if touch_type == 'near':
                            self.logger.info(f"⏱️ 숏 터치 시작 (근접): EMA10(${ema10:.2f}) → EMA21_LOW(${ema21_low:.2f}) 차이: {low_diff*100:.3f}%")
                        else:
                            self.logger.info(f"⏱️ 숏 터치 시작 (하위): EMA10(${ema10:.2f}) < EMA21_LOW(${ema21_low:.2f})")
                    else:
                        # 터치 유지 중 - 시간 체크
                        duration = (current_time - self.touch_state['short']['start_time']).total_seconds()
                        if duration >= 180:
                            # 3분 이상 유지됨 - 다중 타임프레임 정렬 체크
                            # 매도: 모든 경우 2개 이상 필요
                            multi_tf_aligned = await self.check_multi_tf_alignment('bearish', min_required=2)
                            
                            if multi_tf_aligned:
                                # 모든 조건 만족
                                # 이미 숏 포지션이 있으면 신호 무시
                                if has_position and current_position and current_position['side'] in ['sell', 'short']:
                                    self.logger.debug(f"이미 숏 포지션 보유 중 - 숏 신호 무시")
                                    return None
                                
                                # 이미 확정된 신호면 다시 발송하지 않음
                                if self.touch_state['short']['confirmed']:
                                    self.logger.debug(f"이미 확정된 숏 신호 - 추가 신호 무시")
                                    return None
                                
                                # 신호 확정
                                self.touch_state['short']['confirmed'] = True
                                touch_type_msg = "0.2% 이내 터치" if self.touch_state['short']['touch_type'] == 'near' else "EMA21_LOW 아래에 위치"
                                self.logger.info(f"✅ 숏 신호 확정!")
                                self.logger.info(f"   1️⃣ EMA10 < EMA21_OPEN ✓")
                                self.logger.info(f"   2️⃣ {touch_type_msg} ✓")
                                self.logger.info(f"   3️⃣ 3분 이상 유지 ({duration:.0f}초) ✓")
                                self.logger.info(f"   4️⃣ 하위 TF 2개 이상 역배열 3분 이상 유지 ✓")
                                return 'SHORT_SIGNAL'
                            else:
                                self.logger.debug(f"숏 조건 미충족: 하위 TF 정렬 부족 (2개 이상 필요)")
                        elif duration < 180:
                            self.logger.debug(f"숏 터치 유지 중: {duration:.0f}초/180초")
            else:
                # 터치 해제
                if self.touch_state['short']['is_touching']:
                    duration = (current_time - self.touch_state['short']['start_time']).total_seconds()
                    self.logger.info(f"❌ 숏 터치 해제 (유지시간: {duration:.0f}초)")
                self.touch_state['short']['is_touching'] = False
                self.touch_state['short']['start_time'] = None
                self.touch_state['short']['confirmed'] = False
            
            return None
            
        except Exception as e:
            self.logger.error(f"EMA 터치 신호 감지 오류: {str(e)}")
            return None
    
    async def handle_trade_signal(self, signal: str, ema_data: Dict, market_data: Dict):
        """거래 신호 처리 - 터치 기반 진입/청산"""
        try:
            has_position = self.order_executor.has_position()
            position = self.order_executor.current_position
            
            # 실제 포지션 확인
            actual_positions = await self.order_executor.get_open_positions()
            if actual_positions and not position:
                self.logger.warning(f"⚠️ current_position이 None이지만 실제 포지션 존재: {len(actual_positions)}개")
                for pos in actual_positions:
                    normalized_api_symbol = pos['symbol'].replace('/', '').replace(':USDT', '')
                    normalized_config_symbol = self.config.symbol.replace('/', '')
                    if normalized_api_symbol == normalized_config_symbol or 'SOL' in pos['symbol']:
                        position = pos
                        self.order_executor.current_position = pos
                        has_position = True
                        self.logger.info(f"포지션 동기화: {pos['side']} {pos['quantity']:.3f} SOL")
                        break
            
            # 롱 신호 처리 (EMA10이 EMA21_high에 터치)
            if signal == 'LONG_SIGNAL':
                if has_position and position:
                    if position['side'] in ['sell', 'short']:
                        # 숏 포지션에서 EMA10이 EMA21_high에 터치 → 숏 청산 후 롱 진입
                        self.logger.info("🔄 [터치 기반] 숏 포지션 청산 후 롱 전환")
                        self.logger.info(f"   EMA10이 EMA21(고가)에 터치 - 숏 청산 신호")
                        close_result = await self.order_executor.close_position("EMA_TOUCH_EXIT")
                        if close_result:
                            self.logger.info("✅ 숏 포지션 청산 성공")
                            await asyncio.sleep(5)
                            await self.enter_position('buy', market_data)
                    else:
                        self.logger.info("이미 롱 포지션 보유 중")
                else:
                    # 새로운 롱 진입
                    await self.enter_position('buy', market_data)
                    self.last_signal_time = datetime.now()
            
            # 숏 신호 처리 (EMA10이 EMA21_low에 터치)
            elif signal == 'SHORT_SIGNAL':
                if has_position and position:
                    if position['side'] in ['buy', 'long']:
                        # 롱 포지션에서 EMA10이 EMA21_low에 터치 → 롱 청산 후 숏 진입
                        self.logger.info("🔄 [터치 기반] 롱 포지션 청산 후 숏 전환")
                        self.logger.info(f"   EMA10이 EMA21(저가)에 터치 - 롱 청산 신호")
                        close_result = await self.order_executor.close_position("EMA_TOUCH_EXIT")
                        if close_result:
                            self.logger.info("✅ 롱 포지션 청산 성공")
                            await asyncio.sleep(5)
                            await self.enter_position('sell', market_data)
                    else:
                        self.logger.info("이미 숏 포지션 보유 중")
                else:
                    # 새로운 숏 진입
                    await self.enter_position('sell', market_data)
                    self.last_signal_time = datetime.now()
            
            self.last_cross_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"거래 신호 처리 오류: {str(e)}")
    
    async def enter_position(self, side: str, market_data: Dict):
        """포지션 진입"""
        try:
            # 잔고 확인
            balance = await self.order_executor.get_balance()
            if balance <= 0:
                self.logger.error("잔고 부족")
                return
            
            # 포지션 크기 계산
            position_size = balance * (self.config.position_size / 100)
            current_price = market_data['ticker']['last']
            quantity = (position_size * self.config.leverage) / current_price
            
            # 최소 포지션 크기 확인
            if position_size < self.config.min_position_size:
                self.logger.warning(f"포지션 크기 부족: ${position_size:.2f}")
                return
            
            # 주문 실행
            success = await self.order_executor.open_position(
                side=side,
                quantity=quantity,
                leverage=self.config.leverage
            )
            
            if success:
                self.last_signal = side
                self.logger.info(f"✅ {side.upper()} 포지션 진입 성공")
            else:
                self.logger.error(f"❌ {side.upper()} 포지션 진입 실패")
                
        except Exception as e:
            self.logger.error(f"포지션 진입 오류: {str(e)}")
    
    async def track_tf_alignment(self):
        """타임프레임 정렬 상태 추적 (포지션 유무와 관계없이)"""
        try:
            multi_tf_ema = self.data_collector.get_multi_tf_ema()
            if not multi_tf_ema:
                return
            
            # 1m, 3m, 5m만 체크 (긴급청산용)
            emergency_tfs = ['1m', '3m', '5m']
            bullish_count_emergency = 0
            bearish_count_emergency = 0
            
            for tf in emergency_tfs:
                tf_data = multi_tf_ema.get(tf, {})
                if tf_data and tf_data.get('ema10') and tf_data.get('ema21'):
                    ema10_tf = tf_data.get('ema10', 0)
                    ema21_tf = tf_data.get('ema21', 0)
                    
                    if ema10_tf > ema21_tf:
                        bullish_count_emergency += 1
                    else:
                        bearish_count_emergency += 1
            
            # 1m, 3m, 5m, 15m, 1h 체크 (포지션 진입용)
            entry_tfs = ['1m', '3m', '5m', '15m', '1h']
            bullish_count_entry = 0
            bearish_count_entry = 0
            
            for tf in entry_tfs:
                tf_data = multi_tf_ema.get(tf, {})
                if tf_data and tf_data.get('ema10') and tf_data.get('ema21'):
                    ema10_tf = tf_data.get('ema10', 0)
                    ema21_tf = tf_data.get('ema21', 0)
                    
                    if ema10_tf > ema21_tf:
                        bullish_count_entry += 1
                    else:
                        bearish_count_entry += 1
            
            current_time = datetime.now()
            
            # 1m/3m/5m 동시 정렬 상태 업데이트 (긴급청산용)
            if bullish_count_emergency == 3:
                if not self.tf_alignment_state['bullish']['is_aligned']:
                    self.tf_alignment_state['bullish']['is_aligned'] = True
                    self.tf_alignment_state['bullish']['start_time'] = current_time
                    self.logger.info("📊 1m/3m/5m 모두 정배열 시작")
            else:
                if self.tf_alignment_state['bullish']['is_aligned']:
                    duration = (current_time - self.tf_alignment_state['bullish']['start_time']).total_seconds() / 60
                    self.logger.info(f"📊 1m/3m/5m 정배열 종료 (유지시간: {duration:.1f}분)")
                self.tf_alignment_state['bullish']['is_aligned'] = False
                self.tf_alignment_state['bullish']['start_time'] = None
            
            if bearish_count_emergency == 3:
                if not self.tf_alignment_state['bearish']['is_aligned']:
                    self.tf_alignment_state['bearish']['is_aligned'] = True
                    self.tf_alignment_state['bearish']['start_time'] = current_time
                    self.logger.info("📊 1m/3m/5m 모두 역배열 시작")
            else:
                if self.tf_alignment_state['bearish']['is_aligned']:
                    duration = (current_time - self.tf_alignment_state['bearish']['start_time']).total_seconds() / 60
                    self.logger.info(f"📊 1m/3m/5m 역배열 종료 (유지시간: {duration:.1f}분)")
                self.tf_alignment_state['bearish']['is_aligned'] = False
                self.tf_alignment_state['bearish']['start_time'] = None
            
            # 2개 이상 타임프레임 정렬 상태 업데이트 (포지션 진입용)
            if bullish_count_entry >= 2:
                if not self.multi_tf_alignment_state['bullish']['is_aligned']:
                    self.multi_tf_alignment_state['bullish']['is_aligned'] = True
                    self.multi_tf_alignment_state['bullish']['start_time'] = current_time
                    self.multi_tf_alignment_state['bullish']['count'] = bullish_count_entry
                    self.logger.info(f"📊 {bullish_count_entry}개 TF 정배열 시작 (1m/3m/5m/15m)")
                elif self.multi_tf_alignment_state['bullish']['count'] != bullish_count_entry:
                    self.multi_tf_alignment_state['bullish']['count'] = bullish_count_entry
                    self.logger.debug(f"정배열 TF 개수 변경: {bullish_count_entry}개")
            else:
                if self.multi_tf_alignment_state['bullish']['is_aligned']:
                    duration = (current_time - self.multi_tf_alignment_state['bullish']['start_time']).total_seconds() / 60
                    self.logger.info(f"📊 정배열 종료 (유지시간: {duration:.1f}분)")
                self.multi_tf_alignment_state['bullish']['is_aligned'] = False
                self.multi_tf_alignment_state['bullish']['start_time'] = None
                self.multi_tf_alignment_state['bullish']['count'] = 0
            
            if bearish_count_entry >= 2:
                if not self.multi_tf_alignment_state['bearish']['is_aligned']:
                    self.multi_tf_alignment_state['bearish']['is_aligned'] = True
                    self.multi_tf_alignment_state['bearish']['start_time'] = current_time
                    self.multi_tf_alignment_state['bearish']['count'] = bearish_count_entry
                    self.logger.info(f"📊 {bearish_count_entry}개 TF 역배열 시작 (1m/3m/5m/15m)")
                elif self.multi_tf_alignment_state['bearish']['count'] != bearish_count_entry:
                    self.multi_tf_alignment_state['bearish']['count'] = bearish_count_entry
                    self.logger.debug(f"역배열 TF 개수 변경: {bearish_count_entry}개")
            else:
                if self.multi_tf_alignment_state['bearish']['is_aligned']:
                    duration = (current_time - self.multi_tf_alignment_state['bearish']['start_time']).total_seconds() / 60
                    self.logger.info(f"📊 역배열 종료 (유지시간: {duration:.1f}분)")
                self.multi_tf_alignment_state['bearish']['is_aligned'] = False
                self.multi_tf_alignment_state['bearish']['start_time'] = None
                self.multi_tf_alignment_state['bearish']['count'] = 0
                
        except Exception as e:
            self.logger.error(f"타임프레임 정렬 추적 오류: {e}")
    
    async def check_emergency_exit(self, position: Dict) -> bool:
        """긴급청산 체크
        - 롱 포지션: 1m, 3m, 5m이 모두 120분간 연속 역배열시 청산
        - 숏 포지션: 1m, 3m, 5m이 모두 120분간 연속 정배열시 청산
        """
        try:
            if not position:
                return False
            
            multi_tf_ema = self.data_collector.get_multi_tf_ema()
            if not multi_tf_ema:
                return False
            
            # 1m, 3m, 5m만 체크
            emergency_tfs = ['1m', '3m', '5m']
            aligned_count = 0
            
            for tf in emergency_tfs:
                tf_data = multi_tf_ema.get(tf, {})
                if tf_data and tf_data.get('ema10') and tf_data.get('ema21'):
                    ema10_tf = tf_data.get('ema10', 0)
                    ema21_tf = tf_data.get('ema21', 0)
                    
                    if position['side'] in ['buy', 'long']:
                        # 롱 포지션일 때 역배열 체크
                        if ema10_tf < ema21_tf:
                            aligned_count += 1
                    else:  # short
                        # 숏 포지션일 때 정배열 체크
                        if ema10_tf > ema21_tf:
                            aligned_count += 1
            
            current_time = datetime.now()
            side = 'long' if position['side'] in ['buy', 'long'] else 'short'
            opposite_state = '역배열' if side == 'long' else '정배열'
            
            # 3개 모두 반대 배열인 경우
            if aligned_count == 3:
                if not self.emergency_exit_state[side]['checking']:
                    # 긴급청산 체크 시작
                    self.emergency_exit_state[side]['checking'] = True
                    self.emergency_exit_state[side]['start_time'] = current_time
                    self.logger.warning(f"⚠️ 긴급청산 모니터링 시작: {side.upper()} 포지션 - 1m/3m/5m 모두 {opposite_state}")
                else:
                    # 지속 시간 체크
                    duration = (current_time - self.emergency_exit_state[side]['start_time']).total_seconds()
                    
                    if duration >= 7200:  # 120분 = 7200초
                        self.logger.error(f"🚨 긴급청산 발동! {side.upper()} 포지션")
                        self.logger.error(f"   1m, 3m, 5m이 120분간 연속 {opposite_state} 유지")
                        
                        # 텔레그램 알림
                        if self.order_executor.telegram:
                            await self.order_executor.telegram.send_message(
                                f"🚨 <b>긴급청산 발동</b>\n\n"
                                f"• 포지션: {side.upper()}\n"
                                f"• 이유: 1m/3m/5m 120분간 {opposite_state}\n"
                                f"• 즉시 청산 후 신호 대기\n\n"
                                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            )
                        
                        # 포지션 청산
                        await self.order_executor.close_position("EMERGENCY_EXIT")
                        self.last_exit_time = datetime.now()
                        
                        # 상태 초기화
                        self.emergency_exit_state[side]['checking'] = False
                        self.emergency_exit_state[side]['start_time'] = None
                        
                        return True
                    else:
                        remaining = 7200 - duration
                        if int(duration) % 60 == 0:  # 1분마다 로그
                            self.logger.warning(f"⏱️ 긴급청산 대기 중: {duration:.0f}초/7200초 (남은 시간: {remaining:.0f}초)")
            else:
                # 조건 해제
                if self.emergency_exit_state[side]['checking']:
                    duration = (current_time - self.emergency_exit_state[side]['start_time']).total_seconds()
                    self.logger.info(f"✅ 긴급청산 해제: {aligned_count}/3 TF만 {opposite_state} (유지시간: {duration:.0f}초)")
                    self.emergency_exit_state[side]['checking'] = False
                    self.emergency_exit_state[side]['start_time'] = None
            
            return False
            
        except Exception as e:
            self.logger.error(f"긴급청산 체크 오류: {e}")
            return False
    
    async def manage_positions(self, market_data: Dict):
        """포지션 관리 (손절/긴급청산/보유시간)"""
        try:
            if not self.order_executor.has_position():
                return
            
            position = self.order_executor.current_position
            if not position:
                # active_positions에서도 확인
                position = self.active_positions.get(self.config.symbol)
                if not position:
                    return
                # executor에 동기화
                self.order_executor.current_position = position
            
            # 디버깅 로그 추가
            self.logger.info(f"[포지션 관리] 현재 {position['side'].upper()} 포지션 {position['quantity']:.3f} SOL 보유 중")
            
            # 긴급청산 체크 (최우선)
            emergency_exit = await self.check_emergency_exit(position)
            if emergency_exit:
                return  # 긴급청산 실행됨
            
            current_price = market_data['ticker']['last']
            entry_price = position['entry_price']
            
            # ROI 계산 (레버리지 적용)
            side = position['side'].lower()
            leverage = position.get('leverage', self.config.leverage)
            
            if side in ['buy', 'long']:
                roi = ((current_price - entry_price) / entry_price) * 100 * leverage
            else:
                roi = ((entry_price - current_price) / entry_price) * 100 * leverage
            
            # ROI 계산 디버깅
            if roi >= 0:
                self.logger.info(f"[ROI 계산] {side.upper()} | 진입: ${entry_price:.2f} → 현재: ${current_price:.2f} | {leverage}x | ROI: +{roi:.2f}% 📈")
            else:
                self.logger.info(f"[ROI 계산] {side.upper()} | 진입: ${entry_price:.2f} → 현재: ${current_price:.2f} | {leverage}x | ROI: {roi:.2f}% 📉")
            
            # 손절 체크 (ROI 기준)
            leverage = position.get('leverage', self.config.leverage)
            stop_loss_roi = -self.config.base_stop_loss  # -30%
            
            if roi <= stop_loss_roi:
                self.logger.warning(f"🔴 손절 도달! {position['side'].upper()} {position['quantity']:.3f} SOL")
                self.logger.warning(f"   ${position['entry_price']:.2f} → ${current_price:.2f} | ROI: {roi:.2f}% (손절선: {stop_loss_roi}%)")
                
                # 텔레그램 알림
                if self.order_executor.telegram:
                    await self.order_executor.telegram.send_stop_loss_alert(
                        roi=roi,
                        entry_price=position['entry_price'],
                        current_price=current_price
                    )
                
                await self.order_executor.close_position("STOP_LOSS")
                self.last_exit_time = datetime.now()
                return
            
            # 보유 시간 체크 (0 = 무제한)
            if self.config.max_holding_hours > 0:
                holding_time = (datetime.now() - position['opened_at']).total_seconds() / 3600
                if holding_time >= self.config.max_holding_hours:
                    self.logger.warning(f"⏰ 최대 보유시간 초과 ({holding_time:.1f}시간)")
                    
                    # 텔레그램 알림
                    if self.order_executor.telegram:
                        await self.order_executor.telegram.send_max_holding_alert(
                            holding_hours=holding_time
                        )
                    
                    await self.order_executor.close_position("MAX_HOLDING_TIME")
                    self.last_exit_time = datetime.now()
                    return
            
            
            # 트레일링 스톱 업데이트 (현재가 전달)
            position['current_price'] = current_price  # 현재가 추가
            await self.update_trailing_stop(position, roi)
            
        except Exception as e:
            self.logger.error(f"포지션 관리 오류: {str(e)}")
    
    async def update_trailing_stop(self, position: Dict, roi: float):
        """트레일링 스톱 업데이트"""
        try:
            entry_price = position['entry_price']
            current_price = position.get('current_price', 0)
            side = position['side'].lower()
            
            # 최고가/최저가 업데이트
            if side in ['buy', 'long']:
                if self.last_trailing_update['highest_price'] is None or current_price > self.last_trailing_update['highest_price']:
                    self.last_trailing_update['highest_price'] = current_price
                    self.logger.info(f"[최고가 갱신] ${current_price:.4f} (ROI: {roi:.2f}%)")
            else:  # sell/short
                if self.last_trailing_update['lowest_price'] is None or current_price < self.last_trailing_update['lowest_price']:
                    self.last_trailing_update['lowest_price'] = current_price
                    self.logger.info(f"[최저가 갱신] ${current_price:.4f} (ROI: {roi:.2f}%)")
            
            # 현재 SL 가격
            current_sl = position.get('sl_price', 0) or 0
            
            # 1. 30% 수익시 손익분기점으로 이동
            if roi >= 30.0 and not self.last_trailing_update['breakeven_moved']:
                self.logger.info(f"[트레일링 스톱] ROI {roi:.2f}% 도달, 손익분기점으로 SL 이동")
                
                # SL을 진입가로 이동
                success = await self.order_executor.modify_position(sl_price=entry_price)
                if success:
                    position['sl_price'] = entry_price
                    self.last_trailing_update['breakeven_moved'] = True
                    self.last_trailing_update['sl_price'] = entry_price
                    self.last_trailing_update['sl_update_time'] = datetime.now()
                    
                    # DB에 손익분기점 이동 상태 업데이트
                    if position.get('order_id'):
                        cursor = self.db.cursor()
                        cursor.execute('''
                            UPDATE trades 
                            SET breakeven_moved = 1,
                                trailing_sl_price = ?
                            WHERE order_id = ? AND exit_price IS NULL
                        ''', (entry_price, position['order_id']))
                        self.db.commit()
                    
                    # 텔레그램 알림
                    if self.order_executor and self.order_executor.telegram:
                        await self.order_executor.telegram.send_trailing_stop_update(
                            trigger_roi=30.0,
                            new_sl=entry_price,
                            entry_price=entry_price
                        )
                return
            
            # 2. 손익분기점 이동 후, 최고점에서 30% 트레일링
            if self.last_trailing_update['breakeven_moved']:
                if side in ['buy', 'long']:
                    highest_price = self.last_trailing_update['highest_price']
                    # 최고점에서 30% 아래
                    new_sl = highest_price * 0.7
                    
                    # 현재 SL보다 높은 경우만 업데이트
                    if new_sl > current_sl and new_sl > entry_price:
                        self.logger.info(f"[트레일링 스톱] 최고가 ${highest_price:.2f}에서 30% 트레일링 → SL: ${new_sl:.2f}")
                        
                        success = await self.order_executor.modify_position(sl_price=new_sl)
                        if success:
                            position['sl_price'] = new_sl
                            self.last_trailing_update['sl_price'] = new_sl
                            self.last_trailing_update['sl_update_time'] = datetime.now()
                            
                            # DB에 트레일링 SL 및 최고가 업데이트
                            if position.get('order_id'):
                                cursor = self.db.cursor()
                                cursor.execute('''
                                    UPDATE trades 
                                    SET trailing_sl_price = ?,
                                        highest_price = ?
                                    WHERE order_id = ? AND exit_price IS NULL
                                ''', (new_sl, highest_price, position['order_id']))
                                self.db.commit()
                            
                            # 텔레그램 알림
                            if self.order_executor and self.order_executor.telegram:
                                await self.order_executor.telegram.send_trailing_stop_update(
                                    trigger_roi=roi,
                                    new_sl=new_sl,
                                    entry_price=entry_price
                                )
                
                else:  # sell
                    lowest_price = self.last_trailing_update['lowest_price']
                    # 최저점에서 30% 위
                    new_sl = lowest_price * 1.3
                    
                    # 현재 SL보다 낮은 경우만 업데이트
                    if new_sl < current_sl and new_sl < entry_price:
                        self.logger.info(f"[트레일링 스톱] 최저가 ${lowest_price:.2f}에서 30% 트레일링 → SL: ${new_sl:.2f}")
                        
                        success = await self.order_executor.modify_position(sl_price=new_sl)
                        if success:
                            position['sl_price'] = new_sl
                            self.last_trailing_update['sl_price'] = new_sl
                            self.last_trailing_update['sl_update_time'] = datetime.now()
                            
                            # DB에 트레일링 SL 및 최저가 업데이트
                            if position.get('order_id'):
                                cursor = self.db.cursor()
                                cursor.execute('''
                                    UPDATE trades 
                                    SET trailing_sl_price = ?,
                                        lowest_price = ?
                                    WHERE order_id = ? AND exit_price IS NULL
                                ''', (new_sl, lowest_price, position['order_id']))
                                self.db.commit()
                            
                            # 텔레그램 알림
                            if self.order_executor and self.order_executor.telegram:
                                await self.order_executor.telegram.send_trailing_stop_update(
                                    trigger_roi=roi,
                                    new_sl=new_sl,
                                    entry_price=entry_price
                                )
            
        except Exception as e:
            self.logger.error(f"트레일링 스톱 오류: {str(e)}")
    
    
    def calculate_trailing_sl(self, position: Dict, trigger_roi: float, 
                            trailing_percent: float) -> Optional[float]:
        """트레일링 SL 계산"""
        entry_price = position['entry_price']
        leverage = position.get('leverage', self.config.leverage)
        
        if position['side'] == 'buy':
            if trigger_roi == 10.0:  # 진입가로 이동
                return entry_price
            else:  # 일반 트레일링
                return entry_price * (1 + (trigger_roi - trailing_percent) / leverage / 100)
        else:  # sell
            if trigger_roi == 10.0:  # 진입가로 이동
                return entry_price
            else:  # 일반 트레일링
                return entry_price * (1 - (trigger_roi - trailing_percent) / leverage / 100)
    
    async def detect_manual_position_changes(self):
        """수동 포지션 변경 감지 (5초마다)"""
        try:
            current_positions = await self.order_executor.get_open_positions()
            current_position_symbols = {pos['symbol']: pos for pos in current_positions}
            
            # 1. 새로운 수동 포지션 감지
            for symbol, pos in current_position_symbols.items():
                if symbol not in self.active_positions:
                    # 새로운 포지션 발견
                    self.logger.warning(f"🔄 수동 포지션 감지: {pos['side'].upper()} {pos['quantity']:.3f} SOL | 진입가: ${pos['entry_price']:.2f}")
                    
                    # DB에서 이전 부분청산 및 트레일링 상태 확인
                    partial_closed = False
                    breakeven_moved = False
                    highest_price = None
                    lowest_price = None
                    trailing_sl_price = None
                    
                    if pos.get('order_id'):
                        cursor = self.db.cursor()
                        cursor.execute('''
                            SELECT partial_closed, breakeven_moved, highest_price, lowest_price, trailing_sl_price
                            FROM trades 
                            WHERE order_id = ? AND exit_price IS NULL
                            ORDER BY timestamp DESC LIMIT 1
                        ''', (pos['order_id'],))
                        result = cursor.fetchone()
                        if result:
                            partial_closed = result[0] == 1
                            breakeven_moved = result[1] == 1
                            highest_price = result[2]
                            lowest_price = result[3]
                            trailing_sl_price = result[4]
                            
                            if partial_closed:
                                self.logger.info("📌 DB에서 확인: 이전에 부분청산된 포지션")
                            if breakeven_moved:
                                self.logger.info("📌 DB에서 확인: 손익분기점 이동 완료")
                                self.last_trailing_update['breakeven_moved'] = True
                                self.last_trailing_update['sl_price'] = trailing_sl_price
                                if highest_price:
                                    self.last_trailing_update['highest_price'] = highest_price
                                if lowest_price:
                                    self.last_trailing_update['lowest_price'] = lowest_price
                    
                    position_info = {
                        **pos,
                        'opened_at': datetime.now(),
                        'is_manual': True,
                        'partial_closed': partial_closed
                    }
                    self.active_positions[symbol] = position_info
                    
                    # symbol 매칭 확인 (SOLUSDT, SOL/USDT, SOL/USDT:USDT 등)
                    # 디버깅을 위한 로그
                    self.logger.info(f"심볼 매칭 확인: API symbol='{symbol}', Config symbol='{self.config.symbol}'")
                    
                    # 정규화된 심볼 비교
                    normalized_api_symbol = symbol.replace('/', '').replace(':USDT', '')
                    normalized_config_symbol = self.config.symbol.replace('/', '')
                    
                    self.logger.info(f"정규화된 심볼: API='{normalized_api_symbol}', Config='{normalized_config_symbol}'")
                    
                    symbol_match = normalized_api_symbol == normalized_config_symbol
                    
                    if symbol_match:
                        self.logger.info(f"수동 포지션을 current_position에 설정: {symbol}")
                        self.order_executor.current_position = position_info
                        await self.order_executor.track_manual_position(position_info)
                    else:
                        self.logger.warning(f"심볼 불일치: {symbol} != {self.config.symbol}")
                        # 강제로 추적 시작 (SOL 관련 포지션이면)
                        if 'SOL' in symbol:
                            self.logger.warning(f"SOL 포지션이므로 강제 추적 시작")
                            self.order_executor.current_position = position_info
                            await self.order_executor.track_manual_position(position_info)
                    
                    # 텔레그램 알림
                    if self.order_executor.telegram:
                        await self.order_executor.telegram.send_message(
                            f"🔄 <b>수동 포지션 진입 감지</b>\n\n"
                            f"• 방향: {'매수' if pos['side'] == 'buy' else '매도'}\n"
                            f"• 수량: {pos['quantity']:.3f} SOL\n"
                            f"• 진입가: ${pos['entry_price']:.2f}\n"
                            f"• 레버리지: {pos.get('leverage', 10)}x\n\n"
                            f"⚠️ 자동 추적 및 관리를 시작합니다\n\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )
            
            # 2. 수동으로 종료된 포지션 감지
            for symbol in list(self.active_positions.keys()):
                if symbol not in current_position_symbols:
                    # 포지션이 종료됨
                    closed_pos = self.active_positions[symbol]
                    self.logger.warning(f"❌ 수동 포지션 종료 감지: {closed_pos['side']} {closed_pos['quantity']:.3f} SOL")
                    
                    # 텔레그램 알림
                    if self.order_executor.telegram:
                        await self.order_executor.telegram.send_message(
                            f"❌ <b>수동 포지션 종료 감지</b>\n\n"
                            f"• 방향: {'매수' if closed_pos['side'] == 'buy' else '매도'}\n"
                            f"• 수량: {closed_pos['quantity']:.3f} SOL\n"
                            f"• 진입가: ${closed_pos['entry_price']:.2f}\n\n"
                            f"⚠️ 수동으로 포지션이 종료되었습니다\n\n"
                            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
                        )
                    
                    # 내부 상태 정리
                    del self.active_positions[symbol]
                    # symbol 매칭 확인
                    normalized_api_symbol = symbol.replace('/', '').replace(':USDT', '')
                    normalized_config_symbol = self.config.symbol.replace('/', '')
                    symbol_match = normalized_api_symbol == normalized_config_symbol
                    
                    if symbol_match and self.order_executor.current_position:
                        self.logger.info("수동 포지션 종료로 인한 current_position 초기화")
                        # TP/SL 주문 정리
                        self.logger.info("수동 포지션 종료 - TP/SL 주문 취소 중...")
                        await self.order_executor.cancel_all_orders()
                        self.order_executor.current_position = None
                        
        except Exception as e:
            self.logger.error(f"수동 포지션 감지 오류: {str(e)}")
    
    async def check_positions(self):
        """포지션 상태 체크"""
        try:
            positions = await self.order_executor.get_open_positions()
            
            if positions:
                for pos in positions:
                    # 기존 포지션 정보 유지
                    existing = self.active_positions.get(pos['symbol'], {})
                    position_info = {
                        **pos,
                        'opened_at': existing.get('opened_at', datetime.now()),
                        'is_manual': existing.get('is_manual', pos.get('is_manual', False)),
                        'partial_closed': existing.get('partial_closed', False),  # partial_closed 플래그 유지
                        'sl_price': existing.get('sl_price'),  # sl_price 유지
                    }
                    self.active_positions[pos['symbol']] = position_info
                    
                    # order_executor의 current_position 업데이트
                    normalized_api_symbol = pos['symbol'].replace('/', '').replace(':USDT', '')
                    normalized_config_symbol = self.config.symbol.replace('/', '')
                    symbol_match = normalized_api_symbol == normalized_config_symbol
                    
                    if symbol_match:
                        if not self.order_executor.current_position:
                            self.order_executor.current_position = position_info
                            # 수동 포지션 추적 시작
                            if position_info.get('is_manual'):
                                self.logger.warning(f"🔄 수동 포지션 발견: {pos['side']} {pos['quantity']:.3f} SOL @ ${pos['entry_price']:.2f}")
                                await self.order_executor.track_manual_position(position_info)
                        else:
                            # 기존 current_position의 partial_closed 플래그 유지
                            if self.order_executor.current_position.get('partial_closed'):
                                position_info['partial_closed'] = True
                            self.order_executor.current_position = position_info
            else:
                self.active_positions.clear()
                self.order_executor.current_position = None
                # 포지션이 없을 때 최고가/최저가 초기화
                self.last_trailing_update = {
                    'highest_price': None,
                    'lowest_price': None,
                    'breakeven_moved': False,
                    'sl_price': None,
                    'sl_update_time': None
                }
                
        except Exception as e:
            self.logger.error(f"포지션 체크 오류: {str(e)}")
    
    
    async def stop(self):
        """봇 정지"""
        self.logger.info("🛑 봇 정지 시작...")
        self.is_running = False
        
        # 열린 포지션 확인
        if self.order_executor and self.order_executor.has_position():
            self.logger.warning("⚠️  열린 포지션이 있습니다. 수동으로 관리하세요.")
        
        self.logger.info("✅ 봇 정지 완료")