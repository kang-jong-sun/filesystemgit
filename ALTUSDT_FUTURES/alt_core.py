#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 선물 트레이딩 코어 모듈
포지션 관리 및 트레이딩 로직
"""

import os
import asyncio
import logging
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import time
import sqlite3
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from alt_indicators import IndicatorCalculator

@dataclass
class TradingConfig:
    """ALT 트레이딩 설정"""
    
    # API 키
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    
    # 거래 설정
    symbol: str = "ALT/USDT"
    margin_mode: str = "ISOLATED"
    timeframe: str = "5m"  # 5분봉
    
    # 포지션 설정
    position_size: float = 5.0  # 계정 잔액의 5%
    leverage: int = 2  # 2배 레버리지
    min_position_size: float = 10.0  # 최소 포지션 크기 (USDT)
    
    # 손절/익절 설정
    base_stop_loss: float = 10.0  # 기본 손절 -10%
    
    # 트레일링 스톱 설정
    trailing_stop_configs: Dict[float, float] = field(default_factory=lambda: {
        10.0: 5.0,   # 10% 수익시 SL을 5%로 이동
        11.0: 6.0,   # 11% 수익시 SL을 6%로 이동
        12.0: 7.0,   # 12% 수익시 SL을 7%로 이동
        13.0: 8.0,   # 13% 수익시 SL을 8%로 이동
        14.0: 9.0,   # 14% 수익시 SL을 9%로 이동
        15.0: 10.0   # 15% 수익시 SL을 10%로 이동
    })
    trailing_percent: float = 5.0  # 최고점에서 5% 트레일링
    
    # 보유 시간
    max_holding_hours: int = 0  # 0 = 무제한
    
    # 신호 간격
    min_signal_interval_seconds: int = 120  # 동일 방향 신호 최소 간격(초)
    
    # 모니터링
    monitoring_interval: int = 20  # 20초
    check_interval: int = 5  # 5초

@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    side: str  # LONG or SHORT
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    highest_roi: float = 0.0
    lowest_roi: float = 0.0
    trailing_sl_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    
    def calculate_roi(self, current_price: float) -> float:
        """ROI 계산"""
        if self.side == "LONG":
            roi = ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            roi = ((self.entry_price - current_price) / self.entry_price) * 100
        return roi
    
    def update_price(self, current_price: float):
        """가격 업데이트 및 최고/최저가 추적"""
        self.current_price = current_price
        current_roi = self.calculate_roi(current_price)
        
        # 최고/최저가 업데이트
        if self.side == "LONG":
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.highest_roi = current_roi
            if self.lowest_price == 0 or current_price < self.lowest_price:
                self.lowest_price = current_price
                self.lowest_roi = current_roi
        else:  # SHORT
            if self.lowest_price == 0 or current_price < self.lowest_price:
                self.lowest_price = current_price
                self.highest_roi = current_roi
            if current_price > self.highest_price:
                self.highest_price = current_price
                self.lowest_roi = current_roi

def setup_logging(config: TradingConfig) -> Tuple[logging.Logger, sqlite3.Connection]:
    """로깅 및 데이터베이스 설정"""
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로그 파일명 (12시간 간격)
    current_time = datetime.now()
    hour_suffix = "AM" if current_time.hour < 12 else "PM"
    log_file = log_dir / f"alt_trading_{current_time.strftime('%Y%m%d')}_{hour_suffix}.log"
    
    # 로거 설정
    logger = logging.getLogger("ALTTrading")
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
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    # 데이터베이스 설정
    db = sqlite3.connect('alt_trading_bot.db')
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
            highest_price REAL,
            lowest_price REAL,
            highest_roi REAL,
            lowest_roi REAL,
            trailing_sl_price REAL
        )
    ''')
    
    # 신호 기록 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            signal_type TEXT,
            vwma1_dema_angle REAL,
            ema10_vwma_angle REAL,
            wma10_vwma_angle REAL,
            mixed_angle REAL,
            macd_signal_angle REAL,
            action_taken TEXT
        )
    ''')
    
    # 포지션 상태 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS position_state (
            id INTEGER PRIMARY KEY,
            position_data TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    db.commit()
    
    return logger, db

class TradingBot:
    """트레이딩 봇"""
    
    def __init__(self, config: TradingConfig, data_collector, executor, logger, db):
        self.config = config
        self.data_collector = data_collector
        self.executor = executor
        self.logger = logger
        self.db = db
        self.position: Optional[Position] = None
        self.last_signal_time = {}
        
        # 저장된 포지션 복구
        self.load_position_state()
    
    def load_position_state(self):
        """저장된 포지션 상태 로드"""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT position_data FROM position_state WHERE id = 1")
            row = cursor.fetchone()
            
            if row and row[0]:
                position_data = json.loads(row[0])
                # entry_time을 datetime 객체로 변환
                if 'entry_time' in position_data and isinstance(position_data['entry_time'], str):
                    position_data['entry_time'] = datetime.fromisoformat(position_data['entry_time'])
                self.position = Position(**position_data)
                self.logger.info(f"✅ 포지션 상태 복구: {self.position.side} @ {self.position.entry_price}")
        except Exception as e:
            self.logger.error(f"포지션 상태 로드 실패: {e}")
    
    def save_position_state(self):
        """포지션 상태 저장"""
        try:
            cursor = self.db.cursor()
            
            if self.position:
                # datetime을 문자열로 변환
                position_data = asdict(self.position)
                position_data['entry_time'] = position_data['entry_time'].isoformat()
                position_json = json.dumps(position_data)
            else:
                position_json = None
            
            cursor.execute("""
                INSERT OR REPLACE INTO position_state (id, position_data, updated_at)
                VALUES (1, ?, CURRENT_TIMESTAMP)
            """, (position_json,))
            
            self.db.commit()
        except Exception as e:
            self.logger.error(f"포지션 상태 저장 실패: {e}")
    
    async def execute_trading_logic(self):
        """트레이딩 로직 실행"""
        try:
            # 시장 데이터 가져오기
            df = self.data_collector.get_latest_data()
            if df is None or len(df) < 15:
                return
            
            # 지표 계산
            df = IndicatorCalculator.calculate_all_indicators(df)
            
            # 현재 가격
            current_price = df.iloc[-1]['close']
            
            # 포지션 업데이트
            if self.position:
                self.position.update_price(current_price)
                await self.update_trailing_stop()
                
                # 반대 신호 체크
                if await self.check_reverse_signal(df):
                    await self.close_position("역신호")
                    # 새로운 포지션 진입
                    await self.check_entry_signals(df)
            else:
                # 진입 신호 체크
                await self.check_entry_signals(df)
            
            # 신호 로깅
            self.log_signals(df)
            
        except Exception as e:
            self.logger.error(f"트레이딩 로직 오류: {e}")
            traceback.print_exc()
    
    async def check_entry_signals(self, df: pd.DataFrame):
        """진입 신호 체크"""
        # 정배열 체크 (매수)
        if IndicatorCalculator.check_bullish_alignment(df):
            if self.can_enter_position("LONG"):
                await self.enter_position("LONG", df)
        
        # 역배열 체크 (매도)
        elif IndicatorCalculator.check_bearish_alignment(df):
            if self.can_enter_position("SHORT"):
                await self.enter_position("SHORT", df)
    
    async def check_reverse_signal(self, df: pd.DataFrame) -> bool:
        """반대 신호 체크"""
        if not self.position:
            return False
        
        if self.position.side == "LONG":
            # 롱 포지션인데 역배열 신호
            return IndicatorCalculator.check_bearish_alignment(df)
        else:
            # 숏 포지션인데 정배열 신호
            return IndicatorCalculator.check_bullish_alignment(df)
    
    def can_enter_position(self, side: str) -> bool:
        """포지션 진입 가능 여부 체크"""
        # 이미 포지션이 있으면 불가
        if self.position:
            return False
        
        # 최근 신호 시간 체크
        last_time = self.last_signal_time.get(side, 0)
        if time.time() - last_time < self.config.min_signal_interval_seconds:
            return False
        
        return True
    
    async def enter_position(self, side: str, df: pd.DataFrame):
        """포지션 진입"""
        try:
            current_price = df.iloc[-1]['close']
            
            # 포지션 크기 계산
            account_balance = await self.executor.get_account_balance()
            position_value = account_balance * (self.config.position_size / 100)
            
            if position_value < self.config.min_position_size:
                self.logger.warning(f"포지션 크기가 최소값보다 작음: {position_value}")
                return
            
            quantity = position_value / current_price
            
            # 주문 실행
            order = await self.executor.place_order(
                symbol=self.config.symbol,
                side=side,
                quantity=quantity,
                price=current_price
            )
            
            if order:
                # 포지션 생성
                self.position = Position(
                    symbol=self.config.symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=current_price,
                    current_price=current_price,
                    highest_price=current_price,
                    lowest_price=current_price,
                    stop_loss=self.calculate_stop_loss(side, current_price)
                )
                
                # 상태 저장
                self.save_position_state()
                
                # 신호 시간 업데이트
                self.last_signal_time[side] = time.time()
                
                self.logger.info(f"✅ {side} 포지션 진입: {quantity:.4f} @ {current_price:.4f}")
                
                # 모니터에 거래 알림
                if hasattr(self, 'monitor'):
                    self.monitor.log_trade({
                        'side': side,
                        'quantity': quantity,
                        'price': current_price
                    })
                
        except Exception as e:
            self.logger.error(f"포지션 진입 실패: {e}")
    
    async def close_position(self, reason: str = "수동"):
        """포지션 청산"""
        if not self.position:
            return
        
        try:
            # 주문 실행
            close_side = "SELL" if self.position.side == "LONG" else "BUY"
            order = await self.executor.place_order(
                symbol=self.config.symbol,
                side=close_side,
                quantity=self.position.quantity,
                price=self.position.current_price
            )
            
            if order:
                # 수익 계산
                roi = self.position.calculate_roi(self.position.current_price)
                holding_time = (datetime.now() - self.position.entry_time).total_seconds() / 60
                
                # 거래 기록 저장
                self.save_trade_record(reason, roi, holding_time)
                
                self.logger.info(f"✅ 포지션 청산: {reason} | ROI: {roi:.2f}%")
                
                # 포지션 초기화
                self.position = None
                self.save_position_state()
                
        except Exception as e:
            self.logger.error(f"포지션 청산 실패: {e}")
    
    async def update_trailing_stop(self):
        """트레일링 스톱 업데이트
        10% 수익시 SL을 5%으로 이동 
        최고점에서 5% 트레일링으로 수정
        최고점 10%수익시 5% 트레일링 SL을 5% 이동
        최고점 11%수익시 5% 트레일링 SL을 6% 이동
        최고점 12%수익시 5% 트레일링 SL을 7% 이동
        최고점 13%수익시 5% 트레일링 SL을 8% 이동
        최고점 14%수익시 5% 트레일링 SL을 9% 이동
        최고점 15%수익시 5% 트레일링 SL을 10% 이동
        """
        if not self.position:
            return
        
        current_roi = self.position.calculate_roi(self.position.current_price)
        
        # 최고/최저 ROI 업데이트
        self.position.highest_roi = max(self.position.highest_roi, current_roi)
        self.position.lowest_roi = min(self.position.lowest_roi, current_roi)
        
        # 10% 수익 도달시 첫 트레일링 스톱 설정
        if self.position.highest_roi >= 10 and self.position.trailing_sl_price == 0:
            # 5% 수익 위치로 SL 설정
            if self.position.side == "LONG":
                self.position.trailing_sl_price = self.position.entry_price * 1.05
            else:
                self.position.trailing_sl_price = self.position.entry_price * 0.95
            self.position.stop_loss = self.position.trailing_sl_price
            self.logger.info(f"📊 트레일링 스톱 활성화: {self.position.trailing_sl_price:.6f} (최고 ROI: {self.position.highest_roi:.2f}%)")
            self.save_position_state()
        
        # 트레일링 스톱 업데이트 (최고점 기준)
        if self.position.trailing_sl_price > 0:
            # 최고 ROI 기준으로 트레일링 SL 이동
            trailing_distance = 5  # 기본 5% 트레일링
            
            # 11% 이상일 때 트레일링 거리 조정
            if self.position.highest_roi >= 15:
                trailing_distance = 10  # 15% 이상: 10% 트레일링
            elif self.position.highest_roi >= 14:
                trailing_distance = 9   # 14%: 9% 트레일링
            elif self.position.highest_roi >= 13:
                trailing_distance = 8   # 13%: 8% 트레일링
            elif self.position.highest_roi >= 12:
                trailing_distance = 7   # 12%: 7% 트레일링
            elif self.position.highest_roi >= 11:
                trailing_distance = 6   # 11%: 6% 트레일링
            
            # 새로운 SL 계산 (최고점에서 trailing_distance만큼 아래)
            if self.position.side == "LONG":
                if self.position.highest_price > 0:
                    new_sl = self.position.highest_price * (1 - trailing_distance / 100)
                    if new_sl > self.position.trailing_sl_price:
                        self.position.trailing_sl_price = new_sl
                        self.position.stop_loss = new_sl
                        self.logger.info(f"📊 트레일링 스톱 업데이트: {new_sl:.6f} (최고 ROI: {self.position.highest_roi:.2f}%, 거리: {trailing_distance}%)")
                        self.save_position_state()
            else:  # SHORT
                if self.position.lowest_price > 0:
                    new_sl = self.position.lowest_price * (1 + trailing_distance / 100)
                    if new_sl < self.position.trailing_sl_price:
                        self.position.trailing_sl_price = new_sl
                        self.position.stop_loss = new_sl
                        self.logger.info(f"📊 트레일링 스톱 업데이트: {new_sl:.6f} (최고 ROI: {self.position.highest_roi:.2f}%, 거리: {trailing_distance}%)")
                        self.save_position_state()
        
        # 스톱로스 체크
        if self.position.stop_loss > 0:
            if self.position.side == "LONG" and self.position.current_price <= self.position.stop_loss:
                await self.close_position("트레일링 스톱")
            elif self.position.side == "SHORT" and self.position.current_price >= self.position.stop_loss:
                await self.close_position("트레일링 스톱")
    
    def calculate_stop_loss(self, side: str, entry_price: float) -> float:
        """스톱로스 계산"""
        if side == "LONG":
            return entry_price * (1 - self.config.base_stop_loss / 100)
        else:
            return entry_price * (1 + self.config.base_stop_loss / 100)
    
    def save_trade_record(self, close_reason: str, roi: float, holding_time: float):
        """거래 기록 저장"""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    symbol, side, quantity, entry_price, exit_price,
                    profit_percent, holding_time_minutes, close_reason,
                    highest_price, lowest_price, highest_roi, lowest_roi,
                    trailing_sl_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.position.symbol,
                self.position.side,
                self.position.quantity,
                self.position.entry_price,
                self.position.current_price,
                roi,
                holding_time,
                close_reason,
                self.position.highest_price,
                self.position.lowest_price,
                self.position.highest_roi,
                self.position.lowest_roi,
                self.position.trailing_sl_price
            ))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"거래 기록 저장 실패: {e}")
    
    def log_signals(self, df: pd.DataFrame):
        """신호 로깅"""
        try:
            last_row = df.iloc[-1]
            
            # 신호 타입 결정
            signal_type = "NONE"
            if IndicatorCalculator.check_bullish_alignment(df):
                signal_type = "BULLISH"
            elif IndicatorCalculator.check_bearish_alignment(df):
                signal_type = "BEARISH"
            
            # 데이터베이스에 저장
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO signals (
                    signal_type, vwma1_dema_angle, ema10_vwma_angle,
                    wma10_vwma_angle, mixed_angle, macd_signal_angle,
                    action_taken
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_type,
                last_row.get('vwma1_dema_angle', 0),
                last_row.get('ema10_vwma_angle', 0),
                last_row.get('wma10_vwma_angle', 0),
                last_row.get('mixed_angle', 0),
                last_row.get('macd_signal_angle', 0),
                "POSITION_OPENED" if self.position else "NO_ACTION"
            ))
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"신호 로깅 실패: {e}")