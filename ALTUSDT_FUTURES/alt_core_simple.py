#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 단순화된 트레이딩 봇 코어
MACD 신호선 기반 트레이딩
"""

import asyncio
import logging
import sqlite3
import json
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

from alt_data import DataCollector
from alt_executor import OrderExecutor
from alt_indicators_simple import SimpleIndicators
from alt_monitor_simple import SimpleMonitor

@dataclass
class TradingConfig:
    """트레이딩 설정"""
    # API 키
    binance_api_key: str = ""
    binance_api_secret: str = ""
    
    # 거래 설정
    symbol: str = "ALT/USDT"
    margin_mode: str = "ISOLATED"
    timeframe: str = "5m"
    
    # 포지션 설정
    position_size: float = 5.0  # 계정 잔액의 5%
    leverage: int = 2
    min_position_size: float = 10.0  # 최소 10 USDT
    
    # 손절 설정
    base_stop_loss: float = 10.0  # 기본 손절 -10%
    
    # 트레일링 스톱 설정
    trailing_stop_configs: Dict[float, float] = field(default_factory=lambda: {
        10.0: 5.0,   # 10% 수익시 SL을 5%로
        11.0: 6.0,   # 11% → 6%
        12.0: 7.0,   # 12% → 7%
        13.0: 8.0,   # 13% → 8%
        14.0: 9.0,   # 14% → 9%
        15.0: 10.0   # 15% → 10%
    })
    
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
    entry_time: datetime
    current_price: float = 0
    stop_loss: float = 0
    trailing_sl_price: float = 0
    highest_price: float = 0
    lowest_price: float = 0
    highest_roi: float = 0
    lowest_roi: float = 0
    
    def calculate_roi(self, current_price: float) -> float:
        """ROI 계산"""
        if self.side == "LONG":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def update_price(self, current_price: float):
        """가격 업데이트 및 최고/최저 추적"""
        self.current_price = current_price
        current_roi = self.calculate_roi(current_price)
        
        # 최고/최저 가격 업데이트
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

class SimpleTradingBot:
    """단순화된 트레이딩 봇"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger("ALTTrading")
        self.position: Optional[Position] = None
        self.data_collector = DataCollector(config)
        self.executor = OrderExecutor(config)
        self.monitor = SimpleMonitor(self)
        self.db = None
        self.last_signal_time = {"LONG": None, "SHORT": None}
        self.is_running = False
        
        # MACD 신호 추적
        self.last_macd_signal = None
        self.signal_start_time = None
        
    async def initialize(self):
        """시스템 초기화"""
        try:
            # 데이터 수집기 초기화
            await self.data_collector.initialize()
            
            # 주문 실행기 초기화
            await self.executor.initialize()
            
            # 데이터베이스 초기화
            self.init_database()
            
            # 포지션 상태 복구
            self.load_position_state()
            
            self.logger.info("✅ 시스템 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"시스템 초기화 실패: {e}")
            return False
    
    def init_database(self):
        """데이터베이스 초기화"""
        self.db = sqlite3.connect('alt_trading_bot.db')
        cursor = self.db.cursor()
        
        # 거래 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time DATETIME,
                exit_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                entry_price REAL,
                exit_price REAL,
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
        
        # 포지션 상태 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS position_state (
                id INTEGER PRIMARY KEY,
                position_data TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.db.commit()
    
    def load_position_state(self):
        """포지션 상태 로드"""
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT position_data FROM position_state WHERE id = 1")
            row = cursor.fetchone()
            
            if row and row[0]:
                data = json.loads(row[0])
                self.position = Position(
                    symbol=data['symbol'],
                    side=data['side'],
                    quantity=data['quantity'],
                    entry_price=data['entry_price'],
                    entry_time=datetime.fromisoformat(data['entry_time']),
                    stop_loss=data.get('stop_loss', 0),
                    trailing_sl_price=data.get('trailing_sl_price', 0),
                    highest_price=data.get('highest_price', 0),
                    lowest_price=data.get('lowest_price', 0),
                    highest_roi=data.get('highest_roi', 0),
                    lowest_roi=data.get('lowest_roi', 0)
                )
                self.logger.info(f"✅ 포지션 상태 복구: {self.position.side} @ {self.position.entry_price:.6f}")
        except Exception as e:
            self.logger.error(f"포지션 상태 로드 실패: {e}")
    
    def save_position_state(self):
        """포지션 상태 저장"""
        try:
            cursor = self.db.cursor()
            
            if self.position:
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
            if df is None or len(df) < 5:  # 최소 5개 캔들 필요
                return
            
            # 지표 계산 (MACD만)
            df = SimpleIndicators.calculate_all(df)
            
            # 현재 가격
            current_price = df.iloc[-1]['close']
            current_signal = df.iloc[-1]['macd_signal']
            signal_trend = df.iloc[-1]['signal_trend']
            
            # MACD 신호선 값 모니터링에 전달
            self.monitor.last_macd_signal = current_signal
            self.monitor.signal_trend = signal_trend
            
            # 이전 완성된 캔들의 MACD 신호선 값 (마지막에서 두번째)
            if len(df) >= 2:
                self.monitor.previous_macd_signal = df.iloc[-2]['macd_signal']
            
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
            
        except Exception as e:
            self.logger.error(f"트레이딩 로직 오류: {e}")
            traceback.print_exc()
    
    async def check_entry_signals(self, df):
        """진입 신호 체크"""
        signal_trend = df.iloc[-1]['signal_trend']
        
        # 매수 신호 (MACD 신호선 3분 이상 상승)
        if signal_trend == 1:
            await self.enter_position("LONG", df)
        
        # 매도 신호 (MACD 신호선 3분 이상 하락)
        elif signal_trend == -1:
            await self.enter_position("SHORT", df)
    
    async def check_reverse_signal(self, df) -> bool:
        """반대 신호 체크"""
        if not self.position:
            return False
        
        signal_trend = df.iloc[-1]['signal_trend']
        
        # LONG 포지션인데 하락 신호
        if self.position.side == "LONG" and signal_trend == -1:
            return True
        
        # SHORT 포지션인데 상승 신호
        if self.position.side == "SHORT" and signal_trend == 1:
            return True
        
        return False
    
    async def enter_position(self, side: str, df):
        """포지션 진입"""
        try:
            # 계정 잔액 확인
            balance = await self.executor.get_account_balance()
            if balance < self.config.min_position_size:
                self.logger.warning(f"잔액 부족: {balance:.2f} USDT")
                return
            
            # 포지션 크기 계산
            position_value = balance * (self.config.position_size / 100)
            current_price = df.iloc[-1]['close']
            quantity = (position_value * self.config.leverage) / current_price
            
            # 주문 실행
            order = await self.executor.place_order(
                symbol=self.config.symbol,
                side=side,
                quantity=quantity,
                order_type="MARKET"
            )
            
            if order:
                # 포지션 생성
                self.position = Position(
                    symbol=self.config.symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=current_price,
                    entry_time=datetime.now(),
                    current_price=current_price,
                    stop_loss=self.calculate_stop_loss(side, current_price),
                    highest_price=current_price,
                    lowest_price=current_price
                )
                
                self.save_position_state()
                self.logger.info(f"✅ {side} 포지션 진입: {quantity:.4f} @ {current_price:.6f}")
                
        except Exception as e:
            self.logger.error(f"포지션 진입 실패: {e}")
    
    async def close_position(self, reason: str = "Manual"):
        """포지션 청산"""
        if not self.position:
            return
        
        try:
            # 청산 주문
            order = await self.executor.close_position(
                symbol=self.config.symbol,
                side=self.position.side,
                quantity=self.position.quantity
            )
            
            if order:
                # ROI 계산
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
        """트레일링 스톱 업데이트"""
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
            self.logger.info(f"📊 트레일링 스톱 활성화: {self.position.trailing_sl_price:.6f}")
            self.save_position_state()
        
        # 트레일링 스톱 업데이트
        if self.position.trailing_sl_price > 0:
            trailing_distance = 5  # 기본 5%
            
            # ROI에 따른 트레일링 거리 조정
            if self.position.highest_roi >= 15:
                trailing_distance = 10
            elif self.position.highest_roi >= 14:
                trailing_distance = 9
            elif self.position.highest_roi >= 13:
                trailing_distance = 8
            elif self.position.highest_roi >= 12:
                trailing_distance = 7
            elif self.position.highest_roi >= 11:
                trailing_distance = 6
            
            # 새로운 SL 계산
            if self.position.side == "LONG":
                if self.position.highest_price > 0:
                    new_sl = self.position.highest_price * (1 - trailing_distance / 100)
                    if new_sl > self.position.trailing_sl_price:
                        self.position.trailing_sl_price = new_sl
                        self.position.stop_loss = new_sl
                        self.logger.info(f"📊 트레일링 업데이트: {new_sl:.6f}")
                        self.save_position_state()
            else:  # SHORT
                if self.position.lowest_price > 0:
                    new_sl = self.position.lowest_price * (1 + trailing_distance / 100)
                    if new_sl < self.position.trailing_sl_price:
                        self.position.trailing_sl_price = new_sl
                        self.position.stop_loss = new_sl
                        self.logger.info(f"📊 트레일링 업데이트: {new_sl:.6f}")
                        self.save_position_state()
        
        # 스톱로스 체크
        if self.position.stop_loss > 0:
            if self.position.side == "LONG" and self.position.current_price <= self.position.stop_loss:
                await self.close_position("트레일링 스톱")
            elif self.position.side == "SHORT" and self.position.current_price >= self.position.stop_loss:
                await self.close_position("트레일링 스톱")
        
        # 기본 손절 체크 (-10%)
        if current_roi <= -self.config.base_stop_loss:
            await self.close_position("기본 손절")
    
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
                    trailing_sl_price, entry_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                self.position.trailing_sl_price,
                self.position.entry_time
            ))
            self.db.commit()
        except Exception as e:
            self.logger.error(f"거래 기록 저장 실패: {e}")
    
    async def run(self):
        """봇 실행"""
        self.is_running = True
        
        # 모니터링 시작
        asyncio.create_task(self.monitor.run())
        
        while self.is_running:
            try:
                # 데이터 업데이트
                await self.data_collector.update_market_data()
                
                # 트레이딩 로직 실행
                await self.execute_trading_logic()
                
                # 대기
                await asyncio.sleep(self.config.check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("프로그램 종료 요청")
                break
            except Exception as e:
                self.logger.error(f"실행 오류: {e}")
                await asyncio.sleep(5)
        
        self.is_running = False