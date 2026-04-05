#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 데이터 수집 모듈
바이낸스 API를 통한 시장 데이터 수집
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Optional, List, Dict, Any
import time

class DataCollector:
    """시장 데이터 수집 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("ALTTrading.DataCollector")
        self.exchange = None
        self.latest_data = None
        self.kline_data = pd.DataFrame()
        
    async def initialize(self):
        """Exchange 초기화"""
        try:
            # Binance futures 설정
            self.exchange = ccxt.binance({
                'apiKey': self.config.binance_api_key,
                'secret': self.config.binance_api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 선물 거래
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000  # 60초 타임윈도우
                }
            })
            
            # 시간 동기화
            await self.sync_time()
            
            # 테스트넷 사용 여부 (필요시)
            # self.exchange.set_sandbox_mode(True)
            
            # 마진 모드 설정
            await self.set_margin_mode()
            
            # 레버리지 설정
            await self.set_leverage()
            
            self.logger.info("✅ DataCollector 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ DataCollector 초기화 실패: {e}")
            return False
    
    async def sync_time(self):
        """바이낸스 서버와 시간 동기화"""
        try:
            # 서버 시간 가져오기
            server_time = self.exchange.fetch_time()
            local_time = self.exchange.milliseconds()
            time_diff = server_time - local_time
            
            if abs(time_diff) > 1000:  # 1초 이상 차이
                self.logger.warning(f"⚠️ 시간 차이 감지: {time_diff}ms, 동기화 중...")
                # 시간 오프셋 설정
                self.exchange.options['timestamp'] = server_time
                self.exchange.options['serverTime'] = server_time
                self.logger.info(f"✅ 시간 동기화 완료: 오프셋 {time_diff}ms")
            
        except Exception as e:
            self.logger.error(f"시간 동기화 실패: {e}")
    
    async def set_margin_mode(self):
        """마진 모드 설정"""
        try:
            symbol = self.config.symbol.replace('/', '')
            
            # 현재 마진 모드 확인
            positions = self.exchange.fetch_positions([self.config.symbol])
            
            # 격리 마진으로 설정
            if self.config.margin_mode == "ISOLATED":
                self.exchange.fapiPrivate_post_margintype({
                    'symbol': symbol,
                    'marginType': 'ISOLATED'
                })
                self.logger.info(f"마진 모드: ISOLATED")
                
        except Exception as e:
            # 이미 설정되어 있을 수 있음
            if "No need to change margin type" not in str(e):
                self.logger.warning(f"마진 모드 설정 경고: {e}")
    
    async def set_leverage(self):
        """레버리지 설정"""
        try:
            symbol = self.config.symbol.replace('/', '')
            
            self.exchange.fapiPrivate_post_leverage({
                'symbol': symbol,
                'leverage': self.config.leverage
            })
            
            self.logger.info(f"레버리지: {self.config.leverage}x")
            
        except Exception as e:
            self.logger.warning(f"레버리지 설정 경고: {e}")
    
    async def update_market_data(self):
        """시장 데이터 업데이트"""
        try:
            # OHLCV 데이터 가져오기
            timeframe = self.config.timeframe
            limit = 1000  # 최대 1000개 캔들
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.config.symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # DataFrame 변환
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            self.kline_data = df
            self.latest_data = df
            
            return True
            
        except Exception as e:
            self.logger.error(f"시장 데이터 업데이트 실패: {e}")
            return False
    
    def get_latest_data(self, lookback: int = 1000) -> Optional[pd.DataFrame]:
        """최신 데이터 반환"""
        if self.kline_data.empty:
            return None
        
        return self.kline_data.tail(lookback).copy()
    
    async def get_current_price(self) -> float:
        """현재 가격 조회"""
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"현재 가격 조회 실패: {e}")
            return 0.0
    
    async def get_account_info(self) -> Dict[str, Any]:
        """계정 정보 조회"""
        try:
            balance = self.exchange.fetch_balance()
            
            # USDT 잔액
            usdt_balance = balance['USDT']['free'] + balance['USDT']['used']
            
            # 포지션 정보
            positions = self.exchange.fetch_positions([self.config.symbol])
            
            return {
                'balance': usdt_balance,
                'positions': positions,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"계정 정보 조회 실패: {e}")
            return {
                'balance': 0,
                'positions': [],
                'timestamp': datetime.now()
            }
    
    async def get_order_book(self, limit: int = 10) -> Dict[str, Any]:
        """오더북 조회"""
        try:
            orderbook = self.exchange.fetch_order_book(
                symbol=self.config.symbol,
                limit=limit
            )
            
            return {
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': orderbook['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"오더북 조회 실패: {e}")
            return {
                'bids': [],
                'asks': [],
                'timestamp': None
            }
    
    async def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """최근 체결 내역 조회"""
        try:
            trades = self.exchange.fetch_trades(
                symbol=self.config.symbol,
                limit=limit
            )
            
            return trades
            
        except Exception as e:
            self.logger.error(f"최근 체결 조회 실패: {e}")
            return []
    
    def calculate_market_metrics(self) -> Dict[str, float]:
        """시장 지표 계산"""
        if self.kline_data.empty:
            return {}
        
        df = self.kline_data.tail(100)  # 최근 100개 캔들
        
        # 변동성 (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # 거래량
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # 가격 변화율
        price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
        
        return {
            'atr': atr,
            'atr_percent': (atr / df['close'].iloc[-1]) * 100,
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'volume_ratio': volume_ratio,
            'price_change_100': price_change
        }
    
    async def check_connection(self) -> bool:
        """연결 상태 확인"""
        try:
            self.exchange.fetch_time()
            return True
        except Exception:
            return False