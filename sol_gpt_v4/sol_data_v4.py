#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 SOL Futures 자동매매 시스템 v4.0 - Data Collector
Quadruple EMA Touch 전략 데이터 수집기
- EMA10: 시가(Open) 기준
- EMA21: 시가(Open) 기준
- EMA21_HIGH: 고가(High) 기준
- EMA21_LOW: 저가(Low) 기준
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from collections import deque
import logging

class DataCollector:
    """v4 데이터 수집기 - Quadruple EMA 계산
    - EMA10: 시가(Open) 기준
    - EMA21: 시가(Open) 기준  
    - EMA21_HIGH: 고가(High) 기준
    - EMA21_LOW: 저가(Low) 기준
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        # Binance 거래소
        self.exchange = None
        self.symbol = config.symbol
        
        # 데이터 버퍼 (설정된 타임프레임 사용)
        self.candle_buffer = deque(maxlen=1500)  # 최근 1500개 캔들
        
        # 다중 타임프레임 데이터 버퍼 (1h 추가)
        self.multi_tf_buffers = {
            '1m': deque(maxlen=1500),
            '3m': deque(maxlen=1500),
            '5m': deque(maxlen=1500),
            '15m': deque(maxlen=1500),
            '30m': deque(maxlen=1500),
            '1h': deque(maxlen=1500)
        }
        
        # 현재 EMA 값
        self.current_ema = {
            'ema10': 0,
            'ema21': 0,
            'ema21_high': 0,
            'ema21_low': 0,
            'last_update': None
        }
        
        # 다중 타임프레임 EMA 값
        self.multi_tf_ema = {}
        for tf in ['1m', '3m', '5m', '15m', '30m', '1h']:
            self.multi_tf_ema[tf] = {
                'ema10': 0,
                'ema21': 0,
                'ema21_high': 0,
                'ema21_low': 0,
                'last_update': None
            }
        
        # 실시간 가격
        self.current_price = 0
        
        self.logger.info("📊 Data Collector 초기화 완료")
    
    async def initialize(self):
        """거래소 초기화"""
        try:
            # Binance 초기화
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,  # 60초 허용 범위
                    'timeDifference': 0  # 시간 차이 자동 조정
                }
            })
            
            # 시간 동기화
            await self.exchange.load_time_difference()
            
            # 시장 정보 로드
            await self.exchange.load_markets()
            
            # 초기 데이터 수집
            await self.fetch_initial_data()
            
            self.logger.info("✅ 데이터 수집기 초기화 성공")
            
        except Exception as e:
            self.logger.error(f"❌ 초기화 실패: {str(e)}")
            raise
    
    async def fetch_initial_data(self):
        """초기 데이터 수집"""
        try:
            # 설정된 타임프레임으로 1500개 캔들 수집
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, 
                self.config.timeframe,  # 설정값 사용 (현재: 30m)
                limit=1500
            )
            
            # DataFrame 변환
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 다중 타임프레임 데이터 수집 (1h 포함)
            self.logger.info("📊 다중 타임프레임 데이터 수집 중...")
            # 모든 타임프레임에 대해 최대 1500개 캔들 수집
            for tf in ['1m', '3m', '5m', '15m', '30m', '1h']:
                try:
                    # 1500개 캔들 요청 (최대값)
                    tf_ohlcv = await self.exchange.fetch_ohlcv(
                        self.symbol,
                        tf,
                        limit=1500
                    )
                    
                    for candle in tf_ohlcv:
                        self.multi_tf_buffers[tf].append(candle)
                    
                    # 각 타임프레임별 EMA 계산
                    await self.calculate_multi_tf_ema(tf)
                    
                    # 시간 계산
                    if tf == '1m':
                        hours = len(tf_ohlcv) / 60
                    elif tf == '3m':
                        hours = len(tf_ohlcv) * 3 / 60
                    elif tf == '5m':
                        hours = len(tf_ohlcv) * 5 / 60
                    elif tf == '15m':
                        hours = len(tf_ohlcv) * 15 / 60
                    elif tf == '30m':
                        hours = len(tf_ohlcv) * 30 / 60
                    elif tf == '1h':
                        hours = len(tf_ohlcv)
                    else:
                        hours = 0
                    
                    self.logger.info(f"📊 {tf} 타임프레임: {len(tf_ohlcv)}개 캔들 로드 완료 (약 {hours:.1f}시간 데이터)")
                    
                except Exception as e:
                    self.logger.warning(f"타임프레임 {tf} 데이터 수집 실패: {e}")
            
            # 버퍼에 저장
            self.candle_buffer = deque(df.to_dict('records'), maxlen=1500)
            
            # EMA 계산
            self.calculate_ema(df)
            
            self.logger.info(f"✅ 초기 데이터 수집 완료: {self.config.timeframe} {len(self.candle_buffer)}개 캔들 | EMA 계산 완료")
            
        except Exception as e:
            self.logger.error(f"초기 데이터 수집 오류: {str(e)}")
    
    def calculate_ema(self, df: pd.DataFrame):
        """EMA/ATR 계산
        - EMA10: 시가(Open) 기준
        - EMA21: 시가(Open) 기준
        - EMA21_HIGH: 고가(High) 기준
        - EMA21_LOW: 저가(Low) 기준
        - ATR(14): 변동성 지표 및 ATR% (ATR/종가*100)
        """
        try:
            if len(df) < 21:  # 최소 필요 개수
                return
            
            # EMA 계산
            ema10 = df['open'].ewm(span=10, adjust=False).mean()  # 시가 기준
            ema21 = df['open'].ewm(span=21, adjust=False).mean()   # 시가 기준
            ema21_high = df['high'].ewm(span=21, adjust=False).mean()  # 고가 기준
            ema21_low = df['low'].ewm(span=21, adjust=False).mean()    # 저가 기준

            # ATR(14) 계산
            if len(df) >= 15:
                prev_close = df['close'].shift(1)
                high_low = df['high'] - df['low']
                high_prev_close = (df['high'] - prev_close).abs()
                low_prev_close = (df['low'] - prev_close).abs()
                true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
                atr = true_range.ewm(span=14, adjust=False).mean()
                latest_atr = float(atr.iloc[-1]) if not atr.empty else 0.0
                latest_close = float(df['close'].iloc[-1])
                atr_percent = (latest_atr / latest_close * 100) if latest_close > 0 else 0.0
            else:
                latest_atr = 0.0
                atr_percent = 0.0
            
            # 현재 값 저장
            self.current_ema = {
                'ema10': float(ema10.iloc[-1]),
                'ema21': float(ema21.iloc[-1]),
                'ema21_high': float(ema21_high.iloc[-1]),
                'ema21_low': float(ema21_low.iloc[-1]),
                'atr': float(latest_atr),
                'atr_percent': float(atr_percent),
                'last_update': datetime.now()
            }
            
            self.current_price = float(df['close'].iloc[-1])
            
        except Exception as e:
            self.logger.error(f"EMA 계산 오류: {str(e)}")
    
    async def collect_market_data(self) -> Dict:
        """시장 데이터 수집"""
        try:
            # 현재가 정보
            ticker = await self.fetch_ticker()
            
            # 최신 캔들 업데이트
            await self.update_candles()
            
            # DataFrame 생성
            df = pd.DataFrame(list(self.candle_buffer))
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                
                # EMA 재계산
                self.calculate_ema(df)
            
            # 통합 데이터
            market_data = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                '5m': df,
                'ema': self.current_ema,
                'current_price': self.current_price
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 오류: {str(e)}")
            return {}
    
    async def fetch_ticker(self) -> Dict:
        """현재가 정보 조회"""
        try:
            ticker = await self.exchange.fetch_ticker(self.symbol)
            
            return {
                'symbol': self.symbol,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'timestamp': ticker['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Ticker 조회 오류: {str(e)}")
            return {}
    
    async def update_candles(self):
        """최신 캔들 업데이트"""
        try:
            # 최근 5개 캔들만 조회
            ohlcv = await self.exchange.fetch_ohlcv(
                self.symbol, 
                self.config.timeframe,
                limit=5
            )
            
            for candle in ohlcv:
                timestamp = pd.to_datetime(candle[0], unit='ms')
                
                # 새로운 캔들인지 확인
                if not self.candle_buffer or timestamp > self.candle_buffer[-1]['timestamp']:
                    self.candle_buffer.append({
                        'timestamp': timestamp,
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                # 기존 캔들 업데이트
                elif self.candle_buffer and timestamp == self.candle_buffer[-1]['timestamp']:
                    self.candle_buffer[-1].update({
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                    
        except Exception as e:
            self.logger.error(f"캔들 업데이트 오류: {str(e)}")
    
    async def calculate_multi_tf_ema(self, timeframe: str):
        """다중 타임프레임 EMA 계산"""
        try:
            buffer = self.multi_tf_buffers.get(timeframe)
            if not buffer or len(buffer) < 21:  # EMA21 계산을 위한 최소값으로 수정
                self.logger.warning(f"⚠️ {timeframe} 타임프레임: 데이터 부족 ({len(buffer) if buffer else 0}개) - EMA 계산 불가")
                return
            
            # DataFrame 변환
            df = pd.DataFrame(
                list(buffer),
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Quadruple EMA 계산
            df['ema10_open'] = df['open'].ewm(span=10, adjust=False).mean()
            df['ema21_open'] = df['open'].ewm(span=21, adjust=False).mean()
            df['ema21_high'] = df['high'].ewm(span=21, adjust=False).mean()
            df['ema21_low'] = df['low'].ewm(span=21, adjust=False).mean()
            
            # 최신 값 저장
            latest = df.iloc[-1]
            self.multi_tf_ema[timeframe] = {
                'ema10': latest['ema10_open'],
                'ema21': latest['ema21_open'],
                'ema21_high': latest['ema21_high'],
                'ema21_low': latest['ema21_low'],
                'last_update': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"다중 타임프레임 EMA 계산 오류 ({timeframe}): {e}")
    
    async def update_multi_tf_candles(self):
        """다중 타임프레임 캔들 업데이트"""
        try:
            # 1h 포함하여 주기적으로 업데이트 (30m은 메인으로 관리)
            for tf in ['1m', '3m', '5m', '15m', '1h']:
                try:
                    # 최근 2개 캔들만 조회
                    ohlcv = await self.exchange.fetch_ohlcv(
                        self.symbol,
                        tf,
                        limit=2
                    )
                    
                    for candle in ohlcv:
                        timestamp = candle[0]
                        
                        # 버퍼가 비어있거나 새로운 캔들인 경우
                        if not self.multi_tf_buffers[tf] or timestamp > self.multi_tf_buffers[tf][-1][0]:
                            self.multi_tf_buffers[tf].append(candle)
                        # 기존 캔들 업데이트
                        elif self.multi_tf_buffers[tf] and timestamp == self.multi_tf_buffers[tf][-1][0]:
                            self.multi_tf_buffers[tf][-1] = candle
                    
                    # EMA 재계산
                    await self.calculate_multi_tf_ema(tf)
                    
                except Exception as e:
                    self.logger.debug(f"타임프레임 {tf} 업데이트 오류: {e}")
                    
        except Exception as e:
            self.logger.error(f"다중 타임프레임 업데이트 오류: {e}")
    
    def get_current_ema(self) -> Dict:
        """현재 EMA 값 반환"""
        return self.current_ema.copy()
    
    def get_multi_tf_ema(self) -> Dict:
        """다중 타임프레임 EMA 값 반환"""
        return self.multi_tf_ema.copy()
    
    def get_ema_status(self) -> str:
        """EMA 상태 반환"""
        if not self.current_ema['ema10']:
            return "UNKNOWN"
        
        ema10 = self.current_ema['ema10']
        ema21 = self.current_ema['ema21']
        
        if ema10 > ema21:
            return "BULLISH"
        elif ema10 < ema21:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.exchange:
                await self.exchange.close()
            
            self.logger.info("📊 Data Collector 정리 완료")
            
        except Exception as e:
            self.logger.error(f"정리 중 오류: {str(e)}")