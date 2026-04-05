#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 단순화된 지표 계산 모듈
MACD 신호선 기반 트레이딩
"""

import pandas as pd
import numpy as np

class SimpleIndicators:
    """단순화된 지표 계산 클래스"""
    
    @staticmethod
    def calculate_macd(close_prices: pd.Series, fast: int = 5, slow: int = 200, signal: int = 1) -> dict:
        """MACD 계산 (종가 기반)"""
        # Fast EMA
        ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
        
        # Slow EMA
        ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """모든 필요한 지표 계산"""
        # MACD 계산 (종가 기반)
        macd_result = SimpleIndicators.calculate_macd(
            df['close'], 
            fast=5, 
            slow=200, 
            signal=1
        )
        
        # DataFrame에 추가
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['signal']
        df['macd_histogram'] = macd_result['histogram']
        
        # 신호선 변화량 계산 (이전 캔들 대비)
        df['signal_change'] = df['macd_signal'].diff()
        
        # 3분(3개 캔들) 연속 상승/하락 체크
        df['signal_trend'] = 0  # 0: 중립, 1: 상승, -1: 하락
        
        # MACD 신호선 포지션 (0.00001 기준선)
        df['signal_position'] = 0  # 0: 중립, 1: 0.00001 이상, -1: 0.00001 미만
        df['signal_position'] = df['macd_signal'].apply(lambda x: 1 if x >= 0.00001 else -1 if x < 0.00001 else 0)
        
        if len(df) >= 3:
            # 최근 3개 캔들의 신호선 변화 확인
            for i in range(2, len(df)):
                changes = [
                    df['signal_change'].iloc[i-2],
                    df['signal_change'].iloc[i-1],
                    df['signal_change'].iloc[i]
                ]
                
                # 3개 모두 상승 (0.000001 이상)
                if all(c >= 0.000001 for c in changes if not pd.isna(c)):
                    df.loc[df.index[i], 'signal_trend'] = 1  # 상승
                # 3개 모두 하락 (-0.000001 이하)
                elif all(c <= -0.000001 for c in changes if not pd.isna(c)):
                    df.loc[df.index[i], 'signal_trend'] = -1  # 하락
        
        return df
    
    @staticmethod
    def check_buy_signal(df: pd.DataFrame) -> bool:
        """매수 신호 체크"""
        if len(df) < 3:
            return False
        
        # 최근 값
        last_row = df.iloc[-1]
        
        # MACD 신호선이 3분 이상 상승 중
        return last_row['signal_trend'] == 1
    
    @staticmethod
    def check_sell_signal(df: pd.DataFrame) -> bool:
        """매도 신호 체크"""
        if len(df) < 3:
            return False
        
        # 최근 값
        last_row = df.iloc[-1]
        
        # MACD 신호선이 3분 이상 하락 중
        return last_row['signal_trend'] == -1