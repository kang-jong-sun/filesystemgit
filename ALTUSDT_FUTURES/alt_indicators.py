#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 지표 계산 모듈
HMA, DEMA, VWMA, WMA, EMA, MACD 지표 계산
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import math

class IndicatorCalculator:
    """지표 계산 클래스"""
    
    @staticmethod
    def calculate_hma(data: pd.Series, period: int = 2) -> pd.Series:
        """Hull Moving Average (HMA) 계산"""
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        # WMA 계산 함수
        def wma(series, n):
            weights = np.arange(1, n + 1)
            return series.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(data, half_period)
        wma_full = wma(data, period)
        diff = 2 * wma_half - wma_full
        hma = wma(diff, sqrt_period)
        
        return hma
    
    @staticmethod
    def calculate_dema(data: pd.Series, period: int = 10) -> pd.Series:
        """Double Exponential Moving Average (DEMA) 계산"""
        ema1 = data.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        dema = 2 * ema1 - ema2
        return dema
    
    @staticmethod
    def calculate_vwma(data: pd.DataFrame, period: int = 1) -> pd.Series:
        """Volume Weighted Moving Average (VWMA) 계산"""
        if period == 1:
            return data['close']
        
        # 가격 * 거래량의 이동평균 / 거래량의 이동평균
        pv = data['close'] * data['volume']
        vwma = pv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return vwma
    
    @staticmethod
    def calculate_wma(data: pd.Series, period: int = 10) -> pd.Series:
        """Weighted Moving Average (WMA) 계산"""
        weights = np.arange(1, period + 1)
        wma = data.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return wma
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int = 10) -> pd.Series:
        """Exponential Moving Average (EMA) 계산"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast: int = 5, slow: int = 300, signal: int = 1) -> dict:
        """MACD 계산"""
        # MACD Line = Fast EMA - Slow EMA
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        
        # Signal Line = EMA of MACD Line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_angle(data: pd.Series, lookback: int = 15) -> float:
        """
        최근 n개 점을 이은 선의 진입 각도 계산
        단순하게: 첫 점과 마지막 점의 변화율을 각도로 변환
        
        Returns: 각도 (0-360도)
        정배열(상승): 275-355도 
        역배열(하락): 5-85도
        """
        if len(data) < lookback:
            return 180  # 데이터 부족시 수평
        
        # 최근 lookback개 데이터
        recent_data = data.iloc[-lookback:].values
        
        # NaN 체크
        if np.any(np.isnan(recent_data)):
            return 180
        
        # 첫 점과 마지막 점의 변화율 계산
        first_value = recent_data[0]
        last_value = recent_data[-1]
        
        # 절대값이 매우 작은 경우 처리 (MACD의 경우)
        if abs(first_value) < 0.000001:
            # 절대 변화량으로 계산
            change = last_value - first_value
            if change > 0:
                return 315  # 상승
            elif change < 0:
                return 45   # 하락
            else:
                return 180  # 수평
        
        # 변화율 계산 (%)
        change_percent = ((last_value - first_value) / abs(first_value)) * 100
        
        # 더 민감한 각도 변환 (특히 단기 변화에)
        # 상승: +0.5% = 355도, +0.2% = 335도, +0.1% = 315도, +0.05% = 295도
        # 하락: -0.5% = 5도, -0.2% = 25도, -0.1% = 45도, -0.05% = 65도
        
        if change_percent > 0:  # 상승
            # 상승 정도에 따라 275-355도
            if change_percent >= 0.5:
                angle = 355
            elif change_percent >= 0.2:
                angle = 335
            elif change_percent >= 0.1:
                angle = 315
            elif change_percent >= 0.05:
                angle = 295
            elif change_percent >= 0.01:
                angle = 285
            else:
                angle = 280  # 미세 상승
        elif change_percent < 0:  # 하락
            # 하락 정도에 따라 5-85도
            if change_percent <= -0.5:
                angle = 5
            elif change_percent <= -0.2:
                angle = 25
            elif change_percent <= -0.1:
                angle = 45
            elif change_percent <= -0.05:
                angle = 65
            elif change_percent <= -0.01:
                angle = 75
            else:
                angle = 80  # 미세 하락
        else:  # 변화 없음
            angle = 180  # 수평
        
        return angle
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """모든 지표 계산"""
        # HLC3 계산 (High + Low + Close) / 3
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        
        # 1. HMA 계산 (길이: 2, 소스: hlc3)
        hma = IndicatorCalculator.calculate_hma(hlc3, period=2)
        
        # 2. DEMA 계산 (길이: 10, 소스: HMA)
        dema = IndicatorCalculator.calculate_dema(hma, period=10)
        
        # 3. VWMA 계산 (길이: 1, 소스: DEMA)
        # VWMA with period 1 equals to the source itself
        vwma1_dema = dema.copy()
        
        # 4. WMA 계산 (길이: 10, 소스: VWMA)
        wma10_vwma = IndicatorCalculator.calculate_wma(vwma1_dema, period=10)
        
        # 5. EMA 계산 (길이: 10, 소스: VWMA)
        ema10_vwma = IndicatorCalculator.calculate_ema(vwma1_dema, period=10)
        
        # 6. MACD 계산 (소스: vwma1_dema)
        macd_result = IndicatorCalculator.calculate_macd(
            vwma1_dema, 
            fast=5, 
            slow=300, 
            signal=1
        )
        
        # DataFrame에 추가
        df['hma'] = hma
        df['dema'] = dema
        df['vwma1_dema'] = vwma1_dema
        df['wma10_vwma'] = wma10_vwma
        df['ema10_vwma'] = ema10_vwma
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['signal']
        df['macd_histogram'] = macd_result['histogram']
        
        # 각도 계산 (최근 15개 점)
        if len(df) >= 15:
            # 일반 지표는 15개 점 사용
            df['vwma1_dema_angle'] = IndicatorCalculator.calculate_angle(df['vwma1_dema'], 15)
            df['ema10_vwma_angle'] = IndicatorCalculator.calculate_angle(df['ema10_vwma'], 15)
            df['wma10_vwma_angle'] = IndicatorCalculator.calculate_angle(df['wma10_vwma'], 15)
            
            # MACD 신호선 각도 계산 (더 짧은 기간 사용)
            # MACD는 이미 평활화되어 있으므로 5개 점만 사용
            macd_signal = df['macd_signal']
            if len(macd_signal) >= 5:
                df['macd_signal_angle'] = IndicatorCalculator.calculate_angle(macd_signal, 5)
            else:
                df['macd_signal_angle'] = 180  # 수평
            
            # 혼합 각도 계산 (원형 평균)
            angles = np.array([
                df['vwma1_dema_angle'].values,
                df['ema10_vwma_angle'].values,
                df['wma10_vwma_angle'].values
            ])
            
            # 각도를 라디안으로 변환
            angles_rad = np.radians(angles)
            
            # 원형 평균 계산
            sin_mean = np.mean(np.sin(angles_rad), axis=0)
            cos_mean = np.mean(np.cos(angles_rad), axis=0)
            
            # 다시 각도로 변환
            mixed_angle_rad = np.arctan2(sin_mean, cos_mean)
            mixed_angle_deg = np.degrees(mixed_angle_rad)
            
            # 0-360 범위로 정규화
            mixed_angle_deg = np.where(mixed_angle_deg < 0, mixed_angle_deg + 360, mixed_angle_deg)
            
            df['mixed_angle'] = mixed_angle_deg
        else:
            df['vwma1_dema_angle'] = 0
            df['ema10_vwma_angle'] = 0
            df['wma10_vwma_angle'] = 0
            df['macd_signal_angle'] = 0
            df['mixed_angle'] = 0
        
        return df
    
    @staticmethod
    def check_bullish_alignment(df: pd.DataFrame) -> bool:
        """
        정배열 확인 (상승 신호 - LONG 진입)
        좌측 하단에서 우측 상단으로 진입
        - 혼합각도: 275도 ~ 355도 (상승)
        - MACD 신호선 각도: 275도 ~ 355도 (상승)
        """
        if len(df) < 15:
            return False
        
        last_row = df.iloc[-1]
        
        # 혼합 각도 체크 (275-355도: 상승)
        mixed_angle = last_row['mixed_angle']
        mixed_bullish = 275 <= mixed_angle <= 355
        
        # MACD 신호선 각도 체크 (275-355도: 상승)
        macd_angle = last_row['macd_signal_angle']
        macd_bullish = 275 <= macd_angle <= 355
        
        return mixed_bullish and macd_bullish
    
    @staticmethod
    def check_bearish_alignment(df: pd.DataFrame) -> bool:
        """
        역배열 확인 (하락 신호 - SHORT 진입)
        좌측 상단에서 우측 하단으로 진입
        - 혼합각도: 5도 ~ 85도 (하락)
        - MACD 신호선 각도: 5도 ~ 85도 (하락)
        """
        if len(df) < 15:
            return False
        
        last_row = df.iloc[-1]
        
        # 혼합 각도 체크 (5-85도: 하락)
        mixed_angle = last_row['mixed_angle']
        mixed_bearish = 5 <= mixed_angle <= 85
        
        # MACD 신호선 각도 체크 (5-85도: 하락)
        macd_angle = last_row['macd_signal_angle']
        macd_bearish = 5 <= macd_angle <= 85
        
        return mixed_bearish and macd_bearish