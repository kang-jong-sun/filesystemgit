#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 모니터링 모듈 V2
실시간 흐르는 로그 형태의 상태 표시
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
from collections import deque
import time
from alt_indicators import IndicatorCalculator

class Monitor:
    """모니터링 클래스 V2 - 흐르는 로그 스타일"""
    
    def __init__(self, config, bot, data_collector, executor, logger):
        self.config = config
        self.bot = bot
        self.data_collector = data_collector
        self.executor = executor
        self.logger = logger
        self.start_time = datetime.now()
        self.log_buffer = deque(maxlen=50)  # 최근 50개 로그 유지
        self.last_indicators = {}
        
    async def display_status(self):
        """흐르는 로그 형태의 상태 표시"""
        try:
            # 계정 정보
            account_info = await self.data_collector.get_account_info()
            balance = account_info.get('balance', 0)
            
            # 시장 데이터
            df = self.data_collector.get_latest_data()
            if df is not None and len(df) >= 15:
                df = IndicatorCalculator.calculate_all_indicators(df)
                last_row = df.iloc[-1]
                current_price = last_row['close']
                
                # 지표 변화 감지 및 로그
                self._log_indicator_changes(last_row)
                
                # 신호 체크
                self._check_and_log_signals(df, last_row)
            else:
                current_price = 0
            
            # 포지션 상태 로그
            if self.bot.position:
                self._log_position_status(current_price)
            
            # 주기적 요약 (30초마다)
            if int(time.time()) % 30 == 0:
                self._log_summary(balance, current_price)
            
            # 버퍼 출력
            self._print_log_buffer()
            
        except Exception as e:
            self._add_log(f"❌ 모니터링 오류: {e}", "ERROR")
    
    def _add_log(self, message: str, level: str = "INFO"):
        """로그 버퍼에 추가"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] {message}"
        self.log_buffer.append(formatted_msg)
        
        # 로거에도 기록
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _log_indicator_changes(self, last_row: pd.Series):
        """지표 변화 로그"""
        indicators = {
            'vwma1_dema_angle': last_row.get('vwma1_dema_angle', 0),
            'ema10_vwma_angle': last_row.get('ema10_vwma_angle', 0),
            'wma10_vwma_angle': last_row.get('wma10_vwma_angle', 0),
            'mixed_angle': last_row.get('mixed_angle', 0),
            'macd_signal_angle': last_row.get('macd_signal_angle', 0)
        }
        
        # 큰 변화 감지 (5도 이상)
        for key, value in indicators.items():
            if key in self.last_indicators:
                change = value - self.last_indicators[key]
                if abs(change) >= 5:
                    direction = "↑" if change > 0 else "↓"
                    self._add_log(f"📊 {key}: {self.last_indicators[key]:.1f}° → {value:.1f}° ({direction}{abs(change):.1f}°)")
        
        self.last_indicators = indicators
    
    def _check_and_log_signals(self, df: pd.DataFrame, last_row: pd.Series):
        """신호 체크 및 로그"""
        mixed_angle = last_row.get('mixed_angle', 0)
        macd_angle = last_row.get('macd_signal_angle', 0)
        
        # 정배열 체크
        if 273 <= mixed_angle <= 357 and 273 <= macd_angle <= 357:
            self._add_log(f"🟢 정배열 신호! 혼합각: {mixed_angle:.1f}° | MACD각: {macd_angle:.1f}°")
            if not self.bot.position:
                self._add_log("  → 매수 포지션 진입 대기중...")
        
        # 역배열 체크
        elif 3 <= mixed_angle <= 87 and 3 <= macd_angle <= 87:
            self._add_log(f"🔴 역배열 신호! 혼합각: {mixed_angle:.1f}° | MACD각: {macd_angle:.1f}°")
            if not self.bot.position:
                self._add_log("  → 매도 포지션 진입 대기중...")
    
    def _log_position_status(self, current_price: float):
        """포지션 상태 로그"""
        if not self.bot.position:
            return
        
        pos = self.bot.position
        pos.update_price(current_price)
        roi = pos.calculate_roi(current_price)
        
        # ROI 변화 로그
        roi_emoji = "📈" if roi > 0 else "📉"
        roi_color = "🟢" if roi > 0 else "🔴"
        
        # 중요 이벤트 로그
        if abs(roi) >= 1 and int(time.time()) % 10 == 0:  # 10초마다 1% 이상 변화시
            self._add_log(
                f"{roi_emoji} {pos.side} | 현재가: {current_price:.4f} | "
                f"ROI: {roi_color}{roi:+.2f}% | "
                f"최고ROI: {pos.highest_roi:+.2f}% | 최저ROI: {pos.lowest_roi:+.2f}%"
            )
        
        # 트레일링 스톱 업데이트 체크
        if pos.trailing_sl_price > 0:
            sl_distance = abs(current_price - pos.trailing_sl_price) / current_price * 100
            if sl_distance < 1:  # 스톱로스에 1% 이내 접근
                self._add_log(f"⚠️ 스톱로스 근접! 현재가: {current_price:.4f} | SL: {pos.trailing_sl_price:.4f}", "WARNING")
    
    def _log_summary(self, balance: float, current_price: float):
        """주기적 요약 로그"""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self._add_log("=" * 70)
        self._add_log(f"📊 === 요약 === 실행시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self._add_log(f"💰 잔액: {balance:,.2f} USDT | 💱 ALT/USDT: {current_price:.4f}")
        
        if self.bot.position:
            pos = self.bot.position
            roi = pos.calculate_roi(current_price)
            holding_time = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            self._add_log(
                f"📍 포지션: {pos.side} {pos.quantity:.4f} @ {pos.entry_price:.4f} | "
                f"보유시간: {holding_time:.0f}분"
            )
            self._add_log(
                f"   ROI: {roi:+.2f}% | 최고: {pos.highest_roi:+.2f}% | "
                f"최저: {pos.lowest_roi:+.2f}% | SL: {pos.trailing_sl_price:.4f}"
            )
        else:
            self._add_log("📍 포지션: 없음 (신호 대기중)")
        
        # 현재 지표 상태
        if self.last_indicators:
            self._add_log(
                f"📐 각도: VWMA={self.last_indicators.get('vwma1_dema_angle', 0):.1f}° | "
                f"EMA={self.last_indicators.get('ema10_vwma_angle', 0):.1f}° | "
                f"WMA={self.last_indicators.get('wma10_vwma_angle', 0):.1f}°"
            )
            self._add_log(
                f"   혼합={self.last_indicators.get('mixed_angle', 0):.1f}° | "
                f"MACD={self.last_indicators.get('macd_signal_angle', 0):.1f}°"
            )
        
        self._add_log("=" * 70)
    
    def _print_log_buffer(self):
        """로그 버퍼 출력 (화면 클리어 없이)"""
        # 최근 20개 로그만 출력
        recent_logs = list(self.log_buffer)[-20:]
        for log in recent_logs:
            print(log)
    
    def log_trade(self, trade_info: Dict):
        """거래 체결 로그"""
        self._add_log("🔔" + "=" * 68)
        self._add_log(f"✅ 거래 체결: {trade_info.get('side')} {trade_info.get('quantity'):.4f} @ {trade_info.get('price'):.4f}")
        self._add_log("🔔" + "=" * 68)
    
    def log_signal(self, signal_type: str, details: Dict):
        """신호 감지 로그"""
        if signal_type == "BULLISH":
            emoji = "🟢"
            action = "매수"
        elif signal_type == "BEARISH":
            emoji = "🔴"
            action = "매도"
        else:
            return
        
        self._add_log(
            f"{emoji} {action} 신호! 혼합각: {details.get('mixed_angle', 0):.1f}° | "
            f"MACD각: {details.get('macd_angle', 0):.1f}°"
        )
    
    def log_trailing_stop_update(self, new_sl: float, roi: float):
        """트레일링 스톱 업데이트 로그"""
        self._add_log(f"📊 트레일링 스톱 업데이트: {new_sl:.4f} (ROI: {roi:.2f}%)")
    
    def log_position_close(self, reason: str, roi: float):
        """포지션 청산 로그"""
        emoji = "💰" if roi > 0 else "💸"
        self._add_log("=" * 70)
        self._add_log(f"{emoji} 포지션 청산: {reason} | 최종 ROI: {roi:+.2f}%")
        self._add_log("=" * 70)
    
    def log_error(self, error_msg: str):
        """에러 로그"""
        self._add_log(f"❌ 오류: {error_msg}", "ERROR")