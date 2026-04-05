#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 모니터링 모듈 - Flow Style
실시간 흐르는 로그 형태의 상태 표시
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
from collections import deque
import time
import sys
from alt_indicators import IndicatorCalculator

class Monitor:
    """모니터링 클래스 - 흐르는 로그 스타일"""
    
    def __init__(self, config, bot, data_collector, executor, logger):
        self.config = config
        self.bot = bot
        self.data_collector = data_collector
        self.executor = executor
        self.logger = logger
        self.start_time = datetime.now()
        self.last_indicators = {}
        self.last_price = 0
        self.log_count = 0
        self.last_summary_time = 0
        
    async def display_status(self):
        """흐르는 로그 형태의 상태 표시"""
        try:
            current_time = time.time()
            
            # 계정 정보
            account_info = await self.data_collector.get_account_info()
            balance = account_info.get('balance', 0)
            
            # 시장 데이터
            df = self.data_collector.get_latest_data()
            if df is None or len(df) < 15:
                return
                
            # 지표 계산
            df = IndicatorCalculator.calculate_all_indicators(df)
            last_row = df.iloc[-1]
            current_price = last_row['close']
            
            # 가격 변화 표시
            if self.last_price != 0:
                price_change = current_price - self.last_price
                if abs(price_change) > 0.000001:
                    price_emoji = "📈" if price_change > 0 else "📉"
                    self._flow_log(f"{price_emoji} ALT: {self.last_price:.6f} → {current_price:.6f} ({price_change:+.6f})")
            self.last_price = current_price
            
            # 지표 변화 모니터링
            self._monitor_indicator_changes(last_row)
            
            # 신호 체크
            self._check_signals(df, last_row)
            
            # 포지션 상태 업데이트 (포지션 유무 관계없이 현재가 표시)
            await self._update_position_status(current_price)
            
            # 20초마다 요약
            if current_time - self.last_summary_time >= 20:
                self._print_summary(balance, current_price, last_row)
                self.last_summary_time = current_time
                
        except Exception as e:
            self.logger.error(f"모니터링 오류: {e}")
    
    def _flow_log(self, message: str):
        """흐르는 로그 출력 (화면에만 타임스탬프, 파일은 logger가 처리)"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        # 화면 출력용 (타임스탬프 포함)
        print(f"[{timestamp}] {message}")
        # 파일 로그용 (타임스탬프는 logger가 자동 추가)
        self.logger.info(message)
        self.log_count += 1
        
        # 50줄마다 구분선
        if self.log_count % 50 == 0:
            print("─" * 80)
    
    def _monitor_indicator_changes(self, last_row: pd.Series):
        """지표 변화 모니터링"""
        indicators = {
            'VWMA각': last_row.get('vwma1_dema_angle', 0),
            'EMA각': last_row.get('ema10_vwma_angle', 0),
            'WMA각': last_row.get('wma10_vwma_angle', 0),
            '혼합각': last_row.get('mixed_angle', 0),
            'MACD각': last_row.get('macd_signal_angle', 0)
        }
        
        # 변화 감지
        changes = []
        for key, value in indicators.items():
            if key in self.last_indicators:
                change = value - self.last_indicators[key]
                if abs(change) >= 3:  # 3도 이상 변화
                    arrow = "↑" if change > 0 else "↓"
                    changes.append(f"{key}: {arrow}{abs(change):.1f}°")
        
        if changes:
            self._flow_log(f"📐 각도변화: {' | '.join(changes)}")
        
        # 5초마다 현재 각도와 가격 표시 (포지션 정보 포함)
        if int(time.time()) % 5 == 0:
            # 기본 정보
            angle_info = f"📊 ALT: {self.last_price:.6f}"
            
            # 포지션이 있으면 간단한 ROI 표시
            if self.bot.position:
                pos = self.bot.position
                roi = pos.calculate_roi(self.last_price)
                roi_symbol = "↑" if roi > 0 else "↓"
                angle_info += f" | [{pos.side}] 진입:{pos.entry_price:.6f} {roi_symbol}{abs(roi):.1f}%"
            
            # 각도 정보
            angle_info += f" | [V:{indicators['VWMA각']:.1f}° E:{indicators['EMA각']:.1f}° W:{indicators['WMA각']:.1f}°]"
            angle_info += f" 혼합:{indicators['혼합각']:.1f}° MACD:{indicators['MACD각']:.1f}°"
            self._flow_log(angle_info)
        
        self.last_indicators = indicators
    
    def _check_signals(self, df: pd.DataFrame, last_row: pd.Series):
        """신호 체크 (각도 상세 분석)"""
        mixed_angle = last_row.get('mixed_angle', 0)
        macd_angle = last_row.get('macd_signal_angle', 0)
        
        # 정배열 체크 (273° ~ 357°)
        mixed_bull = 273 <= mixed_angle <= 357
        macd_bull = 273 <= macd_angle <= 357
        
        # 역배열 체크 (3° ~ 87°)
        mixed_bear = 3 <= mixed_angle <= 87
        macd_bear = 3 <= macd_angle <= 87
        
        # 완전 정배열
        if mixed_bull and macd_bull:
            if not self.bot.position or self.bot.position.side != "LONG":
                self._flow_log(f"🟢 【정배열 신호】 매수 조건 충족! 혼합:{mixed_angle:.1f}° MACD:{macd_angle:.1f}°")
        
        # 완전 역배열
        elif mixed_bear and macd_bear:
            if not self.bot.position or self.bot.position.side != "SHORT":
                self._flow_log(f"🔴 【역배열 신호】 매도 조건 충족! 혼합:{mixed_angle:.1f}° MACD:{macd_angle:.1f}°")
        
        # 부분 신호 (디버그용)
        elif mixed_bull or macd_bull or mixed_bear or macd_bear:
            conditions = []
            if mixed_bull:
                conditions.append(f"혼합정배열({mixed_angle:.1f}°)")
            elif mixed_bear:
                conditions.append(f"혼합역배열({mixed_angle:.1f}°)")
            if macd_bull:
                conditions.append(f"MACD정배열({macd_angle:.1f}°)")
            elif macd_bear:
                conditions.append(f"MACD역배열({macd_angle:.1f}°)")
            
            # 10초마다 부분 신호 표시
            if int(time.time()) % 10 == 0:
                self._flow_log(f"⚠️ 【부분 신호】 {' / '.join(conditions)}")
    
    async def _update_position_status(self, current_price: float):
        """포지션 상태 업데이트"""
        if not self.bot.position:
            # 포지션이 없을 때도 현재가 표시 (10초마다)
            if int(time.time()) % 10 == 0:
                self._flow_log(f"💱 현재가: ALT/USDT {current_price:.6f} | 포지션: 없음 (신호 대기중)")
            return
            
        pos = self.bot.position
        pos.update_price(current_price)
        roi = pos.calculate_roi(current_price)
        holding_minutes = (datetime.now() - pos.entry_time).total_seconds() / 60
        
        # ROI 색상 및 이모지
        if roi > 10:
            roi_emoji = "🚀"  # 10% 이상
            roi_color = "🟢"
        elif roi > 5:
            roi_emoji = "💰"  # 5-10%
            roi_color = "🟢"
        elif roi > 0:
            roi_emoji = "💚"  # 0-5%
            roi_color = "🟡"
        elif roi > -3:
            roi_emoji = "😰"  # -3% ~ 0%
            roi_color = "🟡"
        else:
            roi_emoji = "💔"  # -3% 이하
            roi_color = "🔴"
        
        # 10초마다 포지션 상태 표시 (진입가와 ROI 강조)
        if int(time.time()) % 10 == 0:
            position_info = f"{roi_emoji} [{pos.side}] "
            position_info += f"진입: {pos.entry_price:.6f} → 현재: {current_price:.6f} "
            position_info += f"| {roi_color} ROI: {roi:+.2f}% | "
            position_info += f"보유: {holding_minutes:.0f}분"
            
            # 최고/최저 ROI
            if abs(pos.highest_roi) > 0 or abs(pos.lowest_roi) > 0:
                position_info += f" [최고:{pos.highest_roi:+.1f}% 최저:{pos.lowest_roi:+.1f}%]"
            
            # 트레일링 SL 상태
            if pos.trailing_sl_price > 0:
                position_info += f" | SL: {pos.trailing_sl_price:.6f}"
            
            self._flow_log(position_info)
        
        # 트레일링 스톱 알림
        if pos.trailing_sl_price > 0:
            if pos.side == "LONG":
                sl_distance = (current_price - pos.trailing_sl_price) / current_price * 100
            else:
                sl_distance = (pos.trailing_sl_price - current_price) / current_price * 100
                
            if 0 < sl_distance < 1:  # 1% 이내 접근
                self._flow_log(f"⚠️ 【SL 근접】 현재:{current_price:.6f} SL:{pos.trailing_sl_price:.6f} 거리:{sl_distance:.2f}%")
        
        # 수익 구간별 알림
        roi_levels = [5, 10, 15, 20, 30, 50]
        for level in roi_levels:
            if roi >= level and not hasattr(pos, f'alerted_{level}'):
                self._flow_log(f"🎯 【수익 {level}% 도달】 ROI: {roi:.2f}%")
                setattr(pos, f'alerted_{level}', True)
    
    def _print_summary(self, balance: float, current_price: float, last_row: pd.Series):
        """주기적 요약"""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print("\n" + "=" * 80)
        self._flow_log(f"📊 【요약】 실행: {hours:02d}:{minutes:02d}:{seconds:02d} | 잔액: {balance:,.2f} USDT | ALT: {current_price:.6f}")
        
        # 포지션 정보
        if self.bot.position:
            pos = self.bot.position
            roi = pos.calculate_roi(current_price)
            holding_time = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            # ROI 색상 결정
            if roi > 5:
                roi_display = f"🟢 {roi:+.2f}%"
            elif roi > 0:
                roi_display = f"🟡 {roi:+.2f}%"
            else:
                roi_display = f"🔴 {roi:+.2f}%"
            
            pos_summary = f"   포지션: [{pos.side}] {pos.quantity:.2f} ALT"
            pos_summary += f" | 진입: {pos.entry_price:.6f} → 현재: {current_price:.6f}"
            pos_summary += f" | ROI: {roi_display} | 보유: {holding_time:.0f}분"
            self._flow_log(pos_summary)
            
            if pos.trailing_sl_price > 0:
                self._flow_log(f"   트레일링SL: {pos.trailing_sl_price:.6f} | 최고ROI: {pos.highest_roi:+.2f}% | 최저ROI: {pos.lowest_roi:+.2f}%")
        else:
            self._flow_log("   포지션: 없음 (신호 대기중)")
        
        # 현재 지표
        indicators = self.last_indicators
        if indicators:
            self._flow_log(f"   각도: V={indicators.get('VWMA각', 0):.1f}° E={indicators.get('EMA각', 0):.1f}° W={indicators.get('WMA각', 0):.1f}° | 혼합={indicators.get('혼합각', 0):.1f}° MACD={indicators.get('MACD각', 0):.1f}°")
        
        # 신호 상태 (각도 조건 체크)
        mixed = indicators.get('혼합각', 0)
        macd = indicators.get('MACD각', 0)
        
        # 각 조건 체크 (올바른 각도 범위)
        mixed_bull = 275 <= mixed <= 355  # 정배열: 좌하→우상 (상승)
        macd_bull = 275 <= macd <= 355    # 정배열: 좌하→우상 (상승)
        mixed_bear = 5 <= mixed <= 85     # 역배열: 좌상→우하 (하락)
        macd_bear = 5 <= macd <= 85       # 역배열: 좌상→우하 (하락)
        
        if mixed_bull and macd_bull:
            self._flow_log(f"   신호: 🟢 정배열 (상승/매수) - 혼합:{mixed:.1f}° MACD:{macd:.1f}°")
        elif mixed_bear and macd_bear:
            self._flow_log(f"   신호: 🔴 역배열 (하락/매도) - 혼합:{mixed:.1f}° MACD:{macd:.1f}°")
        else:
            # 부분 조건 표시
            status = []
            if mixed_bull:
                status.append("혼합↗")
            elif mixed_bear:
                status.append("혼합↘")
            if macd_bull:
                status.append("MACD↗")
            elif macd_bear:
                status.append("MACD↘")
            
            status_str = f" ({', '.join(status)})" if status else ""
            self._flow_log(f"   신호: ⚪ 중립{status_str} - 혼합:{mixed:.1f}° MACD:{macd:.1f}°")
        
        print("=" * 80 + "\n")
    
    def log_trade(self, trade_info: Dict):
        """거래 체결 로그"""
        print("\n" + "🔔" * 40)
        self._flow_log(f"✅ 【거래 체결】 {trade_info.get('side')} {trade_info.get('quantity'):.4f} @ {trade_info.get('price'):.6f}")
        print("🔔" * 40 + "\n")
    
    def log_signal(self, signal_type: str, details: Dict):
        """신호 로그"""
        if signal_type == "BULLISH":
            self._flow_log(f"🟢 【정배열 감지】 혼합:{details.get('mixed_angle', 0):.1f}° MACD:{details.get('macd_angle', 0):.1f}°")
        elif signal_type == "BEARISH":
            self._flow_log(f"🔴 【역배열 감지】 혼합:{details.get('mixed_angle', 0):.1f}° MACD:{details.get('macd_angle', 0):.1f}°")
    
    def log_trailing_stop_update(self, new_sl: float, roi: float):
        """트레일링 스톱 업데이트"""
        self._flow_log(f"📊 【트레일링 업데이트】 새 SL: {new_sl:.6f} (ROI: {roi:.2f}%)")
    
    def log_position_close(self, reason: str, roi: float):
        """포지션 청산"""
        emoji = "💰" if roi > 0 else "💸"
        print("\n" + "=" * 80)
        self._flow_log(f"{emoji} 【포지션 청산】 사유: {reason} | 최종 ROI: {roi:+.2f}%")
        print("=" * 80 + "\n")
    
    def log_error(self, error_msg: str):
        """에러 로그"""
        self._flow_log(f"❌ 【오류】 {error_msg}")