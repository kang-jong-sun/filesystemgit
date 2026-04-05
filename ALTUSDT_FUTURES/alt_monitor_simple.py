#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 단순화된 모니터링
MACD 신호선 기반 표시
"""

import asyncio
from datetime import datetime
from typing import Optional

class SimpleMonitor:
    """단순화된 모니터링"""
    
    def __init__(self, bot):
        self.bot = bot
        self.logger = bot.logger
        self.start_time = datetime.now()
        self.last_price = 0
        self.last_macd_signal = None
        self.previous_macd_signal = None  # 이전 완성된 캔들의 MACD 신호선 값
        self.signal_trend = 0  # -1: 하락, 0: 중립, 1: 상승
        
    def _flow_log(self, message: str):
        """흐르는 로그 출력"""
        timestamp = datetime.now().strftime('[%H:%M:%S]')
        print(f"{timestamp} {message}")
        self.logger.info(message)
    
    async def run(self):
        """모니터링 실행"""
        summary_counter = 0
        
        while self.bot.is_running:
            try:
                # 5초마다 업데이트
                await asyncio.sleep(5)
                
                # 현재 데이터 가져오기
                df = self.bot.data_collector.get_latest_data()
                if df is None or df.empty:
                    continue
                
                current_price = df.iloc[-1]['close']
                
                # 가격 변화 표시
                if self.last_price > 0:
                    price_change = current_price - self.last_price
                    if abs(price_change) > 0.000001:
                        if price_change > 0:
                            self._flow_log(f"📈 ALT: {self.last_price:.6f} → {current_price:.6f} (+{price_change:.6f})")
                        else:
                            self._flow_log(f"📉 ALT: {self.last_price:.6f} → {current_price:.6f} ({price_change:.6f})")
                
                self.last_price = current_price
                
                # 20초마다 요약
                summary_counter += 1
                if summary_counter >= 4:  # 5초 * 4 = 20초
                    self._print_summary(current_price)
                    summary_counter = 0
                    
            except Exception as e:
                self.logger.error(f"모니터링 오류: {e}")
                await asyncio.sleep(5)
    
    def _print_summary(self, current_price: float):
        """주기적 요약"""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 계정 정보
        balance = 0
        try:
            balance = asyncio.run_coroutine_threadsafe(
                self.bot.executor.get_account_balance(),
                asyncio.get_event_loop()
            ).result(timeout=2)
        except:
            pass
        
        print("\n" + "=" * 80)
        self._flow_log(f"📊 【요약】 실행: {hours:02d}:{minutes:02d}:{seconds:02d} | 잔액: {balance:,.2f} USDT | ALT: {current_price:.6f}")
        
        # MACD 신호선 표시
        if self.last_macd_signal is not None:
            # 이전 캔들과 현재 값 표시
            if self.previous_macd_signal is not None:
                signal_str = f"MACD 신호선: 이전 완성된 캔들: {self.previous_macd_signal:.6f} → 현재: {self.last_macd_signal:.6f}"
            else:
                signal_str = f"MACD 신호선: 현재: {self.last_macd_signal:.6f}"
            
            # 0.00001 기준선 대비 위치 표시
            if self.last_macd_signal >= 0.00001:
                signal_str += " (0.00001 이상 - 매수 영역)"
            else:
                signal_str += " (0.00001 미만 - 매도 영역)"
            
            # 트렌드 표시
            if self.signal_trend == 1:
                signal_str += " 📈 상승 (3분+)"
            elif self.signal_trend == -1:
                signal_str += " 📉 하락 (3분+)"
            else:
                signal_str += " ➡️ 중립"
            
            self._flow_log(f"   {signal_str}")
        
        # 포지션 정보
        if self.bot.position:
            pos = self.bot.position
            roi = pos.calculate_roi(current_price)
            holding_time = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            # ROI 색상
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
            
            # 최고/최저 정보
            if pos.highest_price > 0 and pos.lowest_price > 0:
                high_low_info = f"   최고가: {pos.highest_price:.6f} (ROI: {pos.highest_roi:+.2f}%)"
                high_low_info += f" | 최저가: {pos.lowest_price:.6f} (ROI: {pos.lowest_roi:+.2f}%)"
                self._flow_log(high_low_info)
            
            # 트레일링 스톱
            if pos.trailing_sl_price > 0:
                self._flow_log(f"   트레일링 SL: {pos.trailing_sl_price:.6f}")
        else:
            self._flow_log("   포지션: 없음 (신호 대기중)")
        
        # 신호 상태
        if self.signal_trend == 1:
            self._flow_log("   신호: 🟢 매수 신호 (MACD 신호선 상승 3분+)")
        elif self.signal_trend == -1:
            self._flow_log("   신호: 🔴 매도 신호 (MACD 신호선 하락 3분+)")
        else:
            self._flow_log("   신호: ⚪ 중립 (신호 대기)")
        
        print("=" * 80 + "\n")