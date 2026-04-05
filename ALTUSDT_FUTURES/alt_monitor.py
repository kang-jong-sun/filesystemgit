#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALT/USDT 모니터링 모듈
실시간 상태 표시 및 로깅
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from alt_indicators import IndicatorCalculator

class Monitor:
    """모니터링 클래스"""
    
    def __init__(self, config, bot, data_collector, executor, logger):
        self.config = config
        self.bot = bot
        self.data_collector = data_collector
        self.executor = executor
        self.logger = logger
        self.console = Console()
        self.start_time = datetime.now()
        
    async def display_status(self):
        """상태 표시"""
        try:
            # 계정 정보
            account_info = await self.data_collector.get_account_info()
            
            # 시장 데이터
            df = self.data_collector.get_latest_data()
            if df is not None and len(df) >= 15:
                df = IndicatorCalculator.calculate_all_indicators(df)
                last_row = df.iloc[-1]
            else:
                last_row = None
            
            # 콘솔 클리어
            self.console.clear()
            
            # 헤더
            self._print_header()
            
            # 계정 정보
            self._print_account_info(account_info)
            
            # 포지션 정보
            self._print_position_info()
            
            # 지표 정보
            if last_row is not None:
                self._print_indicators(last_row)
            
            # 최근 거래
            self._print_recent_trades()
            
        except Exception as e:
            self.logger.error(f"모니터링 표시 오류: {e}")
    
    def _print_header(self):
        """헤더 출력"""
        runtime = datetime.now() - self.start_time
        hours, remainder = divmod(int(runtime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        header_text = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           🚀 ALT/USDT Futures Trading Bot v1.0                      ║
║           실행 시간: {hours:02d}:{minutes:02d}:{seconds:02d}                                         ║
║           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                    ║
╚══════════════════════════════════════════════════════════════════════╝
        """
        self.console.print(header_text, style="bold cyan")
    
    def _print_account_info(self, account_info: Dict):
        """계정 정보 출력"""
        table = Table(title="💰 계정 정보", show_header=True, header_style="bold magenta")
        table.add_column("항목", style="cyan", width=20)
        table.add_column("값", style="yellow", width=30)
        
        balance = account_info.get('balance', 0)
        table.add_row("USDT 잔액", f"{balance:,.2f} USDT")
        table.add_row("레버리지", f"{self.config.leverage}x")
        table.add_row("포지션 크기", f"{self.config.position_size}%")
        table.add_row("마진 모드", self.config.margin_mode)
        
        self.console.print(table)
        self.console.print()
    
    def _print_position_info(self):
        """포지션 정보 출력"""
        table = Table(title="📊 포지션 정보", show_header=True, header_style="bold magenta")
        table.add_column("항목", style="cyan", width=20)
        table.add_column("값", style="yellow", width=30)
        
        if self.bot.position:
            pos = self.bot.position
            roi = pos.calculate_roi(pos.current_price)
            holding_time = (datetime.now() - pos.entry_time).total_seconds() / 60
            
            # 포지션 방향에 따른 색상
            side_color = "green" if pos.side == "LONG" else "red"
            roi_color = "green" if roi > 0 else "red"
            
            table.add_row("포지션", f"[{side_color}]{pos.side}[/{side_color}]")
            table.add_row("수량", f"{pos.quantity:.4f}")
            table.add_row("진입가", f"{pos.entry_price:.4f}")
            table.add_row("현재가", f"{pos.current_price:.4f}")
            table.add_row("ROI", f"[{roi_color}]{roi:+.2f}%[/{roi_color}]")
            table.add_row("보유 시간", f"{holding_time:.0f}분")
            
            # 최고/최저가 정보
            table.add_row("─" * 20, "─" * 30)
            table.add_row("최고가", f"{pos.highest_price:.4f}")
            table.add_row("최고 ROI", f"{pos.highest_roi:+.2f}%")
            table.add_row("최저가", f"{pos.lowest_price:.4f}")
            table.add_row("최저 ROI", f"{pos.lowest_roi:+.2f}%")
            
            # 트레일링 스톱
            if pos.trailing_sl_price > 0:
                table.add_row("트레일링 SL", f"{pos.trailing_sl_price:.4f}")
            else:
                sl_pct = -self.config.base_stop_loss
                table.add_row("기본 SL", f"{pos.stop_loss:.4f} ({sl_pct:.1f}%)")
        else:
            table.add_row("상태", "포지션 없음")
            table.add_row("대기 중", "진입 신호 대기...")
        
        self.console.print(table)
        self.console.print()
    
    def _print_indicators(self, last_row: pd.Series):
        """지표 정보 출력"""
        table = Table(title="📈 지표 정보 (5분봉)", show_header=True, header_style="bold magenta")
        table.add_column("지표", style="cyan", width=25)
        table.add_column("값", style="yellow", width=15)
        table.add_column("각도", style="green", width=15)
        
        # 지표 값과 각도
        indicators = [
            ("VWMA1_DEMA", last_row.get('vwma1_dema', 0), last_row.get('vwma1_dema_angle', 0)),
            ("EMA10_VWMA", last_row.get('ema10_vwma', 0), last_row.get('ema10_vwma_angle', 0)),
            ("WMA10_VWMA", last_row.get('wma10_vwma', 0), last_row.get('wma10_vwma_angle', 0)),
            ("혼합 각도", "-", last_row.get('mixed_angle', 0)),
            ("MACD 신호선", last_row.get('macd_signal', 0), last_row.get('macd_signal_angle', 0))
        ]
        
        for name, value, angle in indicators:
            if value == "-":
                table.add_row(name, value, f"{angle:.1f}°")
            else:
                table.add_row(name, f"{value:.4f}", f"{angle:.1f}°")
        
        # 신호 상태
        table.add_row("─" * 25, "─" * 15, "─" * 15)
        
        # 정배열/역배열 체크
        mixed_angle = last_row.get('mixed_angle', 0)
        macd_angle = last_row.get('macd_signal_angle', 0)
        
        if 273 <= mixed_angle <= 357 and 273 <= macd_angle <= 357:
            signal = "[green]정배열 (매수)[/green]"
        elif 3 <= mixed_angle <= 87 and 3 <= macd_angle <= 87:
            signal = "[red]역배열 (매도)[/red]"
        else:
            signal = "중립"
        
        table.add_row("신호 상태", signal, "")
        
        self.console.print(table)
        self.console.print()
    
    def _print_recent_trades(self):
        """최근 거래 출력"""
        try:
            cursor = self.bot.db.cursor()
            cursor.execute("""
                SELECT timestamp, side, entry_price, exit_price, 
                       profit_percent, close_reason
                FROM trades
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            
            trades = cursor.fetchall()
            
            if trades:
                table = Table(title="📝 최근 거래", show_header=True, header_style="bold magenta")
                table.add_column("시간", style="cyan", width=20)
                table.add_column("방향", style="yellow", width=10)
                table.add_column("진입가", style="white", width=12)
                table.add_column("청산가", style="white", width=12)
                table.add_column("수익률", style="green", width=12)
                table.add_column("사유", style="blue", width=15)
                
                for trade in trades:
                    timestamp, side, entry, exit, profit, reason = trade
                    
                    # 수익률에 따른 색상
                    profit_color = "green" if profit > 0 else "red"
                    
                    # 시간 포맷
                    trade_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    time_str = trade_time.strftime('%m-%d %H:%M')
                    
                    table.add_row(
                        time_str,
                        side,
                        f"{entry:.4f}",
                        f"{exit:.4f}",
                        f"[{profit_color}]{profit:+.2f}%[/{profit_color}]",
                        reason
                    )
                
                self.console.print(table)
            
        except Exception as e:
            self.logger.error(f"최근 거래 조회 오류: {e}")
    
    def log_trade(self, trade_info: Dict):
        """거래 로그"""
        msg = f"""
        ================== 거래 체결 ==================
        시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        방향: {trade_info.get('side')}
        수량: {trade_info.get('quantity')}
        가격: {trade_info.get('price')}
        ===============================================
        """
        self.logger.info(msg)
        self.console.print(msg, style="bold green")
    
    def log_signal(self, signal_type: str, details: Dict):
        """신호 로그"""
        msg = f"""
        📡 신호 감지: {signal_type}
        혼합 각도: {details.get('mixed_angle', 0):.1f}°
        MACD 각도: {details.get('macd_angle', 0):.1f}°
        """
        self.logger.info(msg)
        
        if signal_type == "BULLISH":
            self.console.print(msg, style="bold green")
        elif signal_type == "BEARISH":
            self.console.print(msg, style="bold red")
    
    def log_error(self, error_msg: str):
        """에러 로그"""
        msg = f"❌ 오류: {error_msg}"
        self.logger.error(msg)
        self.console.print(msg, style="bold red")