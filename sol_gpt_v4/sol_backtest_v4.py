#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 SOL Futures 백테스트 시스템 v4.0
Quadruple EMA Touch 전략 백테스팅
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class QuadrupleEMABacktest:
    """Quadruple EMA Touch 백테스트"""
    
    def __init__(self, symbol: str = "SOL/USDT"):
        self.symbol = symbol
        self.exchange = None
        
        # 백테스트 설정
        self.initial_balance = 1000  # 초기 자본 (USDT)
        self.position_size = 0.2     # 포지션 크기 (20%)
        self.leverage = 10           # 레버리지
        self.fee_rate = 0.0005      # 수수료율 (0.05%)
        self.touch_threshold = 0.002 # 터치 임계값 (0.2%, 기존 0.35%)
        
        # 리스크 관리
        self.stop_loss = 0.20       # 손절 20% (백테스트 최적값)
        self.partial_close_roi = 999.0  # 부분청산 비활성화
        self.partial_close_ratio = 0.0  # 부분청산 안함
        
        # 트레일링 스톱 비활성화
        self.trailing_configs = {}  # 트레일링 스톱 사용 안함
        
        # 데이터 저장 경로
        self.data_dir = Path("backtest_data")
        self.data_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """거래소 초기화"""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        await self.exchange.load_markets()
        
    async def fetch_ohlcv_data(self, timeframe: str, days: int = 30) -> pd.DataFrame:
        """OHLCV 데이터 수집"""
        console.print(f"[yellow]Fetching {timeframe} data for {days} days...[/yellow]")
        
        # 시작 시간 계산
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_ohlcv = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Downloading {timeframe} data...", total=None)
            
            while True:
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(
                        self.symbol, 
                        timeframe, 
                        since=since,
                        limit=1500
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_ohlcv.extend(ohlcv)
                    
                    # 다음 배치 시작점
                    since = ohlcv[-1][0] + 1
                    
                    # 현재까지 도달했으면 종료
                    if since >= datetime.now().timestamp() * 1000:
                        break
                        
                    await asyncio.sleep(self.exchange.rateLimit / 1000)
                    
                except Exception as e:
                    console.print(f"[red]Error fetching data: {e}[/red]")
                    break
        
        # DataFrame 변환
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        console.print(f"[green]Downloaded {len(df)} candles for {timeframe}[/green]")
        return df
    
    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quadruple EMA 계산"""
        df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema21'] = df['open'].ewm(span=21, adjust=False).mean()
        df['ema21_high'] = df['high'].ewm(span=21, adjust=False).mean()
        df['ema21_low'] = df['low'].ewm(span=21, adjust=False).mean()
        return df
    
    def check_touch_signal(self, row: pd.Series, position: Dict = None) -> str:
        """터치 신호 확인 - 올바른 전략"""
        ema10 = row['ema10']
        ema21 = row['ema21']  # EMA21 시가 기준
        ema21_high = row['ema21_high']
        ema21_low = row['ema21_low']
        
        # 터치 확인
        high_diff = abs(ema10 - ema21_high) / ema21_high
        low_diff = abs(ema10 - ema21_low) / ema21_low
        
        # EMA 배열 상태 확인
        is_bullish = ema10 > ema21  # 정배열
        is_bearish = ema10 < ema21  # 역배열
        
        # 포지션이 있는 경우 반대 신호만 체크
        if position:
            if position['side'] == 'long' and is_bearish and low_diff <= self.touch_threshold:
                return 'SHORT'
            elif position['side'] == 'short' and is_bullish and high_diff <= self.touch_threshold:
                return 'LONG'
        else:
            # 포지션이 없는 경우 - EMA 배열에 따른 신호 체크
            if is_bullish and high_diff <= self.touch_threshold:
                # 정배열 상태에서 EMA21_high 터치 → LONG
                return 'LONG'
            elif is_bearish and low_diff <= self.touch_threshold:
                # 역배열 상태에서 EMA21_low 터치 → SHORT
                return 'SHORT'
        
        return None
    
    def calculate_roi(self, position: Dict, current_price: float) -> float:
        """ROI 계산"""
        if position['side'] == 'long':
            roi = ((current_price - position['entry_price']) / position['entry_price']) * 100 * self.leverage
        else:
            roi = ((position['entry_price'] - current_price) / position['entry_price']) * 100 * self.leverage
        return roi
    
    def get_trailing_sl(self, position: Dict, roi: float) -> float:
        """트레일링 스톱 계산"""
        best_sl = None
        
        for trigger_roi, trailing_pct in sorted(self.trailing_configs.items(), reverse=True):
            if roi >= trigger_roi:
                if trigger_roi == 10.0:
                    # 진입가로 이동
                    best_sl = position['entry_price']
                else:
                    # 트레일링 적용
                    if position['side'] == 'long':
                        best_sl = position['entry_price'] * (1 + (trigger_roi - trailing_pct) / self.leverage / 100)
                    else:
                        best_sl = position['entry_price'] * (1 - (trigger_roi - trailing_pct) / self.leverage / 100)
                break
        
        return best_sl
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """백테스트 실행"""
        # 초기화
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []
        
        # EMA 계산
        df = self.calculate_ema(df)
        
        # 백테스트 시작
        for idx, row in df.iterrows():
            current_price = row['close']
            
            # 자산 추적
            equity = balance
            if position:
                pnl = self.calculate_position_pnl(position, current_price)
                equity = balance + pnl
            equity_curve.append({'timestamp': idx, 'equity': equity})
            
            # 포지션이 있는 경우
            if position:
                roi = self.calculate_roi(position, current_price)
                
                # 손절 확인
                if roi <= -self.stop_loss * 100:
                    # 손절 실행
                    trade = self.close_position(position, current_price, 'STOP_LOSS')
                    balance += trade['pnl']
                    trades.append(trade)
                    position = None
                    continue
                
                # 트레일링 스톱 비활성화
                # trailing_sl = self.get_trailing_sl(position, roi)
                # if trailing_sl:
                #     if position['side'] == 'long' and current_price <= trailing_sl:
                #         trade = self.close_position(position, current_price, 'TRAILING_STOP')
                #         balance += trade['pnl']
                #         trades.append(trade)
                #         position = None
                #         continue
                #     elif position['side'] == 'short' and current_price >= trailing_sl:
                #         trade = self.close_position(position, current_price, 'TRAILING_STOP')
                #         balance += trade['pnl']
                #         trades.append(trade)
                #         position = None
                #         continue
                
                # 부분청산 비활성화 (백테스트 결과 기반)
                # if not position.get('partial_closed') and roi >= self.partial_close_roi * 100:
                #     # 50% 부분청산
                #     partial_size = position['size'] * self.partial_close_ratio
                #     partial_pnl = self.calculate_pnl(
                #         position['side'], 
                #         position['entry_price'], 
                #         current_price, 
                #         partial_size
                #     )
                #     balance += partial_pnl
                #     position['size'] *= (1 - self.partial_close_ratio)
                #     position['partial_closed'] = True
            
            # 신호 확인
            signal = self.check_touch_signal(row, position)
            
            if signal:
                # 기존 포지션 청산
                if position:
                    trade = self.close_position(position, current_price, f'{signal}_SIGNAL')
                    balance += trade['pnl']
                    trades.append(trade)
                
                # 새 포지션 진입
                position_value = balance * self.position_size
                position = {
                    'side': 'long' if signal == 'LONG' else 'short',
                    'entry_price': current_price,
                    'entry_time': idx,
                    'size': position_value,
                    'partial_closed': False
                }
        
        # 마지막 포지션 청산
        if position:
            trade = self.close_position(position, df.iloc[-1]['close'], 'END_OF_DATA')
            balance += trade['pnl']
            trades.append(trade)
        
        return {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_balance': balance,
            'df': df
        }
    
    def calculate_pnl(self, side: str, entry_price: float, exit_price: float, size: float) -> float:
        """손익 계산"""
        if side == 'long':
            pnl = size * (exit_price - entry_price) / entry_price * self.leverage
        else:
            pnl = size * (entry_price - exit_price) / entry_price * self.leverage
        
        # 수수료 차감
        fee = size * self.fee_rate * 2  # 진입 + 청산
        return pnl - fee
    
    def calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """포지션 미실현 손익"""
        return self.calculate_pnl(
            position['side'],
            position['entry_price'],
            current_price,
            position['size']
        )
    
    def close_position(self, position: Dict, exit_price: float, reason: str) -> Dict:
        """포지션 청산"""
        pnl = self.calculate_pnl(
            position['side'],
            position['entry_price'],
            exit_price,
            position['size']
        )
        
        roi = ((pnl / position['size']) * 100) if position['size'] > 0 else 0
        
        return {
            'side': position['side'],
            'entry_price': position['entry_price'],
            'entry_time': position['entry_time'],
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'size': position['size'],
            'pnl': pnl,
            'roi': roi,
            'reason': reason
        }
    
    def analyze_results(self, results: Dict, timeframe: str) -> Dict:
        """백테스트 결과 분석"""
        trades = results['trades']
        
        if not trades:
            return {
                'timeframe': timeframe,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'trades': []
            }
        
        # 거래 통계
        long_trades = [t for t in trades if t['side'] == 'long']
        short_trades = [t for t in trades if t['side'] == 'short']
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        # 손익 계산
        total_pnl = sum(t['pnl'] for t in trades)
        total_return = (results['final_balance'] - self.initial_balance) / self.initial_balance * 100
        
        # 평균 손익
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit Factor
        gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # 최대 낙폭
        equity_curve_df = pd.DataFrame(results['equity_curve'])
        if len(equity_curve_df) > 0:
            # timestamp를 index로 설정
            equity_curve_df.set_index('timestamp', inplace=True)
            equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
            equity_curve_df['drawdown'] = (equity_curve_df['equity'] - equity_curve_df['peak']) / equity_curve_df['peak'] * 100
            max_drawdown = equity_curve_df['drawdown'].min()
        else:
            max_drawdown = 0
        
        # trades 데이터의 Timestamp를 문자열로 변환
        trades_serializable = []
        for t in trades:
            trade_copy = t.copy()
            # entry_time과 exit_time을 문자열로 변환
            if 'entry_time' in trade_copy and hasattr(trade_copy['entry_time'], 'strftime'):
                trade_copy['entry_time'] = trade_copy['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            if 'exit_time' in trade_copy and hasattr(trade_copy['exit_time'], 'strftime'):
                trade_copy['exit_time'] = trade_copy['exit_time'].strftime('%Y-%m-%d %H:%M:%S')
            trades_serializable.append(trade_copy)
        
        return {
            'timeframe': timeframe,
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor if profit_factor != float('inf') else 999.99,
            'max_drawdown': max_drawdown,
            'final_balance': results['final_balance'],
            'trades': trades_serializable
        }
    
    def save_results(self, results: Dict, analysis: Dict, timeframe: str):
        """결과 저장"""
        # 결과 파일 저장
        result_file = self.data_dir / f"backtest_result_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # DataFrame의 timestamp를 문자열로 변환
        df_start = results['df'].index[0]
        df_end = results['df'].index[-1]
        
        # pandas Timestamp인 경우 문자열로 변환
        if hasattr(df_start, 'strftime'):
            start_date = df_start.strftime('%Y-%m-%d %H:%M:%S')
        else:
            start_date = str(df_start)
            
        if hasattr(df_end, 'strftime'):
            end_date = df_end.strftime('%Y-%m-%d %H:%M:%S')
        else:
            end_date = str(df_end)
        
        save_data = {
            'timeframe': timeframe,
            'symbol': self.symbol,
            'period': f"{len(results['df'])} candles",
            'start_date': start_date,
            'end_date': end_date,
            'settings': {
                'initial_balance': self.initial_balance,
                'position_size': self.position_size,
                'leverage': self.leverage,
                'touch_threshold': self.touch_threshold,
                'stop_loss': self.stop_loss
            },
            'analysis': analysis,
            'trades': [
                {
                    'side': t['side'],
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'entry_time': t['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time']),
                    'exit_time': t['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(t['exit_time'], 'strftime') else str(t['exit_time']),
                    'pnl': t['pnl'],
                    'roi': t['roi'],
                    'reason': t['reason']
                }
                for t in analysis['trades']
            ]
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        return result_file
    
    def print_results(self, analysis: Dict):
        """결과 출력"""
        table = Table(title=f"Backtest Results - {analysis['timeframe']}")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Trades", str(analysis['total_trades']))
        table.add_row("Long Trades", str(analysis['long_trades']))
        table.add_row("Short Trades", str(analysis['short_trades']))
        table.add_row("Winning Trades", f"{analysis['winning_trades']} ({analysis['win_rate']:.1f}%)")
        table.add_row("Losing Trades", str(analysis['losing_trades']))
        table.add_row("Total P&L", f"${analysis['total_pnl']:.2f}")
        table.add_row("Total Return", f"{analysis['total_return']:.2f}%")
        table.add_row("Avg Win", f"${analysis['avg_win']:.2f}")
        table.add_row("Avg Loss", f"${analysis['avg_loss']:.2f}")
        table.add_row("Profit Factor", f"{analysis['profit_factor']:.2f}")
        table.add_row("Max Drawdown", f"{analysis['max_drawdown']:.2f}%")
        
        console.print(table)
    
    async def run(self, timeframes: List[str], days: int = 30):
        """백테스트 실행"""
        await self.initialize()
        
        all_results = {}
        
        for timeframe in timeframes:
            console.print(f"\n[bold blue]Running backtest for {timeframe}...[/bold blue]")
            
            # 데이터 수집
            df = await self.fetch_ohlcv_data(timeframe, days)
            
            # JSON 파일로 저장
            data_file = self.data_dir / f"ohlcv_{timeframe}_{datetime.now().strftime('%Y%m%d')}.json"
            # DataFrame을 복사하고 인덱스를 리셋하여 timestamp를 컬럼으로 만듦
            df_copy = df.reset_index()
            # timestamp 컬럼을 문자열로 변환
            df_copy['timestamp'] = df_copy['timestamp'].astype(str)
            # JSON으로 저장
            df_copy.to_json(data_file, orient='records', date_format='iso')
            console.print(f"[green]Data saved to {data_file}[/green]")
            
            # 백테스트 실행
            results = self.run_backtest(df)
            
            # 결과 분석
            analysis = self.analyze_results(results, timeframe)
            
            # 결과 저장
            result_file = self.save_results(results, analysis, timeframe)
            console.print(f"[green]Results saved to {result_file}[/green]")
            
            # 결과 출력
            self.print_results(analysis)
            
            all_results[timeframe] = analysis
        
        await self.exchange.close()
        
        return all_results


async def main():
    parser = argparse.ArgumentParser(description='Quadruple EMA Touch Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='SOL/USDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--timeframes', nargs='+', default=['5m', '15m', '30m', '1h'], 
                       help='Timeframes to test')
    
    args = parser.parse_args()
    
    console.print(f"[bold green]🔍 Quadruple EMA Touch Strategy Backtest[/bold green]")
    console.print(f"Symbol: {args.symbol}")
    console.print(f"Period: {args.days} days")
    console.print(f"Timeframes: {', '.join(args.timeframes)}")
    console.print()
    
    backtest = QuadrupleEMABacktest(symbol=args.symbol)
    
    try:
        results = await backtest.run(args.timeframes, args.days)
        
        # 종합 결과
        console.print("\n[bold yellow]📊 Summary of All Timeframes:[/bold yellow]")
        summary_table = Table(title="Comparison Across Timeframes")
        
        summary_table.add_column("Timeframe", style="cyan")
        summary_table.add_column("Trades", style="white")
        summary_table.add_column("Win Rate", style="green")
        summary_table.add_column("Total Return", style="magenta")
        summary_table.add_column("Max DD", style="red")
        summary_table.add_column("Profit Factor", style="yellow")
        
        for tf, analysis in results.items():
            summary_table.add_row(
                tf,
                str(analysis['total_trades']),
                f"{analysis['win_rate']:.1f}%",
                f"{analysis['total_return']:.2f}%",
                f"{analysis['max_drawdown']:.2f}%",
                f"{analysis['profit_factor']:.2f}"
            )
        
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())