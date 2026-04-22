"""
SOL V1 콘솔 실시간 모니터링 (독립 실행)

봇이 실행 중일 때 이 스크립트를 별도 창에서 실행하면
state/sol_state.json + sol_trading_bot.db 읽어서 실시간 표시.

사용법:
  python sol_monitor.py
  SOL_monitor.bat

갱신 주기: 2초
"""
import asyncio
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.progress_bar import ProgressBar
from rich.box import ROUNDED, DOUBLE

STATE_PATH = 'state/sol_state.json'
DB_PATH = 'sol_trading_bot.db'
REFRESH_SEC = 2.0

# SOL V1 constants
COMPOUND_PCT = 0.125
ALLOC_V12 = 0.75
ALLOC_MASS = 0.25
SKIP_THRESHOLD = 4
SKIP_COUNT = 2
LEVERAGE = 2   # 🧪 TEST MODE (원래 10x)


def load_state():
    """state 파일 로드"""
    if not Path(STATE_PATH).exists():
        return None
    try:
        with open(STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def load_recent_trades(n=15):
    """최근 거래 N개 조회"""
    if not Path(DB_PATH).exists():
        return []
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2.0)
        cursor = conn.execute(
            "SELECT timestamp, entry_mode, direction, entry_price, exit_price, exit_type, pnl, roi_pct "
            "FROM trades ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def load_todays_entries():
    """오늘 진입 건수"""
    if not Path(DB_PATH).exists():
        return 0
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2.0)
        today = datetime.now().strftime('%Y-%m-%d')
        cursor = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE timestamp LIKE ?", (f"{today}%",)
        )
        cnt = cursor.fetchone()[0]
        conn.close()
        return cnt
    except Exception:
        return 0


def make_header(state):
    """상단 헤더"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alive = (time.time() - Path(STATE_PATH).stat().st_mtime) < 180 if Path(STATE_PATH).exists() else False
    status = "[bold green]● LIVE[/bold green]" if alive else "[bold red]● OFFLINE[/bold red]"
    last_update = datetime.fromtimestamp(Path(STATE_PATH).stat().st_mtime).strftime('%H:%M:%S') if Path(STATE_PATH).exists() else '?'

    header = Table.grid(expand=True)
    header.add_column(justify="left")
    header.add_column(justify="center")
    header.add_column(justify="right")
    header.add_row(
        f"[bold cyan]🪙 SOL V1 Monitor[/bold cyan]",
        f"[dim]V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% Compound[/dim]",
        f"{status} | Last State: {last_update} | {now}"
    )
    return Panel(header, box=ROUNDED, style="blue", padding=(0, 1))


def make_balance_card(state):
    """잔액/MDD 카드"""
    peak = state.get('peak_capital', 0) if state else 0
    mdd = state.get('max_drawdown', 0) if state else 0
    day_start = state.get('day_start_balance', 0) if state else 0
    # Balance는 봇만 알고 있어서 state에서 유추
    balance_est = peak * (1 - mdd) if peak > 0 else 0

    t = Table.grid(expand=True)
    t.add_column(justify="left", ratio=1)
    t.add_column(justify="right", ratio=1)

    t.add_row("[dim]Peak Capital[/dim]", f"[bold white]${peak:,.2f}[/bold white]")
    t.add_row("[dim]Day Start[/dim]", f"[white]${day_start:,.2f}[/white]")

    mdd_color = "green" if mdd < 0.15 else ("yellow" if mdd < 0.30 else "red")
    mdd_bar = "█" * int(mdd * 50) + "░" * max(0, 50 - int(mdd * 50))
    t.add_row("[dim]MDD[/dim]", f"[bold {mdd_color}]{mdd*100:.2f}%[/bold {mdd_color}]")
    t.add_row("", f"[{mdd_color}]{mdd_bar[:20]}[/{mdd_color}]")

    return Panel(t, title="💰 Balance & Risk", box=ROUNDED, border_style="green")


def make_trades_card(state):
    """거래 통계 카드"""
    if not state:
        return Panel("데이터 없음", title="📊 Trades", box=ROUNDED)

    wins = state.get('win_count', 0)
    losses = state.get('loss_count', 0)
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0

    gp = state.get('gross_profit', 0)
    gl = state.get('gross_loss', 0)
    pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

    sl = state.get('sl_count', 0)
    tsl = state.get('tsl_count', 0)
    rev = state.get('rev_count', 0)
    fc = state.get('fc_count', 0)

    wr_color = "green" if wr >= 30 else ("yellow" if wr >= 20 else "red")
    pf_color = "green" if pf >= 1.5 else ("yellow" if pf >= 1.2 else "red")

    t = Table.grid(expand=True)
    t.add_column(justify="left", ratio=1)
    t.add_column(justify="right", ratio=1)

    t.add_row("[dim]Total Trades[/dim]", f"[bold white]{total}[/bold white]")
    t.add_row("[dim]WR[/dim]", f"[bold {wr_color}]{wr:.1f}%[/bold {wr_color}] ({wins}W / {losses}L)")
    t.add_row("[dim]PF[/dim]", f"[bold {pf_color}]{pf:.2f}[/bold {pf_color}]")
    t.add_row("[dim]GP / GL[/dim]", f"[green]${gp:,.0f}[/green] / [red]${gl:,.0f}[/red]")
    t.add_row("[dim]Exits[/dim]", f"SL [red]{sl}[/red] TSL [green]{tsl}[/green] REV [yellow]{rev}[/yellow] FC [magenta]{fc}[/magenta]")

    return Panel(t, title="📊 Trades", box=ROUNDED, border_style="cyan")


def make_skip_card(state):
    """Skip2@4loss + 알림 카드"""
    if not state:
        return Panel("데이터 없음", title="⚠ Skip2@4loss", box=ROUNDED)

    consec = state.get('consec_losses', 0)
    skip_rem = state.get('skip_remaining', 0)

    t = Table.grid(expand=True)
    t.add_column(justify="center")

    # Consec losses bar
    consec_color = "green" if consec == 0 else ("yellow" if consec < 3 else "red")
    consec_bar = "█" * consec + "░" * (SKIP_THRESHOLD - consec)
    t.add_row(f"[dim]연패 카운터[/dim]")
    t.add_row(f"[bold {consec_color}]{consec} / {SKIP_THRESHOLD}[/bold {consec_color}]")
    t.add_row(f"[{consec_color}]{consec_bar}[/{consec_color}]")

    # Skip status
    if skip_rem > 0:
        t.add_row(f"[bold red]⏸ SKIPPING[/bold red]")
        t.add_row(f"[red]다음 {skip_rem}개 거래 스킵[/red]")
    else:
        t.add_row(f"[green]✓ 진입 가능[/green]")
        t.add_row("")

    return Panel(t, title="⚠ Skip2@4loss", box=ROUNDED, border_style="magenta")


def make_position_card(state):
    """현재 포지션 카드"""
    if not state:
        return Panel("데이터 없음", title="📍 Position", box=ROUNDED)

    pos = state.get('position', {})
    direction = pos.get('direction', 0)

    if direction == 0:
        content = Align.center(Text("\n포지션 없음\n\n[신호 대기 중...]\n", style="dim"), vertical="middle")
        return Panel(content, title="📍 Position", box=ROUNDED, border_style="dim")

    dir_str = "🟢 LONG" if direction == 1 else "🔴 SHORT"
    mode = "V12" if pos.get('entry_mode') == 1 else "MASS"
    entry = pos.get('entry_price', 0)
    size = pos.get('position_size', 0)
    margin = pos.get('margin_used', 0)
    sl = pos.get('sl_price', 0)
    tsl_active = pos.get('tsl_active', False)
    peak_roi = pos.get('peak_roi', 0)
    leg_count = pos.get('leg_count', 1)

    t = Table.grid(expand=True)
    t.add_column(justify="left", ratio=1)
    t.add_column(justify="right", ratio=1)

    dir_color = "green" if direction == 1 else "red"
    t.add_row("[dim]방향[/dim]", f"[bold {dir_color}]{dir_str}[/bold {dir_color}]")
    t.add_row("[dim]Mode[/dim]", f"[bold yellow]{mode}[/bold yellow]")
    t.add_row("[dim]Entry (VWAP)[/dim]", f"[white]${entry:.3f}[/white]")
    t.add_row("[dim]Size[/dim]", f"[white]${size:,.0f}[/white]")
    t.add_row("[dim]Margin[/dim]", f"[white]${margin:,.0f}[/white]")
    t.add_row("[dim]SL[/dim]", f"[red]${sl:.3f}[/red]")
    t.add_row("[dim]Peak ROI[/dim]", f"[green]{peak_roi:+.2f}%[/green]")
    t.add_row("[dim]TSL[/dim]", "[green]✓ Active[/green]" if tsl_active else "[dim]X[/dim]")
    t.add_row("[dim]Leg[/dim]", f"[cyan]{leg_count}/3[/cyan]")

    border = "green" if direction == 1 else "red"
    return Panel(t, title="📍 Active Position", box=DOUBLE, border_style=border)


def make_trades_table():
    """최근 거래 테이블"""
    trades = load_recent_trades(12)

    t = Table(expand=True, show_header=True, box=ROUNDED, border_style="cyan")
    t.add_column("Time", style="dim", width=8)
    t.add_column("Mode", width=5)
    t.add_column("Dir", width=6)
    t.add_column("Entry", justify="right", width=10)
    t.add_column("Exit", justify="right", width=10)
    t.add_column("Type", width=6)
    t.add_column("PnL", justify="right", width=10)
    t.add_column("ROI", justify="right", width=8)

    if not trades:
        t.add_row("[dim]— No trades yet —[/dim]", "", "", "", "", "", "", "")
    else:
        for ts, mode, direction, entry, exit_px, exit_type, pnl, roi in trades:
            time_short = ts[11:19] if ts else "?"
            dir_str = "[green]LONG[/green]" if direction == "LONG" else "[red]SHORT[/red]"
            mode_c = "[yellow]V12[/yellow]" if mode == "V12" else "[magenta]MASS[/magenta]"
            pnl_color = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else ""
            exit_color = {"SL": "red", "TSL": "green", "REV": "yellow", "FC": "magenta"}.get(exit_type, "white")

            t.add_row(
                time_short, mode_c, dir_str,
                f"${entry:.3f}", f"${exit_px:.3f}",
                f"[{exit_color}]{exit_type}[/{exit_color}]",
                f"[{pnl_color}]{pnl_sign}${pnl:.2f}[/{pnl_color}]",
                f"[{pnl_color}]{roi:+.1f}%[/{pnl_color}]",
            )

    return Panel(t, title="📋 최근 거래 (12개)", box=ROUNDED, border_style="blue")


def make_footer(state):
    """하단 정보"""
    today_entries = load_todays_entries()
    last_update = datetime.fromtimestamp(Path(STATE_PATH).stat().st_mtime).strftime('%H:%M:%S') if Path(STATE_PATH).exists() else '?'

    text = Text()
    text.append("💡 ", style="yellow")
    text.append(f"오늘 진입: {today_entries}건 | ", style="white")
    text.append(f"봇 상태 업데이트: {last_update} | ", style="dim")
    text.append("Refresh 2s | Ctrl+C 종료", style="dim")

    return Panel(Align.center(text), box=ROUNDED, style="dim")


def make_layout():
    """전체 레이아웃 구성"""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top_row", size=12),
        Layout(name="mid_row", size=16),
        Layout(name="trades", size=18),
        Layout(name="footer", size=3),
    )

    layout["top_row"].split_row(
        Layout(name="balance"),
        Layout(name="trades_stat"),
        Layout(name="skip"),
    )

    layout["mid_row"].split_row(
        Layout(name="position", ratio=1),
        Layout(name="placeholder", ratio=1),
    )

    return layout


def update_layout(layout, state):
    """레이아웃 내용 업데이트"""
    layout["header"].update(make_header(state))
    layout["balance"].update(make_balance_card(state))
    layout["trades_stat"].update(make_trades_card(state))
    layout["skip"].update(make_skip_card(state))
    layout["position"].update(make_position_card(state))

    # Strategy info panel (placeholder 자리)
    info_table = Table.grid(expand=True)
    info_table.add_column(justify="left", ratio=1)
    info_table.add_column(justify="right", ratio=1)
    info_table.add_row("[bold cyan]Strategy A (V12.1)[/bold cyan]", f"[yellow]{ALLOC_V12*100:.0f}%[/yellow]")
    info_table.add_row("[dim]  EMA9/SMA400 15m + 5중 필터[/dim]", "")
    info_table.add_row("[dim]  Pyramiding 3-Leg + Conf Sigmoid[/dim]", "")
    info_table.add_row("", "")
    info_table.add_row("[bold magenta]Strategy B (Mass)[/bold magenta]", f"[yellow]{ALLOC_MASS*100:.0f}%[/yellow]")
    info_table.add_row("[dim]  Mass Index Reversal[/dim]", "")
    info_table.add_row("[dim]  Mass>27 → <26.5 trigger[/dim]", "")
    info_table.add_row("", "")
    info_table.add_row("[bold]Compound[/bold]", f"[green]{COMPOUND_PCT*100:.1f}%[/green] of balance")
    info_table.add_row("[bold]Leverage[/bold]", f"[green]{LEVERAGE}x[/green] Isolated")

    layout["placeholder"].update(Panel(info_table, title="⚙ Strategy Config", box=ROUNDED, border_style="yellow"))

    layout["trades"].update(make_trades_table())
    layout["footer"].update(make_footer(state))


def main():
    console = Console()

    # 초기 체크
    if not Path(STATE_PATH).exists():
        console.print("[yellow]⚠ state 파일 없음. 봇을 먼저 실행하세요.[/yellow]")
        console.print(f"  경로: {STATE_PATH}")
        console.print("  봇 시작: SOL_start.bat")
        time.sleep(5)
        return

    console.clear()
    console.print("[cyan]🪙 SOL V1 Monitor 시작... (Ctrl+C 종료)[/cyan]")
    time.sleep(1)

    layout = make_layout()

    try:
        with Live(layout, console=console, refresh_per_second=1, screen=True) as live:
            while True:
                state = load_state()
                update_layout(layout, state)
                time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        console.print("\n[yellow]모니터링 종료[/yellow]")


if __name__ == "__main__":
    main()
