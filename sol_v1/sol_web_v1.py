"""
SOL V1 웹 대시보드 - 실시간 모니터링
FastAPI + 자동 새로고침 (5초)

http://localhost:8081 (ETH_V8는 8080 사용중)
비밀번호: .env의 WEB_PASSWORD
"""
import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

logger = logging.getLogger('sol_web')

DB_PATH = 'sol_trading_bot.db'
COOKIE_NAME = 'sol_session'
COOKIE_MAX_AGE = 7 * 24 * 3600


HTML_LOGIN = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Login</title>
<style>
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh}
.box{background:#1a1f2e;padding:40px;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,0.5);width:320px}
h1{margin:0 0 20px;color:#4ade80;font-size:24px;text-align:center}
input{width:100%;padding:12px;margin:8px 0;background:#0d1117;border:1px solid #30363d;color:#e0e0e0;border-radius:6px;box-sizing:border-box;font-size:14px}
button{width:100%;padding:12px;background:#22c55e;color:#fff;border:0;border-radius:6px;cursor:pointer;font-size:14px;font-weight:bold;margin-top:10px}
button:hover{background:#16a34a}
.err{color:#ef4444;font-size:12px;margin-top:8px;text-align:center}
</style></head>
<body><form method="POST" action="/login"><div class="box">
<h1>🪙 SOL V1 Bot</h1>
<input type="password" name="password" placeholder="비밀번호" autofocus required>
<button type="submit">로그인</button>
%%ERR%%
</div></form></body></html>"""


HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;padding:20px}
h1{color:#4ade80;margin:0 0 20px;font-size:24px}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin-bottom:20px}
.card{background:#1a1f2e;padding:16px;border-radius:10px;border:1px solid #2d3748}
.card h3{margin:0 0 10px;color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:0.5px}
.v{font-size:22px;font-weight:bold}
.sub{font-size:11px;color:#64748b;margin-top:4px}
.pos{background:#0f172a;padding:15px;border-radius:8px;margin:15px 0;border-left:4px solid #eab308}
.pos-long{border-left-color:#22c55e}
.pos-short{border-left-color:#ef4444}
.pos-none{border-left-color:#64748b}
.win{color:#22c55e}
.lose{color:#ef4444}
.warn{color:#eab308}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}
th{background:#1e293b;padding:8px;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #2d3748}
td{padding:8px;border-bottom:1px solid #1f2937}
tr:hover{background:rgba(255,255,255,0.02)}
.logo{display:inline-block;margin-right:10px}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:#64748b}
.active{background:#22c55e;box-shadow:0 0 8px #22c55e}
.alert{background:#7c2d12;padding:10px;border-radius:6px;margin:10px 0;border-left:4px solid #ef4444}
.bar{background:#0f172a;height:6px;border-radius:3px;margin-top:5px;overflow:hidden}
.bar-fill{height:100%;background:linear-gradient(90deg,#22c55e,#16a34a)}
.bar-red{background:linear-gradient(90deg,#ef4444,#b91c1c)}
.logout{float:right;color:#64748b;text-decoration:none;font-size:12px}
</style></head>
<body>
<h1>🪙 SOL V1 Dashboard <a href="/logout" class="logout">[로그아웃]</a></h1>
<div style="color:#64748b;font-size:11px;margin-bottom:20px">
  V12:Mass 75:25 Mutex + Skip2@4loss + 12.5% Compound |
  <span class="status-dot %%DOT_CLASS%%"></span>%%STATUS_TEXT%% |
  Last update: %%LAST_UPDATE%% | Auto-refresh 5s
</div>

<div class="g">
  <div class="card"><h3>💰 Balance (USDT)</h3>
    <div class="v">$%%BALANCE%%</div>
    <div class="sub">Available: $%%AVAILABLE%% | Peak: $%%PEAK%%</div>
  </div>
  <div class="card"><h3>📉 Max Drawdown</h3>
    <div class="v %%MDD_CLASS%%">%%MDD%%%</div>
    <div class="bar"><div class="bar-fill bar-red" style="width:%%MDD_PCT_WIDTH%%%"></div></div>
  </div>
  <div class="card"><h3>🎯 SOL Price</h3>
    <div class="v">$%%PRICE%%</div>
    <div class="sub">%%REGIME_EMOJI%% BTC Regime: %%REGIME_STR%%</div>
  </div>
  <div class="card"><h3>📊 Trades</h3>
    <div class="v">%%TOTAL_TRADES%%</div>
    <div class="sub">W %%WINS%% / L %%LOSSES%% | WR %%WR%%%</div>
  </div>
</div>

<div class="g">
  <div class="card"><h3>PF (Profit Factor)</h3>
    <div class="v %%PF_CLASS%%">%%PF%%</div>
    <div class="sub">GP $%%GROSS_PROFIT%% / GL $%%GROSS_LOSS%%</div>
  </div>
  <div class="card"><h3>연패 / Skip 상태</h3>
    <div class="v %%CONSEC_CLASS%%">%%CONSEC%%/4 | Skip %%SKIP%%</div>
    <div class="sub">%%SKIP_NOTE%%</div>
  </div>
  <div class="card"><h3>Exit Types</h3>
    <div class="v">SL %%SL%% · TSL %%TSL%% · REV %%REV%%</div>
    <div class="sub">FC %%FC%%</div>
  </div>
  <div class="card"><h3>📅 일일</h3>
    <div class="v">%%DAILY_PNL_FMT%%</div>
    <div class="sub">Start $%%DAY_START%% | PnL %%DAILY_PCT%%%</div>
  </div>
</div>

%%POSITION_BLOCK%%

<div class="card" style="margin-top:15px">
  <h3>📡 현재 지표 (15m)</h3>
  <table>
    <tr><th>EMA(9)</th><th>SMA(400)</th><th>ADX</th><th>RSI</th><th>LR Slope</th><th>Mass Index</th><th>ATR14</th><th>ATR50</th></tr>
    <tr>
      <td>$%%EMA9%%</td>
      <td>$%%SMA400%%</td>
      <td class="%%ADX_CLASS%%">%%ADX%%</td>
      <td class="%%RSI_CLASS%%">%%RSI%%</td>
      <td>%%LR%%</td>
      <td class="%%MASS_CLASS%%">%%MASS%%</td>
      <td>%%ATR14%%</td>
      <td>%%ATR50%%</td>
    </tr>
  </table>
</div>

<div class="card" style="margin-top:15px">
  <h3>📋 최근 거래 (20개)</h3>
  <table>
    <tr><th>시간</th><th>Mode</th><th>Dir</th><th>진입</th><th>청산</th><th>Exit</th><th>PnL</th><th>ROI</th></tr>
    %%TRADES_ROWS%%
  </table>
</div>
</body></html>"""


class WebDashboard:
    def __init__(self, bot, password: str, secret_key: str = None, port: int = 8081):
        self.bot = bot
        self.password = password
        self.port = port
        self.secret_key = secret_key or hashlib.sha256(os.urandom(32)).hexdigest()
        self.app = FastAPI(docs_url=None, redoc_url=None)
        self._setup_routes()
        self._server_task: Optional[asyncio.Task] = None

    def _make_token(self) -> str:
        ts = str(int(time.time()))
        sig = hmac.new(self.secret_key.encode(), ts.encode(), hashlib.sha256).hexdigest()[:16]
        return f"{ts}:{sig}"

    def _verify_token(self, token: str) -> bool:
        if not token or ':' not in token:
            return False
        try:
            ts, sig = token.split(':', 1)
            expected = hmac.new(self.secret_key.encode(), ts.encode(), hashlib.sha256).hexdigest()[:16]
            if not hmac.compare_digest(sig, expected):
                return False
            if time.time() - int(ts) > COOKIE_MAX_AGE:
                return False
            return True
        except Exception:
            return False

    def _is_auth(self, request: Request) -> bool:
        if not self.password:
            return True
        return self._verify_token(request.cookies.get(COOKIE_NAME, ''))

    async def _fetch_recent_trades(self, n: int = 20) -> list:
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    "SELECT timestamp, entry_mode, direction, entry_price, exit_price, exit_type, pnl, roi_pct "
                    "FROM trades ORDER BY id DESC LIMIT ?", (n,))
                rows = await cursor.fetchall()
                return [
                    {'time': r[0][11:19], 'mode': r[1] or '?', 'dir': r[2],
                     'entry': r[3], 'exit': r[4], 'exit_type': r[5],
                     'pnl': r[6], 'roi': r[7]}
                    for r in rows
                ]
        except Exception:
            return []

    def _render_dashboard(self, trades: list) -> str:
        bot = self.bot
        core = bot.core
        ex = bot.executor
        data = bot.data

        # WebSocket price or last bar close
        price = data.current_price if data and data.current_price > 0 else 0.0
        bar_idx = data.get_latest_index() if data else 0
        ind = data.get_indicators_at(bar_idx) if data else {}

        # BTC regime
        bar_ts = data.df_sol.index[-1].timestamp() if data and data.df_sol is not None else time.time()
        regime_bull = data.get_btc_regime_bull(bar_ts) if data else False
        regime_str = '🟢 BULL (x1.5)' if regime_bull else '🔴 BEAR (x0.5)'
        regime_emoji = '🚀' if regime_bull else '🐻'

        # Status
        ready = ind.get('slow_ma') is not None and ind.get('slow_ma') == ind.get('slow_ma')  # not nan
        import math as _m
        sma_val = ind.get('slow_ma', float('nan'))
        if isinstance(sma_val, float) and _m.isnan(sma_val):
            ready = False
        dot_class = 'active' if ready else ''
        status_text = '진입 대기 중' if ready else '데이터 축적 중 (SMA400 대기)'
        if core.has_position:
            status_text = '포지션 활성'
            dot_class = 'active'

        # MDD
        mdd_class = 'lose' if core.max_drawdown > 0.30 else ('warn' if core.max_drawdown > 0.15 else '')

        # PF
        pf = core.profit_factor if not _m.isinf(core.profit_factor) else 999
        pf_class = 'win' if pf >= 1.5 else ('warn' if pf >= 1.0 else 'lose')

        # ADX class
        adx_val = ind.get('adx', 0) or 0
        adx_class = 'win' if adx_val >= 22 else ''
        rsi_val = ind.get('rsi', 0) or 0
        rsi_class = 'win' if 30 <= rsi_val <= 65 else 'warn'
        mass_val = ind.get('mass', 0) or 0
        mass_class = 'win' if mass_val > 27 else ''

        # Daily
        daily_pnl = ex.balance - core.day_start_balance if core.day_start_balance > 0 else 0
        daily_pct = (ex.balance / core.day_start_balance - 1) * 100 if core.day_start_balance > 0 else 0
        daily_pnl_fmt = f"${daily_pnl:+,.2f}"
        daily_color = 'win' if daily_pnl > 0 else ('lose' if daily_pnl < 0 else '')

        # Consec / Skip
        consec_class = 'lose' if core.consec_losses >= 3 else ('warn' if core.consec_losses >= 2 else '')
        skip_note = f"⏸ 다음 {core.skip_remaining}개 거래 스킵" if core.skip_remaining > 0 else "정상"

        # MDD pct width
        mdd_pct_width = min(core.max_drawdown * 100, 100)

        # Position block
        if core.has_position:
            p = core.position
            dir_str = 'LONG' if p.direction == 1 else 'SHORT'
            dir_class = 'pos-long' if p.direction == 1 else 'pos-short'
            mode_str = 'V12' if p.entry_mode == 1 else 'MASS'
            unrealized_pnl = 0
            if price > 0:
                if p.direction == 1:
                    unrealized_pnl = (price - p.entry_price) / p.entry_price * p.position_size
                else:
                    unrealized_pnl = (p.entry_price - price) / p.entry_price * p.position_size
            pnl_class = 'win' if unrealized_pnl >= 0 else 'lose'
            position_block = f"""
<div class="pos {dir_class}">
  <h3 style="margin:0 0 10px;color:#eab308">📍 현재 포지션 [{mode_str}]</h3>
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:15px">
    <div><div class="sub">방향</div><div class="v">{dir_str}</div></div>
    <div><div class="sub">진입가 (VWAP)</div><div class="v">${p.entry_price:.3f}</div></div>
    <div><div class="sub">현재가</div><div class="v">${price:.3f}</div></div>
    <div><div class="sub">Size</div><div class="v">${p.position_size:,.0f}</div></div>
    <div><div class="sub">Margin</div><div class="v">${p.margin_used:,.0f}</div></div>
    <div><div class="sub">SL</div><div class="v">${p.sl_price:.3f}</div></div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-top:12px">
    <div><div class="sub">Unrealized PnL</div><div class="v {pnl_class}">${unrealized_pnl:+,.2f}</div></div>
    <div><div class="sub">TSL Active</div><div class="v">{'✅' if p.tsl_active else '❌'}</div></div>
    <div><div class="sub">Peak ROI</div><div class="v">{p.peak_roi:+.2f}%</div></div>
    <div><div class="sub">Leg Count</div><div class="v">{p.leg_count}/3</div></div>
  </div>
</div>"""
        else:
            position_block = """<div class="pos pos-none"><h3 style="margin:0;color:#64748b">포지션 없음 - 신호 대기 중</h3></div>"""

        # Trades rows
        trades_rows = ""
        for t in trades[:20]:
            dir_char = '🟢 L' if t['dir'] == 'LONG' else '🔴 S'
            pnl_c = 'win' if t['pnl'] > 0 else 'lose'
            trades_rows += (
                f"<tr><td>{t['time']}</td><td>{t['mode']}</td><td>{dir_char}</td>"
                f"<td>${t['entry']:.3f}</td><td>${t['exit']:.3f}</td><td>{t['exit_type']}</td>"
                f"<td class='{pnl_c}'>${t['pnl']:+,.2f}</td><td class='{pnl_c}'>{t['roi']:+.2f}%</td></tr>"
            )
        if not trades_rows:
            trades_rows = "<tr><td colspan='8' style='text-align:center;color:#64748b'>거래 없음</td></tr>"

        # Format values
        def fmt(v, d=2):
            return f"{v:,.{d}f}" if v is not None else "0.00"

        def safe_num(v, d=3):
            if v is None: return "0.000"
            try:
                if math.isnan(v): return "nan"
            except (TypeError, ValueError): pass
            return f"{v:.{d}f}"

        # %%TOKEN%% 기반 replace 매핑
        repl = {
            '%%DOT_CLASS%%': dot_class,
            '%%STATUS_TEXT%%': status_text,
            '%%LAST_UPDATE%%': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '%%BALANCE%%': fmt(ex.balance),
            '%%AVAILABLE%%': fmt(ex.available_balance),
            '%%PEAK%%': fmt(core.peak_capital),
            '%%MDD%%': f"{core.max_drawdown*100:.2f}",
            '%%MDD_CLASS%%': mdd_class,
            '%%MDD_PCT_WIDTH%%': f"{mdd_pct_width:.1f}",
            '%%PRICE%%': f"{price:.3f}" if price else "0.000",
            '%%REGIME_EMOJI%%': regime_emoji,
            '%%REGIME_STR%%': regime_str,
            '%%TOTAL_TRADES%%': str(core.total_trades),
            '%%WINS%%': str(core.win_count),
            '%%LOSSES%%': str(core.loss_count),
            '%%WR%%': f"{core.win_rate:.1f}",
            '%%PF%%': f"{pf:.2f}" if pf < 999 else "∞",
            '%%PF_CLASS%%': pf_class,
            '%%GROSS_PROFIT%%': f"{core.gross_profit:,.0f}",
            '%%GROSS_LOSS%%': f"{core.gross_loss:,.0f}",
            '%%CONSEC%%': str(core.consec_losses),
            '%%SKIP%%': str(core.skip_remaining),
            '%%CONSEC_CLASS%%': consec_class,
            '%%SKIP_NOTE%%': skip_note,
            '%%SL%%': str(core.sl_count),
            '%%TSL%%': str(core.tsl_count),
            '%%REV%%': str(core.rev_count),
            '%%FC%%': str(core.fc_count),
            '%%DAILY_PNL_FMT%%': daily_pnl_fmt,
            '%%DAY_START%%': f"{core.day_start_balance:,.2f}",
            '%%DAILY_PCT%%': f"{daily_pct:+.2f}",
            '%%POSITION_BLOCK%%': position_block,
            '%%EMA9%%': safe_num(ind.get('fast_ma', 0), 3),
            '%%SMA400%%': safe_num(ind.get('slow_ma', 0), 3),
            '%%ADX%%': f"{adx_val:.1f}",
            '%%ADX_CLASS%%': adx_class,
            '%%RSI%%': f"{rsi_val:.1f}",
            '%%RSI_CLASS%%': rsi_class,
            '%%LR%%': f"{(ind.get('lr_slope', 0) or 0):+.4f}",
            '%%MASS%%': f"{mass_val:.2f}",
            '%%MASS_CLASS%%': mass_class,
            '%%ATR14%%': safe_num(ind.get('atr14', 0), 4),
            '%%ATR50%%': safe_num(ind.get('atr50', 0), 4),
            '%%TRADES_ROWS%%': trades_rows,
        }

        html = HTML_DASHBOARD
        for k, v in repl.items():
            html = html.replace(k, str(v))
        return html

    def _setup_routes(self):
        @self.app.get('/login', response_class=HTMLResponse)
        async def login_page():
            return HTML_LOGIN.replace('%%ERR%%', '')

        @self.app.post('/login')
        async def login(password: str = Form(...)):
            if password == self.password:
                resp = RedirectResponse('/', status_code=303)
                resp.set_cookie(COOKIE_NAME, self._make_token(), max_age=COOKIE_MAX_AGE, httponly=True, samesite='lax')
                return resp
            return HTMLResponse(HTML_LOGIN.replace('%%ERR%%', '<div class="err">❌ 비밀번호 오류</div>'))

        @self.app.get('/logout')
        async def logout():
            resp = RedirectResponse('/login', status_code=303)
            resp.delete_cookie(COOKIE_NAME)
            return resp

        @self.app.get('/', response_class=HTMLResponse)
        async def dashboard(request: Request):
            if not self._is_auth(request):
                return RedirectResponse('/login', status_code=303)
            trades = await self._fetch_recent_trades(20)
            return HTMLResponse(self._render_dashboard(trades))

        @self.app.get('/api/status')
        async def api_status(request: Request):
            if not self._is_auth(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            core = self.bot.core
            ex = self.bot.executor
            return JSONResponse({
                'balance': ex.balance, 'available': ex.available_balance,
                'peak': core.peak_capital, 'mdd': core.max_drawdown,
                'total_trades': core.total_trades, 'win_rate': core.win_rate,
                'pf': core.profit_factor if core.profit_factor != float('inf') else 999,
                'consec_losses': core.consec_losses, 'skip_remaining': core.skip_remaining,
                'has_position': core.has_position,
            })

    async def start(self):
        """웹 서버 시작 (비동기 백그라운드)"""
        config = uvicorn.Config(self.app, host='0.0.0.0', port=self.port, log_level='warning', access_log=False)
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())
        logger.info(f"🌐 웹 대시보드: http://localhost:{self.port} (비밀번호: {self.password[:3]}***)")

    async def stop(self):
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass


def create_dashboard(bot, password: str = None, port: int = 8081) -> WebDashboard:
    pwd = password or os.getenv('WEB_PASSWORD', 'kang1366')
    return WebDashboard(bot, pwd, port=port)
