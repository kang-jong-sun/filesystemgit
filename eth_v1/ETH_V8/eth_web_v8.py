"""
ETH/USDT 선물 자동매매 - 웹 대시보드
V8: FastAPI + Jinja2 + WebSocket
"""

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger('eth_web')

DB_PATH = 'eth_trading_bot.db'
TEMPLATE_DIR = Path(__file__).parent / 'templates'
COOKIE_NAME = 'eth_session'
COOKIE_MAX_AGE = 7 * 24 * 3600  # 7일


class WebDashboard:
    def __init__(self, bot, password: str, secret_key: str = None):
        self.bot = bot
        self.password = password
        self.secret_key = secret_key or hashlib.sha256(os.urandom(32)).hexdigest()
        self.app = FastAPI(docs_url=None, redoc_url=None)
        self.templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
        # 커스텀 필터
        self.templates.env.filters['usd'] = lambda v, d=2: f"${v:,.{d}f}" if v else "$0.00"
        self.templates.env.filters['usd0'] = lambda v: f"${v:,.0f}" if v else "$0"
        self.templates.env.filters['susd'] = lambda v, d=2: f"${v:+,.{d}f}" if v else "$0.00"
        self.templates.env.filters['susd0'] = lambda v: f"${v:+,.0f}" if v else "$0"
        self.templates.env.filters['pct'] = lambda v, d=2: f"{v:+.{d}f}%" if v else "0.00%"
        self._failed_attempts = 0
        self._lockout_until = 0.0
        self._setup_routes()

    # ═══════════════════════════════════════════════
    # 인증
    # ═══════════════════════════════════════════════
    def _make_token(self) -> str:
        ts = str(int(time.time()))
        sig = hmac.new(self.secret_key.encode(), ts.encode(), hashlib.sha256).hexdigest()[:16]
        return f"{ts}:{sig}"

    def _verify_token(self, token: str) -> bool:
        if not token or ':' not in token:
            return False
        ts, sig = token.split(':', 1)
        expected = hmac.new(self.secret_key.encode(), ts.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(sig, expected):
            return False
        # 7일 만료
        try:
            if time.time() - int(ts) > COOKIE_MAX_AGE:
                return False
        except ValueError:
            return False
        return True

    def _is_authenticated(self, request: Request) -> bool:
        if not self.password:  # 비밀번호 미설정 시 인증 없음
            return True
        token = request.cookies.get(COOKIE_NAME, '')
        return self._verify_token(token)

    # ═══════════════════════════════════════════════
    # 데이터 빌드
    # ═══════════════════════════════════════════════
    def _build_status(self) -> dict:
        bot = self.bot
        bar = bot.data.get_current_bar() if bot.data else None
        status = bot.core.get_status() if bot.core else {}
        price = bot.data.current_price if bot.data and bot.data.current_price > 0 else (bar['close'] if bar else 0)

        uptime = time.time() - bot.monitor.start_time if bot.monitor else 0
        h, rem = divmod(int(uptime), 3600)
        m, _ = divmod(rem, 60)

        result = {
            'price': price,
            'ema250': bar['fast_ma'] if bar else 0,
            'ema1575': bar['slow_ma'] if bar else 0,
            'ema_status': 'BULL' if bar and bar['fast_ma'] > bar['slow_ma'] else 'BEAR',
            'ema_gap': abs(bar['fast_ma'] - bar['slow_ma']) / bar['slow_ma'] * 100 if bar and bar['slow_ma'] > 0 else 0,
            'b_ema9': bar.get('b_ema9', 0) if bar else 0,
            'b_ema100': bar.get('b_ema100_15m', 0) if bar else 0,
            'balance': bot.executor.balance if bot.executor else 0,
            'available': bot.executor.available_balance if bot.executor else 0,
            'peak_capital': bot.core.peak_capital if bot.core else 0,
            'uptime': f"{h}h {m}m",
            'uptime_seconds': int(uptime),
            'version': 'V8',
            **status,
        }

        # 포지션 상세
        if status.get('has_position') and bot.core.position:
            pos = bot.core.position
            roi = (price - pos.entry_price) / pos.entry_price * pos.direction * 100 if pos.entry_price > 0 else 0
            pnl = (price - pos.entry_price) / pos.entry_price * pos.position_size * pos.direction if pos.entry_price > 0 else 0
            hold = time.time() - pos.entry_time
            hh, rm = divmod(int(hold), 3600)
            mm, _ = divmod(rm, 60)
            result.update({
                'entry_mode': pos.entry_mode,
                'roi': roi,
                'unrealized_pnl': pnl,
                'hold_time': f"{hh}h {mm}m",
            })

        # 최근 거래
        recent = []
        if bot.core and bot.core.trade_history:
            for tr in bot.core.trade_history[-5:]:
                recent.append({
                    'direction': 'L' if tr.direction == 1 else 'S',
                    'exit_type': tr.exit_type,
                    'entry_price': tr.entry_price,
                    'exit_price': tr.exit_price,
                    'pnl': tr.pnl,
                    'roi_pct': tr.roi_pct,
                    'exit_time': tr.exit_time,
                })
        result['recent_trades'] = recent
        result['total_transferred'] = getattr(bot, '_total_transferred', 0)

        # JSON 호환: Infinity/NaN → 안전한 값
        import math
        for k, v in result.items():
            if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
                result[k] = 0.0

        return result

    # ═══════════════════════════════════════════════
    # 라우트
    # ═══════════════════════════════════════════════
    def _setup_routes(self):
        app = self.app

        @app.get('/login', response_class=HTMLResponse)
        async def login_page(request: Request):
            return self.templates.TemplateResponse(request, 'login.html', {'error': ''})

        @app.post('/login')
        async def login_submit(request: Request, password: str = Form(...)):
            now = time.time()
            if now < self._lockout_until:
                wait = int(self._lockout_until - now)
                return self.templates.TemplateResponse(request, 'login.html',
                    {'error': f'Too many attempts. Wait {wait}s.'})

            if password == self.password:
                self._failed_attempts = 0
                response = RedirectResponse('/', status_code=302)
                response.set_cookie(COOKIE_NAME, self._make_token(),
                                    max_age=COOKIE_MAX_AGE, httponly=True)
                return response
            else:
                self._failed_attempts += 1
                if self._failed_attempts >= 5:
                    self._lockout_until = now + 60
                    self._failed_attempts = 0
                return self.templates.TemplateResponse(request, 'login.html',
                    {'error': 'Invalid password'})

        @app.get('/logout')
        async def logout():
            response = RedirectResponse('/login', status_code=302)
            response.delete_cookie(COOKIE_NAME)
            return response

        @app.get('/', response_class=HTMLResponse)
        async def dashboard(request: Request):
            if not self._is_authenticated(request):
                return RedirectResponse('/login', status_code=302)
            status = self._build_status()
            return self.templates.TemplateResponse(request, 'dashboard.html', status)

        @app.get('/trades', response_class=HTMLResponse)
        async def trades_page(request: Request, page: int = 1):
            if not self._is_authenticated(request):
                return RedirectResponse('/login', status_code=302)
            limit = 20
            offset = (page - 1) * limit
            trades = []
            total = 0
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute('SELECT COUNT(*) as cnt FROM trades') as cur:
                        row = await cur.fetchone()
                        total = row[0]
                    async with db.execute(
                        'SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?',
                        (limit, offset)) as cur:
                        trades = [dict(r) for r in await cur.fetchall()]
            except Exception as e:
                logger.error(f"trades query: {e}")

            total_pages = max(1, (total + limit - 1) // limit)
            return self.templates.TemplateResponse(request, 'trades.html', {
                'trades': trades, 'page': page,
                'total_pages': total_pages, 'total': total})

        @app.get('/balance', response_class=HTMLResponse)
        async def balance_page(request: Request):
            if not self._is_authenticated(request):
                return RedirectResponse('/login', status_code=302)
            return self.templates.TemplateResponse(request, 'balance.html')

        # ─── API ───
        @app.get('/api/candles')
        async def api_candles(request: Request, limit: int = 200):
            """10분봉 캔들 + EMA 데이터"""
            if not self._is_authenticated(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            try:
                bot = self.bot
                if not bot.data or bot.data.df is None:
                    return {'candles': [], 'ema250': [], 'ema1575': []}

                df = bot.data.df
                n = len(df)
                count = min(limit, n) if limit > 0 else n
                start = n - count

                KST_OFFSET = 9 * 3600  # UTC → KST
                candles = []
                for i in range(start, n):
                    ts = int(df.index[i].timestamp()) + KST_OFFSET
                    candles.append({
                        'time': ts,
                        'open': round(float(df['open'].iloc[i]), 2),
                        'high': round(float(df['high'].iloc[i]), 2),
                        'low': round(float(df['low'].iloc[i]), 2),
                        'close': round(float(df['close'].iloc[i]), 2),
                    })

                ema250_data = []
                ema1575_data = []
                if bot.data.fast_ma is not None and bot.data.slow_ma is not None:
                    for i in range(start, n):
                        ts = int(df.index[i].timestamp()) + KST_OFFSET
                        v250 = float(bot.data.fast_ma[i])
                        v1575 = float(bot.data.slow_ma[i])
                        if v250 > 0:
                            ema250_data.append({'time': ts, 'value': round(v250, 2)})
                        if v1575 > 0:
                            ema1575_data.append({'time': ts, 'value': round(v1575, 2)})

                # 포지션 정보 (진입가, SL)
                position = None
                if bot.core and bot.core.has_position:
                    pos = bot.core.position
                    position = {
                        'entry_price': pos.entry_price,
                        'sl_price': pos.sl_price,
                        'direction': pos.direction,
                    }

                return {
                    'candles': candles,
                    'ema250': ema250_data,
                    'ema1575': ema1575_data,
                    'position': position,
                }
            except Exception as e:
                logger.error(f"candles API: {e}")
                return {'candles': [], 'ema250': [], 'ema1575': [], 'error': str(e)}

        @app.get('/api/errors')
        async def api_errors(request: Request, limit: int = 20):
            """최근 에러 로그"""
            if not self._is_authenticated(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            try:
                import glob
                log_files = sorted(glob.glob('logs/eth_trading_*.log'), reverse=True)
                errors = []
                for lf in log_files[:2]:  # 최근 2일 로그만
                    with open(lf, 'r', encoding='utf-8') as f:
                        for line in f:
                            if 'ERROR' in line or 'CRITICAL' in line:
                                errors.append(line.strip())
                return {'errors': errors[-limit:]}
            except Exception as e:
                return {'errors': [], 'error': str(e)}

        @app.get('/api/status')
        async def api_status(request: Request):
            if not self._is_authenticated(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            return self._build_status()

        @app.get('/api/trades')
        async def api_trades(request: Request, page: int = 1, limit: int = 20):
            if not self._is_authenticated(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            offset = (page - 1) * limit
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(
                        'SELECT * FROM trades ORDER BY id DESC LIMIT ? OFFSET ?',
                        (limit, offset)) as cur:
                        trades = [dict(r) for r in await cur.fetchall()]
                    async with db.execute('SELECT COUNT(*) FROM trades') as cur:
                        total = (await cur.fetchone())[0]
                return {'trades': trades, 'page': page, 'total': total}
            except Exception as e:
                return {'error': str(e)}

        @app.get('/api/balance-history')
        async def api_balance_history(request: Request, days: int = 0):
            if not self._is_authenticated(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            try:
                async with aiosqlite.connect(DB_PATH) as db:
                    db.row_factory = aiosqlite.Row
                    query = 'SELECT timestamp, balance_after FROM trades ORDER BY id'
                    if days > 0:
                        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                        query = f"SELECT timestamp, balance_after FROM trades WHERE timestamp >= '{since}' ORDER BY id"
                    async with db.execute(query) as cur:
                        rows = await cur.fetchall()
                    return {
                        'labels': [r['timestamp'] for r in rows],
                        'values': [r['balance_after'] for r in rows],
                    }
            except Exception as e:
                return {'labels': [], 'values': [], 'error': str(e)}

        # ─── WebSocket ───
        @app.websocket('/ws/price')
        async def ws_price(websocket: WebSocket):
            # 쿠키 인증
            if self.password:
                token = websocket.cookies.get(COOKIE_NAME, '')
                if not self._verify_token(token):
                    await websocket.close(code=4001)
                    return

            await websocket.accept()
            try:
                while True:
                    data = self._build_status()
                    await websocket.send_json(data)
                    await asyncio.sleep(2)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass


def create_app(bot) -> FastAPI:
    password = os.getenv('WEB_PASSWORD', '')
    dashboard = WebDashboard(bot, password)
    return dashboard.app


async def start_web_server(app, host='0.0.0.0', port=8080):
    config = uvicorn.Config(app, host=host, port=port, log_level='warning', access_log=False)
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None
    asyncio.create_task(server.serve())
    logger.info(f"Web dashboard: http://localhost:{port}")
    return server
