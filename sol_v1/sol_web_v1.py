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
.logout{color:#64748b;text-decoration:none;font-size:13px;padding:6px 12px;border-radius:6px}
.logout:hover{background:#1e293b;color:#ef4444}
.filter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:10px}
.filter-cell{background:#0f172a;padding:12px;border-radius:8px;border-left:3px solid #64748b}
.filter-cell.pass{border-left-color:#22c55e;background:rgba(34,197,94,0.05)}
.filter-cell.fail{border-left-color:#ef4444;background:rgba(239,68,68,0.05)}
.filter-cell.warn{border-left-color:#eab308;background:rgba(234,179,8,0.05)}
.filter-label{font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.3px}
.filter-value{font-size:18px;font-weight:bold;margin-top:4px}
.filter-cond{font-size:10px;color:#64748b;margin-top:3px}
.filter-status{font-size:12px;margin-top:4px;font-weight:600}
.status-pass{color:#22c55e}
.status-fail{color:#ef4444}
.status-warn{color:#eab308}
.verdict{padding:12px;border-radius:8px;margin-top:12px;text-align:center;font-size:13px;font-weight:600}
.verdict.block{background:rgba(239,68,68,0.1);color:#ef4444;border:1px solid rgba(239,68,68,0.3)}
.verdict.warn{background:rgba(234,179,8,0.1);color:#eab308;border:1px solid rgba(234,179,8,0.3)}
.verdict.ready{background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.3)}
.section-title{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin:14px 0 8px}
.cross-indicator{padding:10px;background:#0f172a;border-radius:6px;margin-top:8px;font-size:13px}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #2d3748}
.brand{color:#4ade80;font-size:22px;font-weight:bold}
.nav{display:flex;gap:4px}
.nav a{padding:8px 16px;color:#94a3b8;text-decoration:none;border-radius:6px;font-size:14px;font-weight:500;transition:all 0.15s}
.nav a:hover{background:#1e293b;color:#e0e0e0}
.nav a.active-tab{background:#1e293b;color:#4ade80;box-shadow:inset 0 -2px 0 #4ade80}
@media (max-width:640px){.topbar{flex-wrap:wrap;gap:10px}.nav{width:100%;overflow-x:auto}.brand{font-size:18px}}
</style></head>
<body>
<div class="topbar">
  <div class="brand">🪙 SOL V1</div>
  <div class="nav">
    <a href="/" class="active-tab">Dashboard</a>
    <a href="/chart">Chart</a>
    <a href="/trades">Trades</a>
    <a href="/balance">Balance</a>
    <a href="/errors">Errors</a>
    <a href="/logout" class="logout">Logout</a>
  </div>
</div>
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
  <h3>📡 현재 지표 + V12/Mass 필터 상태 (15m)</h3>

  <div class="cross-indicator">
    <b>추세 감지:</b> EMA9 <span style="color:#facc15">$%%EMA9%%</span> %%CROSS_SIGN%% SMA400 <span style="color:#f97316">$%%SMA400%%</span>
    → %%CROSS_DIR%%
  </div>

  <div class="section-title">V12 5중 필터</div>
  <div class="filter-grid">
    <div class="filter-cell %%F_ADX_CLASS%%">
      <div class="filter-label">ADX</div>
      <div class="filter-value">%%ADX%%</div>
      <div class="filter-cond">필요: ≥ 22 (추세 강도)</div>
      <div class="filter-status status-%%F_ADX_CLASS%%">%%F_ADX_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_RSI_CLASS%%">
      <div class="filter-label">RSI</div>
      <div class="filter-value">%%RSI%%</div>
      <div class="filter-cond">필요: 30 ~ 65 (중립)</div>
      <div class="filter-status status-%%F_RSI_CLASS%%">%%F_RSI_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_LR_CLASS%%">
      <div class="filter-label">LR Slope</div>
      <div class="filter-value">%%LR%%</div>
      <div class="filter-cond">필요: -0.5 ~ +0.5</div>
      <div class="filter-status status-%%F_LR_CLASS%%">%%F_LR_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_SAME_CLASS%%">
      <div class="filter-label">Skip Same</div>
      <div class="filter-value" style="font-size:14px">%%LAST_EXIT_TXT%%</div>
      <div class="filter-cond">이전 방향 재진입 차단</div>
      <div class="filter-status status-%%F_SAME_CLASS%%">%%F_SAME_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_WATCH_CLASS%%">
      <div class="filter-label">Watch (V12)</div>
      <div class="filter-value" style="font-size:14px">%%WATCH_TXT%%</div>
      <div class="filter-cond">Cross 감지 / Entry Delay 5봉</div>
      <div class="filter-status status-%%F_WATCH_CLASS%%">%%F_WATCH_TEXT%%</div>
    </div>
  </div>

  <div class="section-title">Mass Index & ATR Defense</div>
  <div class="filter-grid">
    <div class="filter-cell %%F_MASS_CLASS%%">
      <div class="filter-label">Mass Index</div>
      <div class="filter-value">%%MASS%%</div>
      <div class="filter-cond">Bulge: ≥ 27 → < 26.5 (반전)</div>
      <div class="filter-status status-%%F_MASS_CLASS%%">%%F_MASS_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_ATR_CLASS%%">
      <div class="filter-label">ATR14 / ATR50</div>
      <div class="filter-value">%%ATR_RATIO%%</div>
      <div class="filter-cond">&lt; 1.5x (정상) / ≥ 1.5x (margin flat)</div>
      <div class="filter-status status-%%F_ATR_CLASS%%">%%F_ATR_TEXT%%</div>
    </div>
  </div>

  <div class="section-title">게이트 (일일 손실 / Skip2@4loss / Mutex)</div>
  <div class="filter-grid">
    <div class="filter-cell %%F_DL_CLASS%%">
      <div class="filter-label">일일 손실</div>
      <div class="filter-value" style="font-size:14px">%%DAILY_PCT%%%</div>
      <div class="filter-cond">한도: ≤ -2.5%</div>
      <div class="filter-status status-%%F_DL_CLASS%%">%%F_DL_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_SKIP_CLASS%%">
      <div class="filter-label">Skip2@4loss</div>
      <div class="filter-value" style="font-size:14px">%%CONSEC%%/4 · skip %%SKIP%%</div>
      <div class="filter-cond">4연패 → 2거래 스킵</div>
      <div class="filter-status status-%%F_SKIP_CLASS%%">%%F_SKIP_TEXT%%</div>
    </div>
    <div class="filter-cell %%F_MUTEX_CLASS%%">
      <div class="filter-label">Mutex</div>
      <div class="filter-value" style="font-size:14px">%%POS_STATE%%</div>
      <div class="filter-cond">포지션 보유 중엔 신규 차단</div>
      <div class="filter-status status-%%F_MUTEX_CLASS%%">%%F_MUTEX_TEXT%%</div>
    </div>
  </div>

  <div class="verdict %%VERDICT_CLASS%%">%%VERDICT_TEXT%%</div>
</div>

<div class="card" style="margin-top:15px">
  <h3>📋 최근 거래 (20개)</h3>
  <table>
    <tr><th>시간</th><th>Mode</th><th>Dir</th><th>진입</th><th>청산</th><th>Exit</th><th>PnL</th><th>ROI</th></tr>
    %%TRADES_ROWS%%
  </table>
</div>
</body></html>"""


HTML_TRADES = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Trades</title>
<meta http-equiv="refresh" content="10">
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;padding:20px}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #2d3748}
.brand{color:#4ade80;font-size:22px;font-weight:bold}
.nav{display:flex;gap:4px}
.nav a{padding:8px 16px;color:#94a3b8;text-decoration:none;border-radius:6px;font-size:14px;font-weight:500;transition:all 0.15s}
.nav a:hover{background:#1e293b;color:#e0e0e0}
.nav a.active-tab{background:#1e293b;color:#4ade80;box-shadow:inset 0 -2px 0 #4ade80}
.logout{color:#64748b;padding:6px 12px;border-radius:6px;text-decoration:none;font-size:13px}
.card{background:#1a1f2e;padding:16px;border-radius:10px;border:1px solid #2d3748;margin-bottom:16px}
.card h3{margin:0 0 12px;color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:0.5px}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.summary .v{font-size:20px;font-weight:bold}
.summary .sub{font-size:11px;color:#64748b;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#1e293b;padding:10px;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #2d3748;position:sticky;top:0}
td{padding:10px;border-bottom:1px solid #1f2937}
tr:hover{background:rgba(255,255,255,0.02)}
.win{color:#22c55e}
.lose{color:#ef4444}
.warn{color:#eab308}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge-v12{background:#1e40af;color:#dbeafe}
.badge-mass{background:#7e22ce;color:#f3e8ff}
.badge-manual{background:#92400e;color:#fed7aa}
.badge-pyramid{background:#065f46;color:#a7f3d0}
.badge-sl{background:#7f1d1d;color:#fecaca}
.badge-tsl{background:#365314;color:#d9f99d}
.badge-rev{background:#78350f;color:#fde68a}
.badge-ext{background:#374151;color:#d1d5db}
.pagination{display:flex;justify-content:center;gap:8px;margin-top:16px}
.pagination a,.pagination span{padding:6px 12px;color:#94a3b8;text-decoration:none;background:#1e293b;border-radius:6px;font-size:13px}
.pagination a:hover{background:#2d3748}
.pagination .current{background:#4ade80;color:#0a0e1a;font-weight:bold}
.empty{text-align:center;color:#64748b;padding:40px}
@media (max-width:640px){.topbar{flex-wrap:wrap}.nav{width:100%;overflow-x:auto}table{font-size:11px}th,td{padding:6px}}
</style></head>
<body>
<div class="topbar">
  <div class="brand">🪙 SOL V1</div>
  <div class="nav">
    <a href="/">Dashboard</a>
    <a href="/chart">Chart</a>
    <a href="/trades" class="active-tab">Trades</a>
    <a href="/balance">Balance</a>
    <a href="/errors">Errors</a>
    <a href="/logout" class="logout">Logout</a>
  </div>
</div>

<div class="summary">
  <div class="card"><h3>Total Trades</h3><div class="v">%%TOTAL%%</div><div class="sub">Wins %%WINS%% / Losses %%LOSSES%%</div></div>
  <div class="card"><h3>Win Rate</h3><div class="v %%WR_CLASS%%">%%WR%%%</div><div class="sub">PF %%PF%%</div></div>
  <div class="card"><h3>Total PnL</h3><div class="v %%PNL_CLASS%%">$%%TOTAL_PNL%%</div><div class="sub">GP $%%GP%% / GL $%%GL%%</div></div>
  <div class="card"><h3>Exit Types</h3><div class="v" style="font-size:14px">SL %%SL%% · TSL %%TSL%% · REV %%REV%% · EXT %%EXT%%</div></div>
</div>

<div class="card">
  <h3>📋 거래 내역 (Page %%PAGE%%)</h3>
  <table>
    <tr><th>#</th><th>Time</th><th>Source</th><th>Mode</th><th>Dir</th><th>Entry</th><th>Exit</th><th>Type</th><th>PnL</th><th>ROI</th><th>Hold</th></tr>
    %%TRADE_ROWS%%
  </table>
  <div class="pagination">%%PAGINATION%%</div>
</div>
</body></html>"""


HTML_BALANCE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Balance</title>
<meta http-equiv="refresh" content="15">
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;padding:20px}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #2d3748}
.brand{color:#4ade80;font-size:22px;font-weight:bold}
.nav{display:flex;gap:4px}
.nav a{padding:8px 16px;color:#94a3b8;text-decoration:none;border-radius:6px;font-size:14px;font-weight:500;transition:all 0.15s}
.nav a:hover{background:#1e293b;color:#e0e0e0}
.nav a.active-tab{background:#1e293b;color:#4ade80;box-shadow:inset 0 -2px 0 #4ade80}
.logout{color:#64748b;padding:6px 12px;border-radius:6px;text-decoration:none;font-size:13px}
.card{background:#1a1f2e;padding:16px;border-radius:10px;border:1px solid #2d3748;margin-bottom:16px}
.card h3{margin:0 0 12px;color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:0.5px}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:20px}
.summary .v{font-size:24px;font-weight:bold}
.summary .sub{font-size:11px;color:#64748b;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:#1e293b;padding:10px;text-align:left;color:#94a3b8;font-weight:600;border-bottom:2px solid #2d3748}
td{padding:10px;border-bottom:1px solid #1f2937}
tr:hover{background:rgba(255,255,255,0.02)}
.win{color:#22c55e}
.lose{color:#ef4444}
.warn{color:#eab308}
.big{font-size:28px;font-weight:bold}
.meta{color:#64748b;font-size:12px;margin-top:6px}
.info-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px}
.info-grid .item{padding:12px;background:#0f172a;border-radius:6px}
.info-grid .label{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.5px}
.info-grid .value{font-size:18px;font-weight:bold;margin-top:4px}
@media (max-width:640px){.topbar{flex-wrap:wrap}.nav{width:100%;overflow-x:auto}}
</style></head>
<body>
<div class="topbar">
  <div class="brand">🪙 SOL V1</div>
  <div class="nav">
    <a href="/">Dashboard</a>
    <a href="/chart">Chart</a>
    <a href="/trades">Trades</a>
    <a href="/balance" class="active-tab">Balance</a>
    <a href="/errors">Errors</a>
    <a href="/logout" class="logout">Logout</a>
  </div>
</div>

<div class="summary">
  <div class="card"><h3>💰 Total Balance (공용 계정)</h3>
    <div class="big">$%%BALANCE%%</div>
    <div class="meta">Available: $%%AVAILABLE%% | Peak: $%%PEAK%%</div>
  </div>
  <div class="card"><h3>📊 SOL 누적 PnL (봇 시작 이후)</h3>
    <div class="big %%CUM_CLASS%%">$%%CUMULATIVE%%</div>
    <div class="meta">ETH 영향 배제한 순수 SOL 성과</div>
  </div>
  <div class="card"><h3>📅 오늘 (UTC 기준)</h3>
    <div class="big %%TODAY_CLASS%%">$%%TODAY_PNL%%</div>
    <div class="meta">Trades %%TODAY_TRADES%% · Start $%%DAY_START%%</div>
  </div>
</div>

<div class="card">
  <h3>🎛 봇 운영 정보</h3>
  <div class="info-grid">
    <div class="item"><div class="label">Leverage</div><div class="value">%%LEVERAGE%%x</div></div>
    <div class="item"><div class="label">Max Capital Cap</div><div class="value">$%%MAX_CAP%%</div></div>
    <div class="item"><div class="label">Compound %</div><div class="value">%%COMPOUND%%%</div></div>
    <div class="item"><div class="label">Daily Loss Limit</div><div class="value">%%DAILY_LIMIT%%%</div></div>
    <div class="item"><div class="label">Consec Losses</div><div class="value %%CONSEC_CLASS%%">%%CONSEC%%/4</div></div>
    <div class="item"><div class="label">Skip Remaining</div><div class="value">%%SKIP%%</div></div>
    <div class="item"><div class="label">Max Drawdown</div><div class="value %%MDD_CLASS%%">%%MDD%%%</div></div>
    <div class="item"><div class="label">Last Exit Dir</div><div class="value">%%LAST_EXIT%%</div></div>
  </div>
</div>

<div class="card">
  <h3>📆 일일 성과 요약 (최근 %%DAILY_COUNT%%일)</h3>
  <table>
    <tr><th>Date</th><th>Start Balance</th><th>End Balance</th><th>Change %</th><th>Trades</th><th>W/L</th><th>SOL PnL</th><th>ConsecLoss</th></tr>
    %%DAILY_ROWS%%
  </table>
  <div class="meta" style="margin-top:10px">※ daily_summary.log 파일에서 읽어옵니다</div>
</div>
</body></html>"""


HTML_ERRORS = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Errors</title>
<meta http-equiv="refresh" content="30">
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;padding:20px}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #2d3748}
.brand{color:#4ade80;font-size:22px;font-weight:bold}
.nav{display:flex;gap:4px;flex-wrap:wrap}
.nav a{padding:8px 16px;color:#94a3b8;text-decoration:none;border-radius:6px;font-size:14px;font-weight:500;transition:all 0.15s}
.nav a:hover{background:#1e293b;color:#e0e0e0}
.nav a.active-tab{background:#1e293b;color:#4ade80;box-shadow:inset 0 -2px 0 #4ade80}
.logout{color:#64748b;padding:6px 12px;border-radius:6px;text-decoration:none;font-size:13px}
.card{background:#1a1f2e;padding:16px;border-radius:10px;border:1px solid #2d3748;margin-bottom:16px}
.card h3{margin:0 0 12px;color:#94a3b8;font-size:13px;text-transform:uppercase;letter-spacing:0.5px}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:20px}
.summary .v{font-size:24px;font-weight:bold}
.summary .sub{font-size:11px;color:#64748b;margin-top:4px}
.log-line{font-family:Consolas,Monaco,monospace;font-size:12px;padding:8px 10px;border-bottom:1px solid #1f2937;white-space:pre-wrap;word-break:break-all}
.log-line:hover{background:rgba(255,255,255,0.03)}
.level-ERROR{color:#ef4444;border-left:3px solid #ef4444}
.level-WARNING{color:#eab308;border-left:3px solid #eab308}
.level-OTHER{color:#94a3b8;border-left:3px solid #64748b}
.file-tag{display:inline-block;font-size:10px;background:#1e293b;padding:2px 6px;border-radius:4px;color:#64748b;margin-right:8px}
.empty{text-align:center;color:#64748b;padding:60px}
.empty-big{font-size:48px;margin-bottom:10px}
.filter-bar{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.filter-btn{padding:6px 14px;background:#1e293b;color:#94a3b8;border:1px solid #2d3748;border-radius:6px;cursor:pointer;font-size:12px}
.filter-btn.active{background:#4ade80;color:#0a0e1a;border-color:#4ade80}
</style></head>
<body>
<div class="topbar">
  <div class="brand">🪙 SOL V1</div>
  <div class="nav">
    <a href="/">Dashboard</a>
    <a href="/chart">Chart</a>
    <a href="/trades">Trades</a>
    <a href="/balance">Balance</a>
    <a href="/errors" class="active-tab">Errors</a>
    <a href="/logout" class="logout">Logout</a>
  </div>
</div>

<div class="summary">
  <div class="card"><h3>⚠ 총 이벤트</h3><div class="v">%%TOTAL%%</div><div class="sub">최근 3일 error_*.log</div></div>
  <div class="card"><h3>🔴 Errors</h3><div class="v" style="color:#ef4444">%%ERRORS%%</div><div class="sub">즉시 대응 필요</div></div>
  <div class="card"><h3>🟡 Warnings</h3><div class="v" style="color:#eab308">%%WARNINGS%%</div><div class="sub">참고용</div></div>
  <div class="card"><h3>📅 조회 기간</h3><div class="v" style="font-size:14px">365일 보관</div><div class="sub">30초 자동 새로고침</div></div>
</div>

<div class="card">
  <h3>🗂 필터</h3>
  <div class="filter-bar">
    <button class="filter-btn active" onclick="filter('all')">All</button>
    <button class="filter-btn" onclick="filter('ERROR')">🔴 Errors Only</button>
    <button class="filter-btn" onclick="filter('WARNING')">🟡 Warnings Only</button>
  </div>
</div>

<div class="card">
  <h3>📋 로그 (최신순, 최근 100개)</h3>
  <div id="log-container">
    %%LOG_LINES%%
  </div>
</div>

<script>
function filter(level) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.log-line').forEach(line => {
    if (level === 'all') line.style.display = '';
    else line.style.display = line.classList.contains('level-' + level) ? '' : 'none';
  });
}
</script>
</body></html>"""


HTML_CHART = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>SOL V1 Chart</title>
<script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
<style>
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,sans-serif;background:#0a0e1a;color:#e0e0e0;margin:0;padding:20px}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #2d3748}
.brand{color:#4ade80;font-size:22px;font-weight:bold}
.nav{display:flex;gap:4px;flex-wrap:wrap}
.nav a{padding:8px 16px;color:#94a3b8;text-decoration:none;border-radius:6px;font-size:14px;font-weight:500;transition:all 0.15s}
.nav a:hover{background:#1e293b;color:#e0e0e0}
.nav a.active-tab{background:#1e293b;color:#4ade80;box-shadow:inset 0 -2px 0 #4ade80}
.logout{color:#64748b;padding:6px 12px;border-radius:6px;text-decoration:none;font-size:13px}
.chart-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:10px}
.chart-title{font-size:16px;color:#e0e0e0}
.chart-title .price{font-size:24px;font-weight:bold;color:#4ade80;margin-left:10px}
.info-bar{display:flex;gap:12px;font-size:12px;color:#94a3b8;flex-wrap:wrap;align-items:center}
.info-bar span{padding:6px 10px;background:#1a1f2e;border-radius:6px;border:1px solid #2d3748}
.legend{display:flex;gap:16px;margin-top:10px;font-size:12px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-dot{width:12px;height:3px;border-radius:1px}
#chart{width:100%;height:calc(100vh - 280px);min-height:500px;background:#0a0e1a;border-radius:10px;border:1px solid #2d3748;margin-top:10px}
#loading{text-align:center;color:#64748b;padding:60px}
.pos-info{background:#1a1f2e;padding:12px;border-radius:8px;border-left:4px solid #eab308;margin-top:10px;font-size:13px}
.pos-long{border-left-color:#22c55e}
.pos-short{border-left-color:#ef4444}
.hint{color:#64748b;font-size:11px;margin-left:10px}
@media (max-width:640px){#chart{height:50vh;min-height:350px}.chart-header{flex-direction:column;align-items:flex-start}}
</style></head>
<body>
<div class="topbar">
  <div class="brand">🪙 SOL V1</div>
  <div class="nav">
    <a href="/">Dashboard</a>
    <a href="/chart" class="active-tab">Chart</a>
    <a href="/trades">Trades</a>
    <a href="/balance">Balance</a>
    <a href="/errors">Errors</a>
    <a href="/logout" class="logout">Logout</a>
  </div>
</div>

<div class="chart-header">
  <div class="chart-title">
    SOL/USDT 15m (6개월 17,280봉)<span class="price" id="current-price">-</span>
    <span class="hint">🖱 마우스 휠로 줌인/아웃, 드래그로 이동 (최근 120봉 기본 표시)</span>
  </div>
</div>

<div class="info-bar" id="info-bar">
  <span>EMA9: <b id="ema9-val">-</b></span>
  <span>SMA400: <b id="sma400-val">-</b></span>
  <span>Bars: <b id="bars-count">0</b></span>
  <span>Position: <b id="position-info">없음</b></span>
</div>

<div id="chart"><div id="loading">📊 차트 로딩 중... (6개월 17,280봉)</div></div>

<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>Up Candle</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Down Candle</div>
  <div class="legend-item"><div class="legend-dot" style="background:#facc15"></div>EMA(9) Fast</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f97316"></div>SMA(400) Slow</div>
  <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div>▲ WIN Trade</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>▼ LOSS Trade</div>
</div>

<div id="pos-block"></div>

<script>
let chart, candleSeries, ema9Series, sma400Series;
let currentBar = null;  // 진행 중 봉 추적 (실시간 업데이트용)

function initChart() {
  const container = document.getElementById('chart');
  container.innerHTML = '';
  chart = LightweightCharts.createChart(container, {
    layout: {
      background: { type: 'solid', color: '#0a0e1a' },
      textColor: '#94a3b8',
      fontSize: 11,
    },
    grid: {
      vertLines: { color: '#1e293b' },
      horzLines: { color: '#1e293b' },
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
      borderColor: '#2d3748',
      scaleMargins: { top: 0.1, bottom: 0.1 },
    },
    timeScale: {
      borderColor: '#2d3748',
      timeVisible: true,
      secondsVisible: false,
      barSpacing: 8,         // 저변동성 봉 가독성 개선 (6 → 8)
      rightOffset: 5,
    },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    height: Math.min(600, window.innerHeight - 250),
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e', downColor: '#ef4444',
    borderUpColor: '#22c55e', borderDownColor: '#ef4444',
    wickUpColor: '#22c55e', wickDownColor: '#ef4444',
  });
  ema9Series = chart.addLineSeries({
    color: '#facc15', lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
    title: 'EMA(9)',
  });
  sma400Series = chart.addLineSeries({
    color: '#f97316', lineWidth: 2, priceLineVisible: false, lastValueVisible: true,
    title: 'SMA(400)',
  });
  window.addEventListener('resize', () => {
    chart.resize(container.clientWidth, container.clientHeight);
  });
}

async function loadChart(limit) {
  if (!chart) initChart();
  try {
    const resp = await fetch('/api/candles?limit=' + limit);
    const data = await resp.json();
    if (data.error) { alert('Error: ' + data.error); return; }

    candleSeries.setData(data.candles);
    ema9Series.setData(data.ema9);
    sma400Series.setData(data.sma400);

    // ★ 마지막 봉을 currentBar로 seed (WebSocket/polling으로 이어서 업데이트)
    if (data.candles.length > 0) {
      const last = data.candles[data.candles.length - 1];
      currentBar = { time: last.time, open: last.open, high: last.high, low: last.low, close: last.close };
    }

    if (data.markers && data.markers.length > 0) {
      candleSeries.setMarkers(data.markers);
    }

    // 현재 포지션 SL 라인
    if (data.position) {
      const p = data.position;
      candleSeries.createPriceLine({
        price: p.entry_price, color: '#eab308', lineWidth: 1, lineStyle: 2,
        axisLabelVisible: true, title: 'ENTRY ' + (p.direction === 1 ? 'L' : 'S'),
      });
      candleSeries.createPriceLine({
        price: p.sl_price, color: '#ef4444', lineWidth: 1, lineStyle: 2,
        axisLabelVisible: true, title: 'SL',
      });
      document.getElementById('pos-block').innerHTML =
        '<div class="pos-info pos-' + (p.direction === 1 ? 'long' : 'short') + '">' +
        '📍 현재 포지션 [' + p.mode + '] ' + (p.direction === 1 ? 'LONG 🟢' : 'SHORT 🔴') +
        ' @ $' + p.entry_price.toFixed(3) + ' | SL $' + p.sl_price.toFixed(3) +
        '</div>';
      document.getElementById('position-info').textContent =
        (p.direction === 1 ? 'LONG' : 'SHORT') + ' @$' + p.entry_price.toFixed(2);
    } else {
      document.getElementById('pos-block').innerHTML = '';
      document.getElementById('position-info').textContent = '없음 (대기 중)';
    }

    // 현재가 + 최신 지표 업데이트
    if (data.candles.length > 0) {
      const last = data.candles[data.candles.length - 1];
      document.getElementById('current-price').textContent = '$' + last.close.toFixed(3);
    }
    if (data.ema9.length > 0) {
      document.getElementById('ema9-val').textContent = '$' + data.ema9[data.ema9.length-1].value.toFixed(3);
    }
    if (data.sma400.length > 0) {
      document.getElementById('sma400-val').textContent = '$' + data.sma400[data.sma400.length-1].value.toFixed(3);
    }
    document.getElementById('bars-count').textContent = data.returned_bars + ' / ' + data.total_bars;

    // ETH V8 방식: 단순한 setVisibleLogicalRange 1회 호출
    // 최근 120봉 (30시간 = 15m × 120) 표시, 과거는 휠/드래그로
    if (data.candles.length > 0) {
      const isMobile = window.innerWidth < 768;
      const visibleBars = isMobile ? 60 : 120;
      chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, data.candles.length - visibleBars),
        to: data.candles.length + 3,
      });
    }
  } catch (e) {
    document.getElementById('chart').innerHTML = '<div id="loading">❌ 차트 로딩 실패: ' + e + '</div>';
  }
}

// 초기 로드: 6개월 전체(17,280봉) 단 1회만 로드
// 이후 실시간 업데이트는 updateCurrentBar() 5초 폴링으로 처리
// 15분 봉 경계 크로싱 시 자동으로 새 봉 생성
// ★ loadChart setInterval 제거: setData/update 충돌 제거
// (페이지 새로고침 시에만 전체 재로드)
initChart();
loadChart(17280);

// ★ 서버 데이터 그대로 반영 (추정 로직 완전 제거)
// 서버의 df_sol 마지막 봉 OHLC를 5초마다 가져와서 차트에 직접 적용
// → 봉 경계 계산/OHLC 추정 등 모든 프론트 추정 로직 제거
async function updateCurrentBar() {
  try {
    const resp = await fetch('/api/status');
    if (!resp.ok) return;
    const d = await resp.json();

    // 현재가 표시 갱신
    if (d.price) {
      document.getElementById('current-price').textContent = '$' + d.price.toFixed(3);
    }

    // 서버의 실제 마지막 봉 데이터를 그대로 update
    // (프론트가 봉 경계 계산/OHLC 추정 안 함 → 깨짐 원천 제거)
    if (d.last_bar && candleSeries) {
      candleSeries.update(d.last_bar);
    }
  } catch(e) {}
}
setInterval(updateCurrentBar, 5000);
</script>
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

        # ═══════════════════════════════════════════════════════════
        # V12/Mass 필터 상태 계산
        # ═══════════════════════════════════════════════════════════
        from sol_core_v1 import (ADX_MIN, RSI_MIN, RSI_MAX, LR_MIN, LR_MAX,
                                  MASS_BULGE_THRESHOLD, MASS_REVERSAL_THRESHOLD,
                                  ATR_EXPANSION_THRESHOLD, DAILY_LOSS_LIMIT)
        ema9_v = ind.get('fast_ma', 0) or 0
        sma400_v = ind.get('slow_ma', 0) or 0
        lr_v = ind.get('lr_slope', 0) or 0
        atr14_v = ind.get('atr14', 0) or 0
        atr50_v = ind.get('atr50', 0.01) or 0.01

        # Cross 상태
        if ema9_v > sma400_v:
            cross_sign = '&gt;'
            cross_dir = '<span class="win">📈 LONG 방향 (EMA9 위)</span>'
        elif ema9_v < sma400_v:
            cross_sign = '&lt;'
            cross_dir = '<span class="lose">📉 SHORT 방향 (EMA9 아래)</span>'
        else:
            cross_sign = '='
            cross_dir = '<span class="warn">— 중립</span>'

        def _fp(ok): return ('pass', '✅ 통과') if ok else ('fail', '❌ 차단')
        def _fw(ok, warn_ok): return ('pass', '✅ 통과') if ok else (('warn', '🟡 대기') if warn_ok else ('fail', '❌ 차단'))

        # 1. ADX
        f_adx_class, f_adx_text = _fp(adx_val >= ADX_MIN)

        # 2. RSI
        f_rsi_class, f_rsi_text = _fp(RSI_MIN <= rsi_val <= RSI_MAX)

        # 3. LR Slope
        f_lr_class, f_lr_text = _fp(LR_MIN <= lr_v <= LR_MAX)

        # 4. Skip Same Direction
        last_exit = core.last_exit_dir
        if last_exit == 0:
            last_exit_txt = "없음 (첫 거래)"
            f_same_class, f_same_text = 'pass', '✅ 제약 없음'
        elif last_exit == 1:
            last_exit_txt = "LONG (↑ 차단)"
            f_same_class, f_same_text = 'warn', '⚠ LONG 진입만 차단'
        else:
            last_exit_txt = "SHORT (↓ 차단)"
            f_same_class, f_same_text = 'warn', '⚠ SHORT 진입만 차단'

        # 5. Watch_v12
        from sol_core_v1 import MONITOR_WIN, ENTRY_DELAY, SKIP_SAME_DIR
        w = core.watch_v12
        bars_since_cross = -1   # -1: no watch
        watch_expired = False
        watch_delay_active = False
        watch_active = False
        # ★ TIME 기반 (재시작 후에도 정확)
        latest_bar_ts = 0
        if data and data.df_sol is not None and len(data.df_sol) > 0:
            latest_bar_ts = data.df_sol.index[-1].timestamp()
        if w.direction == 0:
            watch_txt = "대기 중 (cross 미감지)"
            f_watch_class, f_watch_text = 'fail', '⏳ Cross 대기'
        else:
            # ★ TIME 기반 경과 봉 계산 (15m = 900초)
            if latest_bar_ts > 0 and w.start_time > 0:
                elapsed_sec = latest_bar_ts - w.start_time
                bars_since_cross = int(elapsed_sec // 900)  # 15분 단위
            else:
                bars_since_cross = 0
            watch_dir = 'LONG' if w.direction == 1 else 'SHORT'
            if bars_since_cross > MONITOR_WIN:
                watch_txt = f"{watch_dir} 만료 ({bars_since_cross}봉)"
                f_watch_class, f_watch_text = 'fail', '❌ Window 초과'
                watch_expired = True
            elif bars_since_cross <= ENTRY_DELAY:
                watch_txt = f"{watch_dir} delay {bars_since_cross}/{ENTRY_DELAY}"
                f_watch_class, f_watch_text = 'warn', '⏱ Delay 대기'
                watch_delay_active = True
            else:
                watch_txt = f"{watch_dir} 진입 가능 ({bars_since_cross}봉)"
                f_watch_class, f_watch_text = 'pass', '✅ 활성'
                watch_active = True

        # 6. Mass Index
        if core.mass_bulge_active:
            if mass_val < MASS_REVERSAL_THRESHOLD:
                f_mass_class, f_mass_text = 'pass', '✅ 반전 트리거!'
            else:
                f_mass_class, f_mass_text = 'warn', '🟡 Bulge 형성 중'
        else:
            f_mass_class, f_mass_text = 'fail', '⏳ Bulge 대기 (<27)'

        # 7. ATR Expansion
        atr_ratio = atr14_v / atr50_v if atr50_v > 0 else 0
        atr_expanded = atr_ratio >= ATR_EXPANSION_THRESHOLD
        if atr_expanded:
            f_atr_class, f_atr_text = 'warn', '⚠ Margin flat'
        else:
            f_atr_class, f_atr_text = 'pass', '✅ 정상'

        # 8. Daily Loss (daily_pct는 이 시점에 아직 계산 전이므로 인라인 계산)
        sol_daily_pnl = core.cumulative_sol_pnl - core.day_start_sol_pnl
        daily_limit_hit = core._daily_loss_exceeded(ex.balance)
        _daily_pct_filter = ((ex.balance / core.day_start_balance - 1) * 100
                             if core.day_start_balance > 0 else 0)
        if daily_limit_hit:
            f_dl_class, f_dl_text = 'fail', '❌ 한도 초과'
        elif _daily_pct_filter < -1.5:
            f_dl_class, f_dl_text = 'warn', '🟡 위험 근접'
        else:
            f_dl_class, f_dl_text = 'pass', '✅ 안전'

        # 9. Skip2@4loss
        if core.skip_remaining > 0:
            f_skip_class, f_skip_text = 'fail', f'❌ {core.skip_remaining}거래 스킵'
        elif core.consec_losses >= 3:
            f_skip_class, f_skip_text = 'warn', '⚠ 1패 더하면 스킵'
        else:
            f_skip_class, f_skip_text = 'pass', '✅ 정상'

        # 10. Mutex
        if core.has_position:
            p = core.position
            pos_state = f"{'V12' if p.entry_mode == 1 else 'MASS'} {'LONG' if p.direction == 1 else 'SHORT'} 보유"
            f_mutex_class, f_mutex_text = 'fail', '❌ 포지션 보유 중'
        else:
            pos_state = "없음"
            f_mutex_class, f_mutex_text = 'pass', '✅ 진입 가능'

        # === 진입 판정 ===
        # v12_reasons: 진입 차단 사유 (시간 경과로 해결 가능 / 조건 변화 필요 구분)
        v12_timing_blocks = []     # 시간 지나면 자동 해결 (Delay)
        v12_filter_blocks = []     # 지표 값이 변해야 해결 (RSI, ADX, LR)
        v12_hard_blocks = []       # 새 Cross 또는 리셋 필요

        # Watch 상태 차단
        if w.direction == 0:
            v12_hard_blocks.append('Cross 미감지')
        elif watch_expired:
            v12_hard_blocks.append(f'Window 만료({bars_since_cross}봉)')
        elif watch_delay_active:
            v12_timing_blocks.append(f'Entry Delay {bars_since_cross}/{ENTRY_DELAY}봉')

        # Skip Same Direction (watch 있을 때만)
        if w.direction != 0 and SKIP_SAME_DIR and w.direction == core.last_exit_dir:
            v12_hard_blocks.append('Skip Same 방향')

        # 5중 필터 차단
        if adx_val < ADX_MIN:
            v12_filter_blocks.append(f'ADX {adx_val:.1f}<22')
        if not (RSI_MIN <= rsi_val <= RSI_MAX):
            v12_filter_blocks.append(f'RSI {rsi_val:.1f}범위밖')
        if not (LR_MIN <= lr_v <= LR_MAX):
            v12_filter_blocks.append(f'LR{lr_v:+.2f}범위밖')

        # Gate 차단 (최상위 블록)
        gate_block = None
        if core.has_position:
            gate_block = ('pos', '📍 포지션 보유 중 — 신규 진입 차단 (Mutex)')
        elif daily_limit_hit:
            gate_block = ('dl', '🛑 Daily Loss 한도 도달 — 당일 거래 중지')
        elif core.skip_remaining > 0:
            gate_block = ('skip', f'⏸ Skip2@4loss 발동 — {core.skip_remaining}거래 후 재개')

        mass_ok = core.mass_bulge_active and mass_val < MASS_REVERSAL_THRESHOLD

        # 최종 판정 (우선순위)
        if gate_block:
            verdict_class = 'block'
            verdict_text = gate_block[1]
        elif mass_ok:
            verdict_class = 'ready'
            verdict_text = '🟢 Mass 반전 트리거! 다음 틱 진입 예정'
        elif not v12_timing_blocks and not v12_filter_blocks and not v12_hard_blocks:
            # 모든 V12 조건 충족 → 다음 봉 마감 시 진입
            verdict_class = 'ready'
            verdict_text = f'🟢 V12 진입 준비 완료 — 다음 15m 봉 마감 시 진입 체결 ({watch_txt})'
        elif v12_timing_blocks and not v12_filter_blocks and not v12_hard_blocks:
            # 시간만 지나면 자동 진입 가능 (지표는 모두 통과)
            verdict_class = 'warn'
            remain = ENTRY_DELAY - bars_since_cross + 1
            verdict_text = (f'⏱ V12 {v12_timing_blocks[0]} — '
                            f'{remain}봉({remain*15}분) 후 평가 시작 (지표 모두 통과 중)')
        elif v12_timing_blocks and v12_filter_blocks:
            # Delay + 필터 둘 다 차단
            verdict_class = 'block'
            verdict_text = (f'⏳ V12 이중 차단: {v12_timing_blocks[0]} + '
                            f'필터({", ".join(v12_filter_blocks[:2])}) | Mass ⏳ bulge 대기')
        elif v12_hard_blocks:
            # Cross 미감지 또는 Window 만료
            verdict_class = 'block'
            verdict_text = f'⏳ V12 차단 ({", ".join(v12_hard_blocks[:2])}) | Mass ⏳ bulge 대기'
        else:
            # 필터 차단만 (시간 + Watch OK)
            verdict_class = 'block'
            verdict_text = (f'⏳ V12 필터 차단 ({", ".join(v12_filter_blocks[:3])}) | '
                            f'Watch: {watch_txt} | Mass ⏳ bulge 대기')

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
            # ── 필터 상태 필드 ──
            '%%CROSS_SIGN%%': cross_sign,
            '%%CROSS_DIR%%': cross_dir,
            '%%F_ADX_CLASS%%': f_adx_class,
            '%%F_ADX_TEXT%%': f_adx_text,
            '%%F_RSI_CLASS%%': f_rsi_class,
            '%%F_RSI_TEXT%%': f_rsi_text,
            '%%F_LR_CLASS%%': f_lr_class,
            '%%F_LR_TEXT%%': f_lr_text,
            '%%F_SAME_CLASS%%': f_same_class,
            '%%F_SAME_TEXT%%': f_same_text,
            '%%LAST_EXIT_TXT%%': last_exit_txt,
            '%%F_WATCH_CLASS%%': f_watch_class,
            '%%F_WATCH_TEXT%%': f_watch_text,
            '%%WATCH_TXT%%': watch_txt,
            '%%F_MASS_CLASS%%': f_mass_class,
            '%%F_MASS_TEXT%%': f_mass_text,
            '%%F_ATR_CLASS%%': f_atr_class,
            '%%F_ATR_TEXT%%': f_atr_text,
            '%%ATR_RATIO%%': f"{atr_ratio:.2f}x",
            '%%F_DL_CLASS%%': f_dl_class,
            '%%F_DL_TEXT%%': f_dl_text,
            '%%F_SKIP_CLASS%%': f_skip_class,
            '%%F_SKIP_TEXT%%': f_skip_text,
            '%%F_MUTEX_CLASS%%': f_mutex_class,
            '%%F_MUTEX_TEXT%%': f_mutex_text,
            '%%POS_STATE%%': pos_state,
            '%%VERDICT_CLASS%%': verdict_class,
            '%%VERDICT_TEXT%%': verdict_text,
        }

        html = HTML_DASHBOARD
        for k, v in repl.items():
            html = html.replace(k, str(v))
        return html

    async def _fetch_trades_paginated(self, page: int, per_page: int = 30):
        """페이지네이션된 거래 내역"""
        try:
            offset = (page - 1) * per_page
            async with aiosqlite.connect(DB_PATH) as db:
                # 총 개수
                cur = await db.execute("SELECT COUNT(*) FROM trades")
                total = (await cur.fetchone())[0]
                # 페이지 데이터
                cur = await db.execute(
                    "SELECT timestamp, source, entry_mode, direction, entry_price, exit_price, "
                    "exit_type, pnl, roi_pct, hold_time FROM trades "
                    "ORDER BY id DESC LIMIT ? OFFSET ?", (per_page, offset))
                rows = await cur.fetchall()
                return total, rows
        except Exception as e:
            logger.error(f"trades fetch: {e}")
            return 0, []

    async def _render_trades(self, page: int = 1) -> str:
        import math as _m
        core = self.bot.core
        per_page = 30
        total_trades_db, rows = await self._fetch_trades_paginated(page, per_page)
        total_pages = max(1, _m.ceil(total_trades_db / per_page))
        page = max(1, min(page, total_pages))

        # 통계 요약
        total_pnl = core.gross_profit - core.gross_loss
        pnl_class = 'win' if total_pnl > 0 else ('lose' if total_pnl < 0 else '')
        wr = core.win_rate
        wr_class = 'win' if wr >= 50 else ('warn' if wr >= 30 else 'lose')
        pf = core.profit_factor if core.profit_factor != float('inf') else 999
        pf_str = f"{pf:.2f}" if pf < 999 else "∞"

        # Exit Types counter (EXT = 외부 청산)
        ext_count = 0
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                cur = await db.execute("SELECT COUNT(*) FROM trades WHERE exit_type='EXT'")
                ext_count = (await cur.fetchone())[0]
        except Exception:
            pass

        # 거래 row 생성
        trade_rows = ""
        start_idx = (page - 1) * per_page
        for i, r in enumerate(rows, 1):
            ts, src, mode, direction, entry, exit_px, etype, pnl, roi, hold = r
            ts_short = ts[5:19] if ts and len(ts) >= 19 else (ts or '')
            pnl_c = 'win' if (pnl or 0) > 0 else 'lose'
            src_badge = f'badge-{(src or "bot").lower()}'
            mode_badge = f'badge-{(mode or "v12").lower()}'
            etype_badge = f'badge-{(etype or "fc").lower()}'
            dir_disp = '🟢 L' if direction == 'LONG' else '🔴 S'
            hold_min = (hold or 0) / 60
            hold_str = f"{hold_min:.0f}m" if hold_min < 60 else f"{hold_min/60:.1f}h"
            trade_rows += (
                f"<tr><td>{start_idx+i}</td><td>{ts_short}</td>"
                f"<td><span class='badge {src_badge}'>{src or 'BOT'}</span></td>"
                f"<td><span class='badge {mode_badge}'>{mode or '?'}</span></td>"
                f"<td>{dir_disp}</td>"
                f"<td>${(entry or 0):.3f}</td><td>${(exit_px or 0):.3f}</td>"
                f"<td><span class='badge {etype_badge}'>{etype or '?'}</span></td>"
                f"<td class='{pnl_c}'>${(pnl or 0):+,.2f}</td>"
                f"<td class='{pnl_c}'>{(roi or 0):+.2f}%</td>"
                f"<td>{hold_str}</td></tr>"
            )
        if not trade_rows:
            trade_rows = "<tr><td colspan='11' class='empty'>거래 없음 - V12 크로스 또는 Mass bulge 대기 중</td></tr>"

        # 페이지네이션 렌더
        pagination = ""
        if total_pages > 1:
            if page > 1:
                pagination += f'<a href="/trades?page={page-1}">◀ 이전</a>'
            for p in range(max(1, page-2), min(total_pages+1, page+3)):
                if p == page:
                    pagination += f'<span class="current">{p}</span>'
                else:
                    pagination += f'<a href="/trades?page={p}">{p}</a>'
            if page < total_pages:
                pagination += f'<a href="/trades?page={page+1}">다음 ▶</a>'
        else:
            pagination = f'<span>전체 {total_trades_db}건</span>'

        repl = {
            '%%TOTAL%%': str(core.total_trades),
            '%%WINS%%': str(core.win_count),
            '%%LOSSES%%': str(core.loss_count),
            '%%WR%%': f"{wr:.1f}",
            '%%WR_CLASS%%': wr_class,
            '%%PF%%': pf_str,
            '%%TOTAL_PNL%%': f"{total_pnl:+,.2f}",
            '%%PNL_CLASS%%': pnl_class,
            '%%GP%%': f"{core.gross_profit:,.2f}",
            '%%GL%%': f"{core.gross_loss:,.2f}",
            '%%SL%%': str(core.sl_count),
            '%%TSL%%': str(core.tsl_count),
            '%%REV%%': str(core.rev_count),
            '%%EXT%%': str(ext_count),
            '%%PAGE%%': f"{page}/{total_pages}",
            '%%TRADE_ROWS%%': trade_rows,
            '%%PAGINATION%%': pagination,
        }
        html = HTML_TRADES
        for k, v in repl.items():
            html = html.replace(k, str(v))
        return html

    def _load_daily_summary(self, max_days: int = 30) -> list:
        """daily_summary.log 파일에서 최근 N일 파싱"""
        path = Path('logs/daily_summary.log')
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding='utf-8').strip().splitlines()
        except Exception:
            return []

        out = []
        for line in reversed(lines[-max_days:]):
            # 형식: 2026-04-22 17:33:10 2026-04-21 | StartBal $5,000.00 | EndBal $5,020.00 (+0.40%) | Trades 2 (W1/L1, WR 50%) | SOL_PnL $+20.00 | ...
            try:
                parts = line.split(' | ')
                if len(parts) < 5:
                    continue
                # parts[0]: "2026-04-22 17:33:10 2026-04-21"
                date_part = parts[0].split()[-1] if len(parts[0].split()) >= 3 else parts[0]
                start_bal = parts[1].replace('StartBal', '').replace('$', '').replace(',', '').strip()
                end_info = parts[2].replace('EndBal', '').strip()
                # "$5,020.00 (+0.40%)"
                end_split = end_info.split('(')
                end_bal = end_split[0].replace('$', '').replace(',', '').strip()
                pct = end_split[1].rstrip(')').strip() if len(end_split) > 1 else '0%'
                trades_info = parts[3].replace('Trades', '').strip()
                # "2 (W1/L1, WR 50%)"
                t_count = trades_info.split(' ')[0]
                wl = trades_info.split('(')[1].rstrip(')') if '(' in trades_info else '-'
                sol_pnl = parts[4].replace('SOL_PnL', '').replace('$', '').strip()
                consec = parts[5].replace('ConsecLoss', '').strip() if len(parts) > 5 else '0'
                out.append({
                    'date': date_part, 'start': start_bal, 'end': end_bal, 'pct': pct,
                    'trades': t_count, 'wl': wl, 'pnl': sol_pnl, 'consec': consec,
                })
            except Exception:
                continue
        return out[:max_days]

    def _render_balance(self) -> str:
        import math as _m
        from sol_core_v1 import LEVERAGE, MAX_CAPITAL, COMPOUND_PCT, DAILY_LOSS_LIMIT
        core = self.bot.core
        ex = self.bot.executor

        # SOL 전용 누적 PnL
        cum_pnl = core.cumulative_sol_pnl
        cum_class = 'win' if cum_pnl > 0 else ('lose' if cum_pnl < 0 else '')

        # 오늘 SOL PnL
        today_pnl = core.cumulative_sol_pnl - core.day_start_sol_pnl
        today_class = 'win' if today_pnl > 0 else ('lose' if today_pnl < 0 else '')

        # Direction label
        last_exit = 'LONG' if core.last_exit_dir == 1 else ('SHORT' if core.last_exit_dir == -1 else '-')

        # Consec class
        consec_class = 'lose' if core.consec_losses >= 3 else ('warn' if core.consec_losses >= 2 else '')
        mdd_class = 'lose' if core.max_drawdown > 0.30 else ('warn' if core.max_drawdown > 0.15 else '')

        # Daily rows
        daily_data = self._load_daily_summary(30)
        daily_rows = ""
        for d in daily_data:
            # pct color
            try:
                pct_val = float(d['pct'].replace('%', '').replace('+', ''))
                pct_class = 'win' if pct_val > 0 else ('lose' if pct_val < 0 else '')
            except Exception:
                pct_class = ''
            try:
                pnl_val = float(d['pnl'].replace(',', '').replace('+', ''))
                pnl_class = 'win' if pnl_val > 0 else ('lose' if pnl_val < 0 else '')
            except Exception:
                pnl_class = ''
            daily_rows += (
                f"<tr><td>{d['date']}</td>"
                f"<td>${d['start']}</td>"
                f"<td>${d['end']}</td>"
                f"<td class='{pct_class}'>{d['pct']}</td>"
                f"<td>{d['trades']}</td>"
                f"<td>{d['wl']}</td>"
                f"<td class='{pnl_class}'>${d['pnl']}</td>"
                f"<td>{d['consec']}</td></tr>"
            )
        if not daily_rows:
            daily_rows = "<tr><td colspan='8' style='text-align:center;color:#64748b;padding:40px'>아직 일일 요약 데이터 없음 (UTC 자정 이후 첫 기록 생성)</td></tr>"

        repl = {
            '%%BALANCE%%': f"{ex.balance:,.2f}",
            '%%AVAILABLE%%': f"{ex.available_balance:,.2f}",
            '%%PEAK%%': f"{core.peak_capital:,.2f}",
            '%%CUMULATIVE%%': f"{cum_pnl:+,.2f}",
            '%%CUM_CLASS%%': cum_class,
            '%%TODAY_PNL%%': f"{today_pnl:+,.2f}",
            '%%TODAY_CLASS%%': today_class,
            '%%TODAY_TRADES%%': str(core.daily_trade_count),
            '%%DAY_START%%': f"{core.day_start_balance:,.2f}",
            '%%LEVERAGE%%': str(LEVERAGE),
            '%%MAX_CAP%%': f"{MAX_CAPITAL:,.0f}",
            '%%COMPOUND%%': f"{COMPOUND_PCT*100:.1f}",
            '%%DAILY_LIMIT%%': f"{DAILY_LOSS_LIMIT*100:.1f}",
            '%%CONSEC%%': str(core.consec_losses),
            '%%CONSEC_CLASS%%': consec_class,
            '%%SKIP%%': str(core.skip_remaining),
            '%%MDD%%': f"{core.max_drawdown*100:.2f}",
            '%%MDD_CLASS%%': mdd_class,
            '%%LAST_EXIT%%': last_exit,
            '%%DAILY_COUNT%%': str(len(daily_data)),
            '%%DAILY_ROWS%%': daily_rows,
        }
        html = HTML_BALANCE
        for k, v in repl.items():
            html = html.replace(k, str(v))
        return html

    def _render_errors(self) -> str:
        """최근 error_*.log 파일 파싱 + HTML 렌더링"""
        import glob
        import html as _html
        log_files = sorted(glob.glob('logs/error_*.log'), reverse=True)
        errors_lines = []
        for lf in log_files[:3]:  # 최근 3일
            try:
                with open(lf, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.rstrip()
                        if not s:
                            continue
                        level = 'WARNING' if 'WARNING' in s else ('ERROR' if 'ERROR' in s else 'OTHER')
                        errors_lines.append({
                            'file': os.path.basename(lf),
                            'line': s,
                            'level': level,
                        })
            except Exception:
                continue
        # 최신순 (역순)
        errors_lines = errors_lines[::-1][:100]
        total = len(errors_lines)
        err_count = sum(1 for e in errors_lines if e['level'] == 'ERROR')
        warn_count = sum(1 for e in errors_lines if e['level'] == 'WARNING')

        log_html = ""
        if not errors_lines:
            log_html = '<div class="empty"><div class="empty-big">✅</div>에러/경고 없음 — 봇이 건강하게 작동 중</div>'
        else:
            for e in errors_lines:
                log_html += (
                    f'<div class="log-line level-{e["level"]}">'
                    f'<span class="file-tag">{_html.escape(e["file"])}</span>'
                    f'{_html.escape(e["line"])}'
                    f'</div>'
                )

        repl = {
            '%%TOTAL%%': str(total),
            '%%ERRORS%%': str(err_count),
            '%%WARNINGS%%': str(warn_count),
            '%%LOG_LINES%%': log_html,
        }
        html = HTML_ERRORS
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

        @self.app.get('/trades', response_class=HTMLResponse)
        async def trades_page(request: Request, page: int = 1):
            if not self._is_auth(request):
                return RedirectResponse('/login', status_code=303)
            html = await self._render_trades(page=max(1, page))
            return HTMLResponse(html)

        @self.app.get('/balance', response_class=HTMLResponse)
        async def balance_page(request: Request):
            if not self._is_auth(request):
                return RedirectResponse('/login', status_code=303)
            return HTMLResponse(self._render_balance())

        @self.app.get('/errors', response_class=HTMLResponse)
        async def errors_page(request: Request):
            if not self._is_auth(request):
                return RedirectResponse('/login', status_code=303)
            return HTMLResponse(self._render_errors())

        @self.app.get('/chart', response_class=HTMLResponse)
        async def chart_page(request: Request):
            if not self._is_auth(request):
                return RedirectResponse('/login', status_code=303)
            return HTMLResponse(HTML_CHART)

        @self.app.get('/api/status')
        async def api_status(request: Request):
            if not self._is_auth(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            core = self.bot.core
            ex = self.bot.executor
            data = self.bot.data
            # WebSocket 실시간 가격
            price = 0.0
            if data and data.current_price > 0:
                price = float(data.current_price)
            elif data and data.df_sol is not None and len(data.df_sol) > 0:
                price = float(data.df_sol['close'].iloc[-1])
            # ★ 서버 df_sol의 마지막 봉 OHLC (클라이언트 추정 제거)
            KST_OFFSET = 9 * 3600
            last_bar_info = None
            if data and data.df_sol is not None and len(data.df_sol) > 0:
                last = data.df_sol.iloc[-1]
                ts = data.df_sol.index[-1]
                last_bar_info = {
                    'time': int(ts.timestamp()) + KST_OFFSET,
                    'open': round(float(last['open']), 3),
                    'high': round(float(last['high']), 3),
                    'low': round(float(last['low']), 3),
                    'close': round(float(last['close']), 3),
                }
            return JSONResponse({
                'price': price,
                'last_bar': last_bar_info,
                'balance': ex.balance, 'available': ex.available_balance,
                'peak': core.peak_capital, 'mdd': core.max_drawdown,
                'total_trades': core.total_trades, 'win_rate': core.win_rate,
                'pf': core.profit_factor if core.profit_factor != float('inf') else 999,
                'consec_losses': core.consec_losses, 'skip_remaining': core.skip_remaining,
                'has_position': core.has_position,
            })

        @self.app.get('/api/candles')
        async def api_candles(request: Request, limit: int = 17280):
            """15분봉 캔들 + EMA9/SMA400 (기본 6개월 = 17,280봉)"""
            if not self._is_auth(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            try:
                bot = self.bot
                if not bot.data or bot.data.df_sol is None:
                    return {'candles': [], 'ema9': [], 'sma400': []}
                df = bot.data.df_sol
                n = len(df)
                count = min(limit, n) if limit > 0 else n
                start = n - count
                KST_OFFSET = 9 * 3600
                candles = []
                for i in range(start, n):
                    ts = int(df.index[i].timestamp()) + KST_OFFSET
                    candles.append({
                        'time': ts,
                        'open': round(float(df['open'].iloc[i]), 3),
                        'high': round(float(df['high'].iloc[i]), 3),
                        'low': round(float(df['low'].iloc[i]), 3),
                        'close': round(float(df['close'].iloc[i]), 3),
                    })
                ema9_data = []
                sma400_data = []
                if bot.data.fast_ma is not None and bot.data.slow_ma is not None:
                    for i in range(start, n):
                        ts = int(df.index[i].timestamp()) + KST_OFFSET
                        v9 = float(bot.data.fast_ma[i])
                        v400 = float(bot.data.slow_ma[i])
                        if v9 > 0 and not math.isnan(v9):
                            ema9_data.append({'time': ts, 'value': round(v9, 3)})
                        if v400 > 0 and not math.isnan(v400):
                            sma400_data.append({'time': ts, 'value': round(v400, 3)})
                # 최근 거래 마커 (SL/TSL/REV만, 최대 100개)
                markers = []
                try:
                    async with aiosqlite.connect(DB_PATH) as db:
                        cur = await db.execute(
                            "SELECT timestamp, direction, entry_price, exit_price, exit_type, pnl "
                            "FROM trades ORDER BY id DESC LIMIT 100")
                        rows = await cur.fetchall()
                        for r in rows:
                            try:
                                from datetime import datetime as _dt
                                t = _dt.strptime(r[0], '%Y-%m-%d %H:%M:%S').timestamp() + KST_OFFSET
                                is_long = r[1] == 'LONG'
                                pnl = r[5] or 0
                                color = '#22c55e' if pnl > 0 else '#ef4444'
                                markers.append({
                                    'time': int(t),
                                    'position': 'belowBar' if is_long else 'aboveBar',
                                    'color': color,
                                    'shape': 'arrowUp' if is_long else 'arrowDown',
                                    'text': f"{r[4]} {'+' if pnl>0 else ''}{pnl:.1f}",
                                })
                            except Exception:
                                continue
                except Exception:
                    pass
                # 현재 포지션
                position = None
                if bot.core and bot.core.has_position:
                    p = bot.core.position
                    position = {
                        'entry_price': p.entry_price,
                        'sl_price': p.sl_price,
                        'direction': p.direction,
                        'mode': 'V12' if p.entry_mode == 1 else 'MASS',
                    }
                return {
                    'candles': candles,
                    'ema9': ema9_data,
                    'sma400': sma400_data,
                    'markers': markers,
                    'position': position,
                    'total_bars': n,
                    'returned_bars': count,
                }
            except Exception as e:
                logger.error(f"candles API: {e}")
                return {'candles': [], 'ema9': [], 'sma400': [], 'error': str(e)}

        @self.app.get('/api/errors')
        async def api_errors(request: Request, limit: int = 100):
            """error_*.log 파일에서 최근 에러/경고"""
            if not self._is_auth(request):
                return JSONResponse({'error': 'unauthorized'}, status_code=401)
            try:
                import glob
                log_files = sorted(glob.glob('logs/error_*.log'), reverse=True)
                errors = []
                for lf in log_files[:3]:  # 최근 3일
                    try:
                        with open(lf, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                level = 'WARNING' if 'WARNING' in line else ('ERROR' if 'ERROR' in line else 'OTHER')
                                errors.append({'file': os.path.basename(lf), 'line': line, 'level': level})
                    except Exception:
                        continue
                return {'errors': errors[-limit:][::-1], 'total': len(errors)}
            except Exception as e:
                return {'errors': [], 'error': str(e)}

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
