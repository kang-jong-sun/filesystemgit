"""
v16.2 FINAL Backtest & 30-Run Verification
- Model A: WMA(3)/SMA(300), ADX(20)>=45, ATR SL*4.0, TP1*8.0, Trail*2.5@6%, Lev10x, M20%
- Model B: HMA(5)/EMA(200), ADX(20)>=45, ATR SL*4.0, TP1*4.0, Trail*2.5@8%, Lev10x, M15%
- 30m timeframe, 75 months, 3-stage partial exit
- 30-run deterministic verification
- Full results output to files
"""
import pandas as pd, numpy as np, os, time, json, warnings, sys, io
warnings.filterwarnings('ignore')
# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIR = r"D:\filesystem\futures\btc_V1\test3"
FEE = 0.0004
INIT = 3000.0

# ============================================================
# INDICATORS
# ============================================================
def wma(s, p):
    w = np.arange(1, p+1, dtype=float)
    return s.rolling(p).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def sma(s, p):
    return s.rolling(p).mean()

def hma(s, p):
    h = max(int(p/2), 1); sq = max(int(np.sqrt(p)), 1)
    return wma(2*wma(s, h) - wma(s, p), sq)

def rsi_calc(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(alpha=1/p, min_periods=p).mean()
    al = l.ewm(alpha=1/p, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def adx_calc(h, l, c, p=20):
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, min_periods=p).mean()
    um = h - h.shift(1); dm = l.shift(1) - l
    pd_ = pd.Series(np.where((um>dm)&(um>0), um, 0), index=h.index).ewm(alpha=1/p, min_periods=p).mean()
    md_ = pd.Series(np.where((dm>um)&(dm>0), dm, 0), index=h.index).ewm(alpha=1/p, min_periods=p).mean()
    pdi = 100*pd_/atr.replace(0, np.nan)
    mdi = 100*md_/atr.replace(0, np.nan)
    dx = 100*abs(pdi-mdi)/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/p, min_periods=p).mean(), atr

def atr_calc(h, l, c, p=14):
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p).mean()

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    fs = [os.path.join(DIR, f"btc_usdt_5m_2020_to_now_part{i}.csv") for i in range(1, 4)]
    df = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in fs], ignore_index=True)
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    return df

def build_30m(df5):
    d30 = df5.resample('30min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    return d30

def compute_indicators(d30):
    """Compute indicators for both models on 30m data"""
    c = d30['close']; h = d30['high']; l = d30['low']

    # Model A: WMA(3) / SMA(300)
    d30['a_fast'] = wma(c, 3)
    d30['a_slow'] = sma(c, 300)

    # Model B: HMA(5) / EMA(200)
    d30['b_fast'] = hma(c, 5)
    d30['b_slow'] = ema(c, 200)

    # Common: ADX(20), RSI(14), ATR(14)
    d30['adx'], d30['adx_atr'] = adx_calc(h, l, c, 20)
    d30['rsi'] = rsi_calc(c, 14)
    d30['atr'] = atr_calc(h, l, c, 14)
    d30['atr50'] = d30['atr'].rolling(50).mean()

    # Trend states
    d30['a_trend'] = (d30['a_fast'] > d30['a_slow']).astype(int)
    d30['b_trend'] = (d30['b_fast'] > d30['b_slow']).astype(int)

    return d30

# ============================================================
# MODEL CONFIGS
# ============================================================
MODEL_A = {
    'name': 'Model_A', 'label': 'Aggressive',
    'fast_col': 'a_fast', 'slow_col': 'a_slow', 'trend_col': 'a_trend',
    'adx_min': 45, 'rsi_lo': 30, 'rsi_hi': 65,
    'lev': 10, 'ratio': 0.20,
    'atr_sl': 4.0, 'atr_tp1': 8.0, 'atr_tp2': 16.0,
    'trail_atr': 2.5, 'trail_act': 0.06,
    'sl_max': 0.05, 'sl_min': 0.015,
    'gap': 12,  # 12 * 30m = 6 hours
    'capital_pct': 0.60,
}

MODEL_B = {
    'name': 'Model_B', 'label': 'Stable',
    'fast_col': 'b_fast', 'slow_col': 'b_slow', 'trend_col': 'b_trend',
    'adx_min': 45, 'rsi_lo': 25, 'rsi_hi': 65,
    'lev': 10, 'ratio': 0.15,
    'atr_sl': 4.0, 'atr_tp1': 4.0, 'atr_tp2': 8.0,
    'trail_atr': 2.5, 'trail_act': 0.08,
    'sl_max': 0.04, 'sl_min': 0.015,
    'gap': 24,  # 24 * 30m = 12 hours
    'capital_pct': 0.40,
}

# ============================================================
# POSITION CLASS
# ============================================================
class Position:
    def __init__(self, model, direction, entry_price, entry_time,
                 size, leverage, sl, tp1, tp2, atr_val, sl_pct):
        self.model = model
        self.direction = direction  # 'LONG' or 'SHORT'
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size  # notional
        self.leverage = leverage
        self.sl = sl
        self.tp1 = tp1
        self.tp2 = tp2
        self.atr_val = atr_val
        self.sl_pct = sl_pct

        self.highest = entry_price
        self.lowest = entry_price
        self.peak_roi = 0.0
        self.tp1_hit = False
        self.tp2_hit = False
        self.remaining = 1.0
        self.trail_active = False
        self.trail_sl = None
        self.partial_profits = 0.0

# ============================================================
# SINGLE BACKTEST RUN
# ============================================================
def run_backtest(d30, return_trades=True):
    """Run one complete backtest. Returns (metrics_dict, trades_list)"""
    hi = d30['high'].values
    lo = d30['low'].values
    cl = d30['close'].values
    ts = d30.index
    n = len(d30)

    adx_arr = d30['adx'].values
    rsi_arr = d30['rsi'].values
    atr_arr = d30['atr'].values
    atr50_arr = d30['atr50'].values
    a_trend = d30['a_trend'].values
    b_trend = d30['b_trend'].values

    # State
    bal_a = INIT * MODEL_A['capital_pct']  # $1,800
    bal_b = INIT * MODEL_B['capital_pct']  # $1,200
    peak_a = bal_a
    peak_b = bal_b
    positions = []
    trades = []
    last_entry = {'Model_A': -999, 'Model_B': -999}
    consec_loss = {'Model_A': 0, 'Model_B': 0}
    cooldown = {'Model_A': None, 'Model_B': None}
    monthly_start = {}
    balance_history = []

    warmup = 350  # Need 300+ bars for SMA(300)

    for i in range(warmup, n):
        t = ts[i]; h = hi[i]; l = lo[i]; c = cl[i]

        # ---------- Position Management ----------
        to_remove = []
        for pi, p in enumerate(positions):
            xr = None; xp = c
            cfg = MODEL_A if p.model == 'Model_A' else MODEL_B
            bal = bal_a if p.model == 'Model_A' else bal_b

            if p.direction == 'LONG':
                p.highest = max(p.highest, h)
                cur_roi = (c - p.entry_price) / p.entry_price
                p.peak_roi = max(p.peak_roi, (p.highest - p.entry_price) / p.entry_price)

                # SL
                if l <= p.sl:
                    xp = p.sl; xr = 'SL'
                else:
                    # TP1: 30% partial
                    if not p.tp1_hit and h >= p.tp1:
                        ps = p.size * 0.30
                        pp = ps * (p.tp1 - p.entry_price) / p.entry_price - ps * FEE
                        p.partial_profits += pp
                        p.remaining -= 0.30
                        p.tp1_hit = True
                        p.sl = p.entry_price  # Move to breakeven

                    # TP2: 30% partial
                    if p.tp1_hit and not p.tp2_hit and h >= p.tp2:
                        ps = p.size * 0.30
                        pp = ps * (p.tp2 - p.entry_price) / p.entry_price - ps * FEE
                        p.partial_profits += pp
                        p.remaining -= 0.30
                        p.tp2_hit = True
                        p.sl = p.tp1  # Move SL to TP1

                    # Trailing stop
                    if cur_roi >= cfg['trail_act'] or p.tp1_hit:
                        p.trail_active = True
                        new_tsl = p.highest - p.atr_val * cfg['trail_atr']
                        if p.trail_sl is None or new_tsl > p.trail_sl:
                            p.trail_sl = max(new_tsl, p.sl)

                    if p.trail_active and p.trail_sl and l <= p.trail_sl:
                        xp = p.trail_sl; xr = 'TRAIL'

            else:  # SHORT
                p.lowest = min(p.lowest, l)
                cur_roi = (p.entry_price - c) / p.entry_price
                p.peak_roi = max(p.peak_roi, (p.entry_price - p.lowest) / p.entry_price)

                if h >= p.sl:
                    xp = p.sl; xr = 'SL'
                else:
                    if not p.tp1_hit and l <= p.tp1:
                        ps = p.size * 0.30
                        pp = ps * (p.entry_price - p.tp1) / p.entry_price - ps * FEE
                        p.partial_profits += pp
                        p.remaining -= 0.30
                        p.tp1_hit = True
                        p.sl = p.entry_price

                    if p.tp1_hit and not p.tp2_hit and l <= p.tp2:
                        ps = p.size * 0.30
                        pp = ps * (p.entry_price - p.tp2) / p.entry_price - ps * FEE
                        p.partial_profits += pp
                        p.remaining -= 0.30
                        p.tp2_hit = True
                        p.sl = p.tp1

                    if cur_roi >= cfg['trail_act'] or p.tp1_hit:
                        p.trail_active = True
                        new_tsl = p.lowest + p.atr_val * cfg['trail_atr']
                        if p.trail_sl is None or new_tsl < p.trail_sl:
                            p.trail_sl = min(new_tsl, p.sl)

                    if p.trail_active and p.trail_sl and h >= p.trail_sl:
                        xp = p.trail_sl; xr = 'TRAIL'

            # Close position
            if xr:
                rs = p.size * p.remaining
                if p.direction == 'LONG':
                    rpnl = rs * (xp - p.entry_price) / p.entry_price
                else:
                    rpnl = rs * (p.entry_price - xp) / p.entry_price
                total_pnl = rpnl + p.partial_profits - rs * FEE

                if p.model == 'Model_A':
                    bal_a += total_pnl
                    peak_a = max(peak_a, bal_a)
                else:
                    bal_b += total_pnl
                    peak_b = max(peak_b, bal_b)

                if total_pnl < 0:
                    consec_loss[p.model] += 1
                else:
                    consec_loss[p.model] = 0

                margin = p.size / p.leverage
                roi_pct = total_pnl / margin * 100 if margin > 0 else 0
                hold_min = (t - p.entry_time).total_seconds() / 60

                if return_trades:
                    trades.append({
                        'model': p.model,
                        'direction': p.direction,
                        'entry_time': str(p.entry_time),
                        'exit_time': str(t),
                        'entry_price': round(p.entry_price, 2),
                        'exit_price': round(xp, 2),
                        'exit_reason': xr,
                        'size': round(p.size, 2),
                        'leverage': p.leverage,
                        'margin': round(margin, 2),
                        'pnl': round(total_pnl, 2),
                        'roi_pct': round(roi_pct, 2),
                        'peak_roi_pct': round(p.peak_roi * 100, 2),
                        'tp1_hit': p.tp1_hit,
                        'tp2_hit': p.tp2_hit,
                        'sl_pct': round(p.sl_pct * 100, 2),
                        'hold_hours': round(hold_min / 60, 2),
                        'balance_a': round(bal_a, 2),
                        'balance_b': round(bal_b, 2),
                        'balance_total': round(bal_a + bal_b, 2),
                    })
                to_remove.append(pi)

        for pi in sorted(to_remove, reverse=True):
            positions.pop(pi)

        # Balance history (every 6 bars = 3 hours)
        if i % 6 == 0:
            balance_history.append({
                'timestamp': str(t),
                'balance_a': round(bal_a, 2),
                'balance_b': round(bal_b, 2),
                'total': round(bal_a + bal_b, 2),
            })

        # ---------- Entry Logic ----------
        for cfg in [MODEL_A, MODEL_B]:
            model_name = cfg['name']
            bal = bal_a if model_name == 'Model_A' else bal_b
            pk = peak_a if model_name == 'Model_A' else peak_b

            # Cooldown check
            if cooldown[model_name] and t < cooldown[model_name]:
                continue
            if cooldown[model_name] and t >= cooldown[model_name]:
                cooldown[model_name] = None
                # Reset peak after cooldown
                if model_name == 'Model_A':
                    peak_a = bal_a
                else:
                    peak_b = bal_b
                pk = bal

            # DD check
            dd = (bal - pk) / pk if pk > 0 else 0
            if dd < -0.45:
                cooldown[model_name] = t + pd.Timedelta(hours=48)
                continue

            # Monthly loss limit
            mk = f"{t.year}-{t.month:02d}"
            mk_key = f"{model_name}_{mk}"
            if mk_key not in monthly_start:
                monthly_start[mk_key] = bal
            if monthly_start[mk_key] > 0 and (bal - monthly_start[mk_key]) / monthly_start[mk_key] < -0.20:
                continue

            # Min gap
            if i - last_entry[model_name] < cfg['gap']:
                continue

            # Already have position?
            if any(p.model == model_name for p in positions):
                continue

            # Balance too low
            if bal < 10:
                continue

            # Trend direction
            trend = a_trend[i] if model_name == 'Model_A' else b_trend[i]
            if np.isnan(trend):
                continue
            direction = 'LONG' if trend > 0.5 else 'SHORT'

            # ADX filter
            adx_v = adx_arr[i]
            if np.isnan(adx_v) or adx_v < cfg['adx_min']:
                continue

            # RSI filter
            rsi_v = rsi_arr[i]
            if np.isnan(rsi_v) or not (cfg['rsi_lo'] <= rsi_v <= cfg['rsi_hi']):
                continue

            # ATR
            atr_v = atr_arr[i]
            if np.isnan(atr_v) or atr_v <= 0:
                continue

            # Volatility regime adjustment
            atr50_v = atr50_arr[i]
            vol_ratio = atr_v / atr50_v if (not np.isnan(atr50_v) and atr50_v > 0) else 1.0
            regime_mult = 0.7 if vol_ratio > 1.3 else (1.2 if vol_ratio < 0.7 else 1.0)

            # DD size adjustment
            dd_mult = 0.5 if dd < -0.25 else (0.75 if dd < -0.15 else 1.0)

            # Streak adjustment
            cl_count = consec_loss[model_name]
            streak_mult = max(0.4, 1.0 - cl_count * 0.10)

            # Close opposite direction position
            for pi in sorted([pi for pi, p in enumerate(positions)
                             if p.model == model_name and p.direction != direction], reverse=True):
                p = positions[pi]
                rs = p.size * p.remaining
                if p.direction == 'LONG':
                    rpnl = rs * (c - p.entry_price) / p.entry_price
                else:
                    rpnl = rs * (p.entry_price - c) / p.entry_price
                total_pnl = rpnl + p.partial_profits - rs * FEE

                if model_name == 'Model_A':
                    bal_a += total_pnl; bal = bal_a; peak_a = max(peak_a, bal_a)
                else:
                    bal_b += total_pnl; bal = bal_b; peak_b = max(peak_b, bal_b)

                consec_loss[model_name] = consec_loss[model_name] + 1 if total_pnl < 0 else 0
                margin = p.size / p.leverage
                if return_trades:
                    trades.append({
                        'model': p.model, 'direction': p.direction,
                        'entry_time': str(p.entry_time), 'exit_time': str(t),
                        'entry_price': round(p.entry_price, 2), 'exit_price': round(c, 2),
                        'exit_reason': 'REVERSE', 'size': round(p.size, 2),
                        'leverage': p.leverage, 'margin': round(margin, 2),
                        'pnl': round(total_pnl, 2),
                        'roi_pct': round(total_pnl / margin * 100 if margin > 0 else 0, 2),
                        'peak_roi_pct': round(p.peak_roi * 100, 2),
                        'tp1_hit': p.tp1_hit, 'tp2_hit': p.tp2_hit,
                        'sl_pct': round(p.sl_pct * 100, 2),
                        'hold_hours': round((t - p.entry_time).total_seconds() / 3600, 2),
                        'balance_a': round(bal_a, 2), 'balance_b': round(bal_b, 2),
                        'balance_total': round(bal_a + bal_b, 2),
                    })
                positions.pop(pi)

            # Position sizing
            size = bal * cfg['ratio'] * regime_mult * dd_mult * streak_mult
            if size < 5:
                continue
            lev = cfg['lev']
            notional = size * lev

            # SL/TP calculation
            sl_pct = max(cfg['sl_min'], min(cfg['sl_max'], atr_v * cfg['atr_sl'] / c))
            sl_dist = c * sl_pct
            tp1_dist = atr_v * cfg['atr_tp1']
            tp2_dist = atr_v * cfg['atr_tp2']

            if direction == 'LONG':
                sl_p = c - sl_dist
                tp1_p = c + tp1_dist
                tp2_p = c + tp2_dist
            else:
                sl_p = c + sl_dist
                tp1_p = c - tp1_dist
                tp2_p = c - tp2_dist

            # Entry fee
            if model_name == 'Model_A':
                bal_a -= notional * FEE
            else:
                bal_b -= notional * FEE

            positions.append(Position(
                model_name, direction, c, t, notional, lev,
                sl_p, tp1_p, tp2_p, atr_v, sl_pct
            ))
            last_entry[model_name] = i

    # ---------- Compute Metrics ----------
    final_total = bal_a + bal_b
    total_return = (final_total - INIT) / INIT * 100

    df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()

    metrics = {
        'initial_capital': INIT,
        'final_balance_a': round(bal_a, 2),
        'final_balance_b': round(bal_b, 2),
        'final_balance_total': round(final_total, 2),
        'total_return_pct': round(total_return, 2),
        'total_trades': len(trades),
    }

    if len(df_trades) > 0:
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        gp = wins['pnl'].sum() if len(wins) else 0
        gl = abs(losses['pnl'].sum()) if len(losses) else 0

        # MDD from balance history
        bals = [INIT] + [t['total'] for t in balance_history]
        pk = bals[0]; mdd = 0
        for b in bals:
            pk = max(pk, b)
            dd = (b - pk) / pk
            mdd = min(mdd, dd)

        # Max consecutive losses
        mc = cc = 0
        for t_rec in trades:
            if t_rec['pnl'] <= 0: cc += 1; mc = max(mc, cc)
            else: cc = 0

        metrics.update({
            'profit_factor': round(gp / gl, 4) if gl > 0 else 999,
            'mdd_pct': round(mdd * 100, 2),
            'win_rate': round(len(wins) / len(df_trades) * 100, 2),
            'avg_win_roi': round(wins['roi_pct'].mean(), 2) if len(wins) else 0,
            'avg_loss_roi': round(losses['roi_pct'].mean(), 2) if len(losses) else 0,
            'rr_ratio': round(abs(wins['roi_pct'].mean() / losses['roi_pct'].mean()), 2) if len(losses) and losses['roi_pct'].mean() != 0 else 999,
            'max_consec_loss': mc,
            'gross_profit': round(gp, 2),
            'gross_loss': round(gl, 2),
            'sl_count': len(df_trades[df_trades['exit_reason'] == 'SL']),
            'trail_count': len(df_trades[df_trades['exit_reason'] == 'TRAIL']),
            'reverse_count': len(df_trades[df_trades['exit_reason'] == 'REVERSE']),
            'tp1_hits': int(df_trades['tp1_hit'].sum()),
            'tp2_hits': int(df_trades['tp2_hit'].sum()),
            'trades_model_a': len(df_trades[df_trades['model'] == 'Model_A']),
            'trades_model_b': len(df_trades[df_trades['model'] == 'Model_B']),
        })

        # Per-model metrics
        for m_name in ['Model_A', 'Model_B']:
            mt = df_trades[df_trades['model'] == m_name]
            if len(mt) == 0: continue
            mw = mt[mt['pnl'] > 0]; ml = mt[mt['pnl'] <= 0]
            mgp = mw['pnl'].sum() if len(mw) else 0
            mgl = abs(ml['pnl'].sum()) if len(ml) else 0
            metrics[f'{m_name}_pf'] = round(mgp / mgl, 4) if mgl > 0 else 999
            metrics[f'{m_name}_wr'] = round(len(mw) / len(mt) * 100, 2)
            metrics[f'{m_name}_pnl'] = round(mt['pnl'].sum(), 2)
            metrics[f'{m_name}_avg_win'] = round(mw['roi_pct'].mean(), 2) if len(mw) else 0
            metrics[f'{m_name}_avg_loss'] = round(ml['roi_pct'].mean(), 2) if len(ml) else 0

    return metrics, trades, balance_history


# ============================================================
# FULL REPORT
# ============================================================
def generate_report(metrics, trades, balance_history, d30):
    """Generate comprehensive report string"""
    lines = []
    L = lines.append

    L("=" * 120)
    L("  v16.2 FINAL BACKTEST RESULTS")
    L("  Dual Model: A(WMA3/SMA300) + B(HMA5/EMA200) | 30m | ADX(20)>=45")
    L("=" * 120)

    L(f"""
  OVERALL PERFORMANCE
  {'='*60}
  Initial Capital:     ${metrics['initial_capital']:,.0f}
  Final Balance A:     ${metrics['final_balance_a']:,.2f}  (60% alloc)
  Final Balance B:     ${metrics['final_balance_b']:,.2f}  (40% alloc)
  Final Total:         ${metrics['final_balance_total']:,.2f}
  Total Return:        {metrics['total_return_pct']:+,.1f}%
  Profit Factor:       {metrics.get('profit_factor', 0):.2f}
  MDD:                 {metrics.get('mdd_pct', 0):.1f}%
  Total Trades:        {metrics['total_trades']}
  Win Rate:            {metrics.get('win_rate', 0):.1f}%
  Avg Win ROI:         {metrics.get('avg_win_roi', 0):+.2f}%
  Avg Loss ROI:        {metrics.get('avg_loss_roi', 0):+.2f}%
  R:R Ratio:           {metrics.get('rr_ratio', 0):.2f}
  Max Consec Losses:   {metrics.get('max_consec_loss', 0)}
  Gross Profit:        ${metrics.get('gross_profit', 0):,.2f}
  Gross Loss:          ${metrics.get('gross_loss', 0):,.2f}
  TP1 Hits:            {metrics.get('tp1_hits', 0)} ({metrics.get('tp1_hits', 0)/max(1,metrics['total_trades'])*100:.0f}%)
  TP2 Hits:            {metrics.get('tp2_hits', 0)} ({metrics.get('tp2_hits', 0)/max(1,metrics['total_trades'])*100:.0f}%)
  Exit: SL={metrics.get('sl_count', 0)} TRAIL={metrics.get('trail_count', 0)} REV={metrics.get('reverse_count', 0)}
""")

    # Model breakdown
    L("  MODEL BREAKDOWN")
    L("  " + "=" * 60)
    for mn, ml in [('Model_A', 'Aggressive'), ('Model_B', 'Stable')]:
        pf = metrics.get(f'{mn}_pf', 0)
        wr = metrics.get(f'{mn}_wr', 0)
        pnl = metrics.get(f'{mn}_pnl', 0)
        aw = metrics.get(f'{mn}_avg_win', 0)
        al = metrics.get(f'{mn}_avg_loss', 0)
        tc = metrics.get(f'trades_{mn.lower()}', 0)
        L(f"  {mn} ({ml}): Trades={tc} WR={wr:.1f}% PF={pf:.2f} PnL=${pnl:+,.2f} AvgW={aw:+.1f}% AvgL={al:+.1f}%")

    # Monthly performance
    if trades:
        df = pd.DataFrame(trades)
        df['exit_dt'] = pd.to_datetime(df['exit_time'])
        df['month'] = df['exit_dt'].dt.to_period('M')
        mg = df.groupby('month')

        L(f"\n  MONTHLY PERFORMANCE")
        L("  " + "=" * 110)
        L(f"  {'Month':>8} {'Trd':>4} {'W':>3} {'L':>3} {'WR%':>5} {'GrossP':>10} {'GrossL':>10} {'NetPnL':>10} {'PF':>6} {'Total$':>10} {'MoRet%':>8} {'A':>3} {'B':>3}")
        L("  " + "-" * 110)

        rb = INIT
        yearly = {}
        loss_months = 0
        total_months = 0

        for mo in sorted(mg.groups.keys()):
            g = mg.get_group(mo)
            nt = len(g); nw = len(g[g['pnl'] > 0]); nl = nt - nw
            wr2 = nw/nt*100 if nt else 0
            gp2 = g[g['pnl'] > 0]['pnl'].sum() if nw else 0
            gl2 = abs(g[g['pnl'] <= 0]['pnl'].sum()) if nl else 0
            net = g['pnl'].sum()
            mpf = gp2/gl2 if gl2 > 0 else 999
            sbr = rb; rb += net
            mret = net/sbr*100 if sbr > 0 else 0
            total_months += 1
            if net < 0: loss_months += 1

            ea = len(g[g['model'] == 'Model_A'])
            eb = len(g[g['model'] == 'Model_B'])

            yr = str(mo)[:4]
            if yr not in yearly:
                yearly[yr] = {'p': 0, 't': 0, 'w': 0, 'l': 0, 'gp': 0, 'gl': 0, 'sb': sbr}
            yearly[yr]['p'] += net; yearly[yr]['t'] += nt
            yearly[yr]['w'] += nw; yearly[yr]['l'] += nl
            yearly[yr]['gp'] += gp2; yearly[yr]['gl'] += gl2
            yearly[yr]['eb'] = rb

            pfs = f"{mpf:.1f}" if mpf < 999 else "INF"
            mk = " <<LOSS" if net < 0 else ""
            L(f"  {str(mo):>8} {nt:>4} {nw:>3} {nl:>3} {wr2:>4.0f}% ${gp2:>8,.0f} ${gl2:>8,.0f} ${net:>+8,.0f} {pfs:>5} ${rb:>8,.0f} {mret:>+7.1f}% {ea:>3} {eb:>3}{mk}")

        # Yearly
        L(f"\n  YEARLY PERFORMANCE")
        L("  " + "=" * 90)
        L(f"  {'Year':>6} {'Trd':>5} {'W':>4} {'L':>4} {'WR%':>5} {'GrossP':>11} {'GrossL':>11} {'NetPnL':>11} {'PF':>6} {'YrRet%':>8}")
        L("  " + "-" * 90)

        for yr in sorted(yearly.keys()):
            yd = yearly[yr]
            ywr = yd['w']/yd['t']*100 if yd['t'] else 0
            ypf = yd['gp']/yd['gl'] if yd['gl'] > 0 else 999
            yret = yd['p']/yd['sb']*100 if yd['sb'] > 0 else 0
            pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
            L(f"  {yr:>6} {yd['t']:>4} {yd['w']:>4} {yd['l']:>4} {ywr:>4.0f}% ${yd['gp']:>9,.0f} ${yd['gl']:>9,.0f} ${yd['p']:>+9,.0f} {pfs:>5} {yret:>+7.1f}%")

        pyrs = sum(1 for v in yearly.values() if v['p'] > 0)
        L(f"\n  Loss Months: {loss_months}/{total_months} ({loss_months/max(1,total_months)*100:.0f}%)")
        L(f"  Profitable Years: {pyrs}/{len(yearly)}")

        # Model x Year
        L(f"\n  MODEL x YEAR")
        L("  " + "=" * 80)
        df['year'] = df['exit_dt'].dt.year
        for mn in ['Model_A', 'Model_B']:
            mt = df[df['model'] == mn]
            if len(mt) == 0: continue
            cfg = MODEL_A if mn == 'Model_A' else MODEL_B
            L(f"\n  {mn} ({cfg['label']}) - {cfg['fast_col'].split('_')[0].upper()} / Lev {cfg['lev']}x / M {cfg['ratio']*100:.0f}%:")
            L(f"    {'Year':>6} {'Trd':>4} {'WR%':>5} {'PnL':>10} {'AvgROI':>7} {'PF':>6}")
            for yr in sorted(mt['year'].unique()):
                yt = mt[mt['year'] == yr]
                yw = yt[yt['pnl'] > 0]; yl = yt[yt['pnl'] <= 0]
                ywr = len(yw)/len(yt)*100
                ygp = yw['pnl'].sum() if len(yw) else 0
                ygl = abs(yl['pnl'].sum()) if len(yl) else 0
                ypf = ygp/ygl if ygl > 0 else 999
                pfs = f"{ypf:.2f}" if ypf < 999 else "INF"
                L(f"    {yr:>6} {len(yt):>3} {ywr:>4.0f}% ${yt['pnl'].sum():>+8,.0f} {yt['roi_pct'].mean():>+6.1f}% {pfs:>5}")

        # Entry structure
        L(f"\n  ENTRY STRUCTURE")
        L("  " + "=" * 80)
        for lbl, code in [('LONG', 'LONG'), ('SHORT', 'SHORT')]:
            sub = df[df['direction'] == code]
            if len(sub) == 0: continue
            sw = sub[sub['pnl'] > 0]
            L(f"  {lbl:>5}: {len(sub):>5} ({len(sub)/len(df)*100:.0f}%) WR:{len(sw)/len(sub)*100:.0f}% AvgROI:{sub['roi_pct'].mean():+.2f}% PnL:${sub['pnl'].sum():+,.0f}")

        L(f"\n  [Hold Time]")
        df['hh'] = df['hold_hours']
        for a, b, lb in [(0,1,'<1h'),(1,4,'1-4h'),(4,12,'4-12h'),(12,24,'12-24h'),(24,72,'1-3d'),(72,168,'3-7d'),(168,9999,'7d+')]:
            ht = df[(df['hh'] >= a) & (df['hh'] < b)]
            if len(ht):
                hw = ht[ht['pnl'] > 0]
                L(f"    {lb:>6}: {len(ht):>4} WR:{len(hw)/len(ht)*100:.0f}% AvgROI:{ht['roi_pct'].mean():+.2f}% PnL:${ht['pnl'].sum():+,.0f}")

        # Top/Bottom 10
        ds = df.sort_values('pnl', ascending=False)
        L(f"\n  TOP 10 TRADES")
        L("  " + "-" * 100)
        for idx, (_, r) in enumerate(ds.head(10).iterrows()):
            L(f"    {idx+1:>2} {r['model']:>7} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")

        L(f"\n  BOTTOM 10 TRADES")
        L("  " + "-" * 100)
        for idx, (_, r) in enumerate(ds.tail(10).iterrows()):
            L(f"    {idx+1:>2} {r['model']:>7} {r['direction']:>5} {r['entry_time'][:16]} -> {r['exit_time'][:16]} {r['exit_reason']:>5} ROI:{r['roi_pct']:>+7.1f}% PnL:${r['pnl']:>+8,.0f}")

    L("\n" + "=" * 120)
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 120)
    print("  v16.2 FINAL BACKTEST + 30-RUN VERIFICATION")
    print("=" * 120)

    # Load data
    print("\n[1/4] Loading data...")
    t0 = time.time()
    df5 = load_data()
    d30 = build_30m(df5)
    d30 = compute_indicators(d30)
    print(f"  5m: {len(df5):,} rows | 30m: {len(d30):,} rows | {time.time()-t0:.1f}s")

    # Run main backtest with full trades
    print("\n[2/4] Running main backtest...")
    t0 = time.time()
    metrics, trades, balance_history = run_backtest(d30, return_trades=True)
    print(f"  {time.time()-t0:.1f}s | Trades: {len(trades)} | Final: ${metrics['final_balance_total']:,.2f}")

    # Generate and save report
    print("\n[3/4] Generating report...")
    report_text = generate_report(metrics, trades, balance_history, d30)
    print(report_text)

    # Save results
    report_path = os.path.join(DIR, "v16_2_FINAL_backtest_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    trades_path = os.path.join(DIR, "v16_2_FINAL_trades.csv")
    if trades:
        pd.DataFrame(trades).to_csv(trades_path, index=False, encoding='utf-8-sig')

    balance_path = os.path.join(DIR, "v16_2_FINAL_balance_curve.csv")
    if balance_history:
        pd.DataFrame(balance_history).to_csv(balance_path, index=False, encoding='utf-8-sig')

    metrics_path = os.path.join(DIR, "v16_2_FINAL_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 30-Run verification
    print("\n[4/4] Running 30-run verification...")
    t0 = time.time()
    verification_results = []
    for run_num in range(1, 31):
        m, _, _ = run_backtest(d30, return_trades=False)
        verification_results.append({
            'run': run_num,
            'final_balance': m['final_balance_total'],
            'total_return': m['total_return_pct'],
            'pf': m.get('profit_factor', 0),
            'mdd': m.get('mdd_pct', 0),
            'trades': m['total_trades'],
            'win_rate': m.get('win_rate', 0),
        })
        if run_num % 10 == 0:
            print(f"  Run {run_num}/30 | Bal: ${m['final_balance_total']:,.2f} | PF: {m.get('profit_factor', 0):.2f}")

    vdf = pd.DataFrame(verification_results)
    verify_path = os.path.join(DIR, "v16_2_FINAL_30run_verification.csv")
    vdf.to_csv(verify_path, index=False)

    print(f"\n  30-Run Verification Complete ({time.time()-t0:.1f}s)")
    print(f"  Balance: mean=${vdf['final_balance'].mean():,.2f} std=${vdf['final_balance'].std():,.4f}")
    print(f"  Return:  mean={vdf['total_return'].mean():+.2f}% std={vdf['total_return'].std():.4f}%")
    print(f"  PF:      mean={vdf['pf'].mean():.4f} std={vdf['pf'].std():.6f}")
    print(f"  MDD:     mean={vdf['mdd'].mean():.2f}% std={vdf['mdd'].std():.6f}%")
    print(f"  Trades:  mean={vdf['trades'].mean():.1f} std={vdf['trades'].std():.4f}")
    is_deterministic = vdf['final_balance'].std() < 0.01
    print(f"  Deterministic: {'YES (std < 0.01)' if is_deterministic else 'NO - CHECK FOR BUGS'}")

    print(f"\n  OUTPUT FILES:")
    print(f"    {report_path}")
    print(f"    {trades_path}")
    print(f"    {balance_path}")
    print(f"    {metrics_path}")
    print(f"    {verify_path}")

    print("\n" + "=" * 120)
    print("  BACKTEST + VERIFICATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
