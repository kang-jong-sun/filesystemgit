# -*- coding: utf-8 -*-
"""46개 기획서에서 백테스트 파라미터 자동 추출"""
import os, re, json

SPEC_DIR = 'D:/filesystem/futures/btc_V1/test3/_backup_originals'

def extract(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        c = f.read()

    p = {}

    # Version
    m = re.search(r'v(\d+[\d.]*\w*)', os.path.basename(filepath))
    p['version'] = m.group(1) if m else '?'

    # Timeframe
    if '5분봉' in c and '30분봉' not in c:
        p['tf'] = '5m'
    elif '15분봉' in c and '30분봉' not in c:
        p['tf'] = '15m'
    else:
        p['tf'] = '30m'

    # Fast MA
    fm = re.search(r'(?:FAST_MA|Fast)\s*(?:_TYPE)?\s*=?\s*[\'"]?(EMA|SMA|HMA|WMA|DEMA)', c, re.I)
    p['fast_type'] = fm.group(1).upper() if fm else 'EMA'
    fm2 = re.search(r'(?:FAST_MA|Fast)(?:_PERIOD)?\s*=\s*(\d+)', c)
    if not fm2:
        fm2 = re.search(r'EMA\s*\(\s*(\d+)\s*\)\s*/\s*EMA', c)
    if not fm2:
        fm2 = re.search(r'(?:Fast|fast)\s*(?:MA)?\s*[:=]\s*(\d+)', c)
    p['fast_period'] = int(fm2.group(1)) if fm2 else 3

    # Slow MA
    sm = re.search(r'(?:SLOW_MA|Slow)\s*(?:_TYPE)?\s*=?\s*[\'"]?(EMA|SMA|HMA|WMA|DEMA)', c, re.I)
    p['slow_type'] = sm.group(1).upper() if sm else 'EMA'
    sm2 = re.search(r'(?:SLOW_MA|Slow)(?:_PERIOD)?\s*=\s*(\d+)', c)
    if not sm2:
        sm2 = re.search(r'EMA\s*\(\s*\d+\s*\)\s*/\s*EMA\s*\(\s*(\d+)', c)
    if not sm2:
        sm2 = re.search(r'(?:Slow|slow)\s*(?:MA)?\s*[:=]\s*(\d+)', c)
    p['slow_period'] = int(sm2.group(1)) if sm2 else 200

    # ADX
    adx_p = re.search(r'ADX(?:_PERIOD)?\s*=\s*(\d+)', c)
    p['adx_period'] = int(adx_p.group(1)) if adx_p else 20
    adx_m = re.search(r'ADX(?:_MIN)?\s*(?:>=?\s*|=\s*)(\d+)', c)
    if not adx_m:
        adx_m = re.search(r'ADX\s*[>]=?\s*(\d+)', c)
    p['adx_min'] = float(adx_m.group(1)) if adx_m else 30.0

    # ADX rise bars
    ar = re.search(r'ADX_RISE_BARS\s*=\s*(\d+)', c)
    if not ar:
        ar = re.search(r'av\[i\]\s*>\s*av\[i-(\d+)\]', c)
    p['adx_rise'] = int(ar.group(1)) if ar else 0

    # RSI
    rsi_p = re.search(r'RSI(?:_PERIOD)?\s*=\s*(\d+)', c)
    p['rsi_period'] = int(rsi_p.group(1)) if rsi_p else 14
    rsi_min = re.search(r'RSI(?:_MIN)?\s*(?:>=?\s*|=\s*)(\d+)', c)
    p['rsi_min'] = float(rsi_min.group(1)) if rsi_min else 0.0
    rsi_max = re.search(r'RSI(?:_MAX)?\s*(?:<=?\s*|=\s*)(\d+)', c)
    p['rsi_max'] = float(rsi_max.group(1)) if rsi_max else 100.0

    # SL
    sl = re.search(r'(?:SL|stop.?loss|손절)\s*(?:_PCT|%)?\s*[:=]\s*-?(\d+\.?\d*)', c, re.I)
    if not sl:
        sl = re.search(r'SL\s*(\d+\.?\d*)%', c)
    p['sl_pct'] = float(sl.group(1)) if sl else 3.0

    # TA (TSL activation)
    ta = re.search(r'(?:TA|TRAILING_ACTIVATION|trail.*activ)\s*(?:_PCT|%)?\s*[:=]\s*\+?(\d+\.?\d*)', c, re.I)
    if not ta:
        ta = re.search(r'TA_PCT\s*=\s*(\d+)', c)
    p['ta_pct'] = float(ta.group(1)) if ta else 0.0

    # TSL
    tsl = re.search(r'(?:TSL|TRAILING_STOP|트레일링)\s*(?:_PCT|%)?\s*[:=]\s*-?(\d+\.?\d*)', c, re.I)
    if not tsl:
        tsl = re.search(r'TSL_PCT\s*=\s*(\d+)', c)
    p['tsl_pct'] = float(tsl.group(1)) if tsl else 0.0

    # EMA gap
    gap = re.search(r'(?:EMA_GAP|GAP)(?:_MIN)?\s*=\s*(\d+\.?\d*)', c)
    p['ema_gap'] = float(gap.group(1)) if gap else 0.0

    # Monitor window
    mw = re.search(r'MONITOR(?:_WINDOW)?\s*=\s*(\d+)', c)
    p['monitor'] = int(mw.group(1)) if mw else 0

    # Skip same dir
    p['skip_same'] = 'SKIP_SAME' in c or '동일방향' in c

    # Daily loss limit
    dl = re.search(r'DAILY_LOSS(?:_LIMIT)?\s*=\s*-?(\d+\.?\d*)', c)
    p['daily_loss'] = float(dl.group(1)) if dl else 0.0

    # Margin (original, for reference)
    mg = re.search(r'MARGIN(?:_PCT)?\s*=\s*(\d+\.?\d*)', c)
    p['orig_margin'] = float(mg.group(1)) if mg else 0.2

    # Warmup
    wu = re.search(r'(?:WARMUP|워밍업|start.*index)\s*=?\s*(\d+)', c, re.I)
    p['warmup'] = int(wu.group(1)) if wu else None

    # REV after no continue
    p['rev_no_continue'] = 'CONTINUE 하지 않음' in c or 'NO CONTINUE' in c

    # Fee
    fee = re.search(r'FEE(?:_RATE)?\s*=\s*([\d.]+)', c)
    p['fee'] = float(fee.group(1)) if fee else 0.0004

    # FL protection (forced liquidation)
    p['has_fl_protection'] = 'FL' in c and ('보호' in c or 'protection' in c.lower())

    return p


files = sorted([f for f in os.listdir(SPEC_DIR) if f.startswith('# BTC') and f.endswith('.md')])
print(f"Extracting from {len(files)} specs...")

all_params = []
for f in files:
    fp = os.path.join(SPEC_DIR, f)
    p = extract(fp)
    p['file'] = f
    all_params.append(p)

# Save
with open('D:/filesystem/futures/btc_V1/test3/all_params.json', 'w', encoding='utf-8') as fh:
    json.dump(all_params, fh, indent=2, ensure_ascii=False)

# Print summary
print(f"\n{'Ver':<8} {'TF':<4} {'Fast':<10} {'Slow':<10} {'ADX':>4} {'RSI':<8} {'SL%':>4} {'TA%':>4} {'TSL%':>4} {'Gap':>4} {'Mon':>4} {'Skip':>4} {'Rev':>4}")
print("-"*95)
for p in all_params:
    ft = f"{p['fast_type']}{p['fast_period']}"
    st = f"{p['slow_type']}{p['slow_period']}"
    rsi = f"{p['rsi_min']:.0f}-{p['rsi_max']:.0f}"
    print(f"{p['version']:<8} {p['tf']:<4} {ft:<10} {st:<10} {p['adx_min']:>4.0f} {rsi:<8} {p['sl_pct']:>4.0f} {p['ta_pct']:>4.0f} {p['tsl_pct']:>4.0f} {p['ema_gap']:>4.1f} {p['monitor']:>4} {'Y' if p['skip_same'] else 'N':>4} {'Y' if p['rev_no_continue'] else 'N':>4}")
