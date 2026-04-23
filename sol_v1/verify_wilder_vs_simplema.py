"""
Wilder RSI/ADX 수정 검증 스크립트
=====================================
1. 봇의 새 Wilder 함수와 engine_data의 Wilder 함수가 동일한지 검증
2. Simple MA(14) vs Wilder(10/20) 차이 정량화 (최근 180일 15m 데이터 기준)
3. V12.1 필터 통과율 비교

실행:
  cd D:\\filesystem\\futures\\sol_v1
  python verify_wilder_vs_simplema.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, r"D:\filesystem\futures\CUDA")
sys.path.insert(0, r"D:\filesystem\futures\sol_v1")

import glob
import numpy as np
import pandas as pd

print("="*80)
print("Wilder RSI/ADX 수정 검증")
print("="*80)

# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드 (봇 cache 또는 51개월 백테스트 데이터)
# ─────────────────────────────────────────────────────────────
cache_dir = r"D:\filesystem\futures\sol_v1\cache"
sol_files = sorted(glob.glob(os.path.join(cache_dir, "sol_5m_*.csv")))

if not sol_files:
    # fallback: 51개월 데이터 사용
    sol_files = sorted(glob.glob(r"D:\filesystem\futures\CUDA\data_51m\sol\sol_5m_*.csv"))
    print(f"봇 cache 없음 → 51개월 데이터 사용: {len(sol_files)}개 파일")
else:
    print(f"봇 cache 사용: {len(sol_files)}개 파일")

dfs = []
for f in sol_files:
    df = pd.read_csv(f, parse_dates=['timestamp'])
    dfs.append(df)
df_5m = pd.concat(dfs, ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
print(f"5m 데이터: {len(df_5m):,} 봉, {df_5m['timestamp'].iloc[0]} ~ {df_5m['timestamp'].iloc[-1]}")

# 최근 180일만
cutoff = df_5m['timestamp'].iloc[-1] - pd.Timedelta(days=180)
df_5m = df_5m[df_5m['timestamp'] >= cutoff].reset_index(drop=True)
print(f"최근 180일: {len(df_5m):,} 봉")

# 15m 리샘플
df_5m = df_5m.set_index('timestamp')
df_15m = df_5m.resample('15min').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna().reset_index()
print(f"15m 리샘플: {len(df_15m):,} 봉")

high = df_15m['high'].values.astype(np.float64)
low = df_15m['low'].values.astype(np.float64)
close = df_15m['close'].values.astype(np.float64)

# ─────────────────────────────────────────────────────────────
# 2. 봇의 새 Wilder (sol_data_v1.py 정의와 동일)
# ─────────────────────────────────────────────────────────────
def bot_wilder_rsi(c, period=10):
    series = pd.Series(c)
    d = series.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    a = 1.0 / period
    ag = g.ewm(alpha=a, min_periods=period, adjust=False).mean()
    al = l.ewm(alpha=a, min_periods=period, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, 1e-10))
    return rsi.fillna(50).values.astype(np.float64)

def bot_wilder_adx(h, l, c, period=20):
    h_s = pd.Series(h); l_s = pd.Series(l); c_s = pd.Series(c)
    a = 1.0 / period
    tr = pd.concat([h_s - l_s, (h_s - c_s.shift(1)).abs(), (l_s - c_s.shift(1)).abs()], axis=1).max(axis=1)
    up = h_s - h_s.shift(1); dn = l_s.shift(1) - l_s
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c_s.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c_s.index)
    atr_w = tr.ewm(alpha=a, min_periods=period, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
    mdi = 100 * mdm.ewm(alpha=a, min_periods=period, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-10)
    adx = dx.ewm(alpha=a, min_periods=period, adjust=False).mean()
    return adx.fillna(0).values.astype(np.float64)

# ─────────────────────────────────────────────────────────────
# 3. 봇의 기존 Simple MA (변경 전)
# ─────────────────────────────────────────────────────────────
def bot_old_simple_rsi(c, period=14):
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).rolling(period).mean().values
    al = pd.Series(loss).rolling(period).mean().values
    return 100 - 100 / (1 + ag / (al + 1e-10))

def bot_old_simple_adx(h, l, c, period=14):
    up = h - np.roll(h, 1); dn = np.roll(l, 1) - l
    up[0] = 0; dn[0] = 0
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr_w = pd.Series(tr).rolling(period).mean().values
    pdi = 100 * pd.Series(pdm).rolling(period).mean().values / (atr_w + 1e-10)
    mdi = 100 * pd.Series(mdm).rolling(period).mean().values / (atr_w + 1e-10)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-10)
    return pd.Series(dx).rolling(period).mean().values

# ─────────────────────────────────────────────────────────────
# 4. engine_data 백테스트 함수 (참조)
# ─────────────────────────────────────────────────────────────
try:
    from engine_data import compute_rsi as engine_rsi, compute_adx as engine_adx
    engine_available = True
except Exception as e:
    print(f"engine_data import 실패: {e}")
    engine_available = False

# ─────────────────────────────────────────────────────────────
# 5. 계산 실행
# ─────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("[1] 기존 Simple MA(14) → [2] 새 Wilder(10/20) 비교")
print("="*80)

old_rsi = bot_old_simple_rsi(close, 14)
old_adx = bot_old_simple_adx(high, low, close, 14)
new_rsi = bot_wilder_rsi(close, 10)
new_adx = bot_wilder_adx(high, low, close, 20)

if engine_available:
    eng_rsi = engine_rsi(close, 10)
    eng_adx = engine_adx(high, low, close, 20)
    diff_rsi = np.abs(new_rsi - eng_rsi)
    diff_adx = np.abs(new_adx - eng_adx)
    # NaN 제거
    valid_rsi = ~(np.isnan(diff_rsi))
    valid_adx = ~(np.isnan(diff_adx))
    max_diff_rsi = diff_rsi[valid_rsi].max() if valid_rsi.any() else 0
    max_diff_adx = diff_adx[valid_adx].max() if valid_adx.any() else 0
    print(f"\n[검증 1] 봇 Wilder vs engine_data Wilder 일치성")
    print(f"  RSI 최대 차이: {max_diff_rsi:.6f} ({'✅ PASS (< 0.01)' if max_diff_rsi < 0.01 else '❌ FAIL'})")
    print(f"  ADX 최대 차이: {max_diff_adx:.6f} ({'✅ PASS (< 0.01)' if max_diff_adx < 0.01 else '❌ FAIL'})")

# ─────────────────────────────────────────────────────────────
# 6. Simple MA vs Wilder 값 분포 비교
# ─────────────────────────────────────────────────────────────
print(f"\n[검증 2] Simple MA(14) vs Wilder(10/20) 값 분포")
print(f"           {'Simple MA(14)':^20s}  {'Wilder(10/20)':^20s}  {'차이':^15s}")
print(f"  RSI avg: {np.nanmean(old_rsi):>8.2f}           {np.nanmean(new_rsi):>8.2f}           {np.nanmean(new_rsi) - np.nanmean(old_rsi):+.2f}")
print(f"  RSI std: {np.nanstd(old_rsi):>8.2f}           {np.nanstd(new_rsi):>8.2f}           {np.nanstd(new_rsi) - np.nanstd(old_rsi):+.2f}")
print(f"  ADX avg: {np.nanmean(old_adx):>8.2f}           {np.nanmean(new_adx):>8.2f}           {np.nanmean(new_adx) - np.nanmean(old_adx):+.2f}")
print(f"  ADX std: {np.nanstd(old_adx):>8.2f}           {np.nanstd(new_adx):>8.2f}           {np.nanstd(new_adx) - np.nanstd(old_adx):+.2f}")

# 마지막 10개 봉 비교
print(f"\n[검증 3] 마지막 10개 15m 봉 비교 (최근 시점)")
print(f"  {'Timestamp':^20s}  {'RSI_old':^8s}  {'RSI_new':^8s}  {'ADX_old':^8s}  {'ADX_new':^8s}")
for i in range(-10, 0):
    ts = df_15m['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M')
    print(f"  {ts:20s}  {old_rsi[i]:8.2f}  {new_rsi[i]:8.2f}  {old_adx[i]:8.2f}  {new_adx[i]:8.2f}")

# ─────────────────────────────────────────────────────────────
# 7. V12.1 필터 통과율 차이 (RSI 30-65, ADX ≥ 22)
# ─────────────────────────────────────────────────────────────
print(f"\n[검증 4] V12.1 필터 통과율 (유효 구간만)")
warm = max(100, 50)
valid = slice(warm, None)

# 필터: ADX ≥ 22 AND 30 ≤ RSI ≤ 65
old_filter_pass = (old_adx[valid] >= 22) & (old_rsi[valid] >= 30) & (old_rsi[valid] <= 65)
new_filter_pass = (new_adx[valid] >= 22) & (new_rsi[valid] >= 30) & (new_rsi[valid] <= 65)

n_valid = len(old_filter_pass)
print(f"  유효 봉: {n_valid:,}개")
print(f"  Simple MA(14) 통과율: {old_filter_pass.sum():>6,} / {n_valid:,} ({old_filter_pass.sum()/n_valid*100:.2f}%)")
print(f"  Wilder(10/20) 통과율: {new_filter_pass.sum():>6,} / {n_valid:,} ({new_filter_pass.sum()/n_valid*100:.2f}%)")
print(f"  변화: {new_filter_pass.sum() - old_filter_pass.sum():+,} ({(new_filter_pass.sum()-old_filter_pass.sum())/n_valid*100:+.2f}%p)")

# 개별 필터 통과율
print(f"\n[검증 5] 개별 필터 통과율")
print(f"  ADX ≥ 22:  Simple MA {(old_adx[valid] >= 22).sum()/n_valid*100:.2f}% → Wilder {(new_adx[valid] >= 22).sum()/n_valid*100:.2f}%")
print(f"  RSI 30-65: Simple MA {((old_rsi[valid] >= 30) & (old_rsi[valid] <= 65)).sum()/n_valid*100:.2f}% → Wilder {((new_rsi[valid] >= 30) & (new_rsi[valid] <= 65)).sum()/n_valid*100:.2f}%")

print("\n" + "="*80)
print("완료. 검증 1이 PASS면 봇 Wilder가 백테스트와 완전 일치.")
print("="*80)
