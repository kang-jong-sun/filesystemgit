"""Extract BTC data summary for GPT-5 collaboration"""
import pandas as pd, numpy as np, sys
sys.stdout.reconfigure(line_buffering=True)

df = pd.read_csv('btc_usdt_5m_merged.csv', parse_dates=['timestamp'])
df['year'] = df.timestamp.dt.year

print('=== 75-MONTH BTC/USDT 5MIN DATA SUMMARY ===')
print('Period: %s ~ %s' % (str(df.timestamp.iloc[0]), str(df.timestamp.iloc[-1])))
print('Total 5min candles: %d' % len(df))
print()

print('=== YEARLY SUMMARY ===')
for y, g in df.groupby('year'):
    ret = (g.close.iloc[-1] - g.open.iloc[0]) / g.open.iloc[0] * 100
    avg_range = ((g.high - g.low) / g.close).mean() * 100
    print('%d: $%.0f->$%.0f (%+.1f%%) High=$%.0f Low=$%.0f AvgRange=%.2f%% Bars=%d' % (
        y, g.open.iloc[0], g.close.iloc[-1], ret, g.high.max(), g.low.min(), avg_range, len(g)))

print()
print('=== VOLATILITY DISTRIBUTION (5min candle range / close) ===')
df['range_pct'] = (df.high - df.low) / df.close * 100
print('25th=%.3f%% Median=%.3f%% 75th=%.3f%% 95th=%.3f%%' % (
    df.range_pct.quantile(0.25), df.range_pct.median(),
    df.range_pct.quantile(0.75), df.range_pct.quantile(0.95)))

print()
print('=== MONTHLY RETURNS (30min close) ===')
df30 = df.set_index('timestamp').resample('30min').agg({'open':'first','close':'last'}).dropna()
df30['month'] = df30.index.to_period('M')
for m, g in df30.groupby('month'):
    ret = (g.close.iloc[-1] - g.open.iloc[0]) / g.open.iloc[0] * 100
    print('%s: %+.1f%%' % (str(m), ret))
