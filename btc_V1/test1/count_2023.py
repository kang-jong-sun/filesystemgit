import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np

def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def adx_calc(h,l,c,p=14):
    pdm=h.diff();mdm=-l.diff()
    pdm=pdm.where((pdm>mdm)&(pdm>0),0.0);mdm=mdm.where((mdm>pdm)&(mdm>0),0.0)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    pdi=100*(pdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    mdi=100*(mdm.ewm(alpha=1/p,min_periods=p,adjust=False).mean()/atr)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,1e-10)
    return dx.ewm(alpha=1/p,min_periods=p,adjust=False).mean()

base='D:/filesystem/futures/btc_V1/test2'
dfs=[pd.read_csv(f'{base}/btc_usdt_5m_2020_to_now_part{i}.csv',parse_dates=['timestamp']) for i in [1,2,3]]
df=pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
df.set_index('timestamp',inplace=True)
df=df[['open','high','low','close','volume']].astype(float)
df30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

e3=ema(df30['close'],3); e200=ema(df30['close'],200)
adx=adx_calc(df30['high'],df30['low'],df30['close'],14)

bull=e3>e200; prev=bull.shift(1).fillna(False)
gx=bull&~prev; dx=~bull&prev
cross=(gx|dx)

# 2023 이후만
mask=df30.index>=pd.Timestamp('2023-01-01')
idx=cross[cross&mask].index

total=len(idx)
a20=int(sum(adx[idx]>=20))
a25=int(sum(adx[idx]>=25))
a30=int(sum(adx[idx]>=30))
a35=int(sum(adx[idx]>=35))

print(f'2023~2026 크로스:')
print(f'  전체:    {total}건')
print(f'  ADX>=20: {a20}건 x2 = {a20*2}호출, ~${a20*0.07:.0f}, ~{a20*2*4//60}분')
print(f'  ADX>=25: {a25}건')
print(f'  ADX>=30: {a30}건')
print(f'  ADX>=35: {a35}건 (v14.4 필터)')
