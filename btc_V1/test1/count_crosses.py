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
def rsi_calc(s,p=14):
    d=s.diff();g=d.where(d>0,0.0);l=(-d).where(d<0,0.0)
    ag=g.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    al=l.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    return 100-100/(1+ag/al.replace(0,1e-10))

base='D:/filesystem/futures/btc_V1/test2'
dfs=[pd.read_csv(f'{base}/btc_usdt_5m_2020_to_now_part{i}.csv',parse_dates=['timestamp']) for i in [1,2,3]]
df=pd.concat(dfs).drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
df.set_index('timestamp',inplace=True)
df=df[['open','high','low','close','volume']].astype(float)
df30=df.resample('30min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()

e3=ema(df30['close'],3); e200=ema(df30['close'],200)
adx=adx_calc(df30['high'],df30['low'],df30['close'],14)
rsi=rsi_calc(df30['close'],14)

bull=e3>e200; prev=bull.shift(1).fillna(False)
gx=bull&~prev; dx=~bull&prev
vf=(adx>=35)&(rsi>=30)&(rsi<=65)

all_cross=(gx|dx)
filtered_cross=(gx|dx)&vf

all_c=int(all_cross.iloc[300:].sum())
filt_c=int(filtered_cross.iloc[300:].sum())
unfilt_c=all_c-filt_c
print(f'전체 크로스: {all_c}건')
print(f'필터 통과:   {filt_c}건 (기존 v14.4)')
print(f'필터 미통과: {unfilt_c}건 (AI만 판단)')
print(f'API 호출 예상: {all_c*2}회 (GPT+Claude)')
print(f'예상 비용: ~${all_c*0.07:.0f}')
print(f'예상 시간: ~{all_c*2*3//60}분')

unf=all_cross.iloc[300:]&~vf.iloc[300:]
idx=unf[unf].index
print(f'\n필터 미통과 사유:')
print(f'  ADX<35:  {int(sum(adx[idx]<35))}건')
print(f'  RSI<30:  {int(sum(rsi[idx]<30))}건')
print(f'  RSI>65:  {int(sum(rsi[idx]>65))}건')
