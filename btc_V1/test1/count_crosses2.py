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
all_cross=(gx|dx).iloc[300:]
idx=all_cross[all_cross].index

print("ADX 구간별 크로스 분포:")
for lo,hi,label in [(0,15,"0~15 (매우약)"),(15,20,"15~20"),(20,25,"20~25"),(25,30,"25~30"),(30,35,"30~35 (경계)"),(35,50,"35~50 (v14.4통과)"),(50,100,"50+ (강함)")]:
    cnt=int(sum((adx[idx]>=lo)&(adx[idx]<hi)))
    print(f"  ADX {label:>20}: {cnt:>5}건")

print(f"\n--- 방안별 AI 호출 수 ---")
# 방안1: ADX>=20만 AI 호출
a20=int(sum(adx[idx]>=20))
print(f"방안1 ADX>=20: {a20}건 x 2 = {a20*2}호출, ~${a20*0.07:.0f}, ~{a20*2*3//60}분")
# 방안2: ADX>=25만 AI 호출
a25=int(sum(adx[idx]>=25))
print(f"방안2 ADX>=25: {a25}건 x 2 = {a25*2}호출, ~${a25*0.07:.0f}, ~{a25*2*3//60}분")
# 방안3: ADX>=30만 AI 호출
a30=int(sum(adx[idx]>=30))
print(f"방안3 ADX>=30: {a30}건 x 2 = {a30*2}호출, ~${a30*0.07:.0f}, ~{a30*2*3//60}분")
# 방안4: 전체
print(f"방안4 전체:    {len(idx)}건 x 2 = {len(idx)*2}호출, ~${len(idx)*0.07:.0f}, ~{len(idx)*2*3//60}분")
