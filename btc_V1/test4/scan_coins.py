"""바이낸스 선물 주요 코인 스캔 - v16.0 전략 적합성 평가"""
import requests, pandas as pd, numpy as np, time

URL = 'https://fapi.binance.com/fapi/v1/klines'

# 바이낸스 선물 거래량 상위 코인 (BTC/ETH/SOL 제외)
COINS = [
    'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT',
    'DOTUSDT', 'LINKUSDT', 'MATICUSDT', 'UNIUSDT', 'ATOMUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
    'FILUSDT', 'AAVEUSDT', 'MKRUSDT', 'SUIUSDT', 'PEPEUSDT',
    'WIFUSDT', 'SEIUSDT', 'TIAUSDT', 'INJUSDT', 'FETUSDT',
]

def get_recent_data(symbol, days=90):
    """최근 90일 5분봉 다운로드"""
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    all_data = []
    current = start
    while current < end:
        try:
            resp = requests.get(URL, params={'symbol':symbol,'interval':'5m','startTime':current,'limit':1500}, timeout=10)
            data = resp.json()
            if not data or not isinstance(data, list): break
            for k in data:
                all_data.append({'close':float(k[4]),'high':float(k[2]),'low':float(k[3]),'volume':float(k[5])})
            current = data[-1][0] + 1
            time.sleep(0.05)
        except:
            break
    return all_data

def get_listing_date(symbol):
    """상장일 확인 (첫 캔들)"""
    try:
        resp = requests.get(URL, params={'symbol':symbol,'interval':'1M','startTime':0,'limit':1}, timeout=10)
        data = resp.json()
        if data and isinstance(data, list):
            return pd.Timestamp(data[0][0], unit='ms').strftime('%Y-%m')
    except:
        pass
    return '?'

def analyze(data):
    """코인 특성 분석"""
    if len(data) < 100: return None
    closes = np.array([d['close'] for d in data])
    highs = np.array([d['high'] for d in data])
    lows = np.array([d['low'] for d in data])
    vols = np.array([d['volume'] for d in data])

    # 변동성 (5분봉 평균 변동폭)
    ranges = (highs - lows) / closes * 100
    avg_range = np.mean(ranges)

    # 일간 변동성
    daily_closes = closes[::288]  # 5분 x 288 = 1일
    if len(daily_closes) > 2:
        daily_returns = np.diff(daily_closes) / daily_closes[:-1] * 100
        daily_vol = np.std(daily_returns)
    else:
        daily_vol = 0

    # 추세 강도 (최근 vs 과거)
    mid = len(closes) // 2
    trend = (np.mean(closes[mid:]) / np.mean(closes[:mid]) - 1) * 100

    # 거래량 (USDT 기준)
    avg_vol_usdt = np.mean(vols)

    # ADX 친화도 (큰 움직임 빈도)
    big_moves = np.sum(np.abs(np.diff(closes) / closes[:-1]) > 0.01)  # 1% 이상 움직임
    big_move_pct = big_moves / len(closes) * 100

    return {
        'avg_range': round(avg_range, 3),
        'daily_vol': round(daily_vol, 2),
        'trend': round(trend, 1),
        'avg_vol_usdt': avg_vol_usdt,
        'big_move_pct': round(big_move_pct, 2),
        'price': round(closes[-1], 4),
        'candles': len(data),
    }

def main():
    print("=" * 100)
    print("  바이낸스 선물 코인 스캔 - v16.0 전략 적합성 평가")
    print("=" * 100)

    # BTC/ETH/SOL 기준값 먼저
    refs = {}
    for sym, name in [('BTCUSDT','BTC'),('ETHUSDT','ETH'),('SOLUSDT','SOL')]:
        print(f"  {name} 분석 중...")
        data = get_recent_data(sym)
        r = analyze(data)
        if r:
            r['listing'] = get_listing_date(sym)
            refs[name] = r

    print(f"\n  기준 코인:")
    print(f"  {'코인':>6} {'가격':>12} {'5m변동%':>8} {'일변동%':>8} {'추세%':>8} {'대형이동%':>8} {'거래량(M$)':>10} {'상장'}")
    for name, r in refs.items():
        print(f"  {name:>6} ${r['price']:>10,.2f} {r['avg_range']:>7.3f}% {r['daily_vol']:>7.2f}% {r['trend']:>+7.1f}% {r['big_move_pct']:>7.2f}% ${r['avg_vol_usdt']/1e6:>8.1f}M {r.get('listing','')}")

    # 후보 코인 스캔
    print(f"\n  후보 코인 스캔 중...")
    candidates = []
    for sym in COINS:
        name = sym.replace('USDT','')
        print(f"    {name}...", end=' ')
        data = get_recent_data(sym, days=60)
        r = analyze(data)
        if r:
            r['symbol'] = sym
            r['name'] = name
            r['listing'] = get_listing_date(sym)
            candidates.append(r)
            print(f"OK ({r['candles']} candles, vol ${r['avg_vol_usdt']/1e6:.1f}M)")
        else:
            print("SKIP")
        time.sleep(0.2)

    # 점수 계산 (v16.0 적합성)
    # 높은 변동성 + 높은 거래량 + 충분한 데이터 기간 = 좋음
    btc_range = refs['BTC']['avg_range']
    btc_vol = refs['BTC']['avg_vol_usdt']

    for c in candidates:
        # 변동성 점수 (BTC 대비 0.5~2배가 이상적)
        vol_ratio = c['avg_range'] / btc_range if btc_range > 0 else 1
        if 0.8 <= vol_ratio <= 2.5:
            vol_score = 100
        elif 0.5 <= vol_ratio <= 3.0:
            vol_score = 70
        else:
            vol_score = 30

        # 거래량 점수
        liq_ratio = c['avg_vol_usdt'] / btc_vol if btc_vol > 0 else 0
        liq_score = min(liq_ratio * 200, 100)

        # 대형 이동 빈도 (높을수록 추세추종에 유리)
        move_score = min(c['big_move_pct'] * 20, 100)

        # 상장 기간 (2020 이전이면 만점)
        try:
            yr = int(c['listing'][:4])
            age_score = 100 if yr <= 2020 else (80 if yr <= 2021 else (60 if yr <= 2022 else 40))
        except:
            age_score = 30

        c['score'] = round(vol_score * 0.3 + liq_score * 0.3 + move_score * 0.2 + age_score * 0.2, 1)

    candidates.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*100}")
    print(f"  v16.0 전략 적합성 순위 (BTC/ETH/SOL 제외)")
    print(f"{'='*100}")
    print(f"  {'#':>3} {'코인':>6} {'가격':>12} {'5m변동%':>8} {'일변동%':>8} {'대형이동%':>9} {'거래량(M$)':>10} {'상장':>8} {'점수':>6} 평가")
    print(f"  {'-'*95}")

    for i, c in enumerate(candidates):
        grade = ''
        if c['score'] >= 70: grade = '** 강추 **'
        elif c['score'] >= 55: grade = '* 추천 *'
        elif c['score'] >= 40: grade = '가능'
        else: grade = '부적합'
        print(f"  {i+1:>3} {c['name']:>6} ${c['price']:>10,.4f} {c['avg_range']:>7.3f}% {c['daily_vol']:>7.2f}% {c['big_move_pct']:>8.2f}% ${c['avg_vol_usdt']/1e6:>8.1f}M {c['listing']:>8} {c['score']:>5.1f} {grade}")

    # 저장
    import json
    save = {
        'refs': {k: {kk:vv for kk,vv in v.items()} for k,v in refs.items()},
        'candidates': [{k:v for k,v in c.items()} for c in candidates[:15]],
    }
    with open(r'D:\filesystem\futures\btc_V1\test4\coin_scan_results.json', 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n완료!")

if __name__ == '__main__':
    main()
