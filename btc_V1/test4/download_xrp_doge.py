"""XRP/DOGE 5분봉 데이터 다운로드"""
import requests, pandas as pd, time, os

BASE = r'D:\filesystem\futures\btc_V1\test4'
URL = 'https://fapi.binance.com/fapi/v1/klines'

def download_5m(symbol, start_date='2020-01-01'):
    print(f"\n  [{symbol}] 다운로드...")
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(time.time() * 1000)
    all_data = []; current = start_ms; batch = 0
    while current < end_ms:
        try:
            resp = requests.get(URL, params={'symbol':symbol,'interval':'5m','startTime':current,'limit':1500}, timeout=30)
            data = resp.json()
            if not data or not isinstance(data, list): break
            for k in data:
                all_data.append({
                    'timestamp': pd.Timestamp(k[0], unit='ms'),
                    'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]),
                    'volume': float(k[5]), 'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                })
            current = data[-1][0] + 1; batch += 1
            if batch % 100 == 0: print(f"    {len(all_data):,}행... ({all_data[-1]['timestamp']})")
            time.sleep(0.08)
        except Exception as e:
            print(f"    에러: {e}"); time.sleep(5)
    df = pd.DataFrame(all_data).drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
    print(f"  [{symbol}] {len(df):,}행 ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df

def save_parts(df, name, n=3):
    chunk = len(df) // n + 1
    for i in range(n):
        part = df.iloc[i*chunk:(i+1)*chunk]
        fn = os.path.join(BASE, f'{name}_5m_2020_to_now_part{i+1}.csv')
        part.to_csv(fn, index=False)
        print(f"    {os.path.basename(fn)} ({len(part):,}행)")

def main():
    print("=" * 60)
    print("  XRP/DOGE 5분봉 다운로드")
    print("=" * 60)
    for sym, name in [('XRPUSDT','xrp_usdt'), ('DOGEUSDT','doge_usdt')]:
        df = download_5m(sym)
        save_parts(df, name)
    print("\n완료!")

if __name__ == '__main__':
    main()
