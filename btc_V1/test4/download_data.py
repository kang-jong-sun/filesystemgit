"""바이낸스 공개 API로 ETH/SOL 5분봉 데이터 다운로드 (2020~현재)"""
import requests, pandas as pd, time, os

BASE = r'D:\filesystem\futures\btc_V1\test4'
URL = 'https://fapi.binance.com/fapi/v1/klines'

def download_5m(symbol, start_date='2020-01-01', end_date='2026-03-28'):
    """바이낸스 선물 5분봉 다운로드"""
    print(f"\n  [{symbol}] 다운로드 시작...")
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date).timestamp() * 1000)

    all_data = []
    current = start_ms
    batch = 0

    while current < end_ms:
        params = {
            'symbol': symbol,
            'interval': '5m',
            'startTime': current,
            'limit': 1500,
        }
        try:
            resp = requests.get(URL, params=params, timeout=30)
            data = resp.json()
            if not data or not isinstance(data, list):
                print(f"    빈 응답, 종료")
                break

            for k in data:
                all_data.append({
                    'timestamp': pd.Timestamp(k[0], unit='ms'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'quote_volume': float(k[7]),
                    'trades': int(k[8]),
                })

            current = data[-1][0] + 1
            batch += 1
            if batch % 100 == 0:
                print(f"    {len(all_data):,}행... ({all_data[-1]['timestamp']})")
            time.sleep(0.1)

        except Exception as e:
            print(f"    에러: {e}, 5초 대기 후 재시도")
            time.sleep(5)

    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  [{symbol}] 완료: {len(df):,}행 ({df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]})")
    return df


def save_parts(df, symbol_name, n_parts=3):
    """CSV 분할 저장"""
    chunk = len(df) // n_parts + 1
    for i in range(n_parts):
        part = df.iloc[i*chunk : (i+1)*chunk]
        fn = os.path.join(BASE, f'{symbol_name}_5m_2020_to_now_part{i+1}.csv')
        part.to_csv(fn, index=False)
        print(f"    저장: {os.path.basename(fn)} ({len(part):,}행)")


def main():
    print("=" * 70)
    print("  ETH/SOL 5분봉 데이터 다운로드 (바이낸스 공개 API)")
    print("=" * 70)

    # ETH
    df_eth = download_5m('ETHUSDT')
    save_parts(df_eth, 'eth_usdt')

    # SOL
    df_sol = download_5m('SOLUSDT')
    save_parts(df_sol, 'sol_usdt')

    print(f"\n  BTC: 기존 데이터 사용 (655,399행)")
    print(f"  ETH: {len(df_eth):,}행")
    print(f"  SOL: {len(df_sol):,}행")
    print(f"\n완료!")


if __name__ == '__main__':
    main()
