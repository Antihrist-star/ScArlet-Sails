import pandas as pd
import os

symbols = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
timeframes = ['1m', '15m', '1h']

print("=== Data Quality Report ===\n")

for symbol in symbols:
    for tf in timeframes:
        file = f'data/raw/{symbol}_{tf}.parquet'
        if not os.path.exists(file):
            print(f"MISSING: {file}")
            continue
        
        df = pd.read_parquet(file)
        
        print(f"\n{symbol} {tf}:")
        print(f"  Строк: {len(df)}")
        print(f"  Период: {df.index.min()} -> {df.index.max()}")
        
        nan_count = df.isna().sum().sum()
        print(f"  NaN: {nan_count}")
        
        dupes = df.index.duplicated().sum()
        print(f"  Дубликаты: {dupes}")
        
        zero_close = (df['close'] == 0).sum()
        zero_volume = (df['volume'] == 0).sum()
        print(f"  Нулевые close: {zero_close}")
        print(f"  Нулевые volume: {zero_volume}")

print("\n=== Report Complete ===")