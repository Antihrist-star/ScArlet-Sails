import pandas as pd
import numpy as np

# Загрузить данные
df = pd.read_parquet('data/raw/BTC_USDT_15m.parquet')

# Анализ по кварталам
df['quarter'] = df.index.to_period('Q')
quarterly_stats = []

for quarter in df['quarter'].unique():
    q_data = df[df['quarter'] == quarter]
    start_price = q_data['close'].iloc[0]
    end_price = q_data['close'].iloc[-1]
    return_pct = (end_price - start_price) / start_price * 100
    
    # UP movements ratio (1 час)
    hourly_returns = q_data['close'].pct_change(4)
    up_ratio = (hourly_returns > 0).mean()
    
    quarterly_stats.append({
        'Quarter': str(quarter),
        'Start': f"${start_price:,.0f}",
        'End': f"${end_price:,.0f}",
        'Return': f"{return_pct:+.1f}%",
        'UP_ratio': f"{up_ratio:.1%}",
        'Regime': 'BULL' if return_pct > 10 else 'BEAR' if return_pct < -10 else 'SIDEWAYS'
    })

# Вывод таблицы
print("\n" + "="*70)
print("QUARTERLY MARKET ANALYSIS (Oct 2023 - Oct 2025)")
print("="*70)

for q in quarterly_stats:
    print(f"{q['Quarter']}: {q['Start']} → {q['End']} = {q['Return']:>7} | UP: {q['UP_ratio']:>6} | {q['Regime']}")

# Подсчет режимов
regimes = [q['Regime'] for q in quarterly_stats]
print("\n" + "="*70)
print(f"BULL quarters:     {regimes.count('BULL')}/{len(regimes)}")
print(f"BEAR quarters:     {regimes.count('BEAR')}/{len(regimes)}")
print(f"SIDEWAYS quarters: {regimes.count('SIDEWAYS')}/{len(regimes)}")

if regimes.count('BULL') > len(regimes) * 0.6:
    print("\n🚨 CRITICAL: >60% BULL MARKET BIAS!")
    print("Model WILL FAIL in bear markets!")
    print("\nRECOMMENDATION:")
    print("1. Add synthetic bear market data")
    print("2. Use market regime as feature")
    print("3. Train separate models for BULL/BEAR/SIDEWAYS")