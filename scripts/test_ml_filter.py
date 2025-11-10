"""
Quick test to verify ML filter is working
Creates synthetic data and tests ML filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.hybrid_entry_system import HybridEntrySystem

print("=" * 80)
print("ML FILTER TEST - Using Synthetic Data")
print("=" * 80)

# Create synthetic OHLCV data with RSI and ATR
np.random.seed(42)
n_bars = 1000

dates = pd.date_range('2023-01-01', periods=n_bars, freq='1h')
close_prices = 100 + np.cumsum(np.random.randn(n_bars) * 0.5)

df = pd.DataFrame({
    'open': close_prices + np.random.randn(n_bars) * 0.1,
    'high': close_prices + np.abs(np.random.randn(n_bars) * 0.5),
    'low': close_prices - np.abs(np.random.randn(n_bars) * 0.5),
    'close': close_prices,
    'volume': np.random.randint(1000, 10000, n_bars)
}, index=dates)

# Calculate RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / (loss + 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))

# Calculate ATR
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = ranges.max(axis=1)
df['atr'] = true_range.rolling(14).mean()

# Force some RSI < 30 signals (oversold)
oversold_indices = np.random.choice(range(100, 900), 50, replace=False)
for idx in oversold_indices:
    df.loc[df.index[idx], 'rsi'] = np.random.uniform(15, 29)

df = df.dropna()

print(f"✅ Created synthetic data: {len(df)} bars")
print(f"   RSI < 30: {(df['rsi'] < 30).sum()} bars ({(df['rsi'] < 30).sum() / len(df) * 100:.1f}%)")

# Test 1: Rule-based only (ML disabled)
print("\n" + "=" * 80)
print("TEST 1: Rule-based ONLY (ML filter disabled)")
print("=" * 80)

entry_system = HybridEntrySystem(
    ml_model_path='models/xgboost_model.json',
    enable_ml_filter=False,  # Disabled
    enable_crisis_gate=False
)

signals_without_ml = []
for i in range(100, len(df)):
    should_enter, reason = entry_system.should_enter(df, i, df.index[i])
    if should_enter:
        signals_without_ml.append(i)

print(f"✅ Generated {len(signals_without_ml)} signals (Rule-based only)")

# Test 2: Rule-based + ML filter
print("\n" + "=" * 80)
print("TEST 2: Rule-based + ML FILTER (ML enabled)")
print("=" * 80)

entry_system2 = HybridEntrySystem(
    ml_model_path='models/xgboost_model.json',
    enable_ml_filter=True,  # ✅ ENABLED
    ml_threshold=0.6,
    enable_crisis_gate=False
)

signals_with_ml = []
for i in range(100, len(df)):
    should_enter, reason = entry_system2.should_enter(df, i, df.index[i])
    if should_enter:
        signals_with_ml.append(i)

print(f"✅ Generated {len(signals_with_ml)} signals (Rule-based + ML)")

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Rule-based only:     {len(signals_without_ml)} signals")
print(f"Rule-based + ML:     {len(signals_with_ml)} signals")
print(f"Filtered out:        {len(signals_without_ml) - len(signals_with_ml)} signals")
print(f"Reduction:           {(1 - len(signals_with_ml) / (len(signals_without_ml) + 1e-10)) * 100:.1f}%")

if len(signals_with_ml) < len(signals_without_ml):
    print("\n✅ ML FILTER IS WORKING! Successfully filtered out low-quality signals.")
else:
    print("\n⚠️  ML filter may not be working as expected")

print("\n" + "=" * 80)
print("NEXT STEP: Load real data for full 56-combination test")
print("=" * 80)
print("\nTo load data, you need to either:")
print("1. Install DVC and run: dvc pull data/raw")
print("2. Or manually copy .parquet files to data/raw/")
print("\nExpected files (with underscore):")
print("  - BTC_USDT_15m.parquet, BTC_USDT_1h.parquet, BTC_USDT_4h.parquet, BTC_USDT_1d.parquet")
print("  - ETH_USDT_15m.parquet, ETH_USDT_1h.parquet, ... (and so on for 14 assets)")
