# 🔧 Integration Guide

## Quick Integration

### Install
```bash
git clone https://github.com/Antihrist-star/ScArlet-Sails.git
cd ScArlet-Sails
pip install -r requirements.txt
```

### Test with Real Data
```bash
# Place your data in data/features/
python test_real_data_FIXED.py
```

### Run Backtest
```bash
python backtester.py --symbol BTC_USDT --timeframe 15m
```

## Integration Points

### 1. As a Strategy Provider
```python
from strategies.hybrid_v2 import HybridStrategy

# Load data
df = pd.read_parquet('data/features/BTC_USDT_15m_features.parquet')

# Initialize strategy
strategy = HybridStrategy()

# Generate signals
signals = strategy.generate_signals(df)
```

### 2. As a Backtesting Framework
```python
from backtester import Backtester

backtester = Backtester()
results = backtester.run(strategy, df)
print(results.metrics)
```

### 3. As a Live Trading System
```python
# Coming in Phase 4
from live_trading import LiveTrader

trader = LiveTrader(strategy, api_key, api_secret)
trader.start()
```

## Configuration

### Strategy Config
```python
config = {
    'rule_based': {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    },
    'xgboost_ml': {
        'model_path': 'models/xgboost_trained_v2.json',
        'ood_threshold': 3.0
    },
    'hybrid': {
        'alpha': 0.45,
        'beta': 0.45,
        'gamma_weight': 0.10
    }
}

strategy = HybridStrategy(config)
```

## Troubleshooting

### KeyError: '15m'
Old version of xgboost_ml_v2.py. Update to version with 74 features.

### "Synthetic data" warnings
Remove test_real_data.py and use test_real_data_FIXED.py instead.

### Model incompatibility
XGBoost model trained on 31 features will be retrained in Phase 4 on 74 features.
