# 📊 Scarlet Sails - Data Sources

## Overview

Scarlet Sails uses **REAL MARKET DATA ONLY**. No synthetic data generation.

## Data Structure

### Parquet Files Location
```
data/features/
├── BTC_USDT_15m_features.parquet
├── ETH_USDT_15m_features.parquet
└── [symbol]_[timeframe]_features.parquet
```

### Data Schema (75 columns)

| Category | Count | Examples |
|----------|-------|----------|
| Basic OHLCV | 5 | open, high, low, close, volume |
| Normalized | 12 | norm_close_zscore, norm_rsi_pctile |
| Derivatives | 20 | deriv_close_velocity, deriv_rsi_roc5 |
| Regime | 12 | regime_rsi_low, regime_atr_mid |
| Cross | 12 | cross_macd_atr_ratio |
| Divergences | 6 | div_rsi_bullish |
| Time | 7 | time_hour, time_asian |
| Target | 1 | target (labels) |

## How to Get Data

### Option 1: Binance API
```python
# Download script coming in Phase 3
python scripts/download_data.py --symbol BTC_USDT --timeframe 15m
```

### Option 2: Pre-computed Dataset
Contact project maintainers for access to pre-computed features.

## Data Requirements

- **Minimum bars**: 10,000 (for training)
- **Recommended**: 100,000+ (for robust backtesting)
- **Update frequency**: Daily
- **Storage**: ~100 MB per symbol-timeframe

## Important Notes

⛔ **NO SYNTHETIC DATA**  
This system is designed for REAL MARKET DATA only. Synthetic data generation is prohibited as it creates false confidence in system performance.

✅ **Data Validation**  
All data must pass validation checks before use:
- No missing values in critical columns
- Reasonable value ranges
- Chronological order
- Sufficient history
