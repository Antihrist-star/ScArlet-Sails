# Scarlet Sails - Algorithmic Trading System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Algorithmic cryptocurrency trading system with mathematical proof of strategy dispersion.**

## Overview

Scarlet Sails implements three trading strategies with mathematically rigorous analysis:

| Model | Strategy | Description |
|-------|----------|-------------|
| Model 1 | Rule-Based (P_rb) | Technical indicators + filters |
| Model 2 | XGBoost ML (P_ml) | Machine learning on 74 features |
| Model 3 | Hybrid (P_hyb) | α·P_rb + β·P_ml + γ·V(S) with DQN |

## Quick Start

```bash
# Clone repository
git clone https://github.com/Antihrist-star/ScArlet-Sails.git
cd ScArlet-Sails

# Install dependencies
pip install -r requirements.txt

# Run backtest
python run_backtest.py --strategy hybrid --coin ENA --timeframe 15m
```

## Project Structure

```
scarlet-sails/
├── core/                    # Core modules
│   ├── backtest_engine.py   # Backtesting framework
│   ├── data_loader.py       # OHLCV data loader
│   ├── feature_loader.py    # 75-feature loader
│   └── metrics_calculator.py
├── strategies/              # Trading strategies
│   ├── rule_based_v2.py     # Model 1
│   ├── xgboost_ml_v2.py     # Model 2
│   └── hybrid_v2.py         # Model 3
├── rl/                      # Reinforcement Learning
│   ├── dqn.py               # Deep Q-Network
│   └── trading_environment.py
├── components/              # Strategy components
│   ├── opportunity_scorer.py
│   └── advanced_risk_penalty.py
└── data/
    ├── raw/                 # OHLCV data (via DVC)
    └── features/            # 75-feature datasets
```

## Supported Assets

14 cryptocurrency pairs on Binance:

```
ALGO, AVAX, BTC, DOT, ENA, ETH, HBAR
LDO, LINK, LTC, ONDO, SOL, SUI, UNI
```

Timeframes: `15m`, `1h`, `4h`, `1d`

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Core architecture |
| Phase 2 | ✅ Complete | Backtesting framework |
| Phase 3 | 🔄 In Progress | Feature integration + Model training |
| Phase 4 | ⏳ Planned | Production deployment |

## Documentation

- [Mathematical Framework](docs/MATHEMATICAL_FRAMEWORK.md)
- [System Architecture](docs/SYSTEM_ARCHITECTURE_DETAILED.md)
- [Model Formulas](docs/MODEL_FORMULAS.md)

## Team

- **STAR_ANT** - Project Lead, Strategy Development
- **EGOR 1** - Pattern Validation
- **EGOR 2** - ML Model Training

## License

MIT License - see [LICENSE](LICENSE) for details.
