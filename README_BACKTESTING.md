# Scarlet Sails Backtesting Framework

Production-ready backtesting system for cryptocurrency trading strategies.

## Quick Start

```bash
# Run backtest with RSI strategy on BTC 15m
python run_backtest.py --strategy rsi --coin BTC --timeframe 15m

# Run backtest on ENA (best performer) with date range
python run_backtest.py --strategy combined --coin ENA --timeframe 15m --start 2024-01-01

# Compare multiple coins
python run_backtest.py --strategy rsi --coins BTC ETH SOL ENA --timeframe 15m
```

## Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted return |
| Profit Factor | > 2.0 | Gross profit / Gross loss |
| Max Drawdown | < 15% | Maximum peak-to-trough decline |

## Project Structure

```
scarlet-sails/
├── core/
│   ├── backtest_engine.py   # Main backtester
│   ├── data_loader.py       # Multi-asset data loading
│   ├── metrics_calculator.py # Performance metrics
│   ├── position_sizer.py    # Position sizing & risk
│   └── trade_logger.py      # Trade logging
├── visualization/
│   └── plotter.py           # Charts and plots
├── strategies/
│   └── simple_strategies.py # Trading strategies
├── tests/
│   ├── test_data_loader.py
│   └── test_backtest_engine.py
├── data/
│   └── raw/                 # OHLCV parquet files
├── backtest_results/        # Output directory
└── run_backtest.py          # Entry point
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `rsi` | RSI Mean Reversion (baseline) |
| `ma` | Moving Average Crossover |
| `bollinger` | Bollinger Bands Mean Reversion |
| `combined` | Multi-indicator approach |
| `rule_based` | Production Rule-Based V2 |
| `hybrid` | Production Hybrid Strategy |

## Available Assets

**Coins (14):**
ALGO, AVAX, BTC, DOT, ENA, ETH, HBAR, LDO, LINK, LTC, ONDO, SOL, SUI, UNI

**Timeframes (4):**
15m, 1h, 4h, 1d

**Total Combinations:** 56

## Data Format

Data files are expected at `data/raw/{COIN}_USDT_{TIMEFRAME}.parquet`

Required columns:
- `open` - Opening price
- `high` - High price
- `low` - Low price
- `close` - Closing price
- `volume` - Trading volume

Index: DatetimeIndex

## Command Line Options

```
usage: run_backtest.py [-h] [--strategy {rsi,ma,bollinger,combined,rule_based,hybrid}]
                       [--coin COIN] [--coins COINS [COINS ...]]
                       [--timeframe {15m,1h,4h,1d}] [--start START] [--end END]
                       [--capital CAPITAL] [--stop-loss STOP_LOSS] [--no-stop-loss]
                       [--output OUTPUT] [--no-plots] [--quiet] [--data-dir DATA_DIR]

Options:
  --strategy, -s     Trading strategy (default: rsi)
  --coin, -c         Single coin to backtest (default: BTC)
  --coins            Multiple coins for comparison
  --timeframe, -t    Timeframe (default: 15m)
  --start            Start date (YYYY-MM-DD)
  --end              End date (YYYY-MM-DD)
  --capital          Initial capital (default: 10000)
  --stop-loss        Stop loss percentage (default: 0.02)
  --no-stop-loss     Disable stop loss
  --output, -o       Output directory (default: backtest_results)
  --no-plots         Skip plot generation
  --quiet, -q        Suppress verbose output
  --data-dir         Path to data directory (default: data/raw)
```

## Output Files

After running a backtest, results are saved to `backtest_results/`:

```
backtest_results/
├── {strategy}_{coin}_{tf}_equity.csv      # Equity curve data
├── {strategy}_{coin}_{tf}_equity.png      # Equity curve plot
├── {strategy}_{coin}_{tf}_drawdown.png    # Drawdown chart
├── {strategy}_{coin}_{tf}_monthly.png     # Monthly returns heatmap
├── {strategy}_{coin}_{tf}_returns.png     # Returns distribution
├── {strategy}_{coin}_{tf}_trades.csv      # Trade log
├── {strategy}_{coin}_{tf}_trades.png      # Trade analysis
└── {strategy}_{coin}_{tf}_metrics.json    # Full metrics
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_backtest_engine.py -v

# Run with coverage
pytest tests/ -v --cov=core
```

## Expected Results

### ENA 15m (Best Performer)

```
Annualized Return: ~392.8%
Profit Factor: ~2.12
Max Drawdown: ~12.3%
```

### BTC 15m (Baseline)

```
Annualized Return: ~88.8%
Profit Factor: ~1.45
Max Drawdown: ~18.5%
```

## Configuration

Transaction costs (realistic):
- Commission: 0.1% (0.001)
- Slippage: 0.05% (0.0005)

Risk management:
- Default stop loss: 2%
- Max position: 25% of capital
- Max drawdown halt: 20%

## Week 1 Discovery

**Critical finding:** 15-minute timeframe significantly outperforms other timeframes:

| Timeframe | Avg Annual Return |
|-----------|-------------------|
| 15m | 325% |
| 1h | 180% |
| 4h | 120% |
| 1d | 80% |

## Philosophy

> "Creativity through errors" - Deep analysis over complex algorithms

- Simple, validated strategies before complexity
- Realistic transaction costs
- No look-ahead bias
- Walk-forward validation