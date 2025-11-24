# 📚 Scarlet Sails - API Reference

## Strategies

### RuleBasedStrategy

```python
from strategies.rule_based_v2 import RuleBasedStrategy

strategy = RuleBasedStrategy(
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    ema_short=9,
    ema_long=21
)

# Generate signals
signals = strategy.generate_signals(df)

# Calculate P_rb for a single state
P_rb, components = strategy.calculate_pjs(market_state, ...)
```

**Methods:**
- `generate_signals(df)` - Generate trading signals for DataFrame
- `calculate_pjs(market_state, ...)` - Calculate P_rb(S) for single state

### XGBoostMLStrategy

```python
from strategies.xgboost_ml_v2 import XGBoostMLStrategy

strategy = XGBoostMLStrategy(
    model_path='models/xgboost_trained_v2.json',
    ood_threshold=3.0
)

# Generate signals (requires multi-timeframe data)
df_dict = {
    '15m': df_15m,
    '1h': df_1h,
    '4h': df_4h,
    '1d': df_1d
}
signals = strategy.generate_signals(df_dict)

# Calculate P_ml for a single state
P_ml, components = strategy.calculate_pjs(market_state, ...)
```

**Methods:**
- `generate_signals(df_dict)` - Generate signals from multi-timeframe data
- `calculate_pjs(market_state, ...)` - Calculate P_ml(S) for single state
- `transform_features(df_dict)` - Transform raw data to 74 features

### HybridStrategy

```python
from strategies.hybrid_v2 import HybridStrategy

strategy = HybridStrategy(
    alpha=0.45,
    beta=0.45,
    gamma_weight=0.10
)

# Generate signals
signals = strategy.generate_signals(df, df_dict)

# Calculate P_hyb for a single state
P_hyb, components = strategy.calculate_pjs(market_state, ...)
```

**Methods:**
- `generate_signals(df, df_dict)` - Generate hybrid signals
- `calculate_pjs(market_state, ...)` - Calculate P_hyb(S) for single state
- `update_weights()` - Update adaptive weights α, β based on performance

## Components

### OpportunityScorer

```python
from components.opportunity_scorer import OpportunityScorer

scorer = OpportunityScorer(
    w_vol=0.40,
    w_liq=0.35,
    w_micro=0.25
)

# Calculate opportunity score
W_opportunity = scorer.calculate_opportunity(market_state)
```

**Methods:**
- `calculate_opportunity(market_state)` - Returns W_opportunity ∈ [0,1]
- `calculate_volatility_score(...)` - W_vol component
- `calculate_liquidity_score(...)` - W_liq component
- `calculate_microstructure_score(...)` - W_micro component

### AdvancedRiskPenalty

```python
from components.advanced_risk_penalty import AdvancedRiskPenalty

risk_calculator = AdvancedRiskPenalty(
    w_vol=0.30,
    w_tail=0.20,
    w_liq=0.20,
    w_ood=0.15,
    w_dd=0.15
)

# Calculate total risk penalty
R_penalty = risk_calculator.calculate_total_risk_penalty(market_state)
```

**Methods:**
- `calculate_total_risk_penalty(market_state)` - Returns R_penalty
- `calculate_garch_volatility(...)` - R_vol (GARCH-based)
- `calculate_tail_risk(...)` - R_tail (CVaR)
- `calculate_liquidity_risk(...)` - R_liq
- `calculate_ood_risk(...)` - R_ood (Mahalanobis)
- `calculate_drawdown_penalty(...)` - R_dd (exponential)

## Backtesting

### Backtester

```python
from backtester import Backtester

backtester = Backtester(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005
)

# Run backtest
results = backtester.run(strategy, df)

# Access metrics
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

**Methods:**
- `run(strategy, df)` - Run backtest
- `calculate_metrics()` - Calculate performance metrics
- `plot_equity_curve()` - Visualize equity curve

## Data Structures

### Market State

```python
market_state = {
    'close': float,
    'rsi': float,
    'ema_9': float,
    'ema_21': float,
    'volume': float,
    'atr': float,
    'bb_width': float,
    # ... 74 total features
}
```

### Signal DataFrame

```python
signals = pd.DataFrame({
    'timestamp': datetime,
    'P_j': float,  # Decision value [0, 1]
    'signal': int,  # 1 (buy), 0 (hold), -1 (sell)
    'components': dict  # Breakdown of P_j calculation
})
```

## Error Handling

```python
try:
    signals = strategy.generate_signals(df)
except KeyError as e:
    print(f"Missing required column: {e}")
except ValueError as e:
    print(f"Invalid data format: {e}")
```

## Performance Considerations

- **Rule-Based**: O(1) per bar - fastest
- **XGBoost ML**: O(log n) per bar - model inference
- **Hybrid**: O(log n) per bar - combines both

## Version Compatibility

- Python: 3.11+
- Pandas: 1.5.0+
- NumPy: 1.23.0+
- XGBoost: 1.7.0+
- PyTorch: 2.0.0+ (for Phase 4 RL)
