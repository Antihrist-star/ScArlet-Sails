"""
RULE-BASED STRATEGY - OPTIMIZED FOR LARGE DATASETS
Performance improvements for 273K+ bars

OPTIMIZATIONS:
1. Pre-calculate pct_change() once
2. Use vectorized operations
3. Add progress tracking
4. Cache repeated calculations

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 24, 2025 (OPTIMIZED)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.opportunity_scorer import OpportunityScorer
from components.advanced_risk_penalty import AdvancedRiskPenalty

logger = logging.getLogger(__name__)


class TechnicalFilters:
    """Technical indicator filters for Rule-Based strategy"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # RSI thresholds
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        
        # Trend filters
        self.ema_short = self.config.get('ema_short', 9)
        self.ema_long = self.config.get('ema_long', 21)
        
        logger.info("TechnicalFilters initialized")
    
    def check_rsi_filter(self, rsi: float) -> int:
        """F1: RSI oversold/overbought filter"""
        if rsi < self.rsi_oversold or rsi > self.rsi_overbought:
            return 1
        return 0
    
    def check_trend_filter(self, close: float, ema_short: float, ema_long: float) -> int:
        """F2: Trend alignment filter"""
        if close > ema_short > ema_long:
            return 1
        return 0
    
    def check_volatility_filter(self, bb_width: float) -> int:
        """F3: Volatility (Bollinger Band width) filter"""
        if 0.01 < bb_width < 0.10:
            return 1
        return 0
    
    def calculate_filters(self, market_state: Dict) -> int:
        """Calculate product of all filters"""
        rsi = market_state.get('rsi', 50)
        close = market_state.get('close', 0)
        ema_short = market_state.get('ema_short', 0)
        ema_long = market_state.get('ema_long', 0)
        bb_width = market_state.get('bb_width', 0)
        
        f1 = self.check_rsi_filter(rsi)
        f2 = self.check_trend_filter(close, ema_short, ema_long)
        f3 = self.check_volatility_filter(bb_width)
        
        return f1 * f2 * f3


class RuleBasedStrategy:
    """
    Rule-Based Trading Strategy - OPTIMIZED FOR LARGE DATASETS
    
    Performance improvements:
    - Pre-calculated indicators
    - Vectorized operations
    - Progress tracking
    - Cached computations
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.opportunity_scorer = OpportunityScorer(self.config.get('opportunity', {}))
        self.technical_filters = TechnicalFilters(self.config.get('filters', {}))
        
        # Costs
        self.commission = self.config.get('commission', 0.001)
        self.slippage = self.config.get('slippage', 0.0005)
        self.C_fixed = self.commission + self.slippage
        
        # Risk calculator (optional)
        self.risk_calculator = None
        
        logger.info(f"RuleBasedStrategy initialized: C_fixed={self.C_fixed}")
    
    def set_risk_calculator(self, risk_calc):
        """Set risk calculator instance"""
        self.risk_calculator = risk_calc
        logger.info("Risk calculator set")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate ALL technical indicators ONCE
        
        OPTIMIZATION: Vectorized calculations for entire DataFrame
        """
        logger.info("Pre-calculating technical indicators...")
        
        df_calc = df.copy()
        
        # RSI (if not present)
        if 'rsi' not in df_calc.columns:
            delta = df_calc['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df_calc['rsi'] = 100 - (100 / (1 + rs))
        
        # EMAs (if not present)
        if 'ema_short' not in df_calc.columns:
            df_calc['ema_short'] = df_calc['close'].ewm(span=9, adjust=False).mean()
        if 'ema_long' not in df_calc.columns:
            df_calc['ema_long'] = df_calc['close'].ewm(span=21, adjust=False).mean()
        
        # Bollinger Bands width (if not present)
        if 'bb_width' not in df_calc.columns:
            sma20 = df_calc['close'].rolling(window=20).mean()
            std20 = df_calc['close'].rolling(window=20).std()
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            df_calc['bb_width'] = (bb_upper - bb_lower) / sma20
        
        # ATR (if not present)
        if 'atr' not in df_calc.columns:
            high_low = df_calc['high'] - df_calc['low']
            high_close = abs(df_calc['high'] - df_calc['close'].shift())
            low_close = abs(df_calc['low'] - df_calc['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_calc['atr'] = true_range.rolling(window=14).mean()
        
        # PRE-CALCULATE pct_change ONCE!
        if 'returns' not in df_calc.columns:
            df_calc['returns'] = df_calc['close'].pct_change()
        
        logger.info("✅ Technical indicators pre-calculated!")
        
        return df_calc
    
    def calculate_pjs(self, market_state: Dict, opportunity: float, filters_product: int) -> Tuple[float, Dict]:
        """Calculate P_rb(S)"""
        W_opportunity = opportunity
        
        # Filters product
        filters = filters_product
        
        # Costs
        costs = self.C_fixed
        
        # Risk penalty (if available)
        risk_penalty = 0.0
        if self.risk_calculator:
            risk_penalty = self.risk_calculator.calculate_total_risk_penalty(market_state)
        
        # Calculate P_rb
        P_rb = W_opportunity * filters - costs - risk_penalty
        
        components = {
            'W_opportunity': W_opportunity,
            'filters_product': filters,
            'costs': costs,
            'risk_penalty': risk_penalty,
            'P_rb': P_rb
        }
        
        return P_rb, components
    
    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """
        Generate trading signals - OPTIMIZED VERSION
        
        OPTIMIZATIONS:
        1. Pre-calculate all indicators ONCE
        2. Vectorized operations where possible
        3. Progress bar for large datasets
        4. Efficient memory usage
        """
        logger.info(f"Generating signals for {len(df)} bars...")
        
        # OPTIMIZATION 1: Pre-calculate ALL indicators
        df_calc = self.calculate_technical_indicators(df)
        
        # Drop NaN rows
        df_calc = df_calc.dropna()
        
        logger.info(f"After dropping NaN: {len(df_calc)} bars remain")
        
        results = []
        
        # OPTIMIZATION 2: Progress bar for large datasets
        iterator = tqdm(range(len(df_calc)), desc="Generating signals") if len(df_calc) > 10000 else range(len(df_calc))
        
        for idx in iterator:
            row = df_calc.iloc[idx]
            
            # Market state
            market_state = {
                'close': row['close'],
                'rsi': row['rsi'],
                'ema_short': row['ema_short'],
                'ema_long': row['ema_long'],
                'bb_width': row['bb_width'],
                'atr': row.get('atr', 0),
                'volume': row.get('volume', 0),
                'volatility': row.get('atr', 0) / row['close'] if row['close'] > 0 else 0.02,
                'crisis_level': 0.0,
                'drawdown': 0.0,
                'regime': 'normal'
            }
            
            # Calculate opportunity
            opportunity = self.opportunity_scorer.calculate_opportunity(market_state)
            
            # Calculate filters
            filters_product = self.technical_filters.calculate_filters(market_state)
            
            # Calculate P_rb
            P_rb, components = self.calculate_pjs(market_state, opportunity, filters_product)
            
            # Generate signal
            signal = 1 if P_rb > 0 else 0
            
            results.append({
                'timestamp': df_calc.index[idx],
                'P_rb': P_rb,
                'signal': signal,
                **components
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {results_df['signal'].sum()} signals (out of {len(results_df)} bars)")
        
        return results_df


# Testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("RULE-BASED STRATEGY TEST (OPTIMIZED)")
    print("=" * 80)
    
    # Test with sample data
    np.random.seed(42)
    n_bars = 1000
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n_bars)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    print(f"\nTest data: {len(df)} bars")
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"\n✅ Signals generated: {len(signals_df)}")
    print(f"Buy signals: {signals_df['signal'].sum()}")
    print(f"Signal rate: {signals_df['signal'].mean():.2%}")
    
    print("\nP_rb Statistics:")
    print(signals_df[['P_rb', 'W_opportunity', 'filters_product', 'signal']].describe())
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)