"""
RULE-BASED STRATEGY - FULL P_j(S) IMPLEMENTATION
Mathematical formula from MATHEMATICAL_FRAMEWORK.md

P_rb(S) = W_opportunity(S) · ∏ᵢ Iᵢ(S) - C_fixed(S) - R_penalty(S)

Components:
- W_opportunity: Opportunity scoring (volatility + liquidity + microstructure)
- ∏ᵢ Iᵢ: Technical filters (RSI, EMA, Volume, BB)
- C_fixed: Transaction costs
- R_penalty: Advanced risk penalty (GARCH, CVaR, Liquidity, OOD, Drawdown)

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.opportunity_scorer import OpportunityScorer

logger = logging.getLogger(__name__)


class TechnicalFilters:
    """
    Technical indicator-based filters for Rule-Based strategy
    
    Filters:
    - I₁: RSI filter (20 < RSI < 80)
    - I₂: EMA trend filter (price > EMA9)
    - I₃: Volume confirmation (volume > MA)
    - I₄: Bollinger Band position (-0.5 < BB_pos < 0.5)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # RSI parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_lower = self.config.get('rsi_lower', 20)
        self.rsi_upper = self.config.get('rsi_upper', 80)
        
        # EMA parameters
        self.ema_fast = self.config.get('ema_fast', 9)
        
        # BB parameters
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        
        logger.info("TechnicalFilters initialized")
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate RSI indicator"""
        if period is None:
            period = self.rsi_period
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def rsi_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        I₁: RSI filter
        Returns binary series: 1 if within range, 0 otherwise
        """
        rsi = self.calculate_rsi(df['close'])
        filter_signal = ((rsi > self.rsi_lower) & (rsi < self.rsi_upper)).astype(int)
        return filter_signal
    
    def ema_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        I₂: EMA trend filter
        Returns binary series: 1 if price > EMA, 0 otherwise
        """
        ema = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        filter_signal = (df['close'] > ema).astype(int)
        return filter_signal
    
    def volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        I₃: Volume confirmation filter
        Returns binary series: 1 if volume > MA, 0 otherwise
        """
        volume_ma = df['volume'].rolling(window=14).mean()
        filter_signal = (df['volume'] > volume_ma).astype(int)
        return filter_signal
    
    def bollinger_filter(self, df: pd.DataFrame) -> pd.Series:
        """
        I₄: Bollinger Band position filter
        Returns binary series: 1 if within middle range, 0 otherwise
        """
        sma = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        
        bb_upper = sma + (self.bb_std * std)
        bb_lower = sma - (self.bb_std * std)
        
        # BB position: 0 = lower band, 0.5 = middle, 1 = upper band
        bb_position = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Filter: accept if in middle range [-0.5, 0.5] relative to center
        # Which is [0.25, 0.75] in absolute terms
        filter_signal = ((bb_position > 0.25) & (bb_position < 0.75)).astype(int)
        
        return filter_signal
    
    def calculate_all_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all filters and return as DataFrame
        
        Returns:
        --------
        DataFrame with columns: I1_rsi, I2_ema, I3_volume, I4_bb, product
        """
        filters = pd.DataFrame(index=df.index)
        
        filters['I1_rsi'] = self.rsi_filter(df)
        filters['I2_ema'] = self.ema_filter(df)
        filters['I3_volume'] = self.volume_filter(df)
        filters['I4_bb'] = self.bollinger_filter(df)
        
        # Product of all filters (binary AND)
        filters['product'] = (filters['I1_rsi'] * filters['I2_ema'] * 
                             filters['I3_volume'] * filters['I4_bb'])
        
        return filters


class RuleBasedStrategy:
    """
    Rule-Based Trading Strategy with full P_j(S) formula
    
    Decision function:
    P_rb(S) = W_opportunity(S) · ∏ᵢ Iᵢ(S) - C_fixed - R_penalty(S)
    
    Where:
    - W_opportunity: Market opportunity score [0,1]
    - ∏ᵢ Iᵢ: Product of technical filters {0,1}
    - C_fixed: Fixed transaction costs
    - R_penalty: Comprehensive risk penalty
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Rule-Based Strategy
        
        Parameters:
        -----------
        config : dict
            Configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.opportunity_scorer = OpportunityScorer(self.config.get('opportunity', {}))
        self.technical_filters = TechnicalFilters(self.config.get('filters', {}))
        
        # Costs
        self.commission = self.config.get('commission', 0.001)  # 0.1%
        self.slippage = self.config.get('slippage', 0.0005)    # 0.05%
        self.C_fixed = self.commission + self.slippage
        
        # Risk penalty - will be set externally
        self.risk_calculator = None  # AdvancedRiskPenalty instance
        
        logger.info(f"RuleBasedStrategy initialized: C_fixed={self.C_fixed:.4f}")
    
    def set_risk_calculator(self, risk_calc):
        """Set the AdvancedRiskPenalty calculator instance"""
        self.risk_calculator = risk_calc
        logger.info("Risk calculator set")
    
    def calculate_pjs(self, market_state: Dict, df: pd.DataFrame, idx: int) -> Tuple[float, Dict]:
        """
        Calculate P_rb(S) for a given market state
        
        Parameters:
        -----------
        market_state : dict
            Current market state for opportunity scoring
        df : DataFrame
            Historical OHLCV data for technical filters
        idx : int
            Current index in dataframe
        
        Returns:
        --------
        tuple : (P_rb value, dict of components)
        """
        # Component 1: Opportunity score
        W_opportunity = self.opportunity_scorer.calculate_opportunity(market_state)
        
        # Component 2: Technical filters
        # Calculate filters on historical data up to current point
        df_history = df.iloc[:idx+1]
        
        if len(df_history) < 50:  # Not enough history
            filters_product = 0
        else:
            filters = self.technical_filters.calculate_all_filters(df_history)
            filters_product = filters['product'].iloc[-1]
        
        # Component 3: Costs (fixed)
        costs = self.C_fixed
        
        # Component 4: Risk penalty
        if self.risk_calculator is not None:
            try:
                # Prepare inputs for risk calculator
                current_return_shock = market_state.get('return_shock', 0.0)
                historical_returns = market_state.get('returns', np.array([0]))
                current_state_features = market_state.get('state_features', np.zeros(3))
                current_value = market_state.get('portfolio_value', 1.0)
                spread = market_state.get('spread', 0.001)
                atr = market_state.get('atr', 0.02)
                regime = market_state.get('regime', 'normal')
                
                risk_results = self.risk_calculator.calculate_total_risk_penalty(
                    current_return_shock=current_return_shock,
                    historical_returns=historical_returns,
                    current_state=current_state_features,
                    current_value=current_value,
                    spread=spread,
                    atr=atr,
                    regime=regime.upper()
                )
                
                risk_penalty = risk_results['total_penalty_adjusted']
            except Exception as e:
                logger.warning(f"Risk calculation failed: {e}. Using default penalty.")
                risk_penalty = 0.1  # Default conservative penalty
        else:
            risk_penalty = 0.0  # No risk penalty if calculator not set
        
        # Calculate P_rb(S)
        P_rb = W_opportunity * filters_product - costs - risk_penalty
        
        # Store component breakdown
        components = {
            'W_opportunity': W_opportunity,
            'filters_product': filters_product,
            'costs': costs,
            'risk_penalty': risk_penalty,
            'P_rb': P_rb
        }
        
        logger.debug(f"P_rb={P_rb:.4f}: W_opp={W_opportunity:.3f} × filters={filters_product} - costs={costs:.4f} - risk={risk_penalty:.4f}")
        
        return P_rb, components
    
    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """
        Generate trading signals for entire dataframe
        
        Parameters:
        -----------
        df : DataFrame
            OHLCV data with columns: open, high, low, close, volume
        market_states : list, optional
            Pre-computed market states for each timestamp
            If None, will be computed from df
        
        Returns:
        --------
        DataFrame with columns: P_rb, signal, components
        """
        results = []
        
        logger.info(f"Generating signals for {len(df)} bars...")
        
        for idx in range(len(df)):
            # Prepare market state
            if market_states and idx < len(market_states):
                market_state = market_states[idx]
            else:
                # Create minimal market state from df
                lookback = min(100, idx)
                recent_returns = df['close'].pct_change().iloc[max(0, idx-lookback):idx+1].values
                
                market_state = {
                    'returns': recent_returns,
                    'regime': 'normal',  # Default
                    'crisis_level': 0.0,
                    'orderbook': None,  # Not available from OHLCV
                    'volume': df['volume'].iloc[idx],
                    'volume_ma': df['volume'].rolling(14).mean().iloc[idx],
                    'return_shock': recent_returns[-1] if len(recent_returns) > 0 else 0.0,
                    'state_features': np.zeros(3),  # Placeholder
                    'portfolio_value': 1.0,
                    'spread': 0.001,
                    'atr': 0.02,  # Placeholder
                }
            
            # Calculate P_rb
            P_rb, components = self.calculate_pjs(market_state, df, idx)
            
            # Generate signal
            # Signal = 1 if P_rb > threshold, 0 otherwise
            signal = 1 if P_rb > 0.05 else 0
            
            results.append({
                'timestamp': df.index[idx],
                'P_rb': P_rb,
                'signal': signal,
                **components
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {results_df['signal'].sum()} signals (out of {len(df)} bars)")
        
        return results_df


# Testing and demonstration
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("RULE-BASED STRATEGY TEST")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    
    # Generate realistic OHLCV data
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, 1000)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, 1000))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, 1000))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, 1000)
    }, index=dates)
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"\nSignals generated: {len(signals_df)}")
    print(f"Total signals: {signals_df['signal'].sum()}")
    print(f"Signal rate: {signals_df['signal'].mean():.2%}")
    
    print("\nSample P_rb values:")
    print(signals_df[['P_rb', 'W_opportunity', 'filters_product', 'signal']].describe())
    
    print("\nFirst 10 signals:")
    print(signals_df[signals_df['signal'] == 1].head(10)[['P_rb', 'W_opportunity', 'filters_product']])
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)