"""
HYBRID STRATEGY - FULL P_j(S) IMPLEMENTATION
Mathematical formula from MATHEMATICAL_FRAMEWORK.md

P_hyb(S) = α(t)·P_rb(S) + β(t)·P_ml(S) + γ·𝔼[V_future(S)]

Components:
- α(t), β(t): Adaptive weights based on rolling performance
- P_rb(S): Rule-Based strategy signal
- P_ml(S): XGBoost ML strategy signal
- γ·𝔼[V_future(S)]: RL future value component (placeholder for Phase 4)

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.rule_based_v2 import RuleBasedStrategy
from strategies.xgboost_ml_v2 import XGBoostMLStrategy

logger = logging.getLogger(__name__)


class AdaptiveWeightCalculator:
    """
    Calculate adaptive weights α(t), β(t) based on rolling performance
    
    Methodology:
    - Track recent performance of each strategy
    - Compute rolling Sharpe ratio
    - Softmax to get weights that sum to (1 - γ_weight)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.window = self.config.get('window', 20)  # Rolling window
        self.gamma_weight = self.config.get('gamma_weight', 0.10)  # Reserve for RL
        
        # Performance tracking
        self.rb_returns = []
        self.ml_returns = []
        
        # Default weights (when no performance history)
        self.default_alpha = 0.45
        self.default_beta = 0.45
        
        logger.info(f"AdaptiveWeightCalculator initialized: window={self.window}, γ_weight={self.gamma_weight}")
    
    def update_performance(self, rb_return: float, ml_return: float):
        """
        Update performance tracking
        
        Parameters:
        -----------
        rb_return : float
            Return from Rule-Based strategy
        ml_return : float
            Return from ML strategy
        """
        self.rb_returns.append(rb_return)
        self.ml_returns.append(ml_return)
        
        # Keep only recent window
        if len(self.rb_returns) > self.window:
            self.rb_returns = self.rb_returns[-self.window:]
            self.ml_returns = self.ml_returns[-self.window:]
    
    def calculate_sharpe(self, returns: list) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        sharpe = mean_return / std_return
        return sharpe
    
    def calculate_weights(self) -> Tuple[float, float]:
        """
        Calculate adaptive weights α(t), β(t)
        
        Returns:
        --------
        tuple : (alpha, beta)
        """
        # Not enough history - use defaults
        if len(self.rb_returns) < 5:
            alpha = self.default_alpha
            beta = self.default_beta
            return alpha, beta
        
        # Calculate Sharpe ratios
        sharpe_rb = self.calculate_sharpe(self.rb_returns)
        sharpe_ml = self.calculate_sharpe(self.ml_returns)
        
        # Softmax to get weights
        # Add small epsilon to avoid division by zero
        exp_rb = np.exp(sharpe_rb + 1e-6)
        exp_ml = np.exp(sharpe_ml + 1e-6)
        
        total_exp = exp_rb + exp_ml
        
        # Weights sum to (1 - gamma_weight)
        available_weight = 1.0 - self.gamma_weight
        
        alpha = (exp_rb / total_exp) * available_weight
        beta = (exp_ml / total_exp) * available_weight
        
        # Clip to reasonable bounds
        alpha = np.clip(alpha, 0.1, 0.8)
        beta = np.clip(beta, 0.1, 0.8)
        
        # Renormalize if needed
        total = alpha + beta
        if total > available_weight:
            scale = available_weight / total
            alpha *= scale
            beta *= scale
        
        return float(alpha), float(beta)


class RLComponentPlaceholder:
    """
    Placeholder for Reinforcement Learning component
    Will be replaced with actual DQN in Phase 4
    
    For now: returns constant small value
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.gamma = self.config.get('gamma', 0.95)
        
        logger.info("RLComponentPlaceholder initialized (Phase 4: replace with DQN)")
    
    def get_state_value(self, state: Dict) -> float:
        """
        Placeholder V(S) calculation
        
        In Phase 4, this will be:
        V(S) = max_a Q_θ(S,a) from trained DQN
        
        For now: return small positive value to show structure
        """
        # Simple heuristic based on market state
        regime = state.get('regime', 'normal')
        crisis_level = state.get('crisis_level', 0.0)
        
        # Lower value in crisis
        if regime == 'crisis' or crisis_level > 0.5:
            return 0.01
        else:
            return 0.05
    
    def calculate_rl_component(self, state: Dict) -> float:
        """
        Calculate γ·𝔼[V_future(S)]
        
        Returns:
        --------
        float : RL contribution to P_hyb
        """
        V_future = self.get_state_value(state)
        rl_component = self.gamma * V_future
        
        return rl_component


class HybridStrategy:
    """
    Hybrid Trading Strategy combining Rule-Based, ML, and RL
    
    Decision function:
    P_hyb(S) = α(t)·P_rb(S) + β(t)·P_ml(S) + γ·𝔼[V_future(S)]
    
    Features:
    - Adaptive weighting based on rolling performance
    - Multi-strategy ensemble
    - RL-enhanced decision making (placeholder for Phase 4)
    """
    
    def __init__(self, config: Dict = None, 
                 rb_strategy: RuleBasedStrategy = None,
                 ml_strategy: XGBoostMLStrategy = None):
        """
        Initialize Hybrid Strategy
        
        Parameters:
        -----------
        config : dict
            Configuration parameters
        rb_strategy : RuleBasedStrategy
            Pre-initialized Rule-Based strategy (optional)
        ml_strategy : XGBoostMLStrategy
            Pre-initialized ML strategy (optional)
        """
        self.config = config or {}
        
        # Initialize component strategies
        self.rb_strategy = rb_strategy or RuleBasedStrategy(self.config.get('rule_based', {}))
        self.ml_strategy = ml_strategy or XGBoostMLStrategy(self.config.get('xgboost_ml', {}))
        
        # Initialize adaptive weights
        self.weight_calculator = AdaptiveWeightCalculator(self.config.get('weights', {}))
        
        # Initialize RL component (placeholder)
        self.rl_component = RLComponentPlaceholder(self.config.get('rl', {}))
        
        # Current weights
        self.alpha = self.weight_calculator.default_alpha
        self.beta = self.weight_calculator.default_beta
        self.gamma_weight = self.weight_calculator.gamma_weight
        
        logger.info(f"HybridStrategy initialized: α={self.alpha:.2f}, β={self.beta:.2f}, γ_weight={self.gamma_weight:.2f}")
    
    def calculate_pjs(self, rb_result: Tuple[float, Dict], 
                           ml_result: Tuple[float, Dict],
                           market_state: Dict) -> Tuple[float, Dict]:
        """
        Calculate P_hyb(S) from component strategy results
        
        Parameters:
        -----------
        rb_result : tuple
            (P_rb value, components dict) from Rule-Based strategy
        ml_result : tuple
            (P_ml value, components dict) from ML strategy
        market_state : dict
            Current market state for RL component
        
        Returns:
        --------
        tuple : (P_hyb value, dict of components)
        """
        P_rb, rb_components = rb_result
        P_ml, ml_components = ml_result
        
        # Calculate adaptive weights
        self.alpha, self.beta = self.weight_calculator.calculate_weights()
        
        # Calculate RL component
        rl_value = self.rl_component.calculate_rl_component(market_state)
        
        # Calculate P_hyb
        P_hyb = self.alpha * P_rb + self.beta * P_ml + rl_value
        
        # Store components
        components = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma_weight': self.gamma_weight,
            'P_rb': P_rb,
            'P_ml': P_ml,
            'rl_value': rl_value,
            'P_hyb': P_hyb,
            'rb_components': rb_components,
            'ml_components': ml_components
        }
        
        logger.debug(f"P_hyb={P_hyb:.4f}: α={self.alpha:.2f}×{P_rb:.3f} + β={self.beta:.2f}×{P_ml:.3f} + RL={rl_value:.3f}")
        
        return P_hyb, components
    
    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """
        Generate hybrid trading signals
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 75 columns (advanced features) from REAL DATA
        market_states : list, optional
            Pre-computed market states
        
        Returns:
        --------
        DataFrame with hybrid signals and all components
        """
        logger.info(f"Generating hybrid signals for {len(df)} bars...")
        
        # Generate Rule-Based signals
        rb_signals = self.rb_strategy.generate_signals(df, market_states)
        
        # Generate ML signals
        # NOTE: ML strategy now works with single DataFrame (74 advanced features)
        # No need for df_dict with multiple timeframes
        ml_signals = self.ml_strategy.generate_signals(df, market_states)
        
        # Ensure same index
        common_index = rb_signals.index.intersection(ml_signals.index)
        rb_signals = rb_signals.loc[common_index]
        ml_signals = ml_signals.loc[common_index]
        
        results = []
        
        for idx in common_index:
            # Get component signals
            P_rb = rb_signals.loc[idx, 'P_rb']
            P_ml = ml_signals.loc[idx, 'P_ml']
            
            rb_components = {
                'W_opportunity': rb_signals.loc[idx, 'W_opportunity'],
                'filters_product': rb_signals.loc[idx, 'filters_product'],
                'costs': rb_signals.loc[idx, 'costs'],
                'risk_penalty': rb_signals.loc[idx, 'risk_penalty']
            }
            
            ml_components = {
                'ml_score': ml_signals.loc[idx, 'ml_score'],
                'filters_product': ml_signals.loc[idx, 'filters_product'],
                'costs': ml_signals.loc[idx, 'costs'],
                'R_ood': ml_signals.loc[idx, 'R_ood']
            }
            
            # Market state
            if market_states and len(market_states) > len(results):
                market_state = market_states[len(results)]
            else:
                market_state = {
                    'regime': 'normal',
                    'crisis_level': 0.0
                }
            
            # Calculate hybrid
            P_hyb, hyb_components = self.calculate_pjs(
                (P_rb, rb_components),
                (P_ml, ml_components),
                market_state
            )
            
            # Update performance for adaptive weights
            # (Using P_j as proxy for return - simplified)
            self.weight_calculator.update_performance(P_rb, P_ml)
            
            # Generate signal
            # Lower threshold for testing with fallback models
            signal = 1 if P_hyb > -0.5 else 0
            
            results.append({
                'timestamp': idx,
                'P_hyb': P_hyb,
                'signal': signal,
                'alpha': hyb_components['alpha'],
                'beta': hyb_components['beta'],
                'P_rb': P_rb,
                'P_ml': P_ml,
                'rl_value': hyb_components['rl_value']
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {results_df['signal'].sum()} hybrid signals (out of {len(results_df)} bars)")
        
        return results_df


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("HYBRID STRATEGY TEST")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    n_bars = 500
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    
    close_prices = 50000 * np.exp(np.cumsum(np.random.normal(0.0001, 0.02, n_bars)))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    # Initialize strategy
    strategy = HybridStrategy()
    
    # Generate signals
    signals_df = strategy.generate_signals(df)
    
    print(f"\nSignals generated: {len(signals_df)}")
    print(f"Total hybrid signals: {signals_df['signal'].sum()}")
    print(f"Signal rate: {signals_df['signal'].mean():.2%}")
    
    print("\nP_hyb Statistics:")
    print(signals_df[['P_hyb', 'P_rb', 'P_ml', 'alpha', 'beta', 'signal']].describe())
    
    print("\nSample signals:")
    print(signals_df[signals_df['signal'] == 1].head(10)[['P_hyb', 'P_rb', 'P_ml', 'alpha', 'beta']])
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)