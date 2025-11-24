"""
Advanced Risk Penalty Implementation for Algorithmic Cryptocurrency Trading
================================================================================

This script implements the mathematical framework for R_penalty(S) with all components:
- Volatility Risk (GARCH-based)
- Tail Risk (CVaR)
- Liquidity Risk
- OOD Risk (Mahalanobis Distance)
- Drawdown Risk

Author: Financial Risk Management Specialist
Date: 2025-01-16
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskPenalty:
    """
    Advanced Risk Penalty Calculator for Algorithmic Trading
    
    Implements the comprehensive risk penalty framework with multiple components:
    R_penalty(S) = w_vol*R_vol + w_tail*R_tail + w_liq*R_liq + w_ood*R_ood + w_dd*R_dd
    """
    
    def __init__(self, 
                 weights={'vol': 0.30, 'tail': 0.20, 'liq': 0.20, 'ood': 0.15, 'dd': 0.15},
                 garch_params={'omega': 0.0001, 'alpha': 0.05, 'beta': 0.90},
                 var_confidence=0.95,
                 regime_multipliers={'BULL': 1.0, 'BEAR': 1.5, 'SIDEWAYS': 1.2, 'CRISIS': 3.0}):
        """
        Initialize the risk penalty calculator
        
        Parameters:
        -----------
        weights : dict
            Component weights (must sum to 1.0)
        garch_params : dict
            GARCH(1,1) parameters
        var_confidence : float
            VaR confidence level
        regime_multipliers : dict
            Market regime multipliers
        """
        self.weights = weights
        self.garch_params = garch_params
        self.var_confidence = var_confidence
        self.regime_multipliers = regime_multipliers
        
        # State variables
        self.garch_variance = garch_params['omega'] / (1 - garch_params['alpha'] - garch_params['beta'])
        self.training_mean = None
        self.training_cov = None
        self.training_cov_inv = None
        self.peak_value = 1.0  # Normalized portfolio value
        
        # Validation
        assert abs(sum(weights.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
        assert garch_params['alpha'] + garch_params['beta'] < 1.0, "GARCH parameters must satisfy stationarity"
    
    def update_garch_variance(self, return_shock):
        """
        Update GARCH variance estimate
        
        Parameters:
        -----------
        return_shock : float
            Current return shock (residual)
        """
        omega, alpha, beta = self.garch_params['omega'], self.garch_params['alpha'], self.garch_params['beta']
        self.garch_variance = omega + alpha * (return_shock**2) + beta * self.garch_variance
    
    def calculate_volatility_risk(self, regime='BULL'):
        """
        Calculate volatility risk component using GARCH
        
        R_vol(S) = λ * σ_GARCH(S) * ρ(ψ_t)
        
        Returns:
        --------
        float : Volatility risk penalty
        """
        lambda_param = 10.0  # Risk aversion coefficient
        sigma_garch = np.sqrt(self.garch_variance)
        regime_multiplier = self.regime_multipliers.get(regime, 1.0)
        
        R_vol = lambda_param * sigma_garch * regime_multiplier
        return R_vol
    
    def calculate_tail_risk(self, returns, confidence=None):
        """
        Calculate tail risk using Conditional Value at Risk (CVaR)
        
        R_tail(S) = CVaR_α(returns) = E[X | X ≤ VaR_α]
        
        Parameters:
        -----------
        returns : array-like
            Historical returns for VaR/CVaR calculation
        confidence : float, optional
            VaR confidence level (default: self.var_confidence)
        
        Returns:
        --------
        float : Tail risk penalty
        """
        if confidence is None:
            confidence = self.var_confidence
        
        returns = np.array(returns)
        
        # Calculate VaR
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        
        # Calculate CVaR (expected loss beyond VaR)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
        else:
            cvar = var_threshold  # Fallback
        
        # Return absolute value as penalty
        R_tail = abs(cvar)
        return R_tail
    
    def calculate_liquidity_risk(self, spread, atr, depth_penalty=0.0, impact_cost=0.0, kappa=5.0):
        """
        Calculate liquidity risk component
        
        R_liq(S) = κ * (spread/ATR + depth_penalty + impact_cost)
        
        Parameters:
        -----------
        spread : float
            Current bid-ask spread
        atr : float
            Average True Range for normalization
        depth_penalty : float
            Penalty for insufficient orderbook depth
        impact_cost : float
            Estimated market impact cost
        kappa : float
            Liquidity sensitivity parameter
        
        Returns:
        --------
        float : Liquidity risk penalty
        """
        if atr == 0:
            spread_ratio = 0.0
        else:
            spread_ratio = spread / atr
        
        R_liq = kappa * (spread_ratio + depth_penalty + impact_cost)
        return R_liq
    
    def set_training_distribution(self, training_data):
        """
        Set training distribution for OOD detection
        
        Parameters:
        -----------
        training_data : array-like
            Training data for Mahalanobis distance calculation
        """
        training_data = np.array(training_data)
        self.training_mean = np.mean(training_data, axis=0)
        self.training_cov = np.cov(training_data.T)
        
        # Add small regularization term for numerical stability
        self.training_cov += 1e-6 * np.eye(self.training_cov.shape[0])
        self.training_cov_inv = inv(self.training_cov)
    
    def calculate_ood_risk(self, current_state, threshold=3.0):
        """
        Calculate out-of-distribution risk using Mahalanobis distance
        
        R_ood(S) = D_M(Φ(S), X_train) / τ
        
        Parameters:
        -----------
        current_state : array-like
            Current state features
        threshold : float
            OOD threshold for normalization
        
        Returns:
        --------
        float : OOD risk penalty
        """
        if self.training_mean is None or self.training_cov_inv is None:
            raise ValueError("Training distribution not set. Call set_training_distribution() first.")
        
        current_state = np.array(current_state)
        diff = current_state - self.training_mean
        
        # Mahalanobis distance
        mahal_distance = np.sqrt(np.dot(np.dot(diff.T, self.training_cov_inv), diff))
        
        R_ood = mahal_distance / threshold
        return R_ood
    
    def update_portfolio_peak(self, current_value):
        """
        Update portfolio peak value for drawdown calculation
        
        Parameters:
        -----------
        current_value : float
            Current portfolio value (normalized)
        """
        self.peak_value = max(self.peak_value, current_value)
    
    def calculate_drawdown_risk(self, current_value, dd_limit=0.15, beta=5.0):
        """
        Calculate drawdown risk component
        
        R_dd(S) = exp(β * DD_current / DD_limit) - 1
        
        Parameters:
        -----------
        current_value : float
            Current portfolio value (normalized)
        dd_limit : float
            Maximum allowed drawdown
        beta : float
            Steepness parameter
        
        Returns:
        --------
        float : Drawdown risk penalty
        """
        if self.peak_value == 0:
            current_dd = 0.0
        else:
            current_dd = (self.peak_value - current_value) / self.peak_value
        
        if current_dd <= 0:
            R_dd = 0.0
        else:
            R_dd = np.exp(beta * current_dd / dd_limit) - 1
        
        return R_dd
    
    def calculate_total_risk_penalty(self, 
                                   current_return_shock,
                                   historical_returns,
                                   current_state,
                                   current_value=1.0,
                                   spread=0.001,
                                   atr=0.02,
                                   regime='BULL',
                                   **kwargs):
        """
        Calculate total risk penalty with all components
        
        R_penalty(S) = Σ w_i * Risk_i(S)
        
        Parameters:
        -----------
        current_return_shock : float
            Current return shock for GARCH update
        historical_returns : array-like
            Historical returns for tail risk calculation
        current_state : array-like
            Current state for OOD detection
        current_value : float
            Current portfolio value (normalized)
        spread : float
            Current bid-ask spread
        atr : float
            Average True Range
        regime : str
            Current market regime
        **kwargs : additional parameters for liquidity risk
        
        Returns:
        --------
        dict : Dictionary with all risk components and total penalty
        """
        # Update state variables
        self.update_garch_variance(current_return_shock)
        self.update_portfolio_peak(current_value)
        
        # Calculate individual risk components
        R_vol = self.calculate_volatility_risk(regime)
        R_tail = self.calculate_tail_risk(historical_returns)
        R_liq = self.calculate_liquidity_risk(spread, atr, **kwargs)
        R_ood = self.calculate_ood_risk(current_state)
        R_dd = self.calculate_drawdown_risk(current_value)
        
        # Calculate weighted total penalty
        total_penalty = (self.weights['vol'] * R_vol + 
                        self.weights['tail'] * R_tail + 
                        self.weights['liq'] * R_liq + 
                        self.weights['ood'] * R_ood + 
                        self.weights['dd'] * R_dd)
        
        # Apply regime multiplier
        regime_multiplier = self.regime_multipliers.get(regime, 1.0)
        total_penalty_adj = total_penalty * regime_multiplier
        
        return {
            'R_vol': R_vol,
            'R_tail': R_tail,
            'R_liq': R_liq,
            'R_ood': R_ood,
            'R_dd': R_dd,
            'total_penalty': total_penalty,
            'regime_multiplier': regime_multiplier,
            'total_penalty_adjusted': total_penalty_adj,
            'weights': self.weights
        }


def demonstrate_risk_penalty():
    """
    Demonstrate the risk penalty calculation with sample data
    """
    print("=" * 80)
    print("ADVANCED RISK PENALTY DEMONSTRATION")
    print("=" * 80)
    
    # Initialize risk penalty calculator
    risk_calc = AdvancedRiskPenalty()
    
    # Generate sample training data for OOD detection
    np.random.seed(42)
    training_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=1000
    )
    risk_calc.set_training_distribution(training_data)
    
    # Sample market data
    sample_returns = np.random.normal(0.001, 0.02, 1000)  # 1000 historical returns
    sample_returns[-50:] = sample_returns[-50:] - 0.05  # Add some tail events
    
    print("\n1. VOLATILITY RISK CALCULATION")
    print("-" * 40)
    
    # Simulate GARCH updating
    for i in range(10):
        return_shock = np.random.normal(0, 0.02)
        risk_calc.update_garch_variance(return_shock)
    
    R_vol = risk_calc.calculate_volatility_risk(regime='BEAR')
    print(f"Current GARCH variance: {risk_calc.garch_variance:.6f}")
    print(f"Annualized volatility: {np.sqrt(risk_calc.garch_variance * 365):.2%}")
    print(f"Volatility risk penalty: {R_vol:.4f}")
    
    print("\n2. TAIL RISK CALCULATION")
    print("-" * 40)
    
    R_tail = risk_calc.calculate_tail_risk(sample_returns)
    var_95 = np.percentile(sample_returns, 5)
    print(f"VaR (95%): {var_95:.4f} ({var_95:.2%})")
    print(f"CVaR/Tail risk penalty: {R_tail:.4f} ({R_tail:.2%})")
    
    print("\n3. LIQUIDITY RISK CALCULATION")
    print("-" * 40)
    
    spread = 0.001  # 10 bps spread
    atr = 0.02      # 2% ATR
    depth_penalty = 0.05  # Moderate depth issue
    impact_cost = 0.03    # 3 bps impact cost
    
    R_liq = risk_calc.calculate_liquidity_risk(spread, atr, depth_penalty, impact_cost)
    print(f"Spread/ATR ratio: {spread/atr:.4f}")
    print(f"Total liquidity penalty: {R_liq:.4f}")
    
    print("\n4. OOD RISK CALCULATION")
    print("-" * 40)
    
    # Normal state
    normal_state = np.array([0.1, -0.2, 0.05])
    R_ood_normal = risk_calc.calculate_ood_risk(normal_state)
    print(f"Normal state OOD risk: {R_ood_normal:.4f}")
    
    # Extreme state
    extreme_state = np.array([2.5, -1.8, 1.2])
    R_ood_extreme = risk_calc.calculate_ood_risk(extreme_state)
    print(f"Extreme state OOD risk: {R_ood_extreme:.4f}")
    
    print("\n5. DRAWDOWN RISK CALCULATION")
    print("-" * 40)
    
    # Simulate portfolio decline
    current_value = 0.90  # 10% drawdown
    risk_calc.update_portfolio_peak(1.0)  # Set peak at 1.0
    R_dd = risk_calc.calculate_drawdown_risk(current_value)
    current_dd = (risk_calc.peak_value - current_value) / risk_calc.peak_value
    print(f"Current drawdown: {current_dd:.2%}")
    print(f"Drawdown risk penalty: {R_dd:.4f}")
    
    print("\n6. COMPREHENSIVE RISK PENALTY")
    print("-" * 40)
    
    # Calculate total risk penalty
    results = risk_calc.calculate_total_risk_penalty(
        current_return_shock=0.015,
        historical_returns=sample_returns,
        current_state=extreme_state,
        current_value=current_value,
        spread=spread,
        atr=atr,
        regime='CRISIS',
        depth_penalty=depth_penalty,
        impact_cost=impact_cost
    )
    
    print("Component Breakdown:")
    for component in ['R_vol', 'R_tail', 'R_liq', 'R_ood', 'R_dd']:
        weight = results['weights'][component.split('_')[1]]
        weighted_risk = weight * results[component]
        print(f"  {component}: {results[component]:.4f} (w={weight:.2f}, weighted={weighted_risk:.4f})")
    
    print(f"\nTotal penalty (before regime): {results['total_penalty']:.4f}")
    print(f"Regime multiplier: {results['regime_multiplier']:.1f}x")
    print(f"Final adjusted penalty: {results['total_penalty_adjusted']:.4f}")
    
    print("\n7. TRADE DECISION EXAMPLE")
    print("-" * 40)
    
    # Simulate a trading opportunity
    opportunity_score = 0.25  # 25% expected return
    risk_adjusted_score = opportunity_score - results['total_penalty_adjusted']
    
    print(f"Opportunity score: {opportunity_score:.4f}")
    print(f"Risk penalty: {results['total_penalty_adjusted']:.4f}")
    print(f"Risk-adjusted score: {risk_adjusted_score:.4f}")
    
    if risk_adjusted_score > 0.05:  # Minimum threshold
        print("✅ TRADE APPROVED - Risk-adjusted return positive")
    else:
        print("❌ TRADE REJECTED - Risk penalty too high")
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_risk_penalty()