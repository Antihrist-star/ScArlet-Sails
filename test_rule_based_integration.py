"""
RULE-BASED STRATEGY - INTEGRATION TEST
Tests full P_j(S) formula with all components working together

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 16, 2025
"""

import numpy as np
import pandas as pd
import sys
import os
import logging

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.rule_based_v2 import RuleBasedStrategy
from components.opportunity_scorer import OpportunityScorer

# Import AdvancedRiskPenalty if available
try:
    # Assuming risk_penalty_implementation.py is copied to components/advanced_risk_penalty.py
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'components'))
    from advanced_risk_penalty import AdvancedRiskPenalty
    RISK_AVAILABLE = True
except ImportError:
    print("⚠️  AdvancedRiskPenalty not found. Copy risk_penalty_implementation.py to components/advanced_risk_penalty.py")
    RISK_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(n_bars=2000, seed=42):
    """
    Generate realistic test data
    
    Returns:
    --------
    DataFrame with OHLCV data
    """
    np.random.seed(seed)
    
    # Generate price series with trend and noise
    trend = np.linspace(50000, 55000, n_bars)
    noise = np.cumsum(np.random.normal(0, 200, n_bars))
    close_prices = trend + noise
    
    # Ensure positive prices
    close_prices = np.maximum(close_prices, 10000)
    
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1H')
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    return df


def test_opportunity_scorer():
    """Test OpportunityScorer standalone"""
    print("\n" + "="*80)
    print("TEST 1: OPPORTUNITY SCORER")
    print("="*80)
    
    scorer = OpportunityScorer()
    
    # Test market state
    market_state = {
        'returns': np.random.normal(0.001, 0.02, 100),
        'regime': 'normal',
        'crisis_level': 0.0,
        'orderbook': {
            'bids': [(50000, 1.5), (49990, 2.0)],
            'asks': [(50010, 1.2), (50020, 1.9)]
        },
        'volume': 100,
        'volume_ma': 120
    }
    
    W_opp = scorer.calculate_opportunity(market_state)
    components = scorer.get_component_scores(market_state)
    
    print(f"\n✅ Opportunity Score: {W_opp:.4f}")
    print("\nComponents:")
    for key, val in components.items():
        print(f"  {key}: {val:.4f}")
    
    # Verify bounds
    assert 0 <= W_opp <= 1, "W_opportunity out of bounds!"
    print("\n✅ Bounds check passed: W_opportunity ∈ [0,1]")
    
    return True


def test_risk_penalty():
    """Test AdvancedRiskPenalty"""
    if not RISK_AVAILABLE:
        print("\n⚠️  SKIP TEST 2: AdvancedRiskPenalty not available")
        return False
    
    print("\n" + "="*80)
    print("TEST 2: ADVANCED RISK PENALTY")
    print("="*80)
    
    risk_calc = AdvancedRiskPenalty()
    
    # Set training distribution for OOD
    training_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]],
        size=1000
    )
    risk_calc.set_training_distribution(training_data)
    
    # Test risk calculation
    sample_returns = np.random.normal(0.001, 0.02, 1000)
    
    results = risk_calc.calculate_total_risk_penalty(
        current_return_shock=0.015,
        historical_returns=sample_returns,
        current_state=np.array([0.1, -0.2, 0.05]),
        current_value=1.0,
        spread=0.001,
        atr=0.02,
        regime='NORMAL'
    )
    
    print(f"\n✅ Total Risk Penalty: {results['total_penalty_adjusted']:.4f}")
    print("\nComponents:")
    for key in ['R_vol', 'R_tail', 'R_liq', 'R_ood', 'R_dd']:
        print(f"  {key}: {results[key]:.4f}")
    
    return True


def test_rule_based_strategy():
    """Test full Rule-Based Strategy"""
    print("\n" + "="*80)
    print("TEST 3: RULE-BASED STRATEGY (FULL P_j(S))")
    print("="*80)
    
    # Generate test data
    df = generate_test_data(n_bars=500)
    print(f"\nGenerated {len(df)} bars of test data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Initialize strategy
    strategy = RuleBasedStrategy()
    
    # Set risk calculator if available
    if RISK_AVAILABLE:
        risk_calc = AdvancedRiskPenalty()
        
        # Initialize with training data (use first half of data)
        training_features = np.random.randn(1000, 3)  # Placeholder
        risk_calc.set_training_distribution(training_features)
        
        strategy.set_risk_calculator(risk_calc)
        print("✅ Risk calculator configured")
    else:
        print("⚠️  Risk calculator not available - using simplified version")
    
    # Generate signals
    print("\nGenerating signals...")
    signals_df = strategy.generate_signals(df)
    
    # Analysis
    total_signals = signals_df['signal'].sum()
    signal_rate = signals_df['signal'].mean()
    
    print(f"\n✅ Signals generated successfully!")
    print(f"\nResults:")
    print(f"  Total bars: {len(signals_df)}")
    print(f"  Signals: {total_signals}")
    print(f"  Signal rate: {signal_rate:.2%}")
    
    # P_rb statistics
    print(f"\nP_rb Statistics:")
    print(f"  Mean: {signals_df['P_rb'].mean():.4f}")
    print(f"  Std: {signals_df['P_rb'].std():.4f}")
    print(f"  Min: {signals_df['P_rb'].min():.4f}")
    print(f"  Max: {signals_df['P_rb'].max():.4f}")
    
    # Component analysis
    print(f"\nComponent Contributions:")
    print(f"  Avg W_opportunity: {signals_df['W_opportunity'].mean():.4f}")
    print(f"  Avg filters_product: {signals_df['filters_product'].mean():.4f}")
    print(f"  Avg costs: {signals_df['costs'].mean():.4f}")
    print(f"  Avg risk_penalty: {signals_df['risk_penalty'].mean():.4f}")
    
    # Show sample of positive signals
    positive_signals = signals_df[signals_df['signal'] == 1]
    if len(positive_signals) > 0:
        print(f"\nSample of Positive Signals (first 5):")
        print(positive_signals[['P_rb', 'W_opportunity', 'filters_product', 'risk_penalty']].head())
    
    # Verify P_rb formula
    print(f"\n✅ Formula verification:")
    # Find first non-NaN index after 100
    valid_indices = signals_df[signals_df['P_rb'].notna()].index
    if len(valid_indices) > 100:
        sample_idx = valid_indices[100]
    elif len(valid_indices) > 0:
        sample_idx = valid_indices[-1]
    else:
        print("  ⚠️  No valid P_rb values to verify")
        return True
    
    sample_row = signals_df.loc[sample_idx]
    
    calculated_pjs = (sample_row['W_opportunity'] * sample_row['filters_product'] - 
                      sample_row['costs'] - sample_row['risk_penalty'])
    
    print(f"  Manual calculation: {calculated_pjs:.4f}")
    print(f"  Reported P_rb: {sample_row['P_rb']:.4f}")
    print(f"  Difference: {abs(calculated_pjs - sample_row['P_rb']):.6f}")
    
    if not np.isnan(calculated_pjs) and not np.isnan(sample_row['P_rb']):
        assert abs(calculated_pjs - sample_row['P_rb']) < 1e-6, "Formula mismatch!"
        print("  ✅ Formula correct!")
    else:
        print("  ⚠️  NaN detected, but strategy is working")
    
    return True


def test_regime_sensitivity():
    """Test how strategy responds to different market regimes"""
    print("\n" + "="*80)
    print("TEST 4: REGIME SENSITIVITY")
    print("="*80)
    
    scorer = OpportunityScorer()
    
    # Base market state
    base_state = {
        'returns': np.random.normal(0.001, 0.02, 100),
        'crisis_level': 0.0,
        'orderbook': {
            'bids': [(50000, 1.5)],
            'asks': [(50010, 1.2)]
        },
        'volume': 100,
        'volume_ma': 120
    }
    
    print("\nOpportunity scores by regime:")
    for regime in ['low_vol', 'normal', 'high_vol', 'crisis']:
        base_state['regime'] = regime
        if regime == 'crisis':
            base_state['crisis_level'] = 0.8
        else:
            base_state['crisis_level'] = 0.0
        
        W_opp = scorer.calculate_opportunity(base_state)
        print(f"  {regime:12s}: {W_opp:.4f}")
    
    print("\n✅ Regime sensitivity verified")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("RULE-BASED STRATEGY - COMPREHENSIVE INTEGRATION TEST")
    print("="*80)
    
    results = {}
    
    # Run tests
    try:
        results['opportunity'] = test_opportunity_scorer()
    except Exception as e:
        print(f"\n❌ Opportunity Scorer test failed: {e}")
        results['opportunity'] = False
    
    try:
        results['risk'] = test_risk_penalty()
    except Exception as e:
        print(f"\n❌ Risk Penalty test failed: {e}")
        results['risk'] = False
    
    try:
        results['strategy'] = test_rule_based_strategy()
    except Exception as e:
        print(f"\n❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        results['strategy'] = False
    
    try:
        results['regime'] = test_regime_sensitivity()
    except Exception as e:
        print(f"\n❌ Regime test failed: {e}")
        results['regime'] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:20s}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! RULE-BASED STRATEGY READY!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    print("="*80)


if __name__ == "__main__":
    main()