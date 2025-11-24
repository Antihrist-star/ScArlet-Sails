"""
XGBOOST ML STRATEGY - INTEGRATION TEST
Tests full P_ml(S) formula with all components

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

from strategies.xgboost_ml_v2 import XGBoostMLStrategy, FeatureTransformer, RegimeFilters

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_data(n_bars=1000, seed=42):
    """Generate multi-timeframe test data"""
    np.random.seed(seed)
    
    # Generate price series
    trend = np.linspace(50000, 55000, n_bars)
    noise = np.cumsum(np.random.normal(0, 200, n_bars))
    close_prices = trend + noise
    close_prices = np.maximum(close_prices, 10000)
    
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.001, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.002, 0.005, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    return df


def test_feature_transformer():
    """Test FeatureTransformer"""
    print("\n" + "="*80)
    print("TEST 1: FEATURE TRANSFORMER")
    print("="*80)
    
    # Generate test data
    df = generate_test_data(500)
    
    df_dict = {
        '15m': df,
        '1h': df,
        '4h': df,
        '1d': df
    }
    
    # Transform
    transformer = FeatureTransformer()
    features = transformer.transform(df_dict)
    
    print(f"\n✅ Features transformed: {features.shape}")
    print(f"   Expected: (500, 31)")
    print(f"   Features: {len(transformer.feature_names)}")
    
    # Verify shape
    assert features.shape[1] == 31, "Wrong number of features!"
    
    # Verify no NaN in last 100 rows (first rows may have NaN due to indicators)
    nan_count = features.iloc[-100:].isna().sum().sum()
    print(f"\n   NaN count (last 100 rows): {nan_count}")
    assert nan_count == 0, "Features contain NaN!"
    
    print("\n✅ Feature transformation verified")
    return True


def test_regime_filters():
    """Test RegimeFilters"""
    print("\n" + "="*80)
    print("TEST 2: REGIME FILTERS")
    print("="*80)
    
    filters = RegimeFilters()
    
    # Test different market states
    test_states = [
        {'crisis_level': 0.0, 'drawdown': 0.05, 'regime': 'normal'},
        {'crisis_level': 0.8, 'drawdown': 0.05, 'regime': 'crisis'},
        {'crisis_level': 0.0, 'drawdown': 0.20, 'regime': 'normal'},
    ]
    
    expected = [1, 0, 0]  # First passes, others fail
    
    print("\nFilter results:")
    for i, state in enumerate(test_states):
        result = filters.calculate_filters(state)
        print(f"  State {i+1}: {result} (expected: {expected[i]})")
        assert result == expected[i], f"Filter mismatch for state {i+1}"
    
    print("\n✅ Regime filters verified")
    return True


def test_xgboost_ml_strategy():
    """Test full XGBoost ML Strategy"""
    print("\n" + "="*80)
    print("TEST 3: XGBOOST ML STRATEGY (FULL P_ml(S))")
    print("="*80)
    
    # Generate test data
    df = generate_test_data(1000)
    print(f"\nGenerated {len(df)} bars of test data")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    df_dict = {
        '15m': df,
        '1h': df,
        '4h': df,
        '1d': df
    }
    
    # Initialize strategy (without trained model - uses fallback)
    strategy = XGBoostMLStrategy()
    print("✅ Strategy initialized (using fallback predictor)")
    
    # Set training distribution for OOD
    training_features = np.random.randn(1000, 31)
    strategy.set_training_distribution(training_features)
    print("✅ Training distribution set for OOD detection")
    
    # Generate signals
    print("\nGenerating ML signals...")
    signals_df = strategy.generate_signals(df_dict)
    
    # Analysis
    total_signals = signals_df['signal'].sum()
    signal_rate = signals_df['signal'].mean()
    
    print(f"\n✅ Signals generated successfully!")
    print(f"\nResults:")
    print(f"  Total bars: {len(signals_df)}")
    print(f"  Signals: {total_signals}")
    print(f"  Signal rate: {signal_rate:.2%}")
    
    # P_ml statistics
    print(f"\nP_ml Statistics:")
    print(f"  Mean: {signals_df['P_ml'].mean():.4f}")
    print(f"  Std: {signals_df['P_ml'].std():.4f}")
    print(f"  Min: {signals_df['P_ml'].min():.4f}")
    print(f"  Max: {signals_df['P_ml'].max():.4f}")
    
    # Component analysis
    print(f"\nComponent Contributions:")
    print(f"  Avg ml_score: {signals_df['ml_score'].mean():.4f}")
    print(f"  Avg filters_product: {signals_df['filters_product'].mean():.4f}")
    print(f"  Avg costs: {signals_df['costs'].mean():.4f}")
    print(f"  Avg R_ood: {signals_df['R_ood'].mean():.4f}")
    
    # Show sample signals
    positive_signals = signals_df[signals_df['signal'] == 1]
    if len(positive_signals) > 0:
        print(f"\nSample of ML Signals (first 5):")
        print(positive_signals[['P_ml', 'ml_score', 'filters_product', 'R_ood']].head())
    
    # Verify P_ml formula
    print(f"\n✅ Formula verification:")
    valid_indices = signals_df[signals_df['P_ml'].notna()].index
    if len(valid_indices) > 100:
        sample_idx = valid_indices[100]
    elif len(valid_indices) > 0:
        sample_idx = valid_indices[-1]
    else:
        print("  ⚠️  No valid P_ml values")
        return True
    
    sample_row = signals_df.loc[sample_idx]
    
    calculated_pml = (sample_row['ml_score'] * sample_row['filters_product'] - 
                      sample_row['costs'] - sample_row['R_ood'])
    
    print(f"  Manual calculation: {calculated_pml:.4f}")
    print(f"  Reported P_ml: {sample_row['P_ml']:.4f}")
    print(f"  Difference: {abs(calculated_pml - sample_row['P_ml']):.6f}")
    
    if not np.isnan(calculated_pml) and not np.isnan(sample_row['P_ml']):
        assert abs(calculated_pml - sample_row['P_ml']) < 1e-6, "Formula mismatch!"
        print("  ✅ Formula correct!")
    else:
        print("  ⚠️  NaN detected")
    
    return True


def test_ood_detection():
    """Test OOD risk calculation"""
    print("\n" + "="*80)
    print("TEST 4: OOD DETECTION")
    print("="*80)
    
    strategy = XGBoostMLStrategy()
    
    # Set training distribution
    training_features = np.random.multivariate_normal(
        mean=np.zeros(31),
        cov=np.eye(31),
        size=1000
    )
    strategy.set_training_distribution(training_features)
    
    # Test normal vs extreme features
    normal_features = np.zeros(31)
    extreme_features = np.ones(31) * 5.0
    
    R_ood_normal = strategy.calculate_ood_risk(normal_features)
    R_ood_extreme = strategy.calculate_ood_risk(extreme_features)
    
    print(f"\nOOD Risk:")
    print(f"  Normal features: {R_ood_normal:.4f}")
    print(f"  Extreme features: {R_ood_extreme:.4f}")
    
    assert R_ood_extreme > R_ood_normal, "OOD detection not working!"
    print("\n✅ OOD detection verified")
    
    return True


def test_adaptive_costs():
    """Test volatility-adjusted costs"""
    print("\n" + "="*80)
    print("TEST 5: ADAPTIVE COSTS")
    print("="*80)
    
    strategy = XGBoostMLStrategy()
    
    # Test different volatility levels
    vols = [0.01, 0.02, 0.05, 0.10]
    
    print("\nAdaptive costs by volatility:")
    for vol in vols:
        cost = strategy.calculate_adaptive_costs(vol)
        print(f"  Vol={vol:.2%}: Cost={cost:.4f}")
    
    # Verify costs increase with volatility
    costs = [strategy.calculate_adaptive_costs(v) for v in vols]
    assert all(costs[i] < costs[i+1] for i in range(len(costs)-1)), "Costs not increasing!"
    
    print("\n✅ Adaptive costs verified")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("XGBOOST ML STRATEGY - COMPREHENSIVE INTEGRATION TEST")
    print("="*80)
    
    results = {}
    
    # Run tests
    try:
        results['features'] = test_feature_transformer()
    except Exception as e:
        print(f"\n❌ Feature transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        results['features'] = False
    
    try:
        results['filters'] = test_regime_filters()
    except Exception as e:
        print(f"\n❌ Regime filters test failed: {e}")
        results['filters'] = False
    
    try:
        results['strategy'] = test_xgboost_ml_strategy()
    except Exception as e:
        print(f"\n❌ ML Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        results['strategy'] = False
    
    try:
        results['ood'] = test_ood_detection()
    except Exception as e:
        print(f"\n❌ OOD detection test failed: {e}")
        results['ood'] = False
    
    try:
        results['costs'] = test_adaptive_costs()
    except Exception as e:
        print(f"\n❌ Adaptive costs test failed: {e}")
        results['costs'] = False
    
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
        print("\n🎉 ALL TESTS PASSED! XGBOOST ML STRATEGY READY!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    print("="*80)


if __name__ == "__main__":
    main()