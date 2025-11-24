"""
TEST RULE-BASED STRATEGY (FINAL FIXED!)
Proper handling of DataFrame results

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
Version: FINAL FIX - DataFrame handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from strategies.rule_based_v2 import RuleBasedStrategy

def prepare_data_for_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map advanced features to classic indicator names
    """
    df = df.copy()
    
    # RSI: percentile to [0-100] scale
    if 'norm_rsi_pctile' in df.columns:
        df['RSI_14'] = df['norm_rsi_pctile'] * 100
    
    # MACD: use normalized zscore
    if 'norm_macd_zscore' in df.columns:
        df['MACD_12_26_9'] = df['norm_macd_zscore']
    
    # MACD Signal: use derivative
    if 'deriv_macd_diff1' in df.columns:
        df['MACDs_12_26_9'] = df['deriv_macd_diff1']
    
    # Bollinger Bands: reconstruct from width percentile
    if 'norm_bb_width_pctile' in df.columns and 'close' in df.columns:
        bb_width = df['norm_bb_width_pctile'] * df['close'] * 0.02
        df['BBM_20_2.0'] = df['close']
        df['BBL_20_2.0'] = df['close'] - bb_width
        df['BBU_20_2.0'] = df['close'] + bb_width
    
    # ATR: approximate from percentile
    if 'norm_atr_pctile' in df.columns and 'close' in df.columns:
        df['ATRr_14'] = df['norm_atr_pctile'] * df['close'] * 0.02
    
    return df


def main():
    """Test Rule-Based strategy with proper DataFrame handling"""
    print("="*80)
    print("TESTING RULE-BASED STRATEGY (FINAL FIXED!)")
    print("="*80)
    print()
    
    # Load data
    logger.info("Loading data...")
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    df = pd.read_parquet(data_file)
    logger.info(f"Loaded: {len(df)} bars")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Use 2024 data
    test_df = df[df.index >= '2024-01-01']
    logger.info(f"Test period: {len(test_df)} bars")
    
    # Map advanced features to classic indicators
    print()
    print("MAPPING FEATURES:")
    print("-"*80)
    print("RSI_14 ← norm_rsi_pctile * 100")
    print("MACD_12_26_9 ← norm_macd_zscore")
    print("MACDs_12_26_9 ← deriv_macd_diff1")
    print("BBL/BBM/BBU ← reconstructed from norm_bb_width_pctile")
    print("ATRr_14 ← norm_atr_pctile")
    
    test_df = prepare_data_for_strategy(test_df)
    
    # Check mapped columns
    print()
    print("CHECKING MAPPED COLUMNS:")
    print("-"*80)
    
    required_base = ['open', 'high', 'low', 'close', 'volume']
    required_indicators = ['RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'ATRr_14']
    
    for col in required_base + required_indicators:
        if col in test_df.columns:
            print(f"✅ {col}")
        else:
            print(f"❌ {col} MISSING!")
    
    # Initialize strategy
    print()
    print("INITIALIZING STRATEGY:")
    print("-"*80)
    
    try:
        strategy = RuleBasedStrategy()
        logger.info("✅ Rule-Based strategy initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        return
    
    # Test signal generation
    print()
    print("TESTING SIGNAL GENERATION:")
    print("-"*80)
    
    try:
        # Test on last 1000 bars
        test_sample = test_df.iloc[-1000:]
        logger.info(f"Testing on {len(test_sample)} bars...")
        
        result = strategy.generate_signals(test_sample)
        
        # FIXED: Proper DataFrame handling
        if result is not None:
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    print(f"✅ Signals generated!")
                    print(f"   Result type: DataFrame")
                    print(f"   Signals count: {len(result)}")
                    print(f"   Columns: {list(result.columns)}")
                    
                    # Check signal types
                    if 'action' in result.columns:
                        action_counts = result['action'].value_counts()
                        print(f"   Actions:")
                        for action, count in action_counts.items():
                            print(f"      {action}: {count}")
                    
                    # Show sample
                    print(f"\n   Sample signals:")
                    print(result.head(3).to_string())
                else:
                    print(f"⚠️ DataFrame is empty - no signals generated")
            
            elif isinstance(result, dict):
                print(f"✅ Signal generated!")
                print(f"   Result type: dict")
                print(f"   Keys: {list(result.keys())}")
                print(f"   Action: {result.get('action', 'N/A')}")
            
            else:
                print(f"⚠️ Unexpected result type: {type(result)}")
        else:
            print(f"⚠️ No signals generated (returned None)")
        
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test on multiple windows
    print()
    print("TESTING MULTIPLE WINDOWS:")
    print("-"*80)
    
    signals_count = 0
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    errors = 0
    
    # Test every 100 bars
    windows_to_test = list(range(500, min(len(test_df), 2000), 100))
    logger.info(f"Testing on {len(windows_to_test)} windows...")
    
    for i in windows_to_test:
        try:
            window = test_df.iloc[:i]
            result = strategy.generate_signals(window)
            
            if result is not None and isinstance(result, pd.DataFrame) and not result.empty:
                signals_count += len(result)
                
                if 'action' in result.columns:
                    buy_signals += (result['action'] == 'buy').sum()
                    sell_signals += (result['action'] == 'sell').sum()
                    hold_signals += (result['action'] == 'hold').sum()
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only log first 3 errors
                logger.error(f"Error at bar {i}: {e}")
    
    print(f"Windows tested: {len(windows_to_test)}")
    print(f"Total signals: {signals_count}")
    print(f"  Buy: {buy_signals}")
    print(f"  Sell: {sell_signals}")
    print(f"  Hold: {hold_signals}")
    print(f"Errors: {errors}")
    
    if signals_count == 0:
        print()
        print("⚠️ WARNING: No signals generated across windows!")
    else:
        print()
        print("✅ Strategy is generating signals!")
        
        # Calculate signal ratios
        if signals_count > 0:
            buy_ratio = buy_signals / signals_count * 100
            sell_ratio = sell_signals / signals_count * 100
            hold_ratio = hold_signals / signals_count * 100
            
            print()
            print("SIGNAL DISTRIBUTION:")
            print(f"  Buy:  {buy_ratio:.1f}%")
            print(f"  Sell: {sell_ratio:.1f}%")
            print(f"  Hold: {hold_ratio:.1f}%")
    
    print()
    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    # Summary
    print()
    print("SUMMARY:")
    print("-"*80)
    if signals_count > 0 and buy_signals > 0 and sell_signals > 0:
        print("✅ PASS: Strategy is working!")
        print("   - Signals generated")
        print("   - Buy and Sell signals present")
        print("   - Ready for backtesting")
    elif signals_count > 0:
        print("⚠️ PARTIAL: Strategy generates signals but needs review")
        print("   - Signals generated")
        print("   - Check signal distribution")
    else:
        print("❌ FAIL: Strategy not generating signals")
        print("   - Needs debugging")


if __name__ == "__main__":
    main()