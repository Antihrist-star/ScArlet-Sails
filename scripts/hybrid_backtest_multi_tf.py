"""
HYBRID SYSTEM BACKTEST - MULTI-TIMEFRAME VERSION
====================================================================================================

Tests Hybrid 3-Layer system with PROPER multi-timeframe features:
- Layer 1: Rule-based (RSI < 30)
- Layer 2: ML Filter (XGBoost with 31 multi-TF features) ✅ FIXED!
- Layer 3: Crisis Gate

Compares against baseline Rule-based system.

Key difference from old version:
- Loads ALL 4 timeframes (15m, 1h, 4h, 1d) for each asset
- Extracts proper 31 features matching trained XGBoost model
- ML predictions should be meaningful now!

Author: Scarlet Sails Team
Date: 2025-11-11 (Multi-TF Fix)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

from models.hybrid_entry_system import HybridEntrySystem
from features.multi_timeframe_extractor import MultiTimeframeFeatureExtractor
from models.regime_detector import SimpleRegimeDetector

# ============================================================================
# CONFIG
# ============================================================================

ASSETS = [
    'BTC', 'ETH', 'SOL', 'LINK',   # Major
    'LDO', 'SUI', 'HBAR', 'ENA',   # Mid-cap
    'ALGO', 'AVAX', 'DOT', 'LTC',  # Established
    'ONDO', 'UNI'                   # DeFi
]

TIMEFRAMES = ['15m', '1h', '4h', '1d']

DATA_DIR = Path('data/raw')
OUTPUT_DIR = Path('reports/hybrid_backtest')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Hybrid system config
ML_THRESHOLD = 0.6
ENABLE_ML_FILTER = True   # ✅ Multi-TF features!
ENABLE_CRISIS_GATE = False  # Disabled for now

# ============================================================================
# MULTI-TIMEFRAME DATA LOADER
# ============================================================================

def load_multi_tf_data(asset: str, target_tf: str) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load all 4 timeframes and prepare multi-TF features

    Args:
        asset: Asset symbol (e.g., 'BTC')
        target_tf: Target timeframe for testing (e.g., '15m')

    Returns:
        (all_timeframes_dict, primary_df_with_features)
    """
    print(f"   Loading multi-TF data for {asset}...")

    extractor = MultiTimeframeFeatureExtractor(data_dir=str(DATA_DIR))

    try:
        # Load and prepare all timeframes
        all_tf, primary_df = extractor.prepare_multi_timeframe_data(asset, target_tf)

        print(f"   ✅ Loaded {len(primary_df)} bars")
        print(f"   ✅ Prepared {len(primary_df.columns)} features")

        return all_tf, primary_df

    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return None, None
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# HYBRID BACKTEST ENGINE
# ============================================================================

class HybridBacktestEngine:
    """
    Backtest engine using HybridEntrySystem with Multi-TF features
    """

    def __init__(
        self,
        ml_threshold=0.6,
        enable_ml_filter=True,
        enable_crisis_gate=False,
        all_timeframes=None,
        target_timeframe=None
    ):
        """
        Initialize backtest engine

        Args:
            ml_threshold: ML probability threshold
            enable_ml_filter: Enable Layer 2 ML filtering
            enable_crisis_gate: Enable Layer 3 crisis protection
            all_timeframes: Dict of DataFrames for all 4 TFs
            target_timeframe: Target TF being tested
        """
        self.entry_system = HybridEntrySystem(
            ml_threshold=ml_threshold,
            enable_ml_filter=enable_ml_filter,
            enable_crisis_gate=enable_crisis_gate,
            all_timeframes=all_timeframes,
            target_timeframe=target_timeframe
        )

        # Exit management
        self.atr_multipliers = {
            'Bull Trend': 3.0,
            'Bear Market': 2.0,
            'Sideways/Choppy': 1.5,
            'Crisis Event': 1.0
        }
        self.trailing_activation = 0.08
        self.partial_exits = [0.33, 0.33, 0.34]
        self.tp_levels = [1.05, 1.10, 1.15]

        self.regime_detector = SimpleRegimeDetector()

    def generate_signals(self, df: pd.DataFrame) -> List[int]:
        """Generate entry signals using 3-layer system"""
        signals = []

        for i in range(100, len(df)):
            timestamp = df.index[i] if hasattr(df.index[i], 'timestamp') else None
            should_enter, reason = self.entry_system.should_enter(df, i, timestamp)

            if should_enter:
                signals.append(i)

        return signals

    def backtest(self, df: pd.DataFrame, signals: List[int]) -> dict:
        """
        Simple backtest with ATR-based stops

        Returns:
            Dict with metrics
        """
        if len(signals) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'avg_bars_held': 0
            }

        trades = []
        position = None

        for signal_bar in signals:
            if position is not None:
                continue  # Already in position

            # Entry
            entry_price = df['close'].iloc[signal_bar]
            entry_time = df.index[signal_bar]

            # ATR stop-loss
            atr = df['15m_ATR_14'].iloc[signal_bar] if '15m_ATR_14' in df.columns else df['close'].iloc[signal_bar] * 0.02
            regime = self.regime_detector.detect(df, signal_bar)
            atr_mult = self.atr_multipliers.get(regime.value, 2.0)
            stop_loss = entry_price - (atr * atr_mult)

            # Scan forward for exit
            for exit_bar in range(signal_bar + 1, min(signal_bar + 500, len(df))):
                current_price = df['close'].iloc[exit_bar]
                low = df['low'].iloc[exit_bar]

                # Check stop-loss
                if low <= stop_loss:
                    pnl = (stop_loss - entry_price) / entry_price
                    bars_held = exit_bar - signal_bar
                    trades.append({'pnl': pnl, 'bars_held': bars_held, 'win': False})
                    position = None
                    break

                # Check take-profit (simple: +15%)
                if current_price >= entry_price * 1.15:
                    pnl = 0.15
                    bars_held = exit_bar - signal_bar
                    trades.append({'pnl': pnl, 'bars_held': bars_held, 'win': True})
                    position = None
                    break

            # Force close at end
            if position is not None and exit_bar == min(signal_bar + 499, len(df) - 1):
                current_price = df['close'].iloc[exit_bar]
                pnl = (current_price - entry_price) / entry_price
                bars_held = exit_bar - signal_bar
                trades.append({'pnl': pnl, 'bars_held': bars_held, 'win': pnl > 0})
                position = None

        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_return': 0.0,
                'avg_bars_held': 0
            }

        wins = [t['pnl'] for t in trades if t['win']]
        losses = [t['pnl'] for t in trades if not t['win']]

        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0

        return {
            'total_trades': len(trades),
            'win_rate': len(wins) / len(trades) if trades else 0,
            'profit_factor': total_wins / total_losses if total_losses > 0 else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_return': sum([t['pnl'] for t in trades]) * 100,
            'avg_bars_held': np.mean([t['bars_held'] for t in trades])
        }


# ============================================================================
# MAIN BACKTEST LOOP
# ============================================================================

def main():
    print("=" * 100)
    print("HYBRID SYSTEM BACKTEST - MULTI-TIMEFRAME VERSION")
    print("=" * 100)
    print("\nUsing PROPER 31-feature extraction from all 4 timeframes!")
    print("This should give meaningful ML predictions.\n")
    print("=" * 100)

    results = []
    tested = 0
    failed = 0

    for asset in ASSETS:
        for timeframe in TIMEFRAMES:
            combo = f"{asset}_{timeframe}"
            tested += 1

            print(f"\n{'=' * 100}")
            print(f"Testing: {combo} [{tested}/56]")
            print(f"{'=' * 100}")

            # Load multi-TF data
            all_tf, primary_df = load_multi_tf_data(asset, timeframe)

            if all_tf is None or primary_df is None:
                failed += 1
                print(f"   ⚠️  Skipping {combo} - no data")
                continue

            # Create engine with multi-TF data
            engine = HybridBacktestEngine(
                ml_threshold=ML_THRESHOLD,
                enable_ml_filter=ENABLE_ML_FILTER,
                enable_crisis_gate=ENABLE_CRISIS_GATE,
                all_timeframes=all_tf,
                target_timeframe=timeframe
            )

            # Reset state for new asset
            engine.entry_system.reset_state()

            # Generate signals
            print("   Generating signals...")
            signals = engine.generate_signals(primary_df)
            print(f"   ✅ Generated {len(signals)} signals")

            if len(signals) == 0:
                print(f"   ⚠️  No signals - skipping")
                failed += 1
                continue

            # Backtest
            print("   Backtesting...")
            metrics = engine.backtest(primary_df, signals)
            print(f"   ✅ Executed {metrics['total_trades']} trades")

            print(f"\n   RESULTS:")
            print(f"   Trades:        {metrics['total_trades']}")
            print(f"   Win Rate:      {metrics['win_rate']*100:.1f}%")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Total Return:  {metrics['total_return']:.1f}%")

            # Save result
            results.append({
                'asset': asset,
                'timeframe': timeframe,
                'combination': combo,
                'data_bars': len(primary_df),
                'signals_generated': len(signals),
                'total_trades': metrics['total_trades'],
                'win_rate': metrics['win_rate'],
                'profit_factor': metrics['profit_factor'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss'],
                'total_return': metrics['total_return'],
                'avg_bars_held': metrics['avg_bars_held']
            })

    # Summary
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Tested: {tested} / 56 combinations")
    print(f"Successful: {len(results)}")
    print(f"Failed: {failed}")

    if results:
        print("\n" + "=" * 100)
        print("OVERALL STATISTICS")
        print("=" * 100)

        total_trades = sum(r['total_trades'] for r in results)
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_pf = np.mean([r['profit_factor'] for r in results if r['profit_factor'] > 0])
        avg_return = np.mean([r['total_return'] for r in results])

        print(f"Total trades:        {total_trades:,}")
        print(f"Avg win rate:        {avg_wr*100:.1f}%")
        print(f"Avg profit factor:   {avg_pf:.2f}")
        print(f"Avg return:          {avg_return:.1f}%")

        # Top performers
        print("\n" + "=" * 100)
        print("TOP 10 PERFORMERS")
        print("=" * 100)

        sorted_results = sorted(results, key=lambda x: x['total_return'], reverse=True)[:10]
        for r in sorted_results:
            print(f"{r['combination']:<20} | Trades: {r['total_trades']:<4} | WR: {r['win_rate']*100:>5.1f}% | PF: {r['profit_factor']:>5.2f} | Return: {r['total_return']:>7.1f}%")

        # Save results
        output_file = OUTPUT_DIR / 'hybrid_multi_tf_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")

        csv_file = OUTPUT_DIR / 'hybrid_multi_tf_results.csv'
        pd.DataFrame(results).to_csv(csv_file, index=False)
        print(f"✅ CSV saved to: {csv_file}")

    print("\n" + "=" * 100)
    print("BACKTEST COMPLETE!")
    print("=" * 100)


if __name__ == "__main__":
    main()
