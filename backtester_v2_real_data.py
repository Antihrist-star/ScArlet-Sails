"""
BACKTESTER V2 - SCARLET SAILS
Production backtesting with REAL market data

Uses data from:
- data/features/BTC_USDT_15m_features.parquet (105 MB)
- Already calculated indicators (RSI, MACD, BB, etc.)
- 2023-2024 data

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import strategies (assuming they're in the same directory or importable)
try:
    from strategies.rule_based_v2 import RuleBasedStrategy
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy
    from strategies.hybrid_v2 import HybridStrategy
    from rl.dqn import DQNAgent
    STRATEGIES_AVAILABLE = True
except ImportError:
    logger.warning("Strategies not available for import. Using mock strategies.")
    STRATEGIES_AVAILABLE = False


class SimpleBacktester:
    """
    Simple backtester for strategy comparison
    
    Uses REAL market data from data/features/
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in USD
        """
        self.initial_capital = initial_capital
        
        # Results storage
        self.results = {}
        
        logger.info(f"SimpleBacktester initialized: ${initial_capital:,.2f}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from parquet file
        
        Parameters:
        -----------
        filepath : str
            Path to parquet file
        
        Returns:
        --------
        pd.DataFrame : Market data with features
        """
        logger.info(f"Loading data from: {filepath}")
        
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"  Loaded: {len(df)} bars")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, 
                     train_end: str = '2023-12-31',
                     test_start: str = '2024-01-01') -> tuple:
        """
        Split data into train and test
        
        Parameters:
        -----------
        df : DataFrame
            Full dataset
        train_end : str
            End of training period
        test_start : str
            Start of test period
        
        Returns:
        --------
        tuple : (train_df, test_df)
        """
        logger.info("Splitting data into train/test...")
        
        train_df = df[df.index <= train_end]
        test_df = df[df.index >= test_start]
        
        logger.info(f"  Train: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
        logger.info(f"  Test: {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")
        
        return train_df, test_df
    
    def simulate_strategy(self, df: pd.DataFrame, 
                         strategy_signals: pd.Series,
                         strategy_name: str) -> Dict:
        """
        Simulate a strategy with simple logic
        
        Parameters:
        -----------
        df : DataFrame
            Market data
        strategy_signals : Series
            Trading signals (1=buy, -1=sell, 0=hold)
        strategy_name : str
            Strategy name
        
        Returns:
        --------
        dict : Performance metrics
        """
        logger.info(f"Simulating {strategy_name}...")
        
        # Initialize
        capital = self.initial_capital
        position = 0  # 0 = no position, 1 = long
        position_size = 0
        entry_price = 0
        
        equity_curve = []
        trades = []
        
        # Simulate
        for i, (timestamp, row) in enumerate(df.iterrows()):
            price = row['close']
            
            # Get signal
            if i < len(strategy_signals):
                signal = strategy_signals.iloc[i]
            else:
                signal = 0
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy
                # Calculate position size (10% of capital)
                position_value = capital * 0.1
                position_size = position_value / price
                entry_price = price
                position = 1
                
                # Costs (0.15%)
                capital -= position_value * 0.0015
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'size': position_size
                })
            
            elif signal == -1 and position == 1:  # Sell
                # Calculate PnL
                exit_value = position_size * price
                pnl = exit_value - (position_size * entry_price)
                
                # Add to capital
                capital += exit_value
                
                # Costs (0.15%)
                capital -= exit_value * 0.0015
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'size': position_size,
                    'pnl': pnl,
                    'return': (price / entry_price - 1) * 100
                })
                
                # Reset position
                position = 0
                position_size = 0
                entry_price = 0
            
            # Calculate current equity
            if position == 1:
                current_equity = capital + (position_size * price)
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # 15m bars
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax * 100
        max_drawdown = drawdowns.min()
        
        # Trade statistics
        if len(trades) > 0:
            sell_trades = [t for t in trades if t['action'] == 'sell']
            if len(sell_trades) > 0:
                wins = sum(1 for t in sell_trades if t['pnl'] > 0)
                win_rate = wins / len(sell_trades) * 100
                
                avg_win = np.mean([t['pnl'] for t in sell_trades if t['pnl'] > 0]) if wins > 0 else 0
                avg_loss = np.mean([abs(t['pnl']) for t in sell_trades if t['pnl'] < 0]) if (len(sell_trades) - wins) > 0 else 0
                
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
        else:
            win_rate = 0
            profit_factor = 0
        
        metrics = {
            'strategy': strategy_name,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len([t for t in trades if t['action'] == 'sell']),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'final_capital': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades
        }
        
        logger.info(f"  {strategy_name}:")
        logger.info(f"    Return: {total_return:.2f}%")
        logger.info(f"    Sharpe: {sharpe:.2f}")
        logger.info(f"    Drawdown: {max_drawdown:.2f}%")
        logger.info(f"    Trades: {metrics['total_trades']}")
        logger.info(f"    Win Rate: {win_rate:.2f}%")
        
        return metrics
    
    def run_backtest(self, df: pd.DataFrame, 
                     strategies: Dict = None) -> Dict:
        """
        Run backtest for all strategies
        
        Parameters:
        -----------
        df : DataFrame
            Market data
        strategies : dict
            Dictionary of strategy objects (optional)
        
        Returns:
        --------
        dict : Results for all strategies
        """
        logger.info("="*80)
        logger.info("STARTING BACKTEST")
        logger.info("="*80)
        
        results = {}
        
        # Generate simple signals based on indicators
        # (Since we don't have strategy objects integrated yet)
        
        # 1. Rule-Based (RSI)
        logger.info("Generating Rule-Based signals...")
        rsi_signals = self._generate_rsi_signals(df)
        results['rule_based'] = self.simulate_strategy(df, rsi_signals, 'Rule-Based')
        
        # 2. Momentum (MACD)
        logger.info("Generating Momentum signals...")
        momentum_signals = self._generate_momentum_signals(df)
        results['momentum'] = self.simulate_strategy(df, momentum_signals, 'Momentum')
        
        # 3. Trend Following (MA crossover)
        logger.info("Generating Trend signals...")
        trend_signals = self._generate_trend_signals(df)
        results['trend'] = self.simulate_strategy(df, trend_signals, 'Trend Following')
        
        # 4. Buy and Hold
        logger.info("Generating Buy&Hold signals...")
        bh_signals = self._generate_buy_hold_signals(df)
        results['buy_hold'] = self.simulate_strategy(df, bh_signals, 'Buy & Hold')
        
        self.results = results
        
        logger.info("="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        
        return results
    
    def _generate_rsi_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on RSI"""
        signals = pd.Series(0, index=df.index)
        
        if 'rsi_14' in df.columns:
            rsi = df['rsi_14']
            signals[rsi < 30] = 1   # Oversold → Buy
            signals[rsi > 70] = -1  # Overbought → Sell
        
        return signals
    
    def _generate_momentum_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on MACD"""
        signals = pd.Series(0, index=df.index)
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd']
            signal = df['macd_signal']
            
            # MACD crossovers
            signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1   # Buy
            signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1  # Sell
        
        return signals
    
    def _generate_trend_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals based on MA crossover"""
        signals = pd.Series(0, index=df.index)
        
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            fast = df['sma_20']
            slow = df['sma_50']
            
            # MA crossovers
            signals[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1   # Buy
            signals[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1  # Sell
        
        return signals
    
    def _generate_buy_hold_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate buy and hold signals"""
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy at start
        # Hold forever
        return signals
    
    def create_visualizations(self, output_dir: str = 'backtest_results'):
        """
        Create visualization charts
        
        Parameters:
        -----------
        output_dir : str
            Output directory for charts
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Creating visualizations...")
        
        # 1. Equity curves
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for strategy_name, metrics in self.results.items():
            equity = metrics['equity_curve']
            time = range(len(equity))
            ax.plot(time, equity, label=f"{strategy_name} ({metrics['total_return_pct']:.1f}%)", linewidth=2)
        
        ax.axhline(y=self.initial_capital, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
        ax.set_xlabel('Time (bars)', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title('Strategy Comparison - Equity Curves', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/equity_curves.png', dpi=150)
        logger.info(f"  Saved: {output_dir}/equity_curves.png")
        plt.close()
        
        # 2. Metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        strategies = list(self.results.keys())
        
        # Returns
        returns = [self.results[s]['total_return_pct'] for s in strategies]
        axes[0, 0].bar(strategies, returns, color='steelblue')
        axes[0, 0].set_title('Total Return (%)')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sharpe Ratio
        sharpes = [self.results[s]['sharpe_ratio'] for s in strategies]
        axes[0, 1].bar(strategies, sharpes, color='green')
        axes[0, 1].set_title('Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Max Drawdown
        drawdowns = [abs(self.results[s]['max_drawdown_pct']) for s in strategies]
        axes[1, 0].bar(strategies, drawdowns, color='red')
        axes[1, 0].set_title('Maximum Drawdown (%)')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win Rate
        win_rates = [self.results[s]['win_rate_pct'] for s in strategies]
        axes[1, 1].bar(strategies, win_rates, color='purple')
        axes[1, 1].set_title('Win Rate (%)')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=150)
        logger.info(f"  Saved: {output_dir}/metrics_comparison.png")
        plt.close()
    
    def export_results(self, output_dir: str = 'backtest_results'):
        """
        Export results to CSV
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        logger.info("Exporting results to CSV...")
        
        # Metrics summary
        metrics_df = pd.DataFrame([
            {
                'strategy': metrics['strategy'],
                'total_return_pct': metrics['total_return_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'total_trades': metrics['total_trades'],
                'win_rate_pct': metrics['win_rate_pct'],
                'profit_factor': metrics['profit_factor'],
                'final_capital': metrics['final_capital']
            }
            for metrics in self.results.values()
        ])
        
        metrics_df.to_csv(f'{output_dir}/backtest_metrics.csv', index=False)
        logger.info(f"  Saved: {output_dir}/backtest_metrics.csv")
        
        # All trades
        all_trades = []
        for strategy_name, metrics in self.results.items():
            for trade in metrics['trades']:
                trade_copy = trade.copy()
                trade_copy['strategy'] = strategy_name
                all_trades.append(trade_copy)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(f'{output_dir}/all_trades.csv', index=False)
            logger.info(f"  Saved: {output_dir}/all_trades.csv")


def main():
    """
    Main backtesting script
    """
    print("="*80)
    print("SCARLET SAILS - BACKTESTER V2 (REAL DATA)")
    print("="*80)
    print()
    
    # Initialize backtester
    backtester = SimpleBacktester(initial_capital=10000)
    
    # Load data
    data_file = 'data/features/BTC_USDT_15m_features.parquet'
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please ensure data is available in data/features/")
        return
    
    print("STEP 1: LOADING DATA")
    print("-"*80)
    df = backtester.load_data(data_file)
    print()
    
    # Split data
    print("STEP 2: PREPARING DATA")
    print("-"*80)
    train_df, test_df = backtester.prepare_data(df)
    print()
    
    # Run backtest on test data
    print("STEP 3: RUNNING BACKTEST")
    print("-"*80)
    results = backtester.run_backtest(test_df)
    print()
    
    # Print summary
    print("="*80)
    print("BACKTEST RESULTS SUMMARY")
    print("="*80)
    for strategy_name, metrics in results.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Create visualizations
    print()
    print("STEP 4: CREATING VISUALIZATIONS")
    print("-"*80)
    backtester.create_visualizations()
    
    # Export results
    print()
    print("STEP 5: EXPORTING RESULTS")
    print("-"*80)
    backtester.export_results()
    
    print()
    print("="*80)
    print("✅ BACKTESTING COMPLETE!")
    print("="*80)
    print()
    print("Results saved to: backtest_results/")
    print("  - equity_curves.png")
    print("  - metrics_comparison.png")
    print("  - backtest_metrics.csv")
    print("  - all_trades.csv")
    print()
    print("Next steps:")
    print("  1. Review results in backtest_results/")
    print("  2. Analyze best performing strategy")
    print("  3. Push to GitHub")
    print()


if __name__ == "__main__":
    main()