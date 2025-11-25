"""
Backtest Plotter for Scarlet Sails Framework

Generates visualization for backtest analysis:
- Equity curves
- Drawdown charts
- Monthly returns heatmap
- Strategy comparison
- Trade distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Use non-interactive backend for server environments
plt.switch_backend('Agg')


class BacktestPlotter:
    """
    Generate backtest visualizations.
    
    All plots are saved to files (not displayed interactively).
    """
    
    # Color scheme
    COLORS = {
        'equity': '#2E86AB',
        'drawdown': '#E94F37',
        'positive': '#2ECC71',
        'negative': '#E74C3C',
        'neutral': '#95A5A6',
        'grid': '#E0E0E0',
    }
    
    # Default figure size
    FIGSIZE = (12, 6)
    DPI = 150
    
    def __init__(self, output_dir: str = 'backtest_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = 'Equity Curve',
        benchmark: Optional[pd.Series] = None,
        filename: str = 'equity_curve.png',
    ) -> str:
        """
        Plot equity curve over time.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            DataFrame with 'value' column and DatetimeIndex
        title : str
            Plot title
        benchmark : pd.Series, optional
            Benchmark equity for comparison
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.FIGSIZE, dpi=self.DPI)
        
        # Plot equity
        ax.plot(
            equity_curve.index,
            equity_curve['value'],
            color=self.COLORS['equity'],
            linewidth=1.5,
            label='Strategy'
        )
        
        # Plot benchmark if provided
        if benchmark is not None:
            ax.plot(
                benchmark.index,
                benchmark.values,
                color=self.COLORS['neutral'],
                linewidth=1,
                linestyle='--',
                label='Benchmark'
            )
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.legend(loc='upper left')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add initial and final values
        initial = equity_curve['value'].iloc[0]
        final = equity_curve['value'].iloc[-1]
        returns = (final / initial - 1) * 100
        
        ax.axhline(y=initial, color=self.COLORS['neutral'], linestyle=':', alpha=0.5)
        
        # Annotation
        text = f"Initial: ${initial:,.0f}\nFinal: ${final:,.0f}\nReturn: {returns:+.1f}%"
        ax.annotate(
            text,
            xy=(0.02, 0.98),
            xycoords='axes fraction',
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = 'Drawdown',
        filename: str = 'drawdown.png',
    ) -> str:
        """
        Plot underwater (drawdown) chart.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            DataFrame with 'value' column
        title : str
            Plot title
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.FIGSIZE, dpi=self.DPI)
        
        # Calculate drawdown
        values = equity_curve['value']
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100  # As percentage
        
        # Fill area
        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            color=self.COLORS['drawdown'],
            alpha=0.3
        )
        
        ax.plot(
            drawdown.index,
            drawdown.values,
            color=self.COLORS['drawdown'],
            linewidth=1
        )
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)
        
        # Add max drawdown annotation
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        ax.annotate(
            f'Max DD: {max_dd:.1f}%',
            xy=(max_dd_idx, max_dd),
            xytext=(10, -30),
            textcoords='offset points',
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Horizontal line at -15% (target)
        ax.axhline(y=-15, color='orange', linestyle='--', alpha=0.7, label='Target (-15%)')
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def plot_monthly_returns_heatmap(
        self,
        equity_curve: pd.DataFrame,
        title: str = 'Monthly Returns (%)',
        filename: str = 'monthly_heatmap.png',
    ) -> str:
        """
        Plot monthly returns heatmap.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            DataFrame with 'value' column
        title : str
            Plot title
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        # Calculate monthly returns
        monthly = equity_curve['value'].resample('ME').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Create pivot table (years as rows, months as columns)
        df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        }).dropna()
        
        pivot = df.pivot(index='year', columns='month', values='return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[m-1] for m in pivot.columns]
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8)), dpi=self.DPI)
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Return (%)', fontsize=10)
        
        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        
        # Add values as text
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 10 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=9, color=color)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def plot_returns_distribution(
        self,
        equity_curve: pd.DataFrame,
        title: str = 'Returns Distribution',
        filename: str = 'returns_distribution.png',
    ) -> str:
        """
        Plot histogram of returns.
        
        Parameters
        ----------
        equity_curve : pd.DataFrame
            DataFrame with 'value' column
        title : str
            Plot title
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        fig, ax = plt.subplots(figsize=self.FIGSIZE, dpi=self.DPI)
        
        # Calculate returns
        returns = equity_curve['value'].pct_change().dropna() * 100
        
        # Plot histogram
        n, bins, patches = ax.hist(
            returns,
            bins=50,
            color=self.COLORS['equity'],
            alpha=0.7,
            edgecolor='white'
        )
        
        # Color positive/negative differently
        for patch, left_edge in zip(patches, bins[:-1]):
            if left_edge < 0:
                patch.set_facecolor(self.COLORS['negative'])
            else:
                patch.set_facecolor(self.COLORS['positive'])
        
        # Add statistics
        mean_ret = returns.mean()
        std_ret = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        stats_text = (
            f"Mean: {mean_ret:.3f}%\n"
            f"Std: {std_ret:.3f}%\n"
            f"Skew: {skew:.2f}\n"
            f"Kurt: {kurt:.2f}"
        )
        
        ax.annotate(
            stats_text,
            xy=(0.98, 0.98),
            xycoords='axes fraction',
            verticalalignment='top',
            horizontalalignment='right',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add mean line
        ax.axvline(x=mean_ret, color='black', linestyle='--', label=f'Mean: {mean_ret:.3f}%')
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, pd.DataFrame],
        title: str = 'Strategy Comparison',
        filename: str = 'strategy_comparison.png',
    ) -> str:
        """
        Compare multiple strategies/assets.
        
        Parameters
        ----------
        results : Dict[str, pd.DataFrame]
            Dictionary mapping name -> equity_curve DataFrame
        title : str
            Plot title
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=self.DPI)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        # Plot 1: Equity curves
        ax1 = axes[0]
        for (name, equity), color in zip(results.items(), colors):
            # Normalize to start at 100
            normalized = equity['value'] / equity['value'].iloc[0] * 100
            ax1.plot(normalized.index, normalized.values, label=name, color=color, linewidth=1.5)
        
        ax1.set_title(f'{title} - Normalized Equity', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Value (Base 100)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', ncol=min(4, len(results)))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Plot 2: Cumulative returns bar chart
        ax2 = axes[1]
        names = list(results.keys())
        returns = []
        for name, equity in results.items():
            ret = (equity['value'].iloc[-1] / equity['value'].iloc[0] - 1) * 100
            returns.append(ret)
        
        bars = ax2.bar(names, returns, color=colors)
        
        # Color bars based on return
        for bar, ret in zip(bars, returns):
            if ret >= 0:
                bar.set_color(self.COLORS['positive'])
            else:
                bar.set_color(self.COLORS['negative'])
        
        ax2.set_title('Total Returns Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Strategy/Asset', fontsize=11)
        ax2.set_ylabel('Total Return (%)', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            ax2.annotate(
                f'{ret:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, ret),
                xytext=(0, 5 if ret >= 0 else -15),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def plot_trade_analysis(
        self,
        trades_df: pd.DataFrame,
        title: str = 'Trade Analysis',
        filename: str = 'trade_analysis.png',
    ) -> str:
        """
        Plot trade analysis (PnL distribution, duration, etc.).
        
        Parameters
        ----------
        trades_df : pd.DataFrame
            DataFrame with trade data
        title : str
            Plot title
        filename : str
            Output filename
        
        Returns
        -------
        str
            Path to saved file
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.DPI)
        
        # Plot 1: PnL distribution
        ax1 = axes[0, 0]
        pnl = trades_df['pnl']
        colors = [self.COLORS['positive'] if p > 0 else self.COLORS['negative'] for p in pnl]
        ax1.hist(pnl, bins=30, color=self.COLORS['equity'], alpha=0.7, edgecolor='white')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.axvline(x=pnl.mean(), color='red', linestyle='--', label=f'Mean: ${pnl.mean():.2f}')
        ax1.set_title('PnL Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('PnL ($)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: PnL over time
        ax2 = axes[0, 1]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, color=self.COLORS['equity'])
        ax2.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
        ax2.set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trade #', fontsize=10)
        ax2.set_ylabel('Cumulative PnL ($)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win/Loss ratio
        ax3 = axes[1, 0]
        winners = len(trades_df[trades_df['pnl'] > 0])
        losers = len(trades_df[trades_df['pnl'] <= 0])
        ax3.pie(
            [winners, losers],
            labels=[f'Winners\n({winners})', f'Losers\n({losers})'],
            colors=[self.COLORS['positive'], self.COLORS['negative']],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0)
        )
        ax3.set_title('Win/Loss Ratio', fontsize=12, fontweight='bold')
        
        # Plot 4: Trade duration distribution
        ax4 = axes[1, 1]
        if 'duration_hours' in trades_df.columns:
            durations = trades_df['duration_hours'].dropna()
            ax4.hist(durations, bins=30, color=self.COLORS['equity'], alpha=0.7, edgecolor='white')
            ax4.axvline(x=durations.mean(), color='red', linestyle='--', 
                       label=f'Mean: {durations.mean():.1f}h')
            ax4.set_xlabel('Duration (hours)', fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Duration data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Trade Duration', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved: {filepath}")
        return str(filepath)
    
    def generate_full_report(
        self,
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        strategy_name: str,
        coin: str,
        timeframe: str,
    ) -> List[str]:
        """
        Generate complete visual report.
        
        Returns
        -------
        List[str]
            List of generated file paths
        """
        prefix = f"{strategy_name}_{coin}_{timeframe}"
        
        files = []
        
        # Equity curve
        files.append(self.plot_equity_curve(
            equity_curve,
            title=f'Equity Curve - {strategy_name} on {coin} {timeframe}',
            filename=f'{prefix}_equity.png'
        ))
        
        # Drawdown
        files.append(self.plot_drawdown(
            equity_curve,
            title=f'Drawdown - {strategy_name} on {coin} {timeframe}',
            filename=f'{prefix}_drawdown.png'
        ))
        
        # Monthly heatmap
        files.append(self.plot_monthly_returns_heatmap(
            equity_curve,
            title=f'Monthly Returns - {strategy_name} on {coin} {timeframe}',
            filename=f'{prefix}_monthly.png'
        ))
        
        # Returns distribution
        files.append(self.plot_returns_distribution(
            equity_curve,
            title=f'Returns Distribution - {strategy_name} on {coin} {timeframe}',
            filename=f'{prefix}_returns.png'
        ))
        
        # Trade analysis
        if not trades_df.empty:
            files.append(self.plot_trade_analysis(
                trades_df,
                title=f'Trade Analysis - {strategy_name} on {coin} {timeframe}',
                filename=f'{prefix}_trades.png'
            ))
        
        print(f"\nGenerated {len(files)} plots for {strategy_name} on {coin}_{timeframe}")
        return files