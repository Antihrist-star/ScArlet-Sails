"""Simple threshold-based backtest utilities.

This module evaluates probabilistic signals by applying fixed thresholds and
computing trade-level metrics such as Sharpe ratio. It is intentionally
lightweight compared to HonestBacktestV2 and is meant for quick threshold
sweeps.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: Optional[float] = None,
) -> float:
    """Compute Sharpe ratio with optional annualisation.

    If ``periods_per_year`` is ``None`` the ratio is returned without scaling.
    When provided, the value is scaled by ``sqrt(periods_per_year)`` where the
    periods are typically *trades per year* rather than bars per year.
    This is designed for trade-level returns where the effective frequency is
    unknown until runtime.
    """

    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = mean_return / std_return

    if periods_per_year is not None:
        sharpe = sharpe * np.sqrt(periods_per_year)

    return float(sharpe)


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown percentage based on an equity curve.

    Args:
        equity_curve: Array of portfolio equity values over time (e.g., cumulative wealth)

    Returns:
        Maximum drawdown as percentage (negative number, e.g. -15.0 for -15%)
    """

    if len(equity_curve) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    running_max = np.where(running_max == 0, 1.0, running_max)
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(np.min(drawdown) * 100.0)
    return max_dd


def evaluate_threshold(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    threshold: float,
) -> Dict:
    """Evaluate performance for a given threshold on trade-level returns."""

    signals = (df[proba_col] >= threshold).astype(int)
    trade_mask = signals == 1
    trade_returns = df.loc[trade_mask, fee_ret_col].values

    if len(trade_returns) == 0:
        return {
            "threshold": threshold,
            "n_trades": 0,
            "total_return": 0.0,
            "mean_return": 0.0,
            "sharpe_ratio": 0.0,
            "sharpe_raw": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
        }

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        period_seconds = (df.index.max() - df.index.min()).total_seconds()
        period_days = max(period_seconds / 86400.0, 1e-6)
    else:
        period_days = 365.0

    n_trades = len(trade_returns)
    trades_per_year = n_trades * (365.0 / period_days)

    mean_return = float(np.mean(trade_returns))
    std_return = float(np.std(trade_returns, ddof=1)) if n_trades > 1 else 0.0

    if std_return == 0.0:
        sharpe_raw = 0.0
        sharpe = 0.0
    else:
        sharpe_raw = mean_return / std_return
        sharpe = sharpe_raw * np.sqrt(trades_per_year)

    equity_curve = np.cumprod(1.0 + trade_returns)
    max_dd = calculate_max_drawdown(equity_curve)

    win_rate = float(np.mean(trade_returns > 0) * 100.0)
    total_return = float(np.sum(trade_returns))

    return {
        "threshold": threshold,
        "n_trades": int(n_trades),
        "total_return": total_return,
        "mean_return": mean_return,
        "sharpe_ratio": float(sharpe),
        "sharpe_raw": float(sharpe_raw),
        "max_drawdown_pct": max_dd,
        "win_rate": win_rate,
    }


def evaluate_thresholds(
    df: pd.DataFrame,
    proba_col: str,
    fee_ret_col: str,
    thresholds: Iterable[float],
) -> Dict[float, Dict]:
    """Evaluate a grid of thresholds and return per-threshold metrics."""

    results: Dict[float, Dict] = {}
    for th in thresholds:
        results[th] = evaluate_threshold(df, proba_col, fee_ret_col, th)
    return results


def select_optimal_threshold(
    threshold_results: Dict[float, Dict],
    max_dd_limit: float = 20.0,
    min_trades: int = 10,
) -> Dict:
    """
    Select optimal threshold based on Sharpe ratio with constraints.

    Args:
        threshold_results: Results from evaluate_thresholds
        max_dd_limit: Maximum allowed drawdown percentage (negative number, e.g. 20.0)
        min_trades: Minimum number of trades required

    Returns:
        Dictionary with optimal threshold and its metrics
    """

    best_threshold = None
    best_sharpe = -np.inf

    for th, metrics in threshold_results.items():
        if metrics.get("n_trades", 0) < min_trades:
            continue
        if metrics.get("max_drawdown_pct", 0) <= max_dd_limit and metrics.get("sharpe_ratio", -np.inf) > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_threshold = th

    all_sharpes = [m.get("sharpe_ratio", -np.inf) for m in threshold_results.values()]
    no_profitable = all(s <= 0 for s in all_sharpes)

    return {
        "threshold": best_threshold,
        "sharpe": best_sharpe,
        "backtest_metrics": threshold_results.get(best_threshold, {}),
        "no_profitable_threshold": no_profitable,
    }

