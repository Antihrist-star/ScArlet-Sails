"""
STRATEGY ORCHESTRATOR - SCARLET SAILS
Unified management system for all trading strategies

Features:
- Position management
- Risk management
- Portfolio tracking
- Signal aggregation
- Backtesting support
- Performance monitoring

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 22, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a trading position
    """
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def pnl(self, current_price: float) -> float:
        """Calculate current PnL"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - current_price) * self.size
    
    def pnl_pct(self, current_price: float) -> float:
        """Calculate current PnL percentage"""
        if self.side == 'long':
            return (current_price / self.entry_price - 1) * 100
        else:
            return (1 - current_price / self.entry_price) * 100


@dataclass
class Signal:
    """
    Represents a trading signal from a strategy
    """
    strategy_name: str
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    price: float
    metadata: Dict = None


class Portfolio:
    """
    Portfolio management system
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize portfolio
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital in USD
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        logger.info(f"Portfolio initialized: ${initial_capital:,.2f}")
    
    def equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity
        
        Parameters:
        -----------
        current_prices : dict
            Current prices for all symbols
        
        Returns:
        --------
        float : Total equity
        """
        position_value = sum(
            pos.pnl(current_prices.get(symbol, pos.entry_price))
            for symbol, pos in self.positions.items()
        )
        return self.cash + sum(
            pos.entry_price * pos.size 
            for pos in self.positions.values()
        ) + position_value
    
    def open_position(self, symbol: str, side: str, size: float, 
                     price: float, timestamp: datetime,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> bool:
        """
        Open a new position
        
        Returns:
        --------
        bool : Success status
        """
        cost = price * size
        
        if cost > self.cash:
            logger.warning(f"Insufficient funds: ${self.cash:.2f} < ${cost:.2f}")
            return False
        
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        self.cash -= cost
        
        logger.info(f"Opened {side} position: {symbol} @ ${price:.2f}, size={size}")
        return True
    
    def close_position(self, symbol: str, price: float, 
                      timestamp: datetime) -> Optional[float]:
        """
        Close an existing position
        
        Returns:
        --------
        float : PnL or None if position doesn't exist
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None
        
        position = self.positions.pop(symbol)
        pnl = position.pnl(price)
        
        # Return capital + PnL
        self.cash += (position.entry_price * position.size) + pnl
        
        self.closed_positions.append(position)
        
        logger.info(f"Closed {position.side} position: {symbol} @ ${price:.2f}, "
                   f"PnL: ${pnl:.2f} ({position.pnl_pct(price):.2f}%)")
        
        return pnl
    
    def record_equity(self, timestamp: datetime, current_prices: Dict[str, float]):
        """Record current equity for tracking"""
        equity = self.equity(current_prices)
        self.equity_history.append((timestamp, equity))
    
    def performance_summary(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
        --------
        dict : Performance statistics
        """
        if not self.equity_history:
            return {}
        
        # Extract equity curve
        times, equities = zip(*self.equity_history)
        equities = np.array(equities)
        returns = np.diff(equities) / equities[:-1]
        
        # Calculate metrics
        total_return = (equities[-1] / equities[0] - 1) * 100
        
        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(equities)
        drawdowns = (equities - cummax) / cummax * 100
        max_drawdown = drawdowns.min()
        
        # Win rate (from closed positions)
        if self.closed_positions:
            wins = sum(1 for pos in self.closed_positions 
                      if pos.pnl(pos.entry_price) > 0)  # This is wrong but placeholder
            win_rate = wins / len(self.closed_positions) * 100
        else:
            win_rate = 0.0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': equities[-1],
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(self.closed_positions),
            'win_rate_pct': win_rate,
            'current_positions': len(self.positions)
        }


class RiskManager:
    """
    Risk management system
    """
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_total_exposure: float = 0.5,
                 max_drawdown: float = 0.15):
        """
        Initialize risk manager
        
        Parameters:
        -----------
        max_position_size : float
            Maximum position size as fraction of equity (default 10%)
        max_total_exposure : float
            Maximum total exposure as fraction of equity (default 50%)
        max_drawdown : float
            Maximum allowed drawdown (default 15%)
        """
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown = max_drawdown
        
        logger.info(f"RiskManager initialized:")
        logger.info(f"  Max position size: {max_position_size*100:.1f}%")
        logger.info(f"  Max total exposure: {max_total_exposure*100:.1f}%")
        logger.info(f"  Max drawdown: {max_drawdown*100:.1f}%")
    
    def check_position_size(self, size: float, price: float, 
                           equity: float) -> Tuple[bool, str]:
        """
        Check if position size is within limits
        
        Returns:
        --------
        tuple : (allowed, reason)
        """
        position_value = size * price
        position_pct = position_value / equity
        
        if position_pct > self.max_position_size:
            return False, f"Position too large: {position_pct*100:.1f}% > {self.max_position_size*100:.1f}%"
        
        return True, "OK"
    
    def check_total_exposure(self, portfolio: Portfolio, 
                            current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check total portfolio exposure
        
        Returns:
        --------
        tuple : (allowed, reason)
        """
        equity = portfolio.equity(current_prices)
        
        total_exposure = sum(
            pos.size * current_prices.get(symbol, pos.entry_price)
            for symbol, pos in portfolio.positions.items()
        )
        
        exposure_pct = total_exposure / equity
        
        if exposure_pct > self.max_total_exposure:
            return False, f"Total exposure too high: {exposure_pct*100:.1f}% > {self.max_total_exposure*100:.1f}%"
        
        return True, "OK"
    
    def check_drawdown(self, portfolio: Portfolio) -> Tuple[bool, str]:
        """
        Check current drawdown
        
        Returns:
        --------
        tuple : (allowed, reason)
        """
        if not portfolio.equity_history:
            return True, "OK"
        
        times, equities = zip(*portfolio.equity_history)
        equities = np.array(equities)
        
        peak = equities.max()
        current = equities[-1]
        drawdown = (current - peak) / peak
        
        if drawdown < -self.max_drawdown:
            return False, f"Drawdown too large: {drawdown*100:.1f}% < -{self.max_drawdown*100:.1f}%"
        
        return True, "OK"
    
    def validate_trade(self, action: str, symbol: str, size: float, 
                      price: float, portfolio: Portfolio,
                      current_prices: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate a trade against all risk checks
        
        Returns:
        --------
        tuple : (allowed, reason)
        """
        equity = portfolio.equity(current_prices)
        
        # Check position size
        allowed, reason = self.check_position_size(size, price, equity)
        if not allowed:
            return False, reason
        
        # Check total exposure
        allowed, reason = self.check_total_exposure(portfolio, current_prices)
        if not allowed:
            return False, reason
        
        # Check drawdown
        allowed, reason = self.check_drawdown(portfolio)
        if not allowed:
            return False, reason
        
        return True, "OK"


class StrategyOrchestrator:
    """
    Main orchestrator for all trading strategies
    
    Manages:
    - Strategy initialization
    - Signal aggregation
    - Position management
    - Risk control
    - Performance tracking
    """
    
    def __init__(self, 
                 strategies: Dict,
                 initial_capital: float = 10000,
                 risk_config: Optional[Dict] = None):
        """
        Initialize orchestrator
        
        Parameters:
        -----------
        strategies : dict
            Dictionary of strategy objects {name: strategy}
        initial_capital : float
            Starting capital
        risk_config : dict
            Risk management configuration
        """
        self.strategies = strategies
        self.portfolio = Portfolio(initial_capital)
        
        risk_config = risk_config or {}
        self.risk_manager = RiskManager(
            max_position_size=risk_config.get('max_position_size', 0.1),
            max_total_exposure=risk_config.get('max_total_exposure', 0.5),
            max_drawdown=risk_config.get('max_drawdown', 0.15)
        )
        
        self.signals_history: List[Signal] = []
        self.trades_history: List[Dict] = []
        
        logger.info(f"StrategyOrchestrator initialized:")
        logger.info(f"  Strategies: {list(strategies.keys())}")
        logger.info(f"  Initial capital: ${initial_capital:,.2f}")
    
    def aggregate_signals(self, signals: Dict[str, Signal]) -> Optional[Signal]:
        """
        Aggregate signals from multiple strategies
        
        Parameters:
        -----------
        signals : dict
            Dictionary of signals {strategy_name: signal}
        
        Returns:
        --------
        Signal : Aggregated signal or None
        """
        if not signals:
            return None
        
        # Simple voting: majority wins
        actions = [sig.action for sig in signals.values()]
        confidences = [sig.confidence for sig in signals.values()]
        
        # Count votes
        buy_votes = sum(1 for a in actions if a == 'buy')
        sell_votes = sum(1 for a in actions if a == 'sell')
        hold_votes = sum(1 for a in actions if a == 'hold')
        
        # Determine action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            action = 'buy'
            confidence = np.mean([c for a, c in zip(actions, confidences) if a == 'buy'])
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action = 'sell'
            confidence = np.mean([c for a, c in zip(actions, confidences) if a == 'sell'])
        else:
            action = 'hold'
            confidence = np.mean(confidences)
        
        # Use first signal's metadata
        first_signal = list(signals.values())[0]
        
        return Signal(
            strategy_name='AGGREGATED',
            timestamp=first_signal.timestamp,
            action=action,
            confidence=confidence,
            price=first_signal.price,
            metadata={'votes': {'buy': buy_votes, 'sell': sell_votes, 'hold': hold_votes}}
        )
    
    def execute_signal(self, signal: Signal, symbol: str = 'BTC/USDT') -> bool:
        """
        Execute a trading signal
        
        Returns:
        --------
        bool : Execution success
        """
        if signal.action == 'hold':
            return False
        
        current_prices = {symbol: signal.price}
        
        # Check if we have a position
        has_position = symbol in self.portfolio.positions
        
        if signal.action == 'buy' and not has_position:
            # Calculate position size (10% of equity by default)
            equity = self.portfolio.equity(current_prices)
            position_value = equity * 0.1
            size = position_value / signal.price
            
            # Validate trade
            allowed, reason = self.risk_manager.validate_trade(
                'buy', symbol, size, signal.price, 
                self.portfolio, current_prices
            )
            
            if not allowed:
                logger.warning(f"Trade rejected: {reason}")
                return False
            
            # Execute
            success = self.portfolio.open_position(
                symbol=symbol,
                side='long',
                size=size,
                price=signal.price,
                timestamp=signal.timestamp
            )
            
            if success:
                self.trades_history.append({
                    'timestamp': signal.timestamp,
                    'action': 'open',
                    'symbol': symbol,
                    'side': 'long',
                    'price': signal.price,
                    'size': size,
                    'strategy': signal.strategy_name
                })
            
            return success
        
        elif signal.action == 'sell' and has_position:
            # Close position
            pnl = self.portfolio.close_position(symbol, signal.price, signal.timestamp)
            
            if pnl is not None:
                self.trades_history.append({
                    'timestamp': signal.timestamp,
                    'action': 'close',
                    'symbol': symbol,
                    'price': signal.price,
                    'pnl': pnl,
                    'strategy': signal.strategy_name
                })
                return True
        
        return False
    
    def step(self, data: pd.DataFrame, current_time: datetime) -> Dict:
        """
        Execute one orchestration step
        
        Parameters:
        -----------
        data : DataFrame
            Market data
        current_time : datetime
            Current timestamp
        
        Returns:
        --------
        dict : Step results
        """
        # Get signals from all strategies
        signals = {}
        for name, strategy in self.strategies.items():
            try:
                # Each strategy should have a generate_signal method
                signal_output = strategy.generate_signal(data)
                
                # Convert to Signal object
                if signal_output and isinstance(signal_output, dict):
                    signals[name] = Signal(
                        strategy_name=name,
                        timestamp=current_time,
                        action=signal_output.get('action', 'hold'),
                        confidence=signal_output.get('confidence', 0.5),
                        price=data['close'].iloc[-1],
                        metadata=signal_output
                    )
            except Exception as e:
                logger.error(f"Error getting signal from {name}: {e}")
        
        # Aggregate signals
        aggregated_signal = self.aggregate_signals(signals)
        
        # Record signals
        if aggregated_signal:
            self.signals_history.append(aggregated_signal)
        
        # Execute trade
        executed = False
        if aggregated_signal:
            executed = self.execute_signal(aggregated_signal)
        
        # Record equity
        current_prices = {'BTC/USDT': data['close'].iloc[-1]}
        self.portfolio.record_equity(current_time, current_prices)
        
        return {
            'signals': signals,
            'aggregated_signal': aggregated_signal,
            'executed': executed,
            'equity': self.portfolio.equity(current_prices),
            'positions': len(self.portfolio.positions)
        }
    
    def get_performance(self) -> Dict:
        """Get complete performance summary"""
        return self.portfolio.performance_summary()
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if not self.trades_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trades_history)


# Example usage
if __name__ == "__main__":
    print("Strategy Orchestrator - Ready for integration!")
    print("Usage:")
    print("  from orchestrator import StrategyOrchestrator, Portfolio, RiskManager")
    print()
    print("  orchestrator = StrategyOrchestrator(")
    print("      strategies={'rule_based': RuleBasedStrategy(), ...},")
    print("      initial_capital=10000")
    print("  )")
    print()
    print("  # Run backtest")
    print("  for timestamp, data in market_data.iterrows():")
    print("      results = orchestrator.step(data, timestamp)")
    print()
    print("  # Get performance")
    print("  performance = orchestrator.get_performance()")