"""
core/backtest_engine.py
=======================
BacktestEngine для симуляции торговли с полным P_j(S) расчётом
Включает: TP/SL, regime-based sizing, crisis filtering, metrics
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Информация о одной сделке"""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'long' or 'short'
    position_size: float
    entry_cost: float
    exit_cost: float
    pnl: float
    pnl_pct: float
    tp_sl_reason: str  # 'tp', 'sl', 'time', 'crisis'
    regime: str
    crisis_level: int
    ml_score: float
    pj_s_value: float


class BacktestEngine:
    """
    Backtesting engine для Scarlet Sails системы.
    
    Особенности:
    - Адаптивные TP/SL из config
    - Regime-based position sizing
    - Crisis filtering (stop trading)
    - Cooldown management
    - Полное логирование метрик
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Config dict с backtesting параметрами
        """
        self.config = config
        self.backtest_config = config.get('backtest', {})
        
        # Основные параметры
        self.initial_capital = self.backtest_config.get('initial_capital', 100000)
        self.position_size_pct = self.backtest_config.get('position_size_pct', 0.95)
        self.take_profit = self.backtest_config.get('take_profit', 0.02)  # 2%
        self.stop_loss = self.backtest_config.get('stop_loss', 0.01)  # 1%
        self.max_hold_bars = self.backtest_config.get('max_hold_bars', 288)  # 3 дня для 15m
        self.commission_pct = self.backtest_config.get('commission_pct', 0.001)  # 0.1%
        self.slippage_pct = self.backtest_config.get('slippage_pct', 0.0005)  # 0.05%
        self.cooldown_bars = self.backtest_config.get('cooldown_bars', 10)
        
        # Адаптация по режиму
        self.regime_tp_sl = self.backtest_config.get('regime_tp_sl', {
            'BULL': {'tp': 1.0, 'sl': 1.0},
            'SIDEWAYS': {'tp': 0.8, 'sl': 1.2},
            'BEAR': {'tp': 0.6, 'sl': 1.5}
        })
        
        # Режим-based position sizing
        self.regime_position_size = self.backtest_config.get('regime_position_size', {
            'BULL': 1.1,
            'SIDEWAYS': 0.6,
            'BEAR': 0.3
        })
        
        # Crisis filtering
        self.crisis_filter_enabled = self.backtest_config.get('crisis_filter_enabled', True)
        
        # State
        self.capital = self.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.last_trade_idx = -self.cooldown_bars
        
        logger.info(f"BacktestEngine инициализирован. Capital: ${self.initial_capital}")
    
    def run(self, 
            df: pd.DataFrame,
            signals: np.ndarray,
            regime: np.ndarray = None,
            crisis_level: np.ndarray = None,
            ml_scores: np.ndarray = None,
            pj_s_values: np.ndarray = None) -> Dict:
        """
        Запускает backtest на историческом периоде.
        
        Args:
            df: DataFrame с OHLCV
            signals: массив сигналов (0 или 1)
            regime: массив режимов (BULL, BEAR, SIDEWAYS)
            crisis_level: массив уровней кризиса (0, 1, 2, 3)
            ml_scores: ML scores для каждой свечи
            pj_s_values: P_j(S) values для каждой свечи
        
        Returns:
            Dict с результатами backtesta
        """
        df = df.copy()
        
        # Defaults для массивов
        if regime is None:
            regime = np.array(['BULL'] * len(df))
        if crisis_level is None:
            crisis_level = np.zeros(len(df), dtype=int)
        if ml_scores is None:
            ml_scores = np.ones(len(df))
        if pj_s_values is None:
            pj_s_values = np.ones(len(df))
        
        # Инициализация
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]
        active_position = None
        
        logger.info(f"Начало backtesta на {len(df)} bars")
        
        try:
            for i in range(len(df)):
                current_price = df.iloc[i]['close']
                current_time = df.index[i]
                current_regime = regime[i]
                current_crisis = crisis_level[i]
                current_ml = ml_scores[i]
                current_pj_s = pj_s_values[i]
                
                # Check if position should be closed (TP/SL/Timeout)
                if active_position:
                    close_trade = self._check_exit_position(
                        active_position, current_price, i, 
                        current_time, current_regime, current_crisis
                    )
                    
                    if close_trade:
                        self.trades.append(close_trade)
                        self.capital += close_trade.pnl - (close_trade.exit_cost * self.commission_pct)
                        active_position = None
                        self.last_trade_idx = i
                
                # Check if new position should be opened
                if not active_position and i > self.last_trade_idx + self.cooldown_bars:
                    if signals[i] == 1:
                        # Crisis filtering
                        if self.crisis_filter_enabled and current_crisis >= 2:
                            logger.debug(f"Bar {i}: Signal отклонен (crisis_level={current_crisis})")
                            continue
                        
                        # Adaptive TP/SL
                        tp_mult = self.regime_tp_sl.get(current_regime, {}).get('tp', 1.0)
                        sl_mult = self.regime_tp_sl.get(current_regime, {}).get('sl', 1.0)
                        
                        position_mult = self.regime_position_size.get(current_regime, 1.0)
                        
                        active_position = self._create_position(
                            entry_idx=i,
                            entry_price=current_price,
                            entry_time=current_time,
                            regime=current_regime,
                            crisis_level=current_crisis,
                            ml_score=current_ml,
                            pj_s_value=current_pj_s,
                            tp_multiplier=tp_mult,
                            sl_multiplier=sl_mult,
                            position_size_multiplier=position_mult
                        )
                        
                        logger.debug(f"Bar {i}: Позиция открыта. Price={current_price:.2f}, "
                                   f"TP={active_position['tp']:.2f}, SL={active_position['sl']:.2f}")
                
                # Track equity
                if active_position:
                    unrealized_pnl = (current_price - active_position['entry_price']) * active_position['position_size']
                    current_equity = self.capital + unrealized_pnl
                else:
                    current_equity = self.capital
                
                self.equity_curve.append(current_equity)
            
            # Close final position if still open
            if active_position:
                final_price = df.iloc[-1]['close']
                close_trade = self._close_position(
                    active_position, final_price, len(df)-1, 
                    df.index[-1], 'time'
                )
                self.trades.append(close_trade)
                self.capital += close_trade.pnl - (close_trade.exit_cost * self.commission_pct)
            
            logger.info(f"Backtest завершен. Trades: {len(self.trades)}, WR: {(len([t for t in self.trades if t.pnl > 0])/len(self.trades)*100) if self.trades else 0:.1f}%")
            
        except Exception as e:
            logger.error(f"Ошибка в backtest: {e}")
            raise
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        return {
            'capital': self.capital,
            'trades': self.trades,
            'equity': self.equity_curve,
            'metrics': metrics
        }
    
    def _create_position(self, 
                        entry_idx: int, 
                        entry_price: float,
                        entry_time: datetime,
                        regime: str,
                        crisis_level: int,
                        ml_score: float,
                        pj_s_value: float,
                        tp_multiplier: float = 1.0,
                        sl_multiplier: float = 1.0,
                        position_size_multiplier: float = 1.0) -> Dict:
        """Создаёт новую позицию с адаптивными параметрами"""
        
        # Адаптивные TP/SL
        tp = entry_price * (1 + self.take_profit * tp_multiplier)
        sl = entry_price * (1 - self.stop_loss * sl_multiplier)
        
        # Adaptive position size
        position_size = (
            self.capital * self.position_size_pct * position_size_multiplier / entry_price
        )
        
        # Entry cost (с комиссией и слиппем)
        entry_cost = entry_price * (1 + self.commission_pct + self.slippage_pct)
        
        return {
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'entry_time': entry_time,
            'entry_cost': entry_cost,
            'tp': tp,
            'sl': sl,
            'position_size': position_size,
            'regime': regime,
            'crisis_level': crisis_level,
            'ml_score': ml_score,
            'pj_s_value': pj_s_value
        }
    
    def _check_exit_position(self, 
                            position: Dict, 
                            current_price: float,
                            current_idx: int,
                            current_time: datetime,
                            current_regime: str,
                            current_crisis: int) -> Trade or None:
        """Проверяет, нужно ли закрыть позицию (TP/SL/Timeout/Crisis)"""
        
        # Crisis level вырос сильно
        if self.crisis_filter_enabled and current_crisis >= 3 and position['crisis_level'] < 3:
            logger.debug(f"Bar {current_idx}: Позиция закрыта (crisis stop)")
            return self._close_position(position, current_price, current_idx, current_time, 'crisis')
        
        # TP
        if current_price >= position['tp']:
            logger.debug(f"Bar {current_idx}: Позиция закрыта (TP hit)")
            return self._close_position(position, current_price, current_idx, current_time, 'tp')
        
        # SL
        if current_price <= position['sl']:
            logger.debug(f"Bar {current_idx}: Позиция закрыта (SL hit)")
            return self._close_position(position, current_price, current_idx, current_time, 'sl')
        
        # Timeout
        if current_idx - position['entry_idx'] >= self.max_hold_bars:
            logger.debug(f"Bar {current_idx}: Позиция закрыта (time)")
            return self._close_position(position, current_price, current_idx, current_time, 'time')
        
        return None
    
    def _close_position(self, 
                       position: Dict,
                       exit_price: float,
                       exit_idx: int,
                       exit_time: datetime,
                       reason: str) -> Trade:
        """Закрывает позицию и возвращает Trade объект"""
        
        # Exit cost (с комиссией и слиппем)
        exit_cost = exit_price * (1 + self.commission_pct + self.slippage_pct)
        
        # P&L
        pnl = (exit_price - position['entry_cost']) * position['position_size']
        pnl_pct = ((exit_price - position['entry_cost']) / position['entry_cost']) * 100
        
        return Trade(
            entry_idx=position['entry_idx'],
            exit_idx=exit_idx,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            entry_time=position['entry_time'],
            exit_time=exit_time,
            direction='long',
            position_size=position['position_size'],
            entry_cost=position['entry_cost'],
            exit_cost=exit_cost,
            pnl=pnl,
            pnl_pct=pnl_pct,
            tp_sl_reason=reason,
            regime=position['regime'],
            crisis_level=position['crisis_level'],
            ml_score=position['ml_score'],
            pj_s_value=position['pj_s_value']
        )
    
    def _calculate_metrics(self) -> Dict:
        """Вычисляет метрики backtesta"""
        
        total_trades = len(self.trades)
        
        if total_trades == 0:
            logger.warning("Нет сделок в backtest!")
            return {
                'final_capital': self.capital,
                'total_pnl': self.capital - self.initial_capital,
                'total_pnl_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'equity_curve': self.equity_curve
            }
        
        # Win/Loss statistics
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        total_wins = sum([t.pnl for t in wins]) if wins else 0
        total_losses = sum([abs(t.pnl) for t in losses]) if losses else 0
        
        # Metrics
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        avg_win = (total_wins / len(wins)) if wins else 0
        avg_loss = (total_losses / len(losses)) if losses else 0
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / (running_max + 1e-8)
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Sharpe Ratio (daily returns, assuming 15m timeframe)
        returns = np.diff(self.equity_curve) / np.array(self.equity_curve[:-1])
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'final_capital': self.capital,
            'total_pnl': self.capital - self.initial_capital,
            'total_pnl_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100,
            'trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': abs(max_drawdown),
            'sharpe_ratio': sharpe,
            'equity_curve': self.equity_curve,
            'trades_detail': self.trades
        }