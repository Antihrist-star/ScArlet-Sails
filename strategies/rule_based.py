"""
STRATEGY - RULE BASED
Simple RSI < 30 mean reversion strategy
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RuleBasedStrategy:
    """Rule-based trading strategy (RSI < 30)"""
    
    def __init__(self, config):
        self.config = config
        self.name = "Rule-Based"
        
        # Parameters
        self.rsi_threshold = 30
        self.rsi_period = 14
        self.cooldown_bars = config['trading']['risk_management']['cooldown_bars']
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, df):
        """Generate trading signals"""
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        
        # Generate raw signals
        df['signal_raw'] = (df['rsi'] < self.rsi_threshold).astype(int)
        
        # Apply cooldown
        df['signal'] = 0
        last_signal_bar = -self.cooldown_bars - 1
        
        for i in range(len(df)):
            if df['signal_raw'].iloc[i] == 1:
                if i - last_signal_bar > self.cooldown_bars:
                    df.loc[df.index[i], 'signal'] = 1
                    last_signal_bar = i
        
        # Remove NaN from RSI calculation
        df['signal'] = df['signal'].fillna(0).astype(int)
        
        logger.debug(f"  Generated {df['signal'].sum()} signals")
        
        return df['signal'].values
    
    def backtest(self, df, regime=None, crisis=None):
        """Run backtest with this strategy"""
        # Generate signals
        signals = self.generate_signals(df)
        
        # Apply filters
        if regime is not None:
            regime_config = self.config['trading']['regime_actions']
            # Adjust position size based on regime
            # (implemented in backtest engine)
        
        if crisis is not None:
            crisis_threshold = self.config['models']['crisis_classifier']['threshold']
            # Filter out signals during crisis
            signals = signals * (crisis < crisis_threshold)
        
        # Execute trades (delegated to backtest engine)
        return signals
    
    def get_parameters(self):
        """Get strategy parameters"""
        return {
            'name': self.name,
            'rsi_threshold': self.rsi_threshold,
            'rsi_period': self.rsi_period,
            'cooldown_bars': self.cooldown_bars
        }