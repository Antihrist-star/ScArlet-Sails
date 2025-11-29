"""XGBoost ML Strategy v3
Works with 74 features from advanced feature engineering.
Supports single timeframe or multi-timeframe scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class XGBoostMLStrategyV3:
    """XGBoost ML Strategy for 74 features.
    
    This is an enhanced version supporting:
    - 74 advanced features (vs 31 in v2)
    - Better feature engineering
    - Improved risk management
    """
    
    PHASE = "Phase 2: Production Ready"
    DESCRIPTION = "XGBoost ML model with 74 features"
    
    def __init__(self, config: Dict = None):
        """Initialize the strategy.
        
        Parameters
        ----------
        config : dict
            Configuration with 'model_manager' instance
        """
        self.config = config or {}
        self.model_manager = self.config.get('model_manager')
        self.logger = logging.getLogger(__name__)
        
        # Model parameters (74 features)
        self.n_features = 74
        self.model_name = 'xgboost_v3'
        
        # Performance tracking
        self.returns = []
        self.returns_ml = []
        
        self.logger.info(f"XGBoostMLStrategyV3 initialized ({self.n_features} features)")
    
    def calculate_signals(self, features: pd.Series, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate trading signals using XGBoost ML model.
        
        Parameters
        ----------
        features : pd.Series
            Feature vector (74 features)
        prices : pd.Series
            Price series
        period : int, optional
            Lookback period
            
        Returns
        -------
        pd.Series
            Trading signals (1=buy, 0=hold, -1=sell)
        """
        # Validate input
        if len(features) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(features)}")
        
        # Get model prediction
        if self.model_manager is None:
            self.logger.warning("No model_manager available, returning 0")
            return 0
        
        # Call model
        signal_prob = self.model_manager.predict_xgboost_v3(features)
        
        # Convert probability to signal
        # threshold = 0.5 by default
        if signal_prob > 0.6:
            return 1  # Strong buy signal
        elif signal_prob > 0.5:
            return 1  # Buy signal
        elif signal_prob < 0.4:
            return -1  # Sell signal
        else:
            return 0  # Hold
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Generate trading signals for the entire dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 74 feature columns and prices
            
        Returns
        -------
        tuple
            (signals Series, metadata dict)
        """
        signals = []
        metadata = {
            'model': 'xgboost_v3',
            'n_features': self.n_features,
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0
        }
        
        for idx, row in df.iterrows():
            # Extract 74 features from row
            feature_cols = [col for col in df.columns if col not in ['close', 'volume', 'target']]
            features = row[feature_cols].values
            
            # Get signal
            signal = self.calculate_signals(features, row.get('close'))
            signals.append(signal)
            
            # Track metadata
            if signal == 1:
                metadata['buy_signals'] += 1
            elif signal == -1:
                metadata['sell_signals'] += 1
            else:
                metadata['neutral_signals'] += 1
        
        return pd.Series(signals, index=df.index), metadata
    
    def update_performance(self, actual_return: float, ml_return: float):
        """Track performance metrics."""
        self.returns.append(actual_return)
        self.returns_ml.append(ml_return)
    
    def get_status(self) -> Dict:
        """Get strategy status."""
        return {
            'phase': self.PHASE,
            'description': self.DESCRIPTION,
            'n_features': self.n_features,
            'total_returns': sum(self.returns) if self.returns else 0,
            'model_returns': sum(self.returns_ml) if self.returns_ml else 0
        }
