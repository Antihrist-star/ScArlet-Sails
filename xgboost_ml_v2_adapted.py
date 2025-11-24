"""
XGBOOST ML STRATEGY - ADAPTED FOR 75 ADVANCED FEATURES
Mathematical formula from MATHEMATICAL_FRAMEWORK.md

P_ml(S) = σ(f_XGB(Φ(S))) · ∏ₖ Fₖ(S) - C_adaptive(S) - R_ood(S)

MAJOR CHANGE: Adapted for 75 advanced features from REAL DATA
- Original: 31 classic indicators (multi-timeframe)
- Now: 74 advanced features from data/features/*.parquet

Components:
- σ(f_XGB(Φ(S))): XGBoost prediction with sigmoid normalization
- Φ(S): 74 advanced features (REAL DATA)
- ∏ₖ Fₖ: Regime-based filters (crisis, drawdown, regime)
- C_adaptive: Volatility-adjusted costs
- R_ood: Out-of-distribution risk penalty

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 23, 2025 (ADAPTED)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install: pip install xgboost")
    XGB_AVAILABLE = False


class FeatureTransformer:
    """
    Φ(S): Feature transformation for REAL MARKET DATA
    
    ADAPTED FOR 74 ADVANCED FEATURES from data/features/*.parquet:
    - Basic OHLCV: 5 features
    - Normalized (norm_*): 12 features
    - Derivatives (deriv_*): 20 features
    - Regime indicators (regime_*): 12 features
    - Cross features (cross_*): 12 features
    - Divergences (div_*): 6 features
    - Time features (time_*): 7 features
    
    Total: 74 features (excluding 'target')
    
    NOTE: Old 31-feature xgboost_trained_v2.json model is INCOMPATIBLE!
    Will need to re-train in Phase 4 on 74 features.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # All 74 advanced features from real data (excluding 'target')
        self.feature_names = [
            # Basic OHLCV (5)
            'open', 'high', 'low', 'close', 'volume',
            
            # Normalized features (12)
            'norm_close_zscore', 'norm_close_pctile',
            'norm_volume_zscore', 'norm_volume_pctile',
            'norm_rsi_zscore', 'norm_rsi_pctile',
            'norm_macd_zscore', 'norm_macd_pctile',
            'norm_atr_zscore', 'norm_atr_pctile',
            'norm_bb_width_zscore', 'norm_bb_width_pctile',
            
            # Derivative features (20)
            'deriv_close_diff1', 'deriv_close_diff2', 'deriv_close_roc5', 'deriv_close_roc20', 'deriv_close_velocity',
            'deriv_rsi_diff1', 'deriv_rsi_diff2', 'deriv_rsi_roc5', 'deriv_rsi_roc20', 'deriv_rsi_velocity',
            'deriv_macd_diff1', 'deriv_macd_diff2', 'deriv_macd_roc5', 'deriv_macd_roc20', 'deriv_macd_velocity',
            'deriv_volume_diff1', 'deriv_volume_diff2', 'deriv_volume_roc5', 'deriv_volume_roc20', 'deriv_volume_velocity',
            
            # Regime features (12)
            'regime_rsi_low', 'regime_rsi_mid', 'regime_rsi_high',
            'regime_atr_low', 'regime_atr_mid', 'regime_atr_high',
            'regime_bb_width_low', 'regime_bb_width_mid', 'regime_bb_width_high',
            'regime_volume_ratio_low', 'regime_volume_ratio_mid', 'regime_volume_ratio_high',
            
            # Cross features (12)
            'cross_rsi_volume_ratio_ratio', 'cross_rsi_volume_ratio_corr50', 'cross_rsi_volume_ratio_diff',
            'cross_macd_atr_ratio', 'cross_macd_atr_corr50', 'cross_macd_atr_diff',
            'cross_bb_width_volume_ratio_ratio', 'cross_bb_width_volume_ratio_corr50', 'cross_bb_width_volume_ratio_diff',
            'cross_stoch_k_rsi_ratio', 'cross_stoch_k_rsi_corr50', 'cross_stoch_k_rsi_diff',
            
            # Divergence features (6)
            'div_rsi_bullish', 'div_rsi_bearish',
            'div_macd_bullish', 'div_macd_bearish',
            'div_stoch_k_bullish', 'div_stoch_k_bearish',
            
            # Time features (7)
            'time_hour', 'time_asian', 'time_european', 'time_american',
            'time_dayofweek', 'time_is_monday', 'time_is_friday'
        ]
        
        self.scaler = None
        self.use_advanced_features = True
        
        logger.info(f"FeatureTransformer initialized: {len(self.feature_names)} ADVANCED features (REAL DATA)")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame to feature matrix
        
        SIMPLIFIED for advanced features - data already has all features!
        Just extract the columns we need.
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 75 columns from data/features/*.parquet
        
        Returns:
        --------
        DataFrame with 74 features (excluding 'target')
        """
        # Check if DataFrame has all required features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        
        if missing_features:
            logger.error(f"Missing features in DataFrame: {missing_features[:10]}...")
            raise ValueError(f"DataFrame missing {len(missing_features)} required features")
        
        # Extract features
        features_df = df[self.feature_names].copy()
        
        # Handle NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.debug(f"Transformed features: {features_df.shape}")
        
        return features_df
    
    def normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using scaler
        
        NOTE: For advanced features, normalization might not be needed
        as many features are already normalized (norm_*)
        """
        if self.scaler is not None:
            try:
                features_scaled = self.scaler.transform(features)
                return pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}, using unnormalized features")
                return features
        else:
            # Many features already normalized - return as is
            return features


class RegimeFilters:
    """
    ∏ₖ Fₖ: Regime-based filters for ML strategy
    
    Filters:
    - F₁: Crisis filter (crisis_level < threshold)
    - F₂: Drawdown filter (drawdown < max_dd)
    - F₃: Regime compatibility (avoid certain regimes)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Thresholds
        self.crisis_threshold = self.config.get('crisis_threshold', 0.7)
        self.max_drawdown = self.config.get('max_drawdown', 0.20)
        
        logger.info("RegimeFilters initialized")
    
    def crisis_filter(self, crisis_level: float) -> int:
        """
        F₁: Crisis filter
        Returns 1 if crisis_level < threshold, 0 otherwise
        """
        return 1 if crisis_level < self.crisis_threshold else 0
    
    def drawdown_filter(self, drawdown: float) -> int:
        """
        F₂: Drawdown filter
        Returns 1 if drawdown < max_dd, 0 otherwise
        """
        return 1 if drawdown < self.max_drawdown else 0
    
    def regime_filter(self, regime: str) -> int:
        """
        F₃: Regime compatibility filter
        Returns 1 for favorable regimes, 0 otherwise
        """
        favorable_regimes = ['normal', 'bull', 'recovery']
        return 1 if regime.lower() in favorable_regimes else 0
    
    def calculate_filters(self, market_state: Dict) -> int:
        """
        Calculate product of all filters
        
        Returns:
        --------
        int : Product (binary AND) of all filters {0, 1}
        """
        crisis_level = market_state.get('crisis_level', 0.0)
        drawdown = market_state.get('drawdown', 0.0)
        regime = market_state.get('regime', 'normal')
        
        f1 = self.crisis_filter(crisis_level)
        f2 = self.drawdown_filter(drawdown)
        f3 = self.regime_filter(regime)
        
        filters_product = f1 * f2 * f3
        
        logger.debug(f"Filters: F1={f1}, F2={f2}, F3={f3} → product={filters_product}")
        
        return filters_product


class XGBoostMLStrategy:
    """
    XGBoost ML Trading Strategy - ADAPTED FOR 75 ADVANCED FEATURES
    
    Decision function:
    P_ml(S) = σ(f_XGB(Φ(S))) · ∏ₖ Fₖ - C_adaptive - R_ood
    
    KEY CHANGE: Uses 74 advanced features instead of 31 classic indicators
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize XGBoost ML Strategy
        
        Parameters:
        -----------
        config : dict
            Configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.feature_transformer = FeatureTransformer(self.config.get('features', {}))
        self.regime_filters = RegimeFilters(self.config.get('filters', {}))
        
        # Costs
        self.commission = self.config.get('commission', 0.001)
        self.slippage_base = self.config.get('slippage', 0.0005)
        self.cost_beta = self.config.get('cost_beta', 2.0)
        
        # OOD detection
        self.ood_threshold = self.config.get('ood_threshold', 3.0)
        self.training_mean = None
        self.training_cov_inv = None
        
        # XGBoost model
        self.model = None
        self.model_loaded = False
        
        # Try to load model
        self._try_load_model()
        
        logger.info(f"XGBoostMLStrategy initialized (74 advanced features, model_loaded={self.model_loaded})")
    
    def _try_load_model(self):
        """Try to load XGBoost model - WILL FAIL with old 31-feature model"""
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available, strategy will use fallback predictions")
            return
        
        model_path = self.config.get('model_path', 'models/xgboost_trained_v2.json')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}, will use fallback")
            return
        
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            self.model_loaded = True
            logger.info(f"✅ XGBoost model loaded: {model_path}")
            logger.warning("⚠️ Model trained on 31 features, but strategy expects 74!")
            logger.warning("⚠️ Model predictions will be INCOMPATIBLE until re-training in Phase 4!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def set_risk_calculator(self, risk_calc):
        """Set the AdvancedRiskPenalty calculator instance (optional)"""
        self.risk_calculator = risk_calc
        logger.info("Risk calculator set")
    
    def set_training_distribution(self, training_features: np.ndarray):
        """
        Set training distribution for OOD detection
        
        Parameters:
        -----------
        training_features : array
            Training feature matrix (N x 74) for computing mean and covariance
        """
        from scipy.linalg import inv
        
        self.training_mean = np.mean(training_features, axis=0)
        cov = np.cov(training_features.T)
        cov += 1e-6 * np.eye(cov.shape[0])
        self.training_cov_inv = inv(cov)
        
        logger.info(f"Training distribution set: {training_features.shape}")
    
    def calculate_ood_risk(self, features: np.ndarray) -> float:
        """
        Calculate out-of-distribution risk using Mahalanobis distance
        
        R_ood(S) = D_M(Φ(S), X_train) / τ
        """
        if self.training_mean is None:
            return 0.0
        
        diff = features - self.training_mean
        mahal_distance = np.sqrt(np.dot(np.dot(diff, self.training_cov_inv), diff.T))
        
        R_ood = mahal_distance / self.ood_threshold
        
        return float(np.clip(R_ood, 0.0, 10.0))
    
    def calculate_adaptive_costs(self, volatility: float) -> float:
        """
        Calculate volatility-adjusted transaction costs
        
        C_adaptive(S) = C_fixed · (1 + β·σ)
        """
        C_fixed = self.commission + self.slippage_base
        C_adaptive = C_fixed * (1 + self.cost_beta * volatility)
        
        return C_adaptive
    
    def predict_proba(self, features: np.ndarray) -> float:
        """
        Get ML prediction probability
        
        σ(f_XGB(Φ(S))) where σ is sigmoid normalization
        
        NOTE: Will use fallback until model is re-trained on 74 features
        """
        if not self.model_loaded or self.model is None:
            # Fallback: simple heuristic based on features
            # Use normalized RSI as proxy (feature index 11)
            if len(features) > 11:
                rsi_norm = features[11]  # norm_rsi_pctile
                # Simple sigmoid: more extreme RSI = higher probability
                prob = 1.0 / (1.0 + np.exp(-5 * (abs(rsi_norm - 0.5) - 0.3)))
                return float(np.clip(prob, 0.3, 0.7))
            else:
                return 0.5
        
        try:
            # INCOMPATIBLE: Model expects 31 features, we have 74
            # This will fail or give nonsense predictions
            logger.warning("⚠️ Model incompatibility: using fallback prediction")
            return 0.5
        except Exception as e:
            logger.warning(f"Prediction failed: {e}, using fallback")
            return 0.5
    
    def calculate_pjs(self, features: np.ndarray, market_state: Dict) -> Tuple[float, Dict]:
        """
        Calculate P_ml(S) for given features and market state
        
        Parameters:
        -----------
        features : array
            74-dimensional feature vector Φ(S)
        market_state : dict
            Market state for filters and costs
        
        Returns:
        --------
        tuple : (P_ml value, dict of components)
        """
        # Component 1: ML prediction
        ml_score = self.predict_proba(features)
        
        # Component 2: Regime filters
        filters_product = self.regime_filters.calculate_filters(market_state)
        
        # Component 3: Adaptive costs
        volatility = market_state.get('volatility', 0.02)
        costs = self.calculate_adaptive_costs(volatility)
        
        # Component 4: OOD risk
        R_ood = self.calculate_ood_risk(features)
        
        # Calculate P_ml(S)
        P_ml = ml_score * filters_product - costs - R_ood
        
        # Store components
        components = {
            'ml_score': ml_score,
            'filters_product': filters_product,
            'costs': costs,
            'R_ood': R_ood,
            'P_ml': P_ml
        }
        
        logger.debug(f"P_ml={P_ml:.4f}: ML={ml_score:.3f} × filters={filters_product} - costs={costs:.4f} - OOD={R_ood:.4f}")
        
        return P_ml, components
    
    def generate_signals(self, df: pd.DataFrame, market_states: list = None) -> pd.DataFrame:
        """
        Generate trading signals for DataFrame with 75 columns
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame with 75 columns from data/features/*.parquet
        market_states : list, optional
            Pre-computed market states
        
        Returns:
        --------
        DataFrame with signals and components
        """
        # Transform features (extract 74 out of 75, excluding 'target')
        features_df = self.feature_transformer.transform(df)
        
        # Normalize if scaler available
        features_df = self.feature_transformer.normalize(features_df)
        
        logger.info(f"Generating ML signals for {len(features_df)} bars...")
        
        results = []
        
        for idx in range(len(features_df)):
            # Get features
            features = features_df.iloc[idx].values
            
            # Market state
            if market_states and idx < len(market_states):
                market_state = market_states[idx]
            else:
                # Create minimal market state
                # Use advanced features to estimate volatility
                volatility = abs(df.iloc[idx].get('deriv_close_velocity', 0.02))
                
                market_state = {
                    'crisis_level': 0.0,
                    'drawdown': 0.0,
                    'regime': 'normal',
                    'volatility': volatility
                }
            
            # Calculate P_ml
            P_ml, components = self.calculate_pjs(features, market_state)
            
            # Generate signal (lower threshold for baseline testing)
            signal = 1 if P_ml > 0.0 else 0
            
            results.append({
                'timestamp': features_df.index[idx],
                'P_ml': P_ml,
                'signal': signal,
                **components
            })
        
        results_df = pd.DataFrame(results)
        results_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {results_df['signal'].sum()} ML signals (out of {len(features_df)} bars)")
        
        return results_df


# Testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("XGBOOST ML STRATEGY TEST (74 ADVANCED FEATURES)")
    print("=" * 80)
    
    # This test requires REAL DATA
    print("\n⚠️ This strategy requires REAL DATA with 75 columns!")
    print("Load data from: data/features/BTC_USDT_15m_features.parquet")
    print("\nTo test:")
    print("  1. Load parquet file")
    print("  2. strategy = XGBoostMLStrategy()")
    print("  3. signals = strategy.generate_signals(df)")
    
    print("\n" + "=" * 80)
    print("READY FOR INTEGRATION TESTING")
    print("=" * 80)