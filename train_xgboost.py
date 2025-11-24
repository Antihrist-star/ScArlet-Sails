"""
XGBOOST TRAINING SCRIPT
Train XGBoost model to predict profitable trading opportunities

Target: Forward returns > threshold (profitable trades)
Features: 31 technical indicators across 4 timeframes

Author: STAR_ANT + Claude Sonnet 4.5
Date: November 18, 2025
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from datetime import datetime

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    XGB_AVAILABLE = True
except ImportError:
    print("ERROR: XGBoost not installed!")
    print("Install: pip install xgboost scikit-learn")
    sys.exit(1)

from strategies.xgboost_ml_v2 import FeatureTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data(n_bars=5000, seed=42):
    """
    Generate realistic training data with multiple regimes
    
    Parameters:
    -----------
    n_bars : int
        Number of bars
    seed : int
        Random seed
    
    Returns:
    --------
    DataFrame : OHLCV data
    """
    np.random.seed(seed)
    
    logger.info(f"Generating {n_bars} bars of training data...")
    
    # Multiple regimes for robustness
    regime_lengths = [1000, 1500, 1000, 1000, 500]
    regime_trends = [0.0003, -0.0002, 0.0002, -0.0001, 0.0004]  # Bull, Bear, Bull, Sideways, Strong Bull
    regime_vols = [0.015, 0.025, 0.018, 0.030, 0.020]
    
    close_prices = [50000]
    
    for regime_len, trend, vol in zip(regime_lengths, regime_trends, regime_vols):
        for _ in range(regime_len):
            ret = np.random.normal(trend, vol)
            new_price = close_prices[-1] * (1 + ret)
            close_prices.append(new_price)
    
    close_prices = np.array(close_prices[:n_bars])
    close_prices = np.maximum(close_prices, 1000)
    
    dates = pd.date_range('2020-01-01', periods=n_bars, freq='h')
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.normal(0, 0.0003, n_bars)),
        'high': close_prices * (1 + np.abs(np.random.normal(0.001, 0.002, n_bars))),
        'low': close_prices * (1 - np.abs(np.random.normal(0.001, 0.002, n_bars))),
        'close': close_prices,
        'volume': np.random.lognormal(5, 0.5, n_bars)
    }, index=dates)
    
    logger.info(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def create_training_dataset(df: pd.DataFrame, forward_periods: int = 5, 
                            profit_threshold: float = 0.015):
    """
    Create training dataset with features and labels
    
    ИСПРАВЛЕНО: Правильное создание feature array без NaN
    
    Parameters:
    -----------
    df : DataFrame
        OHLCV data
    forward_periods : int
        Look forward N periods for target
    profit_threshold : float
        Minimum return to be considered profitable (снижен до 1.5%)
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("Creating training dataset...")
    logger.info(f"  Forward periods: {forward_periods}")
    logger.info(f"  Profit threshold: {profit_threshold:.2%}")
    
    def safe_value(val, default=0.0):
        """Safely convert value, replacing NaN/inf with default"""
        if pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    
    def calculate_features_for_bar(df, idx):
        """Calculate 31 features for a single bar"""
        if idx < 50:  # Need enough history
            return None
            
        features = []
        
        # Get window
        window = df.iloc[max(0, idx-50):idx+1]
        close = window['close']
        high = window['high']
        low = window['low']
        volume = window['volume']
        
        try:
            # Feature 0: RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features.append(safe_value(rsi.iloc[-1] / 100, 0.5))
            
            # Features 1-3: Price to EMAs
            ema9 = close.ewm(span=9, min_periods=1).mean()
            ema21 = close.ewm(span=21, min_periods=1).mean()
            sma50 = close.rolling(50, min_periods=20).mean()
            
            features.append(safe_value((close.iloc[-1] - ema9.iloc[-1]) / ema9.iloc[-1], 0.0))
            features.append(safe_value((close.iloc[-1] - ema21.iloc[-1]) / ema21.iloc[-1], 0.0))
            features.append(safe_value((close.iloc[-1] - sma50.iloc[-1]) / sma50.iloc[-1], 0.0))
            
            # Feature 4: Bollinger Bands width
            bb_ma = close.rolling(20, min_periods=10).mean()
            bb_std = close.rolling(20, min_periods=10).std()
            bb_width = bb_std / (bb_ma + 1e-10)
            features.append(safe_value(bb_width.iloc[-1], 0.02))
            
            # Feature 5: ATR percentage
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14, min_periods=5).mean()
            atr_pct = atr / (close + 1e-10)
            features.append(safe_value(atr_pct.iloc[-1], 0.02))
            
            # Features 6-7: Returns
            returns_5 = close.pct_change(5)
            returns_10 = close.pct_change(10)
            features.append(safe_value(returns_5.iloc[-1], 0.0))
            features.append(safe_value(returns_10.iloc[-1], 0.0))
            
            # Features 8-9: Volume ratios
            vol_ma5 = volume.rolling(5, min_periods=2).mean()
            vol_ma10 = volume.rolling(10, min_periods=5).mean()
            features.append(safe_value(volume.iloc[-1] / (vol_ma5.iloc[-1] + 1e-10), 1.0))
            features.append(safe_value(volume.iloc[-1] / (vol_ma10.iloc[-1] + 1e-10), 1.0))
            
            # Features 10-12: Duplicates (for 31-feature compatibility)
            features.append(features[1])  # price_to_EMA9 dup
            features.append(features[2])  # price_to_EMA21 dup
            features.append(features[3])  # price_to_SMA50 dup
            
            # Features 13-30: Multi-timeframe (simplified - use same as 15m)
            # In real implementation, these would be from different timeframes
            for _ in range(3):  # 3 timeframes (1h, 4h, 1d) x 6 features = 18
                features.append(features[0])  # RSI
                features.append(features[6])  # returns
                features.append(features[1])  # EMA9
                features.append(features[2])  # EMA21
                features.append(features[3])  # SMA50
                features.append(features[5])  # ATR
            
            # Verify we have exactly 31 features
            assert len(features) == 31, f"Expected 31 features, got {len(features)}"
            
            return features
            
        except Exception as e:
            logger.warning(f"Error calculating features at idx {idx}: {e}")
            return None
    
    logger.info("  Calculating features for all bars...")
    features_list = []
    valid_indices = []
    
    for idx in range(50, len(df) - forward_periods):
        feats = calculate_features_for_bar(df, idx)
        if feats is not None:
            features_list.append(feats)
            valid_indices.append(idx)
    
    if len(features_list) == 0:
        raise ValueError("No valid features generated! Check data.")
    
    features_array = np.array(features_list)
    logger.info(f"  Generated features shape: {features_array.shape}")
    
    # Create target: forward returns
    logger.info("  Creating target variable...")
    forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)
    
    # Binary classification: profitable (1) or not (0)
    target = (forward_returns > profit_threshold).astype(int)
    target_values = target.iloc[valid_indices].values
    
    # Final validation - remove any remaining NaN
    valid_mask = ~np.isnan(features_array).any(axis=1) & ~pd.isna(target_values)
    
    X = features_array[valid_mask]
    y = target_values[valid_mask]
    
    logger.info(f"  ✓ Valid samples: {len(X)}")
    logger.info(f"  ✓ Positive samples: {y.sum()} ({y.mean():.2%})")
    logger.info(f"  ✓ Negative samples: {len(y) - y.sum()} ({1 - y.mean():.2%})")
    
    if len(X) == 0:
        raise ValueError("No valid samples after filtering! Check thresholds.")
    
    if y.sum() == 0 or y.sum() == len(y):
        logger.warning(f"  ⚠️  Imbalanced dataset! Adjusting threshold...")
        # Use median return as threshold
        median_return = forward_returns.median()
        logger.info(f"  Using median return as threshold: {median_return:.4f}")
        target = (forward_returns > median_return).astype(int)
        target_values = target.iloc[valid_indices].values
        valid_mask = ~np.isnan(features_array).any(axis=1) & ~pd.isna(target_values)
        X = features_array[valid_mask]
        y = target_values[valid_mask]
        logger.info(f"  ✓ Rebalanced - Positive: {y.sum()} ({y.mean():.2%})")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"  ✓ Train samples: {len(X_train)}")
    logger.info(f"  ✓ Test samples: {len(X_test)}")
    
    feature_names = [
        'rsi_15m', 'price_to_ema9_15m', 'price_to_ema21_15m', 'price_to_sma50_15m',
        'bb_width_15m', 'atr_pct_15m', 'returns_5_15m', 'returns_10_15m',
        'vol_ratio_5_15m', 'vol_ratio_10_15m', 'dup_ema9', 'dup_ema21', 'dup_sma50',
        'rsi_1h', 'returns_1h', 'ema9_1h', 'ema21_1h', 'sma50_1h', 'atr_1h',
        'rsi_4h', 'returns_4h', 'ema9_4h', 'ema21_4h', 'sma50_4h', 'atr_4h',
        'rsi_1d', 'returns_1d', 'ema9_1d', 'ema21_1d', 'sma50_1d', 'atr_1d'
    ]
    
    return X_train, X_test, y_train, y_test, feature_names


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    
    Returns:
    --------
    xgb.Booster : Trained model
    """
    logger.info("Training XGBoost model...")
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'seed': 42
    }
    
    # Train
    evals = [(dtrain, 'train'), (dtest, 'test')]
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=20
    )
    
    logger.info(f"  ✓ Best iteration: {model.best_iteration}")
    logger.info(f"  ✓ Best score: {model.best_score:.4f}")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : Trained model
    X_test, y_test : Test data
    feature_names : Feature names
    """
    logger.info("Evaluating model...")
    
    # Predict
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST PERFORMANCE:")
    logger.info(f"{'='*60}")
    logger.info(f"  AUC Score: {auc:.4f}")
    logger.info(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Profitable', 'Profitable']))
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    if importance:
        importance_df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TOP 10 IMPORTANT FEATURES:")
        logger.info(f"{'='*60}")
        for idx, row in importance_df.head(10).iterrows():
            feat_idx = int(row['feature'].replace('f', ''))
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else row['feature']
            logger.info(f"  {feat_name:20s}: {row['importance']:>8.1f}")


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("XGBOOST MODEL TRAINING - SCARLET SAILS")
    print("="*80 + "\n")
    
    # Step 1: Generate data
    print("STEP 1: DATA GENERATION")
    print("-"*80)
    df = generate_training_data(n_bars=5000)
    print()
    
    # Step 2: Create dataset
    print("STEP 2: DATASET CREATION")
    print("-"*80)
    X_train, X_test, y_train, y_test, feature_names = create_training_dataset(
        df, 
        forward_periods=5,
        profit_threshold=0.015
    )
    print()
    
    # Step 3: Train model
    print("STEP 3: MODEL TRAINING")
    print("-"*80)
    model = train_xgboost(X_train, y_train, X_test, y_test)
    print()
    
    # Step 4: Evaluate
    print("STEP 4: EVALUATION")
    print("-"*80)
    evaluate_model(model, X_test, y_test, feature_names)
    print()
    
    # Step 5: Save model
    print("STEP 5: SAVING MODEL")
    print("-"*80)
    
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgboost_trained.json'
    model.save_model(model_path)
    
    logger.info(f"✅ Model saved to: {model_path}")
    logger.info(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print()
    
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Update XGBoostMLStrategy to load this model")
    print("  2. Re-run dispersion analysis with trained model")
    print("  3. Verify Hybrid Strategy performance")
    print()


if __name__ == "__main__":
    main()
