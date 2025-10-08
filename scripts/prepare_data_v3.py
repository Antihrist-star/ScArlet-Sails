import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def create_profitable_target(df, future_bars=16, min_profit=0.008, max_stop=0.004):
    """
    Target = 1 если trade profitable после комиссий и не hit stop-loss
    
    Parameters:
    - future_bars: 16 (4 часа для 15m данных)
    - min_profit: 0.008 (0.8% чистая прибыль после 0.2% комиссий)
    - max_stop: 0.004 (0.4% максимальный stop-loss)
    """
    print("\n=== Creating Profitable Target ===")
    
    # Future prices
    future_close = df['close'].shift(-future_bars)
    
    # Rolling max/min over future window
    future_high_list = []
    future_low_list = []
    for i in range(len(df)):
        if i + future_bars >= len(df):
            future_high_list.append(np.nan)
            future_low_list.append(np.nan)
        else:
            future_high_list.append(df['high'].iloc[i:i+future_bars].max())
            future_low_list.append(df['low'].iloc[i:i+future_bars].min())
    
    future_high = pd.Series(future_high_list, index=df.index)
    future_low = pd.Series(future_low_list, index=df.index)
    
    # Calculate profit and drawdown
    profit_pct = (future_close - df['close']) / df['close']
    max_drawdown = (df['close'] - future_low) / df['close']
    
    # Target = 1 if profitable AND not hit stop
    profitable = (profit_pct > min_profit) & (max_drawdown < max_stop)
    
    df['target_profitable'] = profitable.astype(int)
    
    # Remove rows without future data
    df = df[:-future_bars].copy()
    
    # Statistics
    profit_pct_valid = profit_pct[:-future_bars]
    print(f"Profitable trades: {profitable.sum()} / {len(profitable[:-future_bars])} ({profitable.sum() / len(profitable[:-future_bars]) * 100:.2f}%)")
    print(f"Average profit when profitable: {profit_pct_valid[profitable[:-future_bars]].mean():.4f}")
    print(f"Average loss when not profitable: {profit_pct_valid[~profitable[:-future_bars]].mean():.4f}")
    
    return df

def add_regime_features(df):
    """
    Add market regime features
    """
    print("\n=== Adding Market Regime Features ===")
    
    # Returns over different periods
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_20'] = df['close'].pct_change(20)
    df['returns_60'] = df['close'].pct_change(60)
    
    # Volatility (ATR normalized by price)
    if 'ATR_14' in df.columns:
        df['volatility'] = df['ATR_14'] / df['close']
    
    # Regime classification based on 20-period returns
    df['regime_bull'] = (df['returns_20'] > 0.02).astype(int)
    df['regime_bear'] = (df['returns_20'] < -0.02).astype(int)
    df['regime_sideways'] = ((df['returns_20'] >= -0.02) & (df['returns_20'] <= 0.02)).astype(int)
    
    # Volume regime (high/low volume periods)
    if 'volume' in df.columns:
        volume_ma = df['volume'].rolling(20).mean()
        df['volume_high'] = (df['volume'] > volume_ma * 1.5).astype(int)
        df['volume_low'] = (df['volume'] < volume_ma * 0.5).astype(int)
    
    print(f"Bull periods: {df['regime_bull'].sum() / len(df) * 100:.1f}%")
    print(f"Bear periods: {df['regime_bear'].sum() / len(df) * 100:.1f}%")
    print(f"Sideways periods: {df['regime_sideways'].sum() / len(df) * 100:.1f}%")
    
    return df

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60):
    print("=== Loading Data ===")
    
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)
            print(f"Loaded: {filename}, shape: {df.shape}")
    
    if not all_data:
        raise ValueError("No 15m parquet files found.")
    
    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    print(f"Combined data shape: {combined_df.shape}")
    
    # Add technical indicators (from previous version)
    print("\n=== Adding Technical Indicators ===")
    
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    combined_df['RSI_14'] = calculate_rsi(combined_df['close'], 14)
    combined_df['EMA_9'] = combined_df['close'].ewm(span=9, adjust=False).mean()
    combined_df['EMA_21'] = combined_df['close'].ewm(span=21, adjust=False).mean()
    combined_df['BB_middle'] = combined_df['close'].rolling(window=20).mean()
    combined_df['BB_std'] = combined_df['close'].rolling(window=20).std()
    combined_df['BB_upper'] = combined_df['BB_middle'] + (combined_df['BB_std'] * 2)
    combined_df['BB_lower'] = combined_df['BB_middle'] - (combined_df['BB_std'] * 2)
    
    high_low = combined_df['high'] - combined_df['low']
    high_close = abs(combined_df['high'] - combined_df['close'].shift())
    low_close = abs(combined_df['low'] - combined_df['close'].shift())
    true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    combined_df['ATR_14'] = true_range.rolling(window=14).mean()
    
    # Add regime features BEFORE creating target
    combined_df = add_regime_features(combined_df)
    
    # Create profitable target (AFTER indicators)
    combined_df = create_profitable_target(combined_df, future_bars=16, min_profit=0.008, max_stop=0.004)
    
    # Remove NaN
    combined_df = combined_df.dropna()
    print(f"\nAfter dropna: {combined_df.shape}")
    
    # Features (exclude close and target)
    feature_cols = [col for col in combined_df.columns if col not in ['target_profitable', 'close']]
    print(f"Total features: {len(feature_cols)}")
    
    X_data = combined_df[feature_cols].values
    y_data = combined_df['target_profitable'].values
    
    print(f"\nX shape: {X_data.shape}")
    print(f"y shape: {y_data.shape}")
    print(f"Target distribution: {np.bincount(y_data.astype(int))}")
    
    # Create sequences WITHOUT scaling
    print("\n=== Creating Sequences ===")
    X_sequences = []
    y_targets = []
    for i in range(len(X_data) - sequence_length):
        X_sequences.append(X_data[i:i+sequence_length])
        y_targets.append(y_data[i+sequence_length])
    
    X = np.array(X_sequences)
    y = np.array(y_targets)
    
    print(f"Sequences created: {X.shape}")
    
    # TEMPORAL SPLIT
    split_idx = int(len(X) * 0.8)
    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    print(f"\nTrain: {X_train_raw.shape}, Test: {X_test_raw.shape}")
    print(f"Train target dist: {np.bincount(y_train.astype(int))}")
    print(f"Test target dist: {np.bincount(y_test.astype(int))}")
    
    # FIT SCALER ONLY ON TRAIN
    print("\n=== Scaling Data ===")
    scaler = MinMaxScaler()
    X_train_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train_raw.shape)
    X_test_reshaped = X_test_raw.reshape(-1, X_test_raw.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test_raw.shape)
    
    # To tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler_X_v3.pkl")
    
    print("\n=== Data Preparation Complete ===")
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"\nFinal shapes:")
    print(f"Train X: {X_train.shape}, Train y: {y_train.shape}")
    print(f"Test X: {X_test.shape}, Test y: {y_test.shape}")
    print(f"Features: {X_train.shape[2]}")