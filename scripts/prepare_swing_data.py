import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_prepare_swing_data(horizon_days=3):
    """
    Подготовка данных для swing trading
    Использует ЛУЧШИЙ подход из v4 (multi-timeframe)
    """
    print(f"\n{'='*60}")
    print(f"SWING TRADING DATA PREPARATION - {horizon_days} DAYS")
    print(f"{'='*60}")
    
    # Load multi-timeframe data (как в v4)
    from prepare_data_v4 import load_all_timeframes, create_multitimeframe_features
    
    timeframes = load_all_timeframes(symbol="BTC_USDT")
    df = create_multitimeframe_features(timeframes)
    
    # Load swing labels
    labels = np.load(f'data/processed/swing_{horizon_days}d_labels.npy', allow_pickle=True)
    df['label'] = labels
    
    # Remove None
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)
    
    print(f"\nData after labels: {df.shape}")
    print(f"Label distribution:")
    print(f"  UP: {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"  DOWN: {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    
    # Features (исключить price columns и target)
    exclude_cols = ['label', 'close', 'open', 'high', 'low', 'volume']
    exclude_cols += [col for col in df.columns if 'close' in col.lower() and col != 'label']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_data = df[feature_cols].values
    y_data = df['label'].values
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    # Sequences
    seq_len = 60
    X_seq, y_seq = [], []
    
    for i in range(len(X_data) - seq_len):
        X_seq.append(X_data[i:i+seq_len])
        y_seq.append(y_data[i+seq_len])
    
    X = np.array(X_seq)
    y = np.array(y_seq)
    
    print(f"Sequences: {X.shape}")
    
    # Split (temporal)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Tensors
    X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/scaler_swing_{horizon_days}d.pkl')
    joblib.dump(feature_cols, f'models/features_swing_{horizon_days}d.pkl')
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    for days in [3, 5, 7]:
        print(f"\n{'='*60}")
        X_tr, X_te, y_tr, y_te = load_and_prepare_swing_data(days)