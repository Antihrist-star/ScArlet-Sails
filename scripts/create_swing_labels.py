import pandas as pd
import numpy as np
import os

def create_swing_labels(df, horizon_days=3, threshold=0.03):
    """
    Swing trading labels для 3-дневного прогноза
    
    UP (1): цена через 3 дня выросла >3%
    DOWN (0): всё остальное
    
    Простой binary classification - избегаем NEUTRAL класса
    """
    print(f"\nCreating swing labels: {horizon_days} days, {threshold*100}% threshold")
    
    labels = []
    horizon_bars = horizon_days * 24 * 4  # 3 дня * 24 часа * 4 (15m bars)
    
    for i in range(len(df) - horizon_bars):
        current = df['close'].iloc[i]
        future = df['close'].iloc[i + horizon_bars]
        
        change = (future - current) / current
        
        # Binary: UP если рост >3%, иначе DOWN
        label = 1 if change > threshold else 0
        labels.append(label)
    
    # Pad
    labels += [None] * horizon_bars
    
    # Stats
    valid_labels = [l for l in labels if l is not None]
    up_count = valid_labels.count(1)
    down_count = valid_labels.count(0)
    
    print(f"\nLabel distribution:")
    print(f"  UP (1): {up_count} ({up_count/len(valid_labels)*100:.1f}%)")
    print(f"  DOWN (0): {down_count} ({down_count/len(valid_labels)*100:.1f}%)")
    
    return labels

if __name__ == "__main__":
    # Load BTC
    df = pd.read_parquet('data/raw/BTC_USDT_15m_FULL.parquet')
    print(f"Loaded {len(df)} bars")
    
    # Test different horizons
    for days in [3, 5, 7]:
        labels = create_swing_labels(df, horizon_days=days, threshold=0.03)
        
        # Save
        os.makedirs('data/processed', exist_ok=True)
        np.save(f'data/processed/swing_{days}d_labels.npy', labels)
        print(f"Saved: data/processed/swing_{days}d_labels.npy\n")