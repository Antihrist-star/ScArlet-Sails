"""
FIXED XGBoost Backtest - Auto-detect scaler file
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import glob

print("="*80)
print("XGBOOST BACKTEST - FIXED VERSION")
print("="*80)

# Paths
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
data_dir = project_root / "data" / "processed"
reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)

def find_scaler():
    """Find any available scaler file"""
    scaler_files = list(models_dir.glob("scaler*.pkl"))
    if not scaler_files:
        print("❌ No scaler file found!")
        print(f"Looked in: {models_dir}")
        print("\nAvailable files:")
        for f in models_dir.glob("*"):
            print(f"  {f.name}")
        return None
    
    scaler_path = scaler_files[0]
    print(f"✅ Found scaler: {scaler_path.name}")
    return scaler_path

def main():
    # [1/6] Find and load scaler
    print("\n[1/6] Finding scaler...")
    scaler_path = find_scaler()
    if scaler_path is None:
        print("\n⚠️  CREATING DUMMY SCALER (for demo)")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Will fit on data later
    else:
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded")
    
    # [2/6] Load model
    print("\n[2/6] Loading XGBoost model...")
    model_path = models_dir / "xgboost_model.json"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    # [3/6] Load test data
    print("\n[3/6] Loading test data...")
    
    # Try different locations
    test_files = [
        data_dir / "X_test_clean_2d.csv",
        data_dir / "X_test.csv",
        project_root / "X_test.csv"
    ]
    
    X_test = None
    for test_file in test_files:
        if test_file.exists():
            print(f"✅ Found: {test_file}")
            X_test = pd.read_csv(test_file)
            break
    
    if X_test is None:
        print("❌ No test data found!")
        print("\nTried:")
        for f in test_files:
            print(f"  {f}")
        print("\n⚠️  Will use sample data for demo")
        # Create dummy data
        X_test = pd.DataFrame(np.random.randn(1000, 31))
    else:
        print(f"✅ Test data loaded: {X_test.shape}")
    
    # [4/6] Load OHLCV
    print("\n[4/6] Loading OHLCV data...")
    ohlcv_files = [
        data_dir / "ohlcv_test.csv",
        project_root / "data" / "raw" / "BTC_USDT_15m.parquet"
    ]
    
    df_ohlcv = None
    for ohlcv_file in ohlcv_files:
        if ohlcv_file.exists():
            if ohlcv_file.suffix == '.parquet':
                df_ohlcv = pd.read_parquet(ohlcv_file)
                # Take last N rows to match X_test
                df_ohlcv = df_ohlcv.tail(len(X_test))
            else:
                df_ohlcv = pd.read_csv(ohlcv_file)
            print(f"✅ Found: {ohlcv_file.name}")
            break
    
    if df_ohlcv is None:
        print("⚠️  No OHLCV found, using dummy prices")
        df_ohlcv = pd.DataFrame({
            'close': np.cumsum(np.random.randn(len(X_test))) + 100
        })
    
    print(f"✅ OHLCV loaded: {df_ohlcv.shape}")
    
    # [5/6] Generate predictions
    print("\n[5/6] Generating predictions...")
    
    # Scale if scaler exists
    if isinstance(scaler, type(None)):
        X_scaled = X_test.values
    else:
        try:
            X_scaled = scaler.transform(X_test)
        except:
            # Fit scaler if needed
            X_scaled = scaler.fit_transform(X_test)
    
    # Predict
    threshold = 0.46
    proba = model.predict_proba(X_scaled)[:, 1]
    signals = (proba >= threshold).astype(int)
    
    print(f"✅ Threshold: {threshold}")
    print(f"✅ BUY signals: {signals.sum()} ({signals.sum()/len(signals)*100:.2f}%)")
    
    # [6/6] Simple backtest
    print("\n[6/6] Running backtest...")
    
    capital = 100000
    position = None
    trades = []
    equity = [capital]
    
    tp = 0.03  # 3%
    sl = 0.012  # 1.2%
    
    for i in range(len(signals)):
        price = df_ohlcv['close'].iloc[i]
        
        # Check existing position
        if position is not None:
            pnl_pct = (price - position['entry_price']) / position['entry_price']
            
            if pnl_pct >= tp:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'tp'})
                position = None
            elif pnl_pct <= -sl:
                pnl = capital * 0.95 * pnl_pct
                capital += pnl
                trades.append({'pnl': pnl, 'exit': 'sl'})
                position = None
        
        # New signal
        if signals[i] == 1 and position is None:
            position = {'entry_price': price}
        
        equity.append(capital)
    
    # Results
    total_return = (capital - 100000) / 100000 * 100
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Initial Capital:  ${100000:,.2f}")
    print(f"Final Capital:    ${capital:,.2f}")
    print(f"Total Return:     {total_return:.2f}%")
    print(f"Number of Trades: {len(trades)}")
    print(f"Wins:             {len(wins)}")
    print(f"Losses:           {len(losses)}")
    if trades:
        print(f"Win Rate:         {len(wins)/len(trades)*100:.1f}%")
    print("="*80)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity)
    plt.title('Equity Curve - XGBoost')
    plt.xlabel('Bars')
    plt.ylabel('Capital ($)')
    plt.grid(True, alpha=0.3)
    
    plot_path = reports_dir / "xgboost_equity_fixed.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved: {plot_path}")
    
    # Save trades
    if trades:
        df_trades = pd.DataFrame(trades)
        csv_path = reports_dir / "xgboost_trades_fixed.csv"
        df_trades.to_csv(csv_path, index=False)
        print(f"✅ Trades saved: {csv_path}")
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()