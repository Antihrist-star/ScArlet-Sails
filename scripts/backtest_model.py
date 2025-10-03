import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import json
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Define the 1D-CNN model (copied from train_model.py to avoid circular imports for now)
class CNN1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # Calculate the size of the flattened layer dynamically
        self.fc_input_size = self._get_conv_output_size(input_channels)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, output_size)
        self.sigmoid = nn.Sigmoid()

    def _get_conv_output_size(self, input_channels):
        # Create a dummy input to calculate the output size of conv layers
        dummy_input = torch.randn(1, input_channels, 60) # batch_size, channels, sequence_length
        x = self.conv1(dummy_input)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.flatten().size(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

def load_model(model_path="models/1d_cnn_model.pth", input_channels=5, output_size=1):
    model = CNN1D(input_channels, output_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

import joblib

def load_scalers(model_dir="models"):
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    return scaler_X, scaler_y

def run_backtest(model, scaler_X, scaler_y, data_dir="data/raw", sequence_length=60, initial_capital=10000, target_column="close"):

    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)

    if not all_data:
        raise ValueError("No 15m parquet files found in data/raw for backtesting.")

    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

    features = ["open", "high", "low", "close", "volume"]
    data_to_scale = combined_df[features].values
    scaled_data = scaler_X.transform(data_to_scale)


    predictions = []
    actual_prices = []
    portfolio_value = [initial_capital]
    position = 0 # 0 for no position, 1 for long
    buy_price = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i in range(len(scaled_data) - sequence_length):
        current_sequence = scaled_data[i:i+sequence_length]
        current_price = combined_df[target_column].iloc[i+sequence_length]

        input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)

        with torch.no_grad():
            predicted_scaled_price = model(input_tensor).item()

        # Inverse transform the single predicted scaled price using scaler_y
        predicted_price = scaler_y.inverse_transform(np.array(predicted_scaled_price).reshape(-1, 1))[0, 0]

        predictions.append(predicted_price)
        actual_prices.append(current_price)

        # Debugging prints
        print(f"Time: {combined_df.index[i+sequence_length]}, Current Price: {current_price:.2f}, Predicted Price: {predicted_price:.2f}, Position: {position}")

        if predicted_price > current_price and position == 0: # Buy signal if predicted price is higher
            position = 1
            buy_price = current_price
            print(f"BUY at {current_price:.2f}")
        elif predicted_price < current_price and position == 1: # Sell signal if predicted price is lower
            profit = (current_price - buy_price) / buy_price
            portfolio_value.append(portfolio_value[-1] * (1 + profit))
            position = 0
            buy_price = 0
            print(f"SELL at {current_price:.2f}, Profit: {profit*100:.2f}%")
        
        if position == 1:
            current_profit = (current_price - buy_price) / buy_price
            current_portfolio_value = portfolio_value[-2] * (1 + current_profit) if len(portfolio_value) > 1 else initial_capital * (1 + current_profit)
            portfolio_value[-1] = current_portfolio_value
        elif position == 0 and len(portfolio_value) > 1:
            portfolio_value.append(portfolio_value[-1])
        elif position == 0 and len(portfolio_value) == 1 and i > 0:
            portfolio_value.append(initial_capital)

    if position == 1:
        final_price = combined_df[target_column].iloc[-1]
        profit = (final_price - buy_price) / buy_price
        portfolio_value.append(portfolio_value[-1] * (1 + profit))
        position = 0

    returns = pd.Series(portfolio_value).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 4) # Assuming 15m data, 24*4 = 96 periods per day
    total_pnl = portfolio_value[-1] - initial_capital

    results = {
        "initial_capital": initial_capital,
        "final_portfolio_value": portfolio_value[-1],
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe_ratio,
        "num_trades": (len(portfolio_value) - 1) / 2 if len(portfolio_value) > 1 else 0,
        "predictions": predictions,
        "actual_prices": actual_prices,
        "portfolio_history": portfolio_value
    }

    return results

if __name__ == "__main__":
    model = load_model()
    scaler_X, scaler_y = load_scalers()

    backtest_results = run_backtest(model, scaler_X, scaler_y)


    print("Backtest Results:")
    print(f"Initial Capital: {backtest_results['initial_capital']:.2f}")
    print(f"Final Portfolio Value: {backtest_results['final_portfolio_value']:.2f}")
    print(f"Total PnL: {backtest_results['total_pnl']:.2f}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Number of Trades: {backtest_results['num_trades']}")

    results_path = "reports/backtest_results.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        results_for_json = {
            k: (v.tolist() if isinstance(v, (np.ndarray, torch.Tensor)) else v)
            for k, v in backtest_results.items()
        }
        json.dump(results_for_json, f, indent=4)
    print(f"Backtest results saved to {results_path}")

