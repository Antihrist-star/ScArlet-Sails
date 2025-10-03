import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import torch
import joblib

def load_and_preprocess_data(data_dir="data/raw", sequence_length=60, target_column="close"):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet") and "15m" in filename:
            filepath = os.path.join(data_dir, filename)
            df = pd.read_parquet(filepath)
            all_data.append(df)

    if not all_data:
        raise ValueError("No 15m parquet files found in data/raw.")

    combined_df = pd.concat(all_data).sort_index()
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

    features_list = ["open", "high", "low", "close", "volume"]
    # Exclude target_column from features for scaler_X if it's present
    features_for_X = [f for f in features_list if f != target_column]

    # Prepare data for X and y
    X_data = combined_df[features_list].values # X will still contain all features for sequence creation
    y_data = combined_df[target_column].values.reshape(-1, 1)

    # Scaler for all features (X)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaled_X_data = scaler_X.fit_transform(X_data)

    # Scaler for target (y)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_y_data = scaler_y.fit_transform(y_data)

    X, y = [], []
    for i in range(len(scaled_X_data) - sequence_length):
        X.append(scaled_X_data[i:i+sequence_length])
        y.append(scaled_y_data[i+sequence_length]) # Target is already scaled

    X = np.array(X)
    y = np.array(y)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).squeeze(1) # Remove extra dimension for target

    # Save scalers for later use
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    return X, y, scaler_X, scaler_y

if __name__ == "__main__":
    X, y, scaler_X, scaler_y = load_and_preprocess_data()
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print("Data preparation complete.")


