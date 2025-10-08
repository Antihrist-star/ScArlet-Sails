import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json

class CNN1D(nn.Module):
    def __init__(self, input_features, sequence_length=60):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15, 64)  # 60->30->15 после 2 pooling
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, timesteps, features)
        x = x.transpose(1, 2)  # -> (batch, features, timesteps)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    import sys
    sys.path.append('scripts')
    from prepare_data_v3 import load_and_preprocess_data

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

    input_features = X_train.shape[2]
    model = CNN1D(input_features, sequence_length=60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Conservative pos_weight for better precision
    pos_weight = torch.tensor([3.0]).to(device)
    print(f"Using pos_weight: {pos_weight.item():.1f} (conservative approach)")
    print("Goal: Precision > 40%, fewer false signals")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    print(f"Using device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {input_features}")

    # Training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("\nTraining with PROFITABLE TARGET (conservative strategy)...")
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/30], Loss: {avg_loss:.4f}")

    # Test evaluation with multiple thresholds
    print("\n=== EVALUATION WITH DIFFERENT THRESHOLDS ===")
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        probabilities = torch.sigmoid(outputs)
        
        # Test different thresholds
        thresholds = [0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).float()
            
            tp = ((predictions.squeeze() == 1) & (y_test == 1)).sum().item()
            fp = ((predictions.squeeze() == 1) & (y_test == 0)).sum().item()
            tn = ((predictions.squeeze() == 0) & (y_test == 0)).sum().item()
            fn = ((predictions.squeeze() == 0) & (y_test == 1)).sum().item()
            
            accuracy = (predictions.squeeze() == y_test).float().mean()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n--- Threshold: {threshold} ---")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"Recall: {recall:.4f} ({recall*100:.1f}%)")
            print(f"F1: {f1:.4f}")
            print(f"Predicted Positives: {(predictions.squeeze() == 1).sum().item()} ({(predictions.squeeze() == 1).float().mean()*100:.1f}%)")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    # Final evaluation with best threshold
    print(f"\n=== FINAL RESULTS (Best Threshold: {best_threshold}) ===")
    predictions = (probabilities > best_threshold).float()
    accuracy = (predictions.squeeze() == y_test).float().mean()
    
    tp = ((predictions.squeeze() == 1) & (y_test == 1)).sum().item()
    fp = ((predictions.squeeze() == 1) & (y_test == 0)).sum().item()
    tn = ((predictions.squeeze() == 0) & (y_test == 0)).sum().item()
    fn = ((predictions.squeeze() == 0) & (y_test == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Baseline comparison
    naive_acc = (y_test == 0).float().mean()
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Naive Baseline (always 0): {naive_acc:.4f} ({naive_acc*100:.2f}%)")
    print(f"Improvement over baseline: {(accuracy - naive_acc):.4f} ({(accuracy - naive_acc)*100:.2f}%)")
    
    print(f"\n📈 TRADING METRICS:")
    print(f"Precision: {precision:.4f} (When we predict BUY, {precision*100:.1f}% are profitable)")
    print(f"Recall: {recall:.4f} (We catch {recall*100:.1f}% of all profitable trades)")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"True Positives:  {tp:5d} (Correct BUY signals)")
    print(f"False Positives: {fp:5d} (Wrong BUY signals - LOSSES)")
    print(f"True Negatives:  {tn:5d} (Correct NO-TRADE)")
    print(f"False Negatives: {fn:5d} (Missed profitable trades)")
    
    print(f"\n📊 SIGNAL DISTRIBUTION:")
    print(f"Actual Profitable: {(y_test == 1).sum().item()} ({(y_test == 1).float().mean()*100:.2f}%)")
    print(f"Model Predicts: {(predictions.squeeze() == 1).sum().item()} ({(predictions.squeeze() == 1).float().mean()*100:.2f}%)")
    
    # Trading simulation
    print(f"\n💰 EXPECTED PROFITABILITY:")
    if tp + fp > 0:
        win_rate = tp / (tp + fp)
        avg_win = 0.008  # 0.8% profit target
        avg_loss = 0.004  # 0.4% stop loss
        commission = 0.002  # 0.2% round trip
        
        expected_value = win_rate * (avg_win - commission) - (1 - win_rate) * (avg_loss + commission)
        
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Avg Win: {avg_win*100:.2f}% (before commission)")
        print(f"Avg Loss: {avg_loss*100:.2f}% (before commission)")
        print(f"Commission: {commission*100:.2f}% per trade")
        print(f"Expected Value per Trade: {expected_value*100:.4f}%")
        
        if expected_value > 0:
            print("✅ PROFITABLE STRATEGY!")
            annual_trades = 252 * 4  # ~4 trades per day
            predicted_trades = (predictions.squeeze() == 1).sum().item()
            trade_frequency = predicted_trades / len(y_test)
            actual_annual_trades = annual_trades * trade_frequency
            annual_return = expected_value * actual_annual_trades
            print(f"Estimated Annual Return: {annual_return*100:.1f}% ({actual_annual_trades:.0f} trades/year)")
        else:
            print("❌ NOT PROFITABLE with current performance")
            print(f"Need Win Rate > {(avg_loss + commission)/(avg_win + avg_loss)*100:.1f}% to be profitable")

    # Save model and metrics
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/profitable_cnn_v3.pth")
    
    metrics = {
        'best_threshold': float(best_threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'win_rate': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'expected_value': float(expected_value) if tp + fp > 0 else 0,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }
    
    os.makedirs("reports", exist_ok=True)
    with open('reports/day4_final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Model saved to models/profitable_cnn_v3.pth")
    print(f"📊 Metrics saved to reports/day4_final_metrics.json")
    print(f"🎯 Best threshold for trading: {best_threshold}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)