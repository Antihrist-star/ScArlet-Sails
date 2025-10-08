import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import json
import os
from prepare_swing_data import load_and_prepare_swing_data

class SwingCNN(nn.Module):
    """Simplified CNN - лучшая архитектура из v4"""
    def __init__(self, input_features):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15, 64)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 2)  # Binary: UP/DOWN
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

def train_swing_model(horizon_days=3):
    print(f"\n{'='*60}")
    print(f"TRAINING SWING MODEL - {horizon_days} DAYS")
    print(f"{'='*60}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_swing_data(horizon_days)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model
    model = SwingCNN(input_features=X_train.shape[2])
    model.to(device)
    
    # Balanced loss
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Data
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # OneCycleLR
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=30,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    # Training
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/30], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        
        accuracy = (predicted == y_test).float().mean()
        
        # Win rate (UP predictions)
        up_mask = (predicted == 1)
        if up_mask.sum() > 0:
            up_correct = ((predicted == 1) & (y_test == 1)).sum()
            win_rate = up_correct.float() / up_mask.sum().float()
        else:
            win_rate = 0
        
        # Expected value
        if up_mask.sum() > 0:
            avg_win = 0.03  # 3% target
            avg_loss = 0.015  # ~1.5% average loss
            commission = 0.002
            
            exp_value = win_rate * (avg_win - commission) - (1 - win_rate) * (avg_loss + commission)
        else:
            exp_value = 0
        
        print(f"\n{'='*60}")
        print(f"RESULTS - {horizon_days} DAYS")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Expected Value: {exp_value*100:.3f}%")
        
        if win_rate >= 0.52:
            print("✅ SUCCESS - PROFITABLE!")
        elif win_rate >= 0.48:
            print("⚠️ MARGINAL - Need optimization")
        else:
            print("❌ NOT PROFITABLE")
        
        # Save
        torch.save(model.state_dict(), f'models/swing_{horizon_days}d.pth')
        
        metrics = {
            'horizon_days': horizon_days,
            'accuracy': float(accuracy),
            'win_rate': float(win_rate),
            'expected_value': float(exp_value)
        }
        
        os.makedirs('reports', exist_ok=True)
        with open(f'reports/swing_{horizon_days}d_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    results = {}
    
    for days in [3, 5, 7]:
        metrics = train_swing_model(days)
        results[f'{days}d'] = metrics
        print("\n")
    
    # Summary
    print(f"\n{'='*60}")
    print("SWING TRADING SUMMARY")
    print(f"{'='*60}")
    
    for horizon, metrics in results.items():
        print(f"\n{horizon}:")
        print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  Expected Value: {metrics['expected_value']*100:.3f}%")
    
    # Best
    best = max(results.items(), key=lambda x: x[1]['win_rate'])
    print(f"\n🏆 BEST: {best[0]} with {best[1]['win_rate']*100:.1f}% Win Rate")