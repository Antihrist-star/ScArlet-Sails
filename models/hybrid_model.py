"""
CNN-LSTM Hybrid для Scarlet Sails Phoenix v2
Автор: Сотрудник 2
Создан: Day 9

Архитектура:
- CNN: локальные паттерны (candlestick formations)
- LSTM: temporal dependencies (trend evolution)
- Output: Binary classification (UP/DOWN)
"""

import torch
import torch.nn as nn


class CNNLSTMHybrid(nn.Module):
    """
    Hybrid model для swing trading (3-7 дней).
    
    Parameters:
        input_features (int): Количество features (default: 38)
        seq_len (int): Длина sequence (default: 240)
        cnn_channels (list): CNN output channels (default: [64, 32])
        lstm_hidden (int): LSTM hidden size (default: 64)
        lstm_layers (int): Количество LSTM layers (default: 2)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(
        self,
        input_features=38,
        seq_len=240,
        cnn_channels=[64, 32],
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3
    ):
        super(CNNLSTMHybrid, self).__init__()
        
        self.input_features = input_features
        self.seq_len = seq_len
        
        # CNN Block для feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_features,
                out_channels=cnn_channels[0],
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(
                in_channels=cnn_channels[0],
                out_channels=cnn_channels[1],
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM Block для temporal dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output Block
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
        
        Returns:
            output: Logits (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # CNN expects: (batch, channels, length)
        x = x.transpose(1, 2)
        
        # CNN forward
        x = self.cnn(x)
        
        # LSTM expects: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use последний hidden state
        final_hidden = hidden[-1]
        
        # Output layer
        output = self.fc(final_hidden)
        
        return output
    
    def get_model_info(self):
        """
        Получить информацию о модели.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'input_shape': f"(batch, {self.seq_len}, {self.input_features})",
            'output_shape': "(batch, 2)"
        }
        
        return info


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-LSTM Hybrid Model Test")
    print("=" * 60)
    
    # Create model
    model = CNNLSTMHybrid(
        input_features=38,
        seq_len=240,
        cnn_channels=[64, 32],
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.3
    )
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass (CPU)
    print("\n" + "-" * 60)
    print("Testing forward pass (CPU)...")
    
    batch_size = 16
    x = torch.randn(batch_size, 240, 38)
    
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output sample:\n{output[:2]}")
    
    # Check GPU if available
    if torch.cuda.is_available():
        print("\n" + "-" * 60)
        print("Testing forward pass (GPU)...")
        
        model = model.cuda()
        x = x.cuda()
        
        output = model(x)
        
        print("GPU forward pass successful!")
        print(f"Output shape: {output.shape}")
    else:
        print("\nCUDA not available, skipping GPU test")
    
    # Test with different batch sizes
    print("\n" + "-" * 60)
    print("Testing different batch sizes...")
    
    model = model.cpu()
    
    for bs in [1, 8, 32, 64]:
        x = torch.randn(bs, 240, 38)
        output = model(x)
        print(f"  Batch size {bs:2d}: {output.shape} OK")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)