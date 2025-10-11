import sys
import os
# Добавляем корень проекта в PATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from models.hybrid_model import HybridModel

print("🔧 Creating HybridModel...")

# Создаём модель
model = HybridModel(
    input_features=31,
    sequence_length=60
)

print("✅ Model created successfully!")
print(f"📊 Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Тестовый forward pass
batch_size = 4
seq_len = 60
features = 31

x = torch.randn(batch_size, seq_len, features)
print(f"\n📥 Input tensor shape: {x.shape}")

# Отключаем градиенты для чистого теста
with torch.no_grad():
    output = model(x)

print(f"📤 Output tensor shape: {output.shape}")

# Проверка результата
if output.shape == (batch_size, 2):
    print("\n🎉 SUCCESS: Model forward pass works correctly!")
else:
    print(f"\n❌ ERROR: Expected output shape ({batch_size}, 2), but got {output.shape}")
    sys.exit(1)