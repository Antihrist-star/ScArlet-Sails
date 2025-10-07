import torch
import numpy as np
from prepare_data import load_and_preprocess_data
from train_model import CNN1D
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
from datetime import datetime

def walk_forward_validation():
    """
    Проверяет модель на разных временных периодах
    чтобы выявить overfitting на bull market
    """
    print("="*60)
    print("WALK-FORWARD VALIDATION")
    print("="*60)
    
    # Загрузить все данные
    X_train_full, X_test, y_train_full, y_test, scaler = load_and_preprocess_data()
    
    # Объединить для walk-forward
    X_all = torch.cat([X_train_full, X_test])
    y_all = torch.cat([y_train_full, y_test])
    
    total_samples = len(X_all)
    up_ratio_total = (y_all==1).sum().item() / total_samples
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Overall UP ratio: {up_ratio_total:.1%}")
    print("-"*60)
    
    # Walk-forward параметры
    n_splits = 7  # Количество периодов
    train_size = 5000  # Размер обучающей выборки
    test_size = 1500   # Размер тестовой выборки
    step_size = 2000   # Шаг сдвига окна
    
    results = []
    
    for i in range(n_splits):
        print(f"\n{'='*20} PERIOD {i+1} {'='*20}")
        
        start_idx = i * step_size
        train_end = start_idx + train_size
        test_end = train_end + test_size
        
        # Проверка границ
        if test_end > total_samples:
            print(f"Skipping period {i+1}: not enough data")
            break
        
        # Подготовка данных
        X_train = X_all[start_idx:train_end]
        y_train = y_all[start_idx:train_end]
        X_test_fold = X_all[train_end:test_end]
        y_test_fold = y_all[train_end:test_end]
        
        # Статистика периода
        train_up_ratio = (y_train == 1).float().mean().item()
        test_up_ratio = (y_test_fold == 1).float().mean().item()
        
        print(f"Train range: [{start_idx:,} : {train_end:,}] - UP ratio: {train_up_ratio:.1%}")
        print(f"Test range:  [{train_end:,} : {test_end:,}] - UP ratio: {test_up_ratio:.1%}")
        
        # Быстрое обучение модели
        print("Training model (10 epochs)...")
        model = CNN1D(12, 60)  # 12 features, 60 timesteps
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # DataLoader
        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1).float())
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Обучение (быстрое - 10 эпох)
        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch == 9:  # Последняя эпоха
                avg_loss = total_loss / len(train_loader)
                print(f"Final training loss: {avg_loss:.4f}")
        
        # Тестирование
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_fold)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            # Метрики
            accuracy = (predictions.squeeze() == y_test_fold).float().mean().item()
            
            # Confusion matrix
            tp = ((predictions.squeeze() == 1) & (y_test_fold == 1)).sum().item()
            tn = ((predictions.squeeze() == 0) & (y_test_fold == 0)).sum().item()
            fp = ((predictions.squeeze() == 1) & (y_test_fold == 0)).sum().item()
            fn = ((predictions.squeeze() == 0) & (y_test_fold == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Baseline accuracy (всегда предсказывать majority class)
            baseline_acc = max(test_up_ratio, 1 - test_up_ratio)
        
        print(f"\n📊 RESULTS:")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Baseline (majority class): {baseline_acc:.2%}")
        print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
        print(f"Improvement over baseline: {(accuracy - baseline_acc):.2%}")
        
        # Сохранить результаты
        results.append({
            'period': i + 1,
            'train_range': f"[{start_idx}:{train_end}]",
            'test_range': f"[{train_end}:{test_end}]",
            'train_up_ratio': train_up_ratio,
            'test_up_ratio': test_up_ratio,
            'accuracy': accuracy,
            'baseline_accuracy': baseline_acc,
            'precision': precision,
            'recall': recall,
            'improvement': accuracy - baseline_acc
        })
    
    # Итоговая статистика
    print("\n" + "="*60)
    print("WALK-FORWARD SUMMARY")
    print("="*60)
    
    accuracies = [r['accuracy'] for r in results]
    improvements = [r['improvement'] for r in results]
    
    print(f"\nAccuracy across {len(results)} periods:")
    print(f"  Mean: {np.mean(accuracies):.2%}")
    print(f"  Std:  {np.std(accuracies):.2%}")
    print(f"  Min:  {min(accuracies):.2%}")
    print(f"  Max:  {max(accuracies):.2%}")
    print(f"  Range: {max(accuracies) - min(accuracies):.2%}")
    
    print(f"\nImprovement over baseline:")
    print(f"  Mean: {np.mean(improvements):.2%}")
    print(f"  Std:  {np.std(improvements):.2%}")
    
    # Вывод о результатах
    accuracy_range = max(accuracies) - min(accuracies)
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    
    if accuracy_range > 0.20:
        print("⚠️ HIGH VARIANCE DETECTED (>20%)")
        print("Model performance varies significantly across periods.")
        print("This confirms overfitting to specific market conditions.")
        print("\n🔴 RECOMMENDATION: Need more robust target and features")
    elif accuracy_range > 0.10:
        print("⚠️ MODERATE VARIANCE (10-20%)")
        print("Model shows some instability across periods.")
        print("\n🟡 RECOMMENDATION: Consider ensemble methods")
    else:
        print("✅ LOW VARIANCE (<10%)")
        print("Model is relatively stable across periods.")
        print("\n🟢 RECOMMENDATION: Can proceed with optimization")
    
    # Проверка на bull market bias
    high_up_periods = [r for r in results if r['test_up_ratio'] > 0.55]
    if len(high_up_periods) > len(results) / 2:
        print("\n⚠️ BULL MARKET BIAS DETECTED")
        print(f"{len(high_up_periods)}/{len(results)} periods have >55% UP movements")
        print("Model may not generalize to bear/sideways markets")
    
    # Сохранить результаты
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'reports/walk_forward_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'summary': {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'min_accuracy': float(min(accuracies)),
                'max_accuracy': float(max(accuracies)),
                'range': float(accuracy_range)
            },
            'periods': results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    try:
        results = walk_forward_validation()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nПроверьте что:")
        print("1. Файл prepare_data.py содержит функцию load_and_preprocess_data()")
        print("2. Файл train_model.py содержит класс CNN1D")
        print("3. Данные подготовлены (X_train.pt, y_train.pt существуют)")