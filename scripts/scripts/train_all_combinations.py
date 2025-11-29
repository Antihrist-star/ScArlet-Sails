"""
Массовое обучение XGBoost v3 на всех 56 комбинациях
====================================================

Обучает модель для каждой пары (монета × таймфрейм)
Сохраняет результаты в CSV для анализа

Использование:
    python scripts/train_all_combinations.py
    
Время выполнения: ~3-4 часа (56 × 3-5 минут)
"""

import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

# Конфигурация
COINS = [
    "ALGO", "AVAX", "BTC", "DOT", "ENA", "ETH", "HBAR",
    "LDO", "LINK", "LTC", "ONDO", "SOL", "SUI", "UNI"
]

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

DATA_DIR = Path("data/features")
RESULTS_FILE = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def check_data_exists(coin, tf):
    """Проверить что файл данных существует"""
    file_path = DATA_DIR / f"{coin}_USDT_{tf}_features.parquet"
    return file_path.exists()

def train_combination(coin, tf):
    """
    Обучить модель для одной комбинации
    
    Returns:
        dict: Результаты обучения или None если ошибка
    """
    print(f"\n{'='*80}")
    print(f"🚀 ОБУЧЕНИЕ: {coin}/{tf}")
    print(f"{'='*80}")
    
    # Проверить данные
    if not check_data_exists(coin, tf):
        print(f"⚠️ SKIP: Файл данных не найден")
        return {
            'coin': coin,
            'timeframe': tf,
            'status': 'NO_DATA',
            'auc': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'samples': None
        }
    
    # Запустить обучение
    cmd = [
        'python', 'scripts/train_xgboost_v3.py',
        '--coin', coin,
        '--tf', tf
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 минут timeout
        )
        
        # Парсить вывод для извлечения метрик
        output = result.stdout
        
        # Извлечь метрики (примерный парсинг)
        auc = None
        f1 = None
        precision = None
        recall = None
        samples = None
        
        for line in output.split('\n'):
            if 'AUC:' in line:
                try:
                    auc = float(line.split('AUC:')[1].strip())
                except:
                    pass
            elif 'F1:' in line:
                try:
                    f1 = float(line.split('F1:')[1].strip())
                except:
                    pass
            elif 'Precision:' in line:
                try:
                    precision = float(line.split('Precision:')[1].strip())
                except:
                    pass
            elif 'Recall:' in line:
                try:
                    recall = float(line.split('Recall:')[1].strip())
                except:
                    pass
            elif 'Test:' in line and 'samples' in line:
                try:
                    samples = int(line.split()[1].replace(',', ''))
                except:
                    pass
        
        status = '✅ SUCCESS' if auc is not None else '⚠️ PARSE_ERROR'
        
        print(f"✅ Завершено: AUC={auc:.4f if auc else 'N/A'}, F1={f1:.4f if f1 else 'N/A'}")
        
        return {
            'coin': coin,
            'timeframe': tf,
            'status': status,
            'auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'samples': samples
        }
        
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: Обучение превысило 10 минут")
        return {
            'coin': coin,
            'timeframe': tf,
            'status': 'TIMEOUT',
            'auc': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'samples': None
        }
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            'coin': coin,
            'timeframe': tf,
            'status': f'ERROR: {str(e)[:50]}',
            'auc': None,
            'f1': None,
            'precision': None,
            'recall': None,
            'samples': None
        }

def main():
    """Основной цикл обучения"""
    
    print("\n" + "="*80)
    print("🚀 МАССОВОЕ ОБУЧЕНИЕ: 56 КОМБИНАЦИЙ")
    print("="*80)
    print(f"Монеты: {len(COINS)}")
    print(f"Таймфреймы: {len(TIMEFRAMES)}")
    print(f"Всего комбинаций: {len(COINS) * len(TIMEFRAMES)}")
    print(f"Результаты: {RESULTS_FILE}")
    print("="*80)
    
    results = []
    total = len(COINS) * len(TIMEFRAMES)
    processed = 0
    
    start_time = datetime.now()
    
    for coin in COINS:
        for tf in TIMEFRAMES:
            processed += 1
            
            print(f"\n[{processed}/{total}] {coin}/{tf}")
            
            result = train_combination(coin, tf)
            results.append(result)
            
            # Сохранять промежуточные результаты каждые 5 комбинаций
            if processed % 5 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(RESULTS_FILE.replace('.csv', '_temp.csv'), index=False)
                print(f"\n💾 Промежуточное сохранение: {processed}/{total}")
    
    # Финальное сохранение
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)
    
    # Удалить временный файл
    temp_file = RESULTS_FILE.replace('.csv', '_temp.csv')
    if Path(temp_file).exists():
        Path(temp_file).unlink()
    
    # Статистика
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*80)
    print("✅ ЗАВЕРШЕНО")
    print("="*80)
    print(f"Обработано: {processed}/{total}")
    print(f"Время: {elapsed}")
    print(f"Результаты: {RESULTS_FILE}")
    
    # Краткая статистика
    successful = results_df[results_df['status'] == '✅ SUCCESS']
    print(f"\n📊 СТАТИСТИКА:")
    print(f"  Успешно обучено: {len(successful)}/{total}")
    
    if len(successful) > 0:
        print(f"  Лучший AUC: {successful['auc'].max():.4f}")
        best_row = successful.loc[successful['auc'].idxmax()]
        print(f"  Лучшая пара: {best_row['coin']}/{best_row['timeframe']}")
        
        print(f"\n🏆 ТОП-5 ПО AUC:")
        top5 = successful.nlargest(5, 'auc')[['coin', 'timeframe', 'auc', 'f1']]
        print(top5.to_string(index=False))

if __name__ == "__main__":
    main()
