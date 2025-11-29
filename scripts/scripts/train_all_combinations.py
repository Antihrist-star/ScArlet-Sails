"""
Массовое обучение XGBoost v3 на всех 56 комбинациях
====================================================

Читает метрики из сохраненных metadata.json файлов
"""

import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

# Конфигурация
COINS = [
    "ALGO", "AVAX", "BTC", "DOT", "ENA", "ETH", "HBAR",
    "LDO", "LINK", "LTC", "ONDO", "SOL", "SUI", "UNI"
]

TIMEFRAMES = ["15m", "1h", "4h", "1d"]

DATA_DIR = Path("data/features")
MODELS_DIR = Path("models")
RESULTS_FILE = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

def check_data_exists(coin, tf):
    """Проверить что файл данных существует"""
    file_path = DATA_DIR / f"{coin}_USDT_{tf}_features.parquet"
    return file_path.exists()

def read_metadata(coin, tf):
    """
    Прочитать метрики из metadata.json
    
    Returns:
        dict: Метрики или None если файл не найден
    """
    metadata_path = MODELS_DIR / f"xgboost_v3_{coin.lower()}_{tf}_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Извлечь метрики
        metrics = metadata.get('metrics', {})
        optimal = metadata.get('optimal_threshold', {})
        
        return {
            'auc': metrics.get('auc'),
            'f1': optimal.get('best_f1'),
            'precision': optimal.get('precision_at_best'),
            'recall': optimal.get('recall_at_best'),
            'samples': None  # Можно добавить в metadata если нужно
        }
    except Exception as e:
        print(f"   ⚠️ Ошибка чтения metadata: {e}")
        return None

def train_combination(coin, tf):
    """
    Обучить модель для одной комбинации
    
    Returns:
        dict: Результаты обучения
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
        
        # Попытка 1: Прочитать из metadata файла
        metrics_dict = read_metadata(coin, tf)
        
        if metrics_dict and metrics_dict['auc'] is not None:
            status = '✅ SUCCESS'
            auc_str = f"{metrics_dict['auc']:.4f}"
            f1_str = f"{metrics_dict['f1']:.4f}" if metrics_dict['f1'] is not None else "N/A"
            print(f"✅ Завершено: AUC={auc_str}, F1={f1_str}")
            
            return {
                'coin': coin,
                'timeframe': tf,
                'status': status,
                **metrics_dict
            }
        else:
            # Модель не обучилась или metadata не создан
            status = '❌ TRAIN_FAILED'
            print(f"❌ Обучение не удалось")
            
            # Диагностика
            if result.stderr:
                print(f"STDERR (первые 500 символов):\n{result.stderr[:500]}")
            if result.returncode != 0:
                print(f"Return code: {result.returncode}")
            
            return {
                'coin': coin,
                'timeframe': tf,
                'status': status,
                'auc': None,
                'f1': None,
                'precision': None,
                'recall': None,
                'samples': None
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
        import traceback
        traceback.print_exc()
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
    print(f"Метод: Чтение из metadata.json файлов")
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
    else:
        print("  ⚠️ Нет успешных обучений")
        print("\n🔍 ПЕРВЫЕ 5 РЕЗУЛЬТАТОВ:")
        print(results_df.head()[['coin', 'timeframe', 'status']])

if __name__ == "__main__":
    main()
