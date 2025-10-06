# Архитектура системы - День 1

## Данные ✅
- Источник: Binance (ccxt)
- Символы: BTC/USDT, ETH/USDT, SOL/USDT
- Таймфреймы: 1m, 15m, 1h  
- Период: 2 года (2023-10-04 → 2025-10-03)
- Качество: Идеальное (0 NaN, 0 дубликатов)

## Pipeline (prepare_data.py) ⚠️ ПРОБЛЕМЫ
- Загружает только 15m 
- 5 features: open,high,low,close,volume
- Sequence=60, target=close_price regression

## НУЖНЫЕ ИСПРАВЛЕНИЯ
- Day 2: +25 технических индикаторов  
- Day 3: binary classification target
- Day 4: sequence=100, все таймфреймы

## Model (train_model.py) ⚠️ ПРОБЛЕМЫ  
- Архитектура: Conv1D правильная
- НО: MSELoss вместо BCE
- НО: sigmoid+regression вместо classification

## Инфраструктура ✅
- MSI RTX 3050: CUDA работает
- GitHub: готов к commit