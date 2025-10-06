#!/usr/bin/env python3
"""
Анализ торговых паттернов в криптовалютных данных
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PatternAnalyzer:
    def __init__(self):
        self.patterns_found = []
    
    def load_recent_data(self, symbol='BTC_USDT', timeframe='15m', days=30):
        """Загрузка данных за последние N дней"""
        filepath = f'data/raw/{symbol}_{timeframe}.parquet'
        df = pd.read_parquet(filepath)
        
        # Последние 30 дней
        end_date = df.index.max()
        start_date = end_date - timedelta(days=days)
        df_recent = df[df.index >= start_date].copy()
        
        return df_recent
    
    def is_hammer(self, row):
        """Определение паттерна Hammer"""
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        lower_shadow = row['open'] - row['low'] if row['close'] > row['open'] else row['close'] - row['low']
        upper_shadow = row['high'] - max(row['open'], row['close'])
        
        # Условия для Hammer
        if (lower_shadow > 2 * body_size and  # Длинная нижняя тень
            upper_shadow < body_size and      # Короткая верхняя тень
            total_range > 0):                 # Не doji
            return True
        return False
    
    def is_engulfing_bullish(self, prev_row, curr_row):
        """Определение Bullish Engulfing"""
        # Предыдущая красная, текущая зеленая
        prev_bearish = prev_row['close'] < prev_row['open']
        curr_bullish = curr_row['close'] > curr_row['open']
        
        # Текущая поглощает предыдущую
        engulfs = (curr_row['open'] < prev_row['close'] and 
                   curr_row['close'] > prev_row['open'])
        
        return prev_bearish and curr_bullish and engulfs
    
    def is_doji(self, row):
        """Определение Doji"""
        body_size = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        # Тело меньше 10% от всего диапазона
        if total_range > 0 and body_size / total_range < 0.1:
            return True
        return False
    
    def check_future_movement(self, df, index, hours=4):
        """Проверка движения цены через N часов"""
        try:
            current_price = df.loc[index, 'close']
            future_index = index + pd.Timedelta(hours=hours)
            
            # Найти ближайшую будущую цену
            future_data = df[df.index >= future_index]
            if len(future_data) > 0:
                future_price = future_data.iloc[0]['close']
                return_pct = (future_price - current_price) / current_price * 100
                return return_pct
        except:
            return None
        return None
    
    def analyze_symbol(self, symbol, timeframe='15m'):
        """Анализ паттернов для одного символа"""
        print(f"\nАнализ {symbol} {timeframe}...")
        
        df = self.load_recent_data(symbol, timeframe, days=30)
        patterns_count = {
            'hammer': 0,
            'bullish_engulfing': 0,
            'doji': 0
        }
        
        successful_patterns = {
            'hammer': 0,
            'bullish_engulfing': 0,
            'doji': 0
        }
        
        for i in range(1, len(df)):
            current_idx = df.index[i]
            current_row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Проверка Hammer
            if self.is_hammer(current_row):
                patterns_count['hammer'] += 1
                future_return = self.check_future_movement(df, current_idx, hours=4)
                if future_return and future_return > 0.5:  # Рост >0.5%
                    successful_patterns['hammer'] += 1
                
                self.patterns_found.append({
                    'symbol': symbol,
                    'pattern': 'Hammer',
                    'datetime': current_idx,
                    'price': current_row['close'],
                    'future_return': future_return,
                    'successful': future_return > 0.5 if future_return else False
                })
            
            # Проверка Bullish Engulfing
            if self.is_engulfing_bullish(prev_row, current_row):
                patterns_count['bullish_engulfing'] += 1
                future_return = self.check_future_movement(df, current_idx, hours=4)
                if future_return and future_return > 0.5:
                    successful_patterns['bullish_engulfing'] += 1
                
                self.patterns_found.append({
                    'symbol': symbol,
                    'pattern': 'Bullish Engulfing',
                    'datetime': current_idx,
                    'price': current_row['close'],
                    'future_return': future_return,
                    'successful': future_return > 0.5 if future_return else False
                })
            
            # Проверка Doji
            if self.is_doji(current_row):
                patterns_count['doji'] += 1
                future_return = self.check_future_movement(df, current_idx, hours=4)
                # Doji может предсказывать любое направление
                if future_return and abs(future_return) > 0.5:
                    successful_patterns['doji'] += 1
                
                self.patterns_found.append({
                    'symbol': symbol,
                    'pattern': 'Doji',
                    'datetime': current_idx,
                    'price': current_row['close'],
                    'future_return': future_return,
                    'successful': abs(future_return) > 0.5 if future_return else False
                })
        
        # Статистика
        print(f"Найдено паттернов:")
        for pattern, count in patterns_count.items():
            success_rate = (successful_patterns[pattern] / count * 100) if count > 0 else 0
            print(f"  {pattern}: {count} (успешных: {success_rate:.1f}%)")
        
        return patterns_count, successful_patterns
    
    def generate_report(self):
        """Создание отчета"""
        if not self.patterns_found:
            return "Паттерны не найдены"
        
        report = "# Анализ торговых паттернов - День 1\n\n"
        
        # Группировка по символам
        symbols = set([p['symbol'] for p in self.patterns_found])
        
        for symbol in symbols:
            report += f"## {symbol}\n\n"
            symbol_patterns = [p for p in self.patterns_found if p['symbol'] == symbol]
            
            for pattern in symbol_patterns[:5]:  # Первые 5 для каждого символа
                dt_str = pattern['datetime'].strftime('%Y-%m-%d %H:%M')
                return_str = f"{pattern['future_return']:.2f}%" if pattern['future_return'] else "N/A"
                success_str = "ДА" if pattern['successful'] else "НЕТ"
                
                report += f"- **{pattern['pattern']}** - {dt_str} - Цена: ${pattern['price']:.2f} - Результат через 4ч: {return_str} - Успешный: {success_str}\n"
            
            report += "\n"
        
        # Общая статистика
        total_patterns = len(self.patterns_found)
        successful = len([p for p in self.patterns_found if p['successful']])
        success_rate = (successful / total_patterns * 100) if total_patterns > 0 else 0
        
        report += f"## Общая статистика\n"
        report += f"- Всего паттернов найдено: {total_patterns}\n"
        report += f"- Успешных предсказаний: {successful} ({success_rate:.1f}%)\n"
        report += f"- Самый надежный паттерн: [требует дополнительного анализа]\n"
        
        return report

def main():
    analyzer = PatternAnalyzer()
    
    # Анализ трех символов
    symbols = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
    
    for symbol in symbols:
        try:
            analyzer.analyze_symbol(symbol)
        except Exception as e:
            print(f"Ошибка анализа {symbol}: {e}")
    
    # Создание отчета
    report = analyzer.generate_report()
    
    # Сохранение
    with open('docs/patterns_observed.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nОтчет сохранен в docs/patterns_observed.md")
    print(f"Найдено паттернов: {len(analyzer.patterns_found)}")

if __name__ == "__main__":
    main()