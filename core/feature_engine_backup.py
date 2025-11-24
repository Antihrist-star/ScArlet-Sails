"""
core/feature_engine.py
======================
FeatureEngine для генерации 31 признака для XGBoost модели
Совместимо с нормализацией и backtesting pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Генерирует 31 признак для ML модели.
    
    Категории признаков:
    - Momentum (9): RSI, Stochastic, CCI, MACD, etc.
    - Trend (8): SMA, EMA, ADX, Ichimoku, etc.
    - Volume (6): OBV, ChaikinMF, VWAP, etc.
    - Volatility (8): ATR, BBands, Keltner, HistVol, etc.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Config dict с параметрами (periods, normalization, etc.)
        """
        self.config = config.get('feature_engine', {})
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Параметры
        self.rsi_period = self.config.get('rsi_period', 14)
        self.ema_periods = self.config.get('ema_periods', [5, 10, 20])
        self.sma_periods = self.config.get('sma_periods', [5, 10, 20])
        self.atr_period = self.config.get('atr_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        
        self.normalize = self.config.get('normalize', True)
        
        logger.info(f"FeatureEngine инициализирован. Нормализация: {self.normalize}")
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет все 31 признак.
        
        Args:
            df: DataFrame с OHLCV (open, high, low, close, volume)
        
        Returns:
            DataFrame с original OHLCV + 31 признак
        """
        df = df.copy()
        
        try:
            # Momentum признаки (9)
            df = self._add_momentum_features(df)
            
            # Trend признаки (8)
            df = self._add_trend_features(df)
            
            # Volume признаки (6)
            df = self._add_volume_features(df)
            
            # Volatility признаки (8)
            df = self._add_volatility_features(df)
            
            # Нормализация
            if self.normalize:
                df = self._normalize_features(df)
            
            logger.info(f"Признаков создано: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при расчёте признаков: {e}")
            raise
    
    # ========== MOMENTUM (9 признаков) ==========
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, CCI, MACD, Momentum"""
        
        # 1. RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # 2-3. Stochastic %K, %D
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
            df['high'], df['low'], df['close'], period=14
        )
        
        # 4. CCI
        df['cci'] = self._calculate_cci(df['high'], df['low'], df['close'], period=20)
        
        # 5-7. MACD, Signal, Histogram
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(
            df['close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        
        # 8. Momentum (ROC)
        df['momentum'] = df['close'].pct_change(10) * 100
        
        # 9. Rate of Change
        df['roc'] = df['close'].pct_change(12) * 100
        
        return df
    
    # ========== TREND (8 признаков) ==========
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """SMA, EMA, ADX, Trend direction"""
        
        # SMA (3 признака)
        for period in self.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        
        # EMA (3 признака)
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # ADX
        df['adx'] = self._calculate_adx(df['high'], df['low'], df['close'], period=14)
        
        # Trend strength (SMA50 - SMA200 нормализованный)
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        df['trend_strength'] = ((sma_50 - sma_200) / df['close']) * 100
        
        return df
    
    # ========== VOLUME (6 признаков) ==========
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, ChaikinMF, VWAP, Volume Rate"""
        
        # OBV
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        # OBV normalized
        df['obv_norm'] = df['obv'].rolling(20).mean()
        
        # Chaikin Money Flow
        df['chaikin_mf'] = self._calculate_chaikin_money_flow(
            df['high'], df['low'], df['close'], df['volume'], period=20
        )
        
        # VWAP
        df['vwap'] = self._calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Volume Rate of Change
        df['vroc'] = df['volume'].pct_change(10) * 100
        
        # Volume MA Ratio
        vol_ma = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (vol_ma + 1e-8)
        
        return df
    
    # ========== VOLATILITY (8 признаков) ==========
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, BBands, Keltner, Historical Volatility"""
        
        # ATR
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Bollinger Bands
        df['bb_high'], df['bb_mid'], df['bb_low'], df['bb_width'] = self._calculate_bollinger_bands(
            df['close'], self.bb_period, self.bb_std
        )
        
        # BB Position (где цена внутри полос)
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-8)
        
        # Historical Volatility (std dev returns)
        df['hist_vol'] = df['close'].pct_change().rolling(20).std() * 100
        
        # Close to SMA (Keltner-like)
        sma_20 = df['close'].rolling(20).mean()
        df['close_sma_ratio'] = (df['close'] - sma_20) / (sma_20 + 1e-8)
        
        return df
    
    # ========== HELPERS ==========
    
    @staticmethod
    def _calculate_rsi(close, period=14):
        """Relative Strength Index"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_stochastic(high, low, close, period=14):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    @staticmethod
    def _calculate_cci(high, low, close, period=20):
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad + 1e-8)
        return cci
    
    @staticmethod
    def _calculate_macd(close, fast=12, slow=26, signal=9):
        """MACD"""
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    @staticmethod
    def _calculate_adx(high, low, close, period=14):
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - close.shift()), 
                                 np.abs(low - close.shift())))
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-8))
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def _calculate_obv(close, volume):
        """On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def _calculate_chaikin_money_flow(high, low, close, volume, period=20):
        """Chaikin Money Flow"""
        mfv = ((close - low) - (high - close)) / (high - low + 1e-8) * volume
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        return cmf
    
    @staticmethod
    def _calculate_vwap(high, low, close, volume):
        """Volume Weighted Average Price"""
        tp = (high + low + close) / 3
        vwap = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()
        return vwap
    
    @staticmethod
    def _calculate_atr(high, low, close, period=14):
        """Average True Range"""
        tr = np.maximum(high - low,
                       np.maximum(np.abs(high - close.shift()),
                                 np.abs(low - close.shift())))
        atr = tr.rolling(period).mean()
        return atr
    
    @staticmethod
    def _calculate_bollinger_bands(close, period=20, std_dev=2):
        """Bollinger Bands"""
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        bb_high = sma + (std * std_dev)
        bb_low = sma - (std * std_dev)
        bb_width = bb_high - bb_low
        return bb_high, sma, bb_low, bb_width
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Z-score нормализация всех признаков (кроме OHLCV).
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Удалить NaN для нормализации
        df_features = df[feature_cols].dropna()
        
        if len(df_features) > 0:
            try:
                if not self.is_fitted:
                    df_features_normalized = self.scaler.fit_transform(df_features)
                    self.is_fitted = True
                else:
                    df_features_normalized = self.scaler.transform(df_features)
                
                df.loc[df_features.index, feature_cols] = df_features_normalized
                logger.info("Признаки нормализованы успешно")
            except Exception as e:
                logger.warning(f"Ошибка при нормализации: {e}")
        
        return df
    
    def get_feature_names(self) -> list:
        """Возвращает список имён всех 31 признака"""
        features = []
        
        # Momentum
        features.extend(['rsi', 'stoch_k', 'stoch_d', 'cci', 'macd', 'macd_signal', 'macd_hist', 'momentum', 'roc'])
        
        # Trend
        for p in self.sma_periods:
            features.append(f'sma_{p}')
        for p in self.ema_periods:
            features.append(f'ema_{p}')
        features.extend(['adx', 'trend_strength'])
        
        # Volume
        features.extend(['obv', 'obv_norm', 'chaikin_mf', 'vwap', 'vroc', 'vol_ratio'])
        
        # Volatility
        features.extend(['atr', 'atr_pct', 'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_position', 'hist_vol', 'close_sma_ratio'])
        
        return features[:31]  # Первые 31