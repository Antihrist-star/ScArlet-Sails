"""
CORE - DATA LOADER
Unified data loading and preprocessing
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Loads and preprocesses market data"""
    
    def __init__(self, config):
        self.config = config
        self.data_dir = Path(config['data']['data_dir'])
        self.cache = {}
        
    def load_ohlcv(self, asset, timeframe):
        """Load OHLCV data for asset/timeframe"""
        cache_key = f"{asset}_{timeframe}"
        
        if cache_key in self.cache:
            logger.debug(f"  Cache hit: {cache_key}")
            return self.cache[cache_key]
        
        filename = f"{asset}_USDT_{timeframe}.parquet"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.error(f"  ❌ File not found: {filepath}")
            return None
        
        try:
            df = pd.read_parquet(filepath)
            
            # Validate columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"  ❌ Missing required columns: {filepath}")
                return None
            
            # Cache
            self.cache[cache_key] = df
            
            logger.info(f"  ✅ Loaded {len(df):,} bars: {asset} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Error loading {filepath}: {e}")
            return None
    
    def load_multiple(self, assets, timeframes):
        """Load multiple asset/timeframe combinations"""
        data = {}
        
        for asset in assets:
            for tf in timeframes:
                df = self.load_ohlcv(asset, tf)
                if df is not None:
                    data[f"{asset}_{tf}"] = df
        
        logger.info(f"✅ Loaded {len(data)} datasets")
        return data
    
    def load_multi_timeframe(self, asset):
        """Load all 4 timeframes for an asset"""
        df_15m = self.load_ohlcv(asset, '15m')
        df_1h = self.load_ohlcv(asset, '1h')
        df_4h = self.load_ohlcv(asset, '4h')
        df_1d = self.load_ohlcv(asset, '1d')
        
        return {
            '15m': df_15m,
            '1h': df_1h,
            '4h': df_4h,
            '1d': df_1d
        }
    
    def get_test_period(self, df, start=None, end=None):
        """Extract test period from dataframe"""
        if start is None:
            start = self.config['backtest']['test_period_start']
        
        df_test = df[df.index >= start].copy()
        
        if end is not None:
            df_test = df_test[df_test.index <= end].copy()
        
        return df_test
    
    def calculate_basic_features(self, df):
        """Calculate basic technical features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Volume ratio
        df['volume_ma'] = df['volume'].rolling(14).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def validate_data(self, df):
        """Validate data quality"""
        issues = []
        
        # Check for NaN
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"NaN values in columns: {nan_cols}")
        
        # Check for zeros
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Zero/negative values in {col}")
        
        # Check for duplicates
        if df.index.duplicated().any():
            issues.append("Duplicate timestamps")
        
        if issues:
            logger.warning(f"  ⚠️  Data quality issues:")
            for issue in issues:
                logger.warning(f"    - {issue}")
            return False
        
        return True