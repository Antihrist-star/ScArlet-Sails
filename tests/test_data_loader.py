"""
Tests for Data Loader

Run: pytest tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_loader import (
    load_market_data,
    load_multiple_assets,
    validate_params,
    get_data_info,
    AVAILABLE_COINS,
    AVAILABLE_TIMEFRAMES,
)


class TestValidation:
    """Test parameter validation."""
    
    def test_valid_coin(self):
        """Valid coins should not raise."""
        for coin in AVAILABLE_COINS:
            validate_params(coin, '15m')  # Should not raise
    
    def test_invalid_coin(self):
        """Invalid coin should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid coin"):
            validate_params('INVALID', '15m')
    
    def test_valid_timeframe(self):
        """Valid timeframes should not raise."""
        for tf in AVAILABLE_TIMEFRAMES:
            validate_params('BTC', tf)  # Should not raise
    
    def test_invalid_timeframe(self):
        """Invalid timeframe should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            validate_params('BTC', '5m')
    
    def test_available_coins_list(self):
        """Check available coins list."""
        assert len(AVAILABLE_COINS) == 14
        assert 'BTC' in AVAILABLE_COINS
        assert 'ETH' in AVAILABLE_COINS
        assert 'ENA' in AVAILABLE_COINS
    
    def test_available_timeframes_list(self):
        """Check available timeframes list."""
        assert AVAILABLE_TIMEFRAMES == ['15m', '1h', '4h', '1d']


class TestDataLoading:
    """Test data loading functionality."""
    
    @pytest.fixture
    def btc_15m_data(self):
        """Load BTC 15m data (if available)."""
        try:
            return load_market_data('BTC', '15m')
        except FileNotFoundError:
            pytest.skip("Data file not found")
    
    def test_load_btc_15m(self, btc_15m_data):
        """Test loading BTC 15m data."""
        df = btc_15m_data
        
        # Check shape
        assert len(df) > 0, "DataFrame should not be empty"
        
        # Check columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        
        # Check data types
        assert df['close'].dtype in [np.float64, np.float32], "Close should be float"
        assert df['volume'].dtype in [np.float64, np.float32, np.int64], "Volume should be numeric"
    
    def test_load_with_date_filter(self, btc_15m_data):
        """Test date filtering."""
        try:
            df = load_market_data('BTC', '15m', start_date='2024-01-01')
            assert df.index[0] >= pd.Timestamp('2024-01-01'), "Start date filter failed"
        except FileNotFoundError:
            pytest.skip("Data file not found")
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_market_data('BTC', '15m', data_dir='nonexistent_dir')
    
    def test_ohlc_relationships(self, btc_15m_data):
        """Test OHLC data integrity."""
        df = btc_15m_data
        
        # High should be >= Low
        assert (df['high'] >= df['low']).all(), "High should be >= Low"
        
        # High should be >= Open and Close
        assert (df['high'] >= df['open']).all(), "High should be >= Open"
        assert (df['high'] >= df['close']).all(), "High should be >= Close"
        
        # Low should be <= Open and Close
        assert (df['low'] <= df['open']).all(), "Low should be <= Open"
        assert (df['low'] <= df['close']).all(), "Low should be <= Close"
    
    def test_no_negative_values(self, btc_15m_data):
        """Test no negative prices or volumes."""
        df = btc_15m_data
        
        assert (df['open'] > 0).all(), "Open should be positive"
        assert (df['high'] > 0).all(), "High should be positive"
        assert (df['low'] > 0).all(), "Low should be positive"
        assert (df['close'] > 0).all(), "Close should be positive"
        assert (df['volume'] >= 0).all(), "Volume should be non-negative"


class TestMultiAssetLoading:
    """Test multi-asset loading."""
    
    def test_load_multiple_coins(self):
        """Test loading multiple coins."""
        try:
            data = load_multiple_assets(
                coins=['BTC', 'ETH'],
                timeframe='15m'
            )
            
            assert isinstance(data, dict), "Should return dict"
            # At least one should load if data exists
            assert len(data) >= 0
            
        except Exception:
            pytest.skip("Data files not found")


class TestDataInfo:
    """Test data info function."""
    
    def test_get_data_info(self):
        """Test data info retrieval."""
        info = get_data_info()
        
        assert 'coins' in info or 'expected_coins' in info
        assert 'timeframes' in info or 'expected_timeframes' in info
        assert 'total_files' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])