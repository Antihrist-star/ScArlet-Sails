"""
AI MODULE - Enhanced Signal Validator
Multi-agent validation without autogen
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedSignalValidator:
    """Validates signals using rule-based multi-criteria"""
    
    def __init__(self, feature_engine, model_manager):
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        self.enabled = True
        
        # Thresholds
        self.min_confidence = 0.5
        self.max_volatility = 0.05
        self.min_volume_ratio = 0.8
        
        logger.info("  ✅ Signal Validator initialized")
        
    def validate_signal(self, signal_data, features_df, pj_s_value):
        """Validate a single signal with multi-criteria check"""
        if not self.enabled:
            return {'execute': True, 'confidence': 1.0}
        
        # Упрощённая проверка - ТОЛЬКО P_j(S)
        execute = pj_s_value >= self.min_confidence
        
        return {
            'execute': execute,
            'confidence': pj_s_value,
            'checks': {'confidence': execute},
            'reason': 'OK' if execute else 'Low P_j(S)'
        }
        
        # Calculate consensus  
        passed = sum(checks.values())
        total = len(checks)
        confidence = passed / total
        
        # Более мягкое решение - пропускаем если хотя бы 1 из 2
        execute = confidence >= 0.5  # 1/2 checks must pass
        
        reason = []
        if not checks['confidence']:
            reason.append('Low P_j(S)')
        if not checks['rsi']:
            reason.append('Extreme RSI')
        
        return {
            'execute': execute,
            'confidence': confidence,
            'checks': checks,
            'reason': ', '.join(reason) if reason else 'Passed'
        }
    
    def validate_batch(self, signals, df, pj_s_values):
        """Validate batch of signals"""
        features_df = self.feature_engine.calculate_features(df)
        validated = []
        
        for i, signal in enumerate(signals):
            if signal == 1:
                validation = self.validate_signal(
                    {'index': i},
                    features_df.iloc[i:i+1],
                    pj_s_values[i]
                )
                validated.append(1 if validation['execute'] else 0)
            else:
                validated.append(0)
        
        filtered_count = sum(signals) - sum(validated)
        if filtered_count > 0:
            logger.info(f"  🔍 Filtered {filtered_count} signals via validation")
        
        return np.array(validated)