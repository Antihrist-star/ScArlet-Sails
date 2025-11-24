# ai_modules/enhanced_signal_validator.py
"""
Интегрируется МЕЖДУ XGBoostStrategy и BacktestEngine
Каждый сигнал проходит multi-agent проверку
"""

import autogen
from typing import Dict
import numpy as np

class EnhancedSignalValidator:
    """
    3 AI агента проверяют КАЖДЫЙ сигнал перед исполнением
    """
    
    def __init__(self, feature_engine, model_manager):
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        
        # Создаем специализированных агентов
        self.technical_analyst = autogen.AssistantAgent(
            name="technical_analyst",
            system_message="""You analyze technical indicators from FeatureEngine.
            RSI, MACD, Bollinger Bands, Volume patterns.
            Flag oversold/overbought conditions."""
        )
        
        self.risk_assessor = autogen.AssistantAgent(
            name="risk_assessor",
            system_message="""You assess P_j(S) components and market regime.
            Check if signal aligns with current crisis_level.
            Validate position sizing based on volatility (ATR)."""
        )
        
        self.market_scanner = autogen.AssistantAgent(
            name="market_scanner",
            system_message="""You check external factors:
            - Binance order book imbalance
            - Funding rates
            - Open interest changes
            Real-time market microstructure."""
        )
        
    def validate_signal(self, 
                       signal_data: Dict,
                       features_df: pd.DataFrame,
                       pj_s_value: float) -> Dict:
        """
        Проверяет сигнал через консенсус 3 агентов
        
        Args:
            signal_data: {'symbol', 'direction', 'ml_score', 'timestamp'}
            features_df: Строка с 31 признаком из FeatureEngine
            pj_s_value: Значение P_j(S) из XGBoostStrategy
        """
        
        # Извлекаем ключевые метрики для агентов
        technical_context = f"""
        Signal: {signal_data['direction']} at {signal_data['ml_score']:.3f}
        P_j(S): {pj_s_value:.4f}
        
        Technical Indicators:
        - RSI: {features_df['rsi'].iloc[-1]:.1f}
        - MACD Hist: {features_df['macd_hist'].iloc[-1]:.4f}
        - BB Position: {features_df['bb_position'].iloc[-1]:.2f}
        - Volume Ratio: {features_df['vol_ratio'].iloc[-1]:.2f}
        - ATR %: {features_df['atr_pct'].iloc[-1]:.2f}
        - ADX: {features_df['adx'].iloc[-1]:.1f}
        """
        
        # Групповое обсуждение
        groupchat = autogen.GroupChat(
            agents=[self.technical_analyst, self.risk_assessor, self.market_scanner],
            messages=[],
            max_round=3
        )
        
        manager = autogen.GroupChatManager(groupchat=groupchat)
        
        # Запрос консенсуса
        query = f"""
        VALIDATE THIS SIGNAL:
        {technical_context}
        
        Each agent must provide:
        1. GO/NO-GO decision
        2. Confidence (0-1)
        3. Key risks identified
        
        FINAL OUTPUT FORMAT:
        DECISION: [EXECUTE/REJECT/REDUCE]
        CONFIDENCE: [0.0-1.0]
        POSITION_SIZE_MULTIPLIER: [0.0-1.0]
        """
        
        result = manager.initiate_chat(message=query)
        
        # Парсинг решения
        decision = self._parse_consensus(groupchat.messages)
        
        return {
            'original_signal': signal_data,
            'features_snapshot': features_df.iloc[-1].to_dict(),
            'pj_s': pj_s_value,
            'ai_decision': decision['action'],
            'ai_confidence': decision['confidence'],
            'position_multiplier': decision['size_mult'],
            'risks': decision['risks'],
            'execute': decision['action'] == 'EXECUTE'
        }
    
    def _parse_consensus(self, messages) -> Dict:
        """Извлекает консенсус из дискуссии агентов"""
        # Simplified - в production нужен NLP parser
        votes = {'EXECUTE': 0, 'REJECT': 0, 'REDUCE': 0}
        confidences = []
        
        for msg in messages[-3:]:  # Last 3 messages = agent decisions
            if 'EXECUTE' in msg.content.upper():
                votes['EXECUTE'] += 1
            elif 'REJECT' in msg.content.upper():
                votes['REJECT'] += 1
            elif 'REDUCE' in msg.content.upper():
                votes['REDUCE'] += 1
        
        # Majority vote
        decision = max(votes, key=votes.get)
        
        return {
            'action': decision,
            'confidence': 0.7 if decision == 'EXECUTE' else 0.3,
            'size_mult': 1.0 if decision == 'EXECUTE' else 0.5 if decision == 'REDUCE' else 0.0,
            'risks': ['volatility_spike'] if votes['REDUCE'] > 0 else []
        }