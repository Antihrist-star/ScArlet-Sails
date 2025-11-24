"""
strategies/xgboost_ml.py
========================
XGBoostStrategy для генерации ML-based сигналов с P_j(S) интеграцией
Использует Feature Engine и Model Manager
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostStrategy:
    """
    ML Strategy на основе XGBoost.
    
    Pipeline:
    1. calculate_features() - генерируем 31 признак
    2. generate_signals() - предсказываем UP/DOWN
    3. применяем threshold фильтрацию
    
    Интегрируется с FeatureEngine и ModelManager.
    """
    
    def __init__(self, config: Dict, feature_engine, model_manager):
        """
        Args:
            config: Config dict
            feature_engine: FeatureEngine instance для расчёта признаков
            model_manager: ModelManager instance с загруженной моделью
        """
        self.config = config
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        
        self.strategy_config = config.get('ml_strategy', {})
        self.ml_threshold = self.strategy_config.get('threshold', 0.5)
        self.use_probability = self.strategy_config.get('use_probability', True)
        
        logger.info(f"XGBoostStrategy инициализирована. Threshold: {self.ml_threshold}")
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует ML сигналы (0/1) и вероятности (0-1).
        
        Args:
            df: DataFrame с OHLCV
        
        Returns:
            signals: Массив (0 или 1) - итоговые сигналы
            probabilities: Массив (0-1) - вероятности UP
        """
        
        try:
            # 1. Calculate features
            logger.info("Calculating features...")
            df_featured = self.feature_engine.calculate_features(df)
            
            # 2. Get feature names
            feature_names = self.feature_engine.get_feature_names()
            X = df_featured[feature_names].copy()
            
            # Handle NaN
            X = X.fillna(0)
            
            # 3. Get predictions from model
            logger.info("Getting predictions from XGBoost...")
            probabilities = self.model_manager.predict_proba(X)
            
            # 4. Apply threshold
            signals = (probabilities >= self.ml_threshold).astype(int)
            
            logger.info(f"Signals generated. Total signals: {np.sum(signals)}/{len(signals)}")
            
            return signals, probabilities
            
        except Exception as e:
            logger.error(f"Error in generate_signals: {e}")
            raise
    
    def generate_signals_with_pj_s(self, 
                                  df: pd.DataFrame,
                                  regime: np.ndarray = None,
                                  crisis_level: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует ML сигналы с P_j(S) компонентами.
        
        Возвращает также:
        - ML score (вероятность)
        - P_j(S) value (для анализа компонентов)
        
        Args:
            df: DataFrame с OHLCV
            regime: Массив режимов (BULL, BEAR, SIDEWAYS)
            crisis_level: Массив уровней кризиса
        
        Returns:
            signals: Массив (0 или 1)
            ml_scores: ML вероятности (0-1)
            pj_s_values: Расчётные P_j(S) значения
        """
        
        try:
            # Generate ML signals
            signals, ml_scores = self.generate_signals(df)
            
            # Calculate P_j(S) components
            pj_s_values = self._calculate_pj_s_components(
                ml_scores, regime, crisis_level
            )
            
            return signals, ml_scores, pj_s_values
            
        except Exception as e:
            logger.error(f"Error in generate_signals_with_pj_s: {e}")
            raise
    
    def _calculate_pj_s_components(self, 
                                  ml_scores: np.ndarray,
                                  regime: np.ndarray = None,
                                  crisis_level: np.ndarray = None) -> np.ndarray:
        """
        Вычисляет P_j(S) компоненты для каждой свечи.
        
        P_j(S) = ML_score × Filters - Costs - RiskPenalty
        
        Где:
        - ML_score: XGBoost вероятность
        - Filters: режим и кризис фильтры
        - Costs: комиссии и слипп
        - RiskPenalty: штраф за риск
        """
        
        pj_s = np.zeros(len(ml_scores))
        
        if regime is None:
            regime = np.array(['BULL'] * len(ml_scores))
        if crisis_level is None:
            crisis_level = np.zeros(len(ml_scores), dtype=int)
        
        for i in range(len(ml_scores)):
            ml_component = ml_scores[i]
            
            # Filter component (regime-based)
            regime_filter = self._get_regime_filter(regime[i])
            crisis_filter = self._get_crisis_filter(crisis_level[i])
            filter_product = regime_filter * crisis_filter
            
            # Cost component
            costs = 0.003  # 0.1% commission + 0.2% slippage
            
            # Risk penalty component
            risk_penalty = self._calculate_risk_penalty(crisis_level[i])
            
            # P_j(S) = ML × Filters - Costs - RiskPenalty
            pj_s[i] = (ml_component * filter_product) - costs - risk_penalty
            
        return pj_s
    
    def _get_regime_filter(self, regime: str) -> float:
        """Фильтр по режиму"""
        regime_filters = {
            'BULL': 1.0,
            'SIDEWAYS': 0.8,
            'BEAR': 0.3
        }
        return regime_filters.get(regime, 0.5)
    
    def _get_crisis_filter(self, crisis_level: int) -> float:
        """Фильтр по кризису"""
        crisis_filters = {
            0: 1.0,    # No crisis
            1: 0.9,    # Low crisis
            2: 0.5,    # Mid crisis
            3: 0.0     # High crisis - stop trading
        }
        return crisis_filters.get(crisis_level, 0.5)
    
    def _calculate_risk_penalty(self, crisis_level: int) -> float:
        """Штраф за риск в зависимости от кризиса"""
        base_penalty = 0.002
        crisis_multipliers = {
            0: 1.0,
            1: 1.5,
            2: 2.5,
            3: 5.0
        }
        multiplier = crisis_multipliers.get(crisis_level, 1.0)
        return base_penalty * multiplier


class ModelManager:
    """
    Менеджер ML модели XGBoost.
    
    Отвечает за:
    - Загрузку/сохранение модели
    - Предсказания
    - Управление версиями
    """
    
    def __init__(self, config: Dict, model_path: str = None):
        """
        Args:
            config: Config dict
            model_path: Путь к сохранённой модели (опционально)
        """
        self.config = config
        self.model_config = config.get('ml_model', {})
        self.model = None
        self.model_path = model_path
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
        else:
            logger.warning("No model loaded. Use load_model() or train_model()")
    
    def load_model(self, model_path: str) -> None:
        """
        Загружает XGBoost модель.
        
        Args:
            model_path: Путь к JSON модели
        """
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def save_model(self, model_path: str) -> None:
        """
        Сохраняет XGBoost модель.
        
        Args:
            model_path: Путь для сохранения
        """
        if self.model is None:
            logger.error("No model to save")
            return
        
        try:
            self.model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def train_model(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Обучает новую XGBoost модель.
        
        Args:
            X_train: Training features (31 признак)
            y_train: Training labels (0 или 1)
        """
        
        try:
            # XGBoost parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': self.model_config.get('max_depth', 5),
                'learning_rate': self.model_config.get('learning_rate', 0.1),
                'n_estimators': self.model_config.get('n_estimators', 100),
                'subsample': self.model_config.get('subsample', 0.8),
                'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
                'random_state': 42,
                'scale_pos_weight': self.model_config.get('scale_pos_weight', 2.13)
            }
            
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train, verbose=False)
            
            logger.info(f"Model trained. Shape: {X_train.shape}")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает вероятности (0-1).
        
        Args:
            X: Features (должны быть в том же порядке, что при обучении)
        
        Returns:
            Массив вероятностей для класса 1 (UP)
        """
        
        if self.model is None:
            logger.error("No model loaded")
            return np.zeros(len(X))
        
        try:
            proba = self.model.predict_proba(X)
            # Вероятность для класса 1 (UP)
            return proba[:, 1]
        except Exception as e:
            logger.error(f"Error in predict_proba: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает классы (0 или 1).
        
        Args:
            X: Features
        
        Returns:
            Массив предсказаний (0 или 1)
        """
        
        if self.model is None:
            logger.error("No model loaded")
            return np.zeros(len(X), dtype=int)
        
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise
    
    def get_feature_importance(self) -> Dict:
        """
        Получает важность признаков.
        
        Returns:
            Dict {feature_name: importance_score}
        """
        
        if self.model is None:
            logger.error("No model loaded")
            return {}
        
        try:
            importances = self.model.feature_importances_
            feature_names = self.model.get_booster().feature_names
            
            return dict(zip(feature_names, importances))
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def get_model_info(self) -> Dict:
        """Возвращает информацию о модели"""
        
        if self.model is None:
            return {'status': 'No model loaded'}
        
        return {
            'type': 'XGBClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'learning_rate': self.model.learning_rate,
            'model_path': self.model_path
        }