# ai_modules/advanced_model_manager.py
"""
Расширяет существующий ModelManager автоматическим retraining
"""

import mlflow
import optuna
from pathlib import Path
import joblib
import json

class AdvancedModelManager(ModelManager):  # Наследуем от вашего ModelManager
    """
    Добавляет:
    - Автоматический retraining при деградации
    - A/B testing моделей
    - Версионирование через MLflow
    """
    
    def __init__(self, config: Dict, model_path: str = None):
        super().__init__(config, model_path)
        
        # MLflow setup
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("scarlet_sails_xgboost")
        
        # Performance tracking
        self.performance_history = []
        self.degradation_threshold = 0.85  # 15% degradation triggers retrain
        self.last_retrain_date = None
        
    def monitor_and_retrain(self, 
                           recent_performance: Dict,
                           X_recent: pd.DataFrame,
                           y_recent: np.ndarray) -> bool:
        """
        Мониторит производительность и запускает retraining при деградации
        
        Args:
            recent_performance: {'pf': 1.8, 'win_rate': 0.58, 'sharpe': 1.2}
            X_recent: Последние features для retraining
            y_recent: Последние labels
        
        Returns:
            True если произошел retrain
        """
        
        # Сравниваем с baseline
        baseline_pf = self.model_config.get('baseline_pf', 2.0)
        current_pf = recent_performance['pf']
        
        if current_pf < baseline_pf * self.degradation_threshold:
            logger.warning(f"Performance degraded: PF {current_pf:.2f} < {baseline_pf * 0.85:.2f}")
            
            with mlflow.start_run():
                # Hyperparameter optimization
                study = optuna.create_study(direction='maximize')
                study.optimize(
                    lambda trial: self._optuna_objective(trial, X_recent, y_recent),
                    n_trials=50
                )
                
                # Train new model with best params
                best_params = study.best_params
                new_model = xgb.XGBClassifier(**best_params)
                new_model.fit(X_recent, y_recent)
                
                # A/B test against current
                if self._ab_test_models(self.model, new_model, X_recent, y_recent):
                    # New model is better
                    self._deploy_new_model(new_model, best_params)
                    
                    # Log to MLflow
                    mlflow.log_params(best_params)
                    mlflow.log_metric("new_pf", current_pf)
                    mlflow.sklearn.log_model(new_model, "model")
                    
                    return True
        
        return False
    
    def _optuna_objective(self, trial, X, y):
        """Optuna objective для hyperparameter tuning"""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        
        # Cross-validation score
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
        return scores.mean()
    
    def _ab_test_models(self, model_a, model_b, X_test, y_test):
        """A/B testing двух моделей"""
        score_a = model_a.score(X_test, y_test)
        score_b = model_b.score(X_test, y_test)
        
        # Statistical significance test
        from scipy import stats
        _, p_value = stats.ttest_ind(
            [score_a] * 100,  # Mock repeated measurements
            [score_b] * 100
        )
        
        return score_b > score_a and p_value < 0.05
    
    def _deploy_new_model(self, model, params):
        """Deploy новой модели с версионированием"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = Path(f"models/production/xgboost_{timestamp}.pkl")
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'params': params,
            'feature_importance': dict(zip(
                self.feature_engine.get_feature_names(),
                model.feature_importances_
            ))
        }
        
        with open(f"models/production/metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update current
        self.model = model
        self.last_retrain_date = pd.Timestamp.now()
        
        logger.info(f"✅ New model deployed: {timestamp}")