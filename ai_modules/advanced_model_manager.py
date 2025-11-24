"""
AI MODULE - Advanced Model Manager
Without MLflow for now
"""
import logging
from pathlib import Path
import json
from datetime import datetime
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)

class AdvancedModelManager:
    """Advanced model lifecycle management"""
    
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = Path(model_path)
        
        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        
        # Try to load scaler
        scaler_path = self.model_path.parent / "xgboost_normalized_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            scaler_path = self.model_path.parent / "xgboost_multi_tf_scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
        
        # Performance tracking
        self.performance_history = []
        self.degradation_threshold = 0.15
        
        logger.info("  ✅ Advanced Model Manager initialized")
    
    def predict_proba(self, X):
        """Get predictions"""
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def predict_xgboost(self, X):
        """Get binary predictions with threshold"""
        proba = self.predict_proba(X)[:, 1]
        threshold = self.config['models']['xgboost']['threshold']
        return (proba >= threshold).astype(int)
    
    def monitor_and_retrain(self, metrics, features=None, labels=None):
        """Monitor performance and flag if retraining needed"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Check if retraining needed
        if len(self.performance_history) < 10:
            return False
        
        # Compare recent vs baseline
        recent = self.performance_history[-5:]
        baseline = self.performance_history[:5]
        
        avg_recent_wr = sum(m['metrics'].get('win_rate', 0) for m in recent) / 5
        avg_baseline_wr = sum(m['metrics'].get('win_rate', 0) for m in baseline) / 5
        
        if avg_baseline_wr == 0:
            return False
        
        degradation = (avg_baseline_wr - avg_recent_wr) / avg_baseline_wr
        
        if degradation > self.degradation_threshold:
            logger.warning(f"  ⚠️  Performance degraded {degradation*100:.1f}%")
            logger.warning(f"  ⚠️  Retraining recommended!")
            
            # Save performance log
            self._save_performance_log()
            
            return True
        
        return False
    
    def _save_performance_log(self):
        """Save performance history to file"""
        log_path = self.model_path.parent / "performance_log.json"
        
        log_data = {
            'last_update': datetime.now().isoformat(),
            'performance_history': [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'win_rate': p['metrics'].get('win_rate', 0),
                    'profit_factor': p['metrics'].get('profit_factor', 0)
                }
                for p in self.performance_history
            ]
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"  💾 Performance log saved: {log_path}")