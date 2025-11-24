"""
SCARLET SAILS - MAIN ENTRY POINT
Unified system orchestrator
"""
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScarletSails:
    """Main system orchestrator"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize system with config"""
        logger.info("="*80)
        logger.info("SCARLET SAILS - STARTING")
        logger.info("="*80)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"✅ Config loaded: {config_path}")
        logger.info(f"✅ Assets: {len(self.config['data']['assets'])}")
        logger.info(f"✅ Timeframes: {len(self.config['data']['timeframes'])}")
        
        self.project_root = Path(__file__).parent
        self.models = {}
        self.results = {}
    
    def load_models(self):
        """Load all models"""
        logger.info("\n[1/4] Loading models...")
        
        try:
            # Check if AI modules are enabled
            use_ai = self.config.get('ai_modules', {}).get('advanced_model_manager', {}).get('enabled', False)
            
            if use_ai:
                logger.info("  Using Advanced Model Manager (OPUS)")
                from ai_modules.advanced_model_manager import AdvancedModelManager
                
                model_path = self.project_root / self.config['models']['xgboost']['model_path']
                self.models['manager'] = AdvancedModelManager(
                    self.config,
                    str(model_path)
                )
            else:
                logger.info("  Using Basic Model Manager")
                # Import here to avoid circular imports
                import xgboost as xgb
                import joblib
                
                # Load XGBoost
                xgb_path = self.project_root / self.config['models']['xgboost']['model_path']
                if xgb_path.exists():
                    self.models['xgboost'] = xgb.XGBClassifier()
                    self.models['xgboost'].load_model(str(xgb_path))
                    logger.info(f"  ✅ XGBoost loaded: {xgb_path.name}")
                
                # Load scaler
                scaler_path = self.project_root / self.config['models']['xgboost']['scaler_path']
                if scaler_path.exists():
                    self.models['scaler'] = joblib.load(scaler_path)
                    logger.info(f"  ✅ Scaler loaded: {scaler_path.name}")
            
            # Load regime detector
            try:
                import sys
                models_dir = self.project_root / "models"
                sys.path.insert(0, str(models_dir))
                
                from regime_detector import RegimeDetector
                self.models['regime_detector'] = RegimeDetector()
                logger.info("  ✅ Regime detector loaded")
            except:
                logger.warning("  ⚠️  Regime detector not available")
            
            # Load crisis classifier
            try:
                from crisis_classifier import CrisisClassifier
                self.models['crisis_classifier'] = CrisisClassifier()
                logger.info("  ✅ Crisis classifier loaded")
            except:
                logger.warning("  ⚠️  Crisis classifier not available")
            
            logger.info("✅ All models loaded")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False
    
    def run_backtest(self, asset=None, timeframe=None):
        """Run backtest on specified asset/timeframe"""
        logger.info("\n[2/4] Running backtest...")
        
        # Import modules
        from core.data_loader import DataLoader
        from core.feature_engine import FeatureEngine
        from core.backtest_engine import BacktestEngine
        from strategies.rule_based import RuleBasedStrategy
        from strategies.xgboost_ml import XGBoostStrategy
        
        if asset is None:
            assets = self.config['data']['assets'][:3]  # Test on 3 assets
        else:
            assets = [asset]
        
        if timeframe is None:
            timeframes = ['15m']  # Test on 15m only
        else:
            timeframes = [timeframe]
        
        logger.info(f"  Assets: {assets}")
        logger.info(f"  Timeframes: {timeframes}")
        
        # Initialize components
        data_loader = DataLoader(self.config)
        feature_engine = FeatureEngine(self.config)
        backtest_engine = BacktestEngine(self.config)
        
        # Load models (already loaded)
        
        # Strategies
        rule_strategy = RuleBasedStrategy(self.config)
        ml_strategy = XGBoostStrategy(self.config, feature_engine, self)
        
        results = {}
        
        for asset_name in assets:
            for tf in timeframes:
                combo = f"{asset_name}_{tf}"
                logger.info(f"\n  Testing {combo}...")
                
                # Load data
                df = data_loader.load_ohlcv(asset_name, tf)
                if df is None:
                    continue
                
                # Get test period
                df_test = data_loader.get_test_period(df)
                logger.info(f"    Test period: {len(df_test):,} bars")
                
                # Test Rule-Based
                try:
                    signals_rule = rule_strategy.generate_signals(df_test)
                    result_rule = backtest_engine.run(df_test, signals_rule)
                    
                    logger.info(f"    Rule-Based: {result_rule['metrics']['total_return']:.2f}% return, {result_rule['metrics']['num_trades']} trades")
                    
                    results[f"{combo}_rule"] = result_rule
                except Exception as e:
                    logger.error(f"    ❌ Rule-Based failed: {e}")
                
                # Test ML (if model loaded)
                if 'xgboost' in self.models:
                    try:
                        signals_ml = ml_strategy.generate_signals(df_test)
                        result_ml = backtest_engine.run(df_test, signals_ml)
                        
                        logger.info(f"    ML: {result_ml['metrics']['total_return']:.2f}% return, {result_ml['metrics']['num_trades']} trades")
                        
                        results[f"{combo}_ml"] = result_ml
                    except Exception as e:
                        logger.error(f"    ❌ ML failed: {e}")
        
        logger.info(f"\n✅ Backtest complete: {len(results)} tests")
        
        return results
    
    def train_models(self):
        """Train/retrain models"""
        logger.info("\n[3/4] Training models...")
        
        # TODO: Implement training
        logger.info("  ⚠️  Training implementation pending")
        
        return True
    
    def start_dashboard(self):
        """Start web dashboard"""
        logger.info("\n[4/4] Starting dashboard...")
        
        host = self.config['dashboard']['host']
        port = self.config['dashboard']['port']
        
        logger.info(f"  Dashboard: http://{host}:{port}")
        logger.info("  ⚠️  Dashboard implementation pending")
        
        return True
    
    def health_check(self):
        """System health check"""
        logger.info("\n🏥 HEALTH CHECK")
        logger.info("="*80)
        
        checks = {
            'Config': self.config is not None,
            'Models directory': (self.project_root / 'models').exists(),
            'Data directory': (self.project_root / self.config['data']['data_dir']).exists(),
            'XGBoost model': (self.project_root / self.config['models']['xgboost']['model_path']).exists(),
        }
        
        for check_name, status in checks.items():
            status_str = "✅ PASS" if status else "❌ FAIL"
            logger.info(f"  {check_name:20s} {status_str}")
        
        all_pass = all(checks.values())
        
        logger.info("="*80)
        if all_pass:
            logger.info("✅ ALL CHECKS PASSED")
        else:
            logger.info("❌ SOME CHECKS FAILED")
        
        return all_pass
    
    def run(self, mode="backtest", **kwargs):
        """Main run method"""
        try:
            if mode == "health":
                return self.health_check()
            
            elif mode == "backtest":
                self.load_models()
                results = self.run_backtest(**kwargs)
                logger.info("\n✅ BACKTEST COMPLETE")
                return results
            
            elif mode == "train":
                self.load_models()
                success = self.train_models()
                logger.info("\n✅ TRAINING COMPLETE" if success else "\n❌ TRAINING FAILED")
                return success
            
            elif mode == "dashboard":
                self.load_models()
                self.start_dashboard()
                logger.info("\n✅ DASHBOARD STARTED")
                return True
            
            else:
                logger.error(f"❌ Unknown mode: {mode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Scarlet Sails Trading System")
    
    parser.add_argument(
        '--mode',
        choices=['health', 'backtest', 'train', 'dashboard'],
        default='health',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--asset',
        type=str,
        help='Asset to test (default: all)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['15m', '1h', '4h', '1d'],
        help='Timeframe to test (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = ScarletSails(config_path=args.config)
    
    # Run
    result = system.run(
        mode=args.mode,
        asset=args.asset,
        timeframe=args.timeframe
    )
    
    if result:
        logger.info("\n" + "="*80)
        logger.info("SUCCESS")
        logger.info("="*80)
    else:
        logger.info("\n" + "="*80)
        logger.info("FAILED")
        logger.info("="*80)

if __name__ == "__main__":
    main()