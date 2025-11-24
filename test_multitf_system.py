"""
BLOCK 3 - TEST MULTI-TIMEFRAME SYSTEM
Tests XGBoost ML strategy with new FeatureEngine v2
"""
import yaml
import logging
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from core.data_loader import DataLoader
from core.feature_engine import FeatureEngine
from core.backtest_engine import BacktestEngine
from strategies.xgboost_ml import XGBoostStrategy
from ai_modules.advanced_model_manager import AdvancedModelManager
from ai_modules.enhanced_signal_validator import EnhancedSignalValidator

def main():
    logger.info("="*80)
    logger.info("БЛОК 3 - ТЕСТ MULTI-TIMEFRAME СИСТЕМЫ")
    logger.info("="*80)
    
    # 1. Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("\n[1/7] Configuration loaded")
    
    # 2. Initialize components
    logger.info("\n[2/7] Initializing components...")
    
    data_loader = DataLoader(config)
    feature_engine = FeatureEngine(config)
    
    # Model Manager
    model_path = Path(config['models']['xgboost']['model_path'])
    model_manager = AdvancedModelManager(config, str(model_path))
    
    # Signal Validator
    validator = EnhancedSignalValidator(feature_engine, model_manager)
    
    # Strategy
    strategy = XGBoostStrategy(config, feature_engine, model_manager)
    
    # Backtest Engine
    backtest = BacktestEngine(config)
    
    logger.info("  ✅ All components initialized")
    
    # 3. Load multi-timeframe data
    logger.info("\n[3/7] Loading multi-timeframe data...")
    
    test_asset = "BTC"
    
    data_dict = data_loader.load_multi_timeframe(test_asset)
    
    if any(df is None for df in data_dict.values()):
        logger.error("  ❌ Failed to load all timeframes")
        return
    
    # Use last 30 days for quick test
    df_15m = data_dict['15m'].tail(30 * 24 * 4).copy()
    df_1h = data_dict['1h'].tail(30 * 24).copy()
    df_4h = data_dict['4h'].tail(30 * 6).copy()
    df_1d = data_dict['1d'].tail(30).copy()
    
    logger.info(f"  Loaded:")
    logger.info(f"    15m: {len(df_15m):,} bars")
    logger.info(f"    1h:  {len(df_1h):,} bars")
    logger.info(f"    4h:  {len(df_4h):,} bars")
    logger.info(f"    1d:  {len(df_1d):,} bars")
    logger.info(f"  Period: {df_15m.index[0]} → {df_15m.index[-1]}")
    
    # 4. Calculate features
    logger.info("\n[4/7] Calculating multi-timeframe features...")
    
    features = feature_engine.calculate_features(df_15m, df_1h, df_4h, df_1d)
    
    logger.info(f"  Features created: {features.shape}")
    logger.info(f"  Feature names: {list(features.columns[:5])}... (showing 5/{len(features.columns)})")
    
    # Validate feature count
    if len(features.columns) != 31:
        logger.error(f"  ❌ Expected 31 features, got {len(features.columns)}")
        return
    
    logger.info(f"  ✅ Feature count matches model: 31")
    
    # 5. Generate signals
    logger.info("\n[5/7] Generating ML signals...")
    
    # Note: strategy.generate_signals_with_pj_s expects single DF
    # We need to update it for multi-TF, but for now test basic signals
    
    signals = strategy.generate_signals(df_15m)
    
    logger.info(f"  Raw signals: {signals.sum()}")
    
    # 6. Validate signals
    logger.info("\n[6/7] Validating signals...")
    
    # Generate P_j(S) values manually for now
    ml_scores = model_manager.predict_proba(features.values)[:, 1]
    pj_s_values = ml_scores.copy()
    
    validated_signals = validator.validate_batch(signals, df_15m, pj_s_values)
    
    logger.info(f"  Validated signals: {validated_signals.sum()}")
    logger.info(f"  Filtered: {signals.sum() - validated_signals.sum()}")
    
    # 7. Run backtest
    logger.info("\n[7/7] Running backtest...")
    
    results = backtest.run(df_15m, validated_signals)
    
    # Extract metrics
    metrics = results['metrics']
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    
    logger.info(f"\nCapital:")
    logger.info(f"  Initial:  ${config['trading']['initial_capital']:,.0f}")
    logger.info(f"  Final:    ${results['capital']:,.0f}")
    logger.info(f"  Return:   {metrics['total_pnl_pct']:.2f}%")
    
    logger.info(f"\nTrades:")
    logger.info(f"  Total:        {metrics['trades']}")
    logger.info(f"  Wins:         {metrics['wins']}")
    logger.info(f"  Losses:       {metrics['losses']}")
    logger.info(f"  Win Rate:     {metrics['win_rate']:.1f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    
    logger.info(f"\nRisk:")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Performance assessment
    logger.info("\n" + "="*80)
    logger.info("ASSESSMENT")
    logger.info("="*80)
    
    if metrics['profit_factor'] > 2.0:
        logger.info("✅ EXCELLENT - PF > 2.0")
    elif metrics['profit_factor'] > 1.5:
        logger.info("✅ GOOD - PF > 1.5")
    elif metrics['profit_factor'] > 1.0:
        logger.info("⚠️  MARGINAL - PF > 1.0")
    else:
        logger.info("❌ LOSING - PF < 1.0")
    
    # Feature analysis
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINE V2 VALIDATION")
    logger.info("="*80)
    
    logger.info(f"Feature count:     {len(features.columns)}")
    logger.info(f"Expected:          31")
    logger.info(f"Match:             {'✅ YES' if len(features.columns) == 31 else '❌ NO'}")
    
    logger.info(f"\nTimeframes used:")
    logger.info(f"  15m features: {len([c for c in features.columns if c.startswith('15m_')])}")
    logger.info(f"  1h features:  {len([c for c in features.columns if c.startswith('1h_')])}")
    logger.info(f"  4h features:  {len([c for c in features.columns if c.startswith('4h_')])}")
    logger.info(f"  1d features:  {len([c for c in features.columns if c.startswith('1d_')])}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ БЛОК 3 ШАГ 2 ЗАВЕРШЁН!")
    logger.info("="*80)
    
    return results

if __name__ == "__main__":
    results = main()