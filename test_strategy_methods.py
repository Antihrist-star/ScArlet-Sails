"""
ДИАГНОСТИКА СТРАТЕГИЙ
Проверка какие методы есть у стратегий
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("="*80)
print("ТЕСТ: Проверка методов стратегий")
print("="*80)

# Test RuleBasedStrategy
try:
    from strategies.rule_based_v2 import RuleBasedStrategy
    strategy = RuleBasedStrategy()
    
    print("\n✅ RuleBasedStrategy imported")
    print(f"   Type: {type(strategy)}")
    
    # Check methods
    methods = [m for m in dir(strategy) if not m.startswith('_')]
    print(f"   Available methods ({len(methods)}): {methods[:10]}")
    
    # Check specific methods
    print(f"   has 'generate_signal': {hasattr(strategy, 'generate_signal')}")
    print(f"   has 'generate_signals': {hasattr(strategy, 'generate_signals')}")
    print(f"   has 'get_signal': {hasattr(strategy, 'get_signal')}")
    
except Exception as e:
    print(f"❌ RuleBasedStrategy failed: {e}")
    import traceback
    traceback.print_exc()

# Test XGBoostMLStrategy
try:
    from strategies.xgboost_ml_v2 import XGBoostMLStrategy
    strategy = XGBoostMLStrategy()
    
    print("\n✅ XGBoostMLStrategy imported")
    print(f"   Type: {type(strategy)}")
    
    methods = [m for m in dir(strategy) if not m.startswith('_')]
    print(f"   Available methods ({len(methods)}): {methods[:10]}")
    
    print(f"   has 'generate_signal': {hasattr(strategy, 'generate_signal')}")
    print(f"   has 'generate_signals': {hasattr(strategy, 'generate_signals')}")
    
except Exception as e:
    print(f"❌ XGBoostMLStrategy failed: {e}")
    import traceback
    traceback.print_exc()

# Test HybridStrategy
try:
    from strategies.hybrid_v2 import HybridStrategy
    strategy = HybridStrategy()
    
    print("\n✅ HybridStrategy imported")
    print(f"   Type: {type(strategy)}")
    
    methods = [m for m in dir(strategy) if not m.startswith('_')]
    print(f"   Available methods ({len(methods)}): {methods[:10]}")
    
    print(f"   has 'generate_signal': {hasattr(strategy, 'generate_signal')}")
    print(f"   has 'generate_signals': {hasattr(strategy, 'generate_signals')}")
    
except Exception as e:
    print(f"❌ HybridStrategy failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("ГОТОВО!")
print("="*80)