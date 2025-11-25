"""
Модуль стратегий для торгового бота
"""

from .simple_strategies import (
    SimpleRSIStrategy,
    SimpleMAStrategy,
    SimpleBollingerStrategy,
    CombinedStrategy,
    RuleBasedStrategy,
    HybridStrategy
)

all = [
    'SimpleRSIStrategy',
    'SimpleMAStrategy', 
    'SimpleBollingerStrategy',
    'CombinedStrategy',
    'RuleBasedStrategy',
    'HybridStrategy'
]