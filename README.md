# SCARLET SAILS - Autonomous Trading System

**Status:** ✅ Core System Complete (Day 12)
**Next Phase:** Production Readiness (Week 2)
**Contact:** bigmandmitriy777@gmail.com

---

## 📊 QUICK SUMMARY

**Scarlet Sails** is a regime-aware algorithmic trading system that trades cryptocurrency mean-reversion patterns across multiple assets and timeframes.

### Key Results (Backtest):
- ✅ **Tested:** 14 assets × 4 timeframes = 56 combinations
- ✅ **Best Performer:** ALGO 15m (342.3% annual, 7.81 years of data)
- ✅ **Average (Top 10):** ~200% annual (backtest, before production adjustments)
- ⚠️ **Realistic Expectation:** 50-110% annual (after production costs)

### Strategy:
- **Entry:** RSI < 30 (oversold mean reversion)
- **Management:** Hybrid Position Manager with adaptive stops
- **Exit:** Regime-aware take-profits, trailing stops, max hold time
- **Costs:** 0.15% per trade (backtest), 0.3-0.5% expected (production)

---

## 🎯 PROJECT STATUS

### ✅ Completed (Week 1 - Day 1-12):

1. **Core Trading System**
   - HybridPositionManager with regime awareness
   - Adaptive stop-loss (ATR-based)
   - Trailing stops and partial exits
   - Max holding time management (7 days)

2. **Backtesting & Validation**
   - Master Comprehensive Audit (56 combinations)
   - Day 11 Forensic Analysis (10 hardcore tests)
   - Walk-forward validation
   - Historical crisis testing

3. **Feature Engineering**
   - Base indicators (RSI, MA, ATR, volatility)
   - Advanced features (liquidity, correlation)
   - Long-term memory features
   - Crisis detection

4. **Analysis Framework**
   - Honest cost accounting
   - Slippage estimation
   - Regime performance analysis
   - Best/worst trade forensics

### 🔄 In Progress (Week 2):

1. **Production Readiness**
   - CCXT integration for live trading
   - Order management system
   - Real slippage tracking
   - Execution failure handling

2. **Paper Trading**
   - 2-week testnet validation
   - Real cost measurement
   - Latency tracking

3. **Risk Management**
   - Dynamic position sizing
   - Portfolio correlation limits
   - Drawdown monitoring
   - Emergency stop system

---

## 📁 PROJECT STRUCTURE

```
scarlet-sails/
├── models/                    # Trading models
│   ├── hybrid_position_manager.py    # Main position manager
│   ├── regime_detector.py            # Market regime classification
│   ├── regime_gate.py                # Entry filtering (Day 12)
│   └── entry_confluence.py           # Entry scoring (Day 12)
│
├── features/                  # Feature engineering
│   ├── base_features.py              # RSI, MA, ATR, etc.
│   ├── advanced_features.py          # Liquidity, correlation
│   └── crisis_detection.py           # Crisis detection
│
├── scripts/                   # Analysis & backtest scripts
│   ├── master_comprehensive_audit.py # 56 combination test
│   ├── day11_forensic_analysis.py    # 10 hardcore tests
│   └── test_week1_improvements.py    # Week 1 validation
│
├── backtesting/               # Backtest engine
│   ├── honest_backtest.py            # Main engine
│   └── honest_backtest_v2.py         # Enhanced version
│
├── reports/                   # Generated reports
│   ├── master_audit/                 # 56 combination results
│   └── day11_forensics/              # Forensic analysis
│
├── data/
│   ├── raw/                          # OHLCV data (NOT in git)
│   ├── features/                     # Engineered features (NOT in git)
│   └── processed/                    # Processed data (NOT in git)
│
└── docs/                      # Documentation
```

---

## 🚀 QUICK START

### Prerequisites:
- Python 3.9+
- 20GB free disk space (for data)
- Git

### Installation:

```bash
# 1. Clone repository
git clone https://github.com/Antihrist-star/scarlet-sails.git
cd scarlet-sails

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (separately - not in repo)
# Contact for data access: bigmandmitriy777@gmail.com

# 4. Run master audit
python scripts/master_comprehensive_audit.py
```

### Quick Test:

```bash
# Test Week 1 improvements
python scripts/test_week1_improvements.py

# Run Day 11 forensic analysis
python scripts/day11_forensic_analysis.py
```

---

## 📊 KEY RESULTS

### Top 10 Performers (Backtest)

| Rank | Asset | Timeframe | Annual Return | Win Rate | Sharpe | Years |
|------|-------|-----------|--------------|----------|--------|-------|
| 1 | ALGO | 15m | 342.3% | 20.0% | 3.09 | 7.81 |
| 2 | ETH | 15m | 181.2% | 25.1% | 3.12 | 7.81 |
| 3 | HBAR | 15m | 175.5% | 25.0% | 2.98 | 7.81 |
| 4 | LINK | 15m | 147.5% | 22.1% | 2.96 | 7.80 |
| 5 | AVAX | 15m | 130.2% | 22.6% | 3.02 | 5.59 |
| 6 | LTC | 15m | 115.3% | 20.2% | 2.99 | 7.81 |
| 7 | LDO | 15m | 106.2% | 22.6% | 2.87 | 2.76 |
| 8 | BTC | 15m | 88.8% | 19.9% | 3.06 | 7.81 |
| 9 | SOL | 15m | 84.2% | 22.7% | 2.89 | 5.59 |
| 10 | DOT | 15m | 83.4% | 22.8% | 2.92 | 5.59 |

**⚠️ Note:** These are backtest results. Realistic production expectations: 50-110% annual.

### Reality Adjustments:

```
Backtest Average (Top 10):     ~200% annual
After cherry-pick adjustment:  -20% → 160%
After real slippage (0.3%):    -10% → 150%
After execution issues:        -8%  → 142%
After correlation reality:     ÷1.3 → 109%
After operational losses:      -10% → 99%

REALISTIC RANGE: 50-110% annual
```

---

## 🧪 VALIDATION RESULTS

### Day 11: 10 Hardcore Tests

```
✅ Passed: 6/9 tests
⚠️  Marginal: 2/9 tests
❌ Failed: 1/9 tests

Decision: ✅ EDGE EXISTS - Proceed with optimization
```

### Key Findings:
1. ✅ **OOS Validation:** +2.3% monthly (Jan-Jun 2024)
2. ✅ **Diamond Test:** 24.6% annual net return (full 9-year period)
3. ✅ **Regime Robust:** Profitable in 3/4 market regimes
4. ⚠️ **Losing Streaks:** Up to 7 losses in a row (manageable)
5. ✅ **Parameter Robust:** 14% sensitivity (acceptable)

*See: reports/day11_forensics/forensic_report.md for details*

---

## 🛠️ CORE COMPONENTS

### 1. Hybrid Position Manager
- **Regime Detection:** Bull/Bear/Sideways classification
- **Adaptive Stops:** ATR-based with regime multipliers
- **Trailing Stops:** Follow price in profit zones
- **Partial Exits:** Take profits at key levels
- **Max Hold Time:** 7 days (168 hours)

**File:** `models/hybrid_position_manager.py`

### 2. Entry System
- **Primary Signal:** RSI < 30 (mean reversion)
- **Regime Gate:** Filter entries by market regime
- **Entry Confluence:** Multi-factor scoring (planned for Week 2)

**Files:**
- `models/regime_gate.py`
- `models/entry_confluence.py`

### 3. Feature Engineering
- **Technical:** RSI, ATR, Moving Averages
- **Volatility:** Rolling std, ATR ratios
- **Volume:** Volume ratio, liquidity metrics
- **Long-term:** 200-bar memory features

**Files:**
- `features/base_features.py`
- `features/advanced_features.py`

---

## 📈 STRATEGY DETAILS

### Entry Criteria:
1. RSI < 30 (oversold)
2. Regime filter passes (not in extreme bear)
3. Anti-clustering: Min 1 hour between signals
4. Confluence score > threshold (optional, Week 2)

### Position Management:
1. **Entry:** Market order at signal
2. **Stop-Loss:** 2-3 ATR below entry (regime-dependent)
3. **Take-Profit:** 3-5% above entry (regime-dependent)
4. **Trailing:** Activate at +2%, trail at -1%
5. **Max Hold:** Exit after 7 days

### Exit Criteria:
1. Stop-loss hit → Exit 100%
2. Take-profit hit → Exit 100%
3. Trailing stop hit → Exit 100%
4. Max hold time → Exit 100%
5. Regime change → Exit 100%

---

## ⚠️ IMPORTANT LIMITATIONS

### 1. Backtest vs Reality Gap
- **Backtest:** Perfect fills, 0.15% costs
- **Reality:** Slippage, latency, partial fills, 0.3-0.5% costs
- **Impact:** -30% to -50% of backtest performance

### 2. Cherry-Picked Assets
- Some assets (ENA, ONDO) have <2 years data
- Missing bear markets = inflated returns
- Production: Focus on assets with 5+ years history

### 3. Correlation Risk
- All crypto assets follow BTC (0.85+ correlation)
- Portfolio diversification benefit limited
- Risk: All positions can lose together

### 4. High Frequency on 15m
- Best performers trade 4,000+ times/year
- Requires robust execution infrastructure
- High operational complexity

### 5. Psychological Challenge
- Win rate ~20-40% (most trades lose!)
- Expect 5-10 losing trades in a row
- Requires discipline to follow system

---

## 📚 DOCUMENTATION

### Getting Started:
- **README.md** (this file) - Project overview
- **QUICK_START_GUIDE.md** - How to resume work
- **DAY12_FINAL_SUMMARY.md** - Complete 12-day summary

### Reference:
- **FILE_INVENTORY.md** - All files in project
- **COMMIT_CHECKLIST.md** - Git workflow guide
- **reports/master_audit/comprehensive_report.txt** - Full results

### Technical:
- **docs/COMPREHENSIVE_AUDIT_GUIDE.md** - Audit framework
- **docs/architecture.md** - System architecture
- **docs/patterns_observed.md** - Trading patterns

---

## 🔐 SECURITY & RISK

### Never Commit:
- ❌ API keys, secrets (.env files)
- ❌ Private keys
- ❌ Personal data
- ❌ Large data files (>10MB)

### Always:
- ✅ Use `.gitignore`
- ✅ Test on Testnet first
- ✅ Start with small position sizes
- ✅ Monitor 24/7 in early stages

### Risk Management:
- Max 2% risk per trade (recommended)
- Max 20% total portfolio exposure
- Correlation limits (<0.9 between positions)
- Drawdown limits (stop if down >20%)

---

## 🗺️ ROADMAP

### Week 1 (Complete): ✅
- [x] Build core trading system
- [x] Backtest 56 combinations
- [x] Validate with 10 hardcore tests
- [x] Honest reality assessment

### Week 2 (In Progress): 🔄
- [ ] CCXT integration
- [ ] Paper trading (2 weeks)
- [ ] Real slippage measurement
- [ ] Risk management system

### Week 3 (Planned):
- [ ] ML entry filter (XGBoost)
- [ ] Regime detector V2
- [ ] Position sizing optimizer
- [ ] Correlation monitor

### Week 4 (Planned):
- [ ] Portfolio optimization
- [ ] Asset selection refinement
- [ ] Live trading preparation
- [ ] Monitoring dashboard

### Week 5+ (Future):
- [ ] Live trading (small size)
- [ ] Performance tracking
- [ ] Continuous optimization
- [ ] Scale up gradually

---

## 📊 PERFORMANCE EXPECTATIONS

### Conservative (50% annual):
```
Starting: $10,000
Year 1:   $15,000
Year 2:   $22,500
Year 3:   $33,750
```

### Base Case (80% annual):
```
Starting: $10,000
Year 1:   $18,000
Year 2:   $32,400
Year 3:   $58,320
```

### Optimistic (110% annual):
```
Starting: $10,000
Year 1:   $21,000
Year 2:   $44,100
Year 3:   $92,610
```

**Assumptions:**
- Proper risk management
- Execution layer working
- Real slippage ~0.3-0.5%
- No major operational failures

---

## 🤝 CONTRIBUTING

This is currently a private research project. For collaboration inquiries:
- Email: bigmandmitriy777@gmail.com

---

## ⚖️ LICENSE

All rights reserved. Private research project.

---

## 📞 CONTACT

**Email:** bigmandmitriy777@gmail.com
**GitHub:** https://github.com/Antihrist-star/scarlet-sails
**Branch:** claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH

---

## ⚠️ DISCLAIMER

**IMPORTANT:** This software is for research and educational purposes only.

- Trading cryptocurrencies carries significant risk
- Past performance does not guarantee future results
- Backtest results are theoretical and not indicative of live trading
- Only trade with money you can afford to lose
- Always use proper risk management
- Consult a financial advisor before trading

**No warranty or guarantee of profitability is provided.**

---

*Last Updated: Day 12 (2025-11-10)*
*Status: Core system complete, entering production readiness phase*
