# FILE INVENTORY - SCARLET SAILS PROJECT

**Complete list of all files in the project**
**Updated:** 2025-11-10 (Day 12)
**Purpose:** Track what exists, what's in git, what's excluded

---

## 📋 SUMMARY

```
TOTAL FILES COMMITTED TO GIT: ~80 files (~100 MB)
TOTAL DATA FILES (NOT in git): ~130 files (~20 GB)

Git Repository Size: ~100 MB
Local Project Size: ~20 GB
```

---

## ✅ CATEGORY 1: DOCUMENTATION (ROOT)

**All committed to git:**

```
✅ README.md                               # Project overview
✅ DAY12_FINAL_SUMMARY.md                  # Complete summary with honest assessment
✅ QUICK_START_GUIDE.md                    # How to resume work
✅ FILE_INVENTORY.md                       # This file
✅ COMMIT_CHECKLIST.md                     # Git workflow guide
✅ PROJECT_INVENTORY_DAY7.txt              # Day 7 inventory (150KB)
✅ requirements.txt                        # Python dependencies
✅ .gitignore                              # Git ignore rules
✅ .dvcignore                              # DVC ignore rules
✅ pyproject.toml                          # Python project config
✅ rules.yml                               # Project rules
```

**Project Documents (docs/):**
```
✅ docs/COMPREHENSIVE_AUDIT_GUIDE.md       # Audit framework guide
✅ docs/architecture.md                    # System architecture
✅ docs/patterns_observed.md               # Trading patterns
✅ docs/week2_plan.md                      # Week 2 plan
✅ docs/mac_setup_ml_engineer.md           # ML setup guide
```

**Additional Docs (various):**
```
✅ ARCHE-30-Final-Plan-v2.md               # 30-day plan
✅ WEEK2_VISIONARY_ARCHITECTURE_PLAN.md    # Architecture plan
✅ STRATEGIC-PRINCIPLE-SAFETY-EDGE__1_.md  # Strategic principles
✅ DAY6_BLOCK2_FINAL_DECISIONS.md          # Day 6 decisions
```

---

## ✅ CATEGORY 2: TRADING MODELS (models/)

**Core Position Management:**
```
✅ models/hybrid_position_manager.py       # Main position manager (16KB)
✅ models/position_manager.py              # Alternative position manager (15KB)
✅ models/exit_strategy.py                 # Exit logic (18KB)
```

**Entry & Filtering:**
```
✅ models/regime_gate.py                   # Day 12: Regime-based entry filter (4KB)
✅ models/entry_confluence.py              # Day 12: Multi-factor entry scoring (6KB)
✅ models/regime_detector.py               # Market regime detection (9KB)
```

**ML Models:**
```
✅ models/xgboost_model.py                 # XGBoost implementation (9KB)
✅ models/hybrid_model.py                  # CNN-LSTM hybrid (2KB)
✅ models/logistic_model.py                # Logistic regression (1KB)
✅ models/logistic_baseline.py             # Baseline model (3KB)
```

**Advanced Models:**
```
✅ models/governance.py                    # Trading governance (22KB)
✅ models/crisis_classifier.py             # Crisis classification (17KB)
✅ models/opportunity_scorer.py            # Opportunity scoring (16KB)
✅ models/decision_formula_v2.py           # Decision formula (12KB)
✅ models/ml_training_pipeline.py          # ML pipeline (18KB)
✅ models/bot_detector.py                  # Bot detection (15KB)
```

**Trained Models (committed):**
```
✅ models/best_cnn_model.pth               # Trained CNN (168KB)
✅ models/xgboost_model.json               # Trained XGBoost (137KB)
✅ models/scaler_X_v3.pkl                  # Feature scaler (1KB)
✅ models/scaler_y.pkl                     # Target scaler (<1KB)
✅ models/best_tp_sl_config.json           # Best TP/SL config (<1KB)
✅ models/xgboost_best_threshold.txt       # Threshold file (<1KB)
✅ models/logistic_enriched_v2_metadata.json # Metadata (<1KB)
```

---

## ✅ CATEGORY 3: FEATURE ENGINEERING (features/)

```
✅ features/__init__.py
✅ features/base_features.py               # Basic indicators (RSI, MA, etc)
✅ features/advanced_features.py           # Advanced features
✅ features/long_term_features.py          # Long-term memory features
✅ features/crisis_detection.py            # Crisis detection (Day 7)
✅ features/liquidity_monitor.py           # Liquidity monitoring (Day 7)
✅ features/portfolio_correlation.py       # Correlation tracking (Day 7)
✅ features/post_entry_validator.py        # Post-entry validation (Day 7)
✅ features/resume_validator.py            # Resume validation (Day 7)
```

---

## ✅ CATEGORY 4: ANALYSIS SCRIPTS (scripts/)

**Master Audit Scripts (Day 11-12):**
```
✅ scripts/master_comprehensive_audit.py   # Day 12: 56 combination audit (16KB)
✅ scripts/day11_forensic_analysis.py      # Day 11: 10 hardcore tests (27KB)
✅ scripts/test_week1_improvements.py      # Week 1 test (10KB)
```

**Comprehensive Audit (Phase 0-4):**
```
✅ scripts/run_comprehensive_audit.py      # Main audit runner (5KB)
✅ scripts/phase0_load_real_data.py        # Phase 0 (8KB)
✅ scripts/phase1_1_validate_crisis_detection.py    # Phase 1.1 (9KB)
✅ scripts/phase1_2_validate_regime_detection.py    # Phase 1.2 (9KB)
✅ scripts/phase1_3_validate_entry_signals.py       # Phase 1.3 (10KB)
✅ scripts/phase1_5_validate_ml_models.py           # Phase 1.5 (7KB)
✅ scripts/phase2_walk_forward_validation.py        # Phase 2 (11KB)
✅ scripts/phase3_root_cause_analysis.py            # Phase 3 (9KB)
✅ scripts/phase4_decision_matrix.py                # Phase 4 (10KB)
✅ scripts/verify_system_integrity.py      # System check (10KB)
```

**ML Training & Optimization:**
```
✅ scripts/train_xgboost.py                # XGBoost training (11KB)
✅ scripts/optimize_tp_sl_xgboost.py       # TP/SL optimization (6KB)
✅ scripts/final_backtest_xgboost.py       # Final backtest (5KB)
✅ scripts/model_comparison.py             # Model comparison (5KB)
✅ scripts/optimize_threshold_by_pf.py     # Threshold optimization (9KB)
✅ scripts/optimize_threshold_by_pf_v2.py  # Threshold optimization v2 (9KB)
```

**Data Preparation:**
```
✅ scripts/prepare_data.py
✅ scripts/prepare_data_v2.py
✅ scripts/prepare_data_v3.py
✅ scripts/prepare_data_v4.py
✅ scripts/prepare_data_with_features_v2.py
✅ scripts/prepare_enhanced_features.py
✅ scripts/clean_nan_data.py
```

**Backtesting:**
```
✅ scripts/run_backtest.py
✅ scripts/run_backtest_v2.py
✅ scripts/run_backtest_xgboost.py
✅ scripts/run_backtest_enriched_v2.py
✅ scripts/comprehensive_exit_test_REAL.py  # Exit strategy test (11KB)
```

**Testing & Validation:**
```
✅ scripts/test_historical_crises.py       # Crisis testing (12KB)
✅ scripts/mutable_phases_testing.py       # Phase testing (29KB)
✅ scripts/walk_forward_test.py            # Walk-forward (8KB)
```

**Data Download:**
```
✅ scripts/fetch_ohlcv.py
✅ scripts/download_full_history.py
✅ scripts/validate_downloaded_data.py
```

**Analysis Tools:**
```
✅ scripts/analyze_market_regime.py
✅ scripts/analyze_periods.py
✅ scripts/pattern_analyzer.py
✅ scripts/check_data_quality.py
✅ scripts/check_leakage.py
✅ scripts/audit_look_ahead.py
```

**Training:**
```
✅ scripts/train_model.py
✅ scripts/train_model_v2.py
✅ scripts/train_model_experiment.py
✅ scripts/train_model_enriched_v2.py
✅ scripts/train_v5_improved.py
✅ scripts/train_swing_models.py
✅ scripts/train_swing_simple.py
```

**Utilities:**
```
✅ scripts/debug_data.py
✅ scripts/quick_data_check.py
✅ scripts/gen_status.py
✅ scripts/tg_alert.py
✅ scripts/uptime_check.py
✅ scripts/test_model.py
✅ scripts/test_model_with_data.py
✅ scripts/torch_gpu_test.py
```

---

## ✅ CATEGORY 5: BACKTESTING ENGINE (backtesting/)

```
✅ backtesting/__init__.py
✅ backtesting/honest_backtest.py          # Main backtest engine
✅ backtesting/honest_backtest_v2.py       # Backtest v2
✅ backtesting/README.md                   # Backtest docs
```

---

## ✅ CATEGORY 6: TESTS (tests/)

```
✅ tests/__init__.py
✅ tests/test_data_integrity.py
✅ tests/test_features_msi.py
```

---

## ✅ CATEGORY 7: REPORTS (reports/)

**Master Audit (Day 12):**
```
✅ reports/master_audit/raw_results.json          # All 56 results (34KB)
✅ reports/master_audit/comprehensive_report.txt  # Summary report (6KB)
```

**Day 11 Forensics:**
```
✅ reports/day11_forensics/forensic_report.md     # Full report (2KB)
✅ reports/day11_forensics/test_results.json      # Test results (<1KB)
✅ reports/day11_forensics/best_10_trades.csv     # Best trades (2KB)
✅ reports/day11_forensics/worst_10_trades.csv    # Worst trades (2KB)
✅ reports/day11_forensics/all_trades_detailed.csv # All trades (1.4MB)
```

**Other Reports (committed):**
```
✅ reports/grid_search_results_day9.json           # Grid search (3KB)
✅ reports/validation_results_day9.json            # Validation (2KB)
✅ reports/final_xgboost_trades.csv                # Final trades (3KB)
✅ reports/xgboost_backtest_trades.csv             # Backtest trades (1KB)
✅ reports/xgboost_046_trades.csv                  # XGBoost trades (2KB)
✅ reports/xgboost_tp_sl_optimization.csv          # TP/SL optimization (2KB)
✅ reports/tp_sl_optimization_results.csv          # TP/SL results (3KB)
✅ reports/threshold_optimization_v2.csv           # Threshold optimization (3KB)
✅ reports/week3_final_summary.csv                 # Week 3 summary (<1KB)
✅ reports/week2_validation_results.csv            # Week 2 results (3KB)
```

**Walk-Forward Results:**
```
✅ reports/walk_forward_results_20251007_181719.json
✅ reports/walk_forward_results_20251007_181919.json
```

**Day Reports:**
```
✅ reports/day1_final_report.md
✅ reports/day1_status.md
✅ reports/today_status.md
✅ reports/week1_final_report.md
✅ reports/data_quality_day1.txt
✅ reports/day4_final_metrics.json
✅ reports/day5_experiment_metrics.json
✅ reports/day5_v4_metrics.json
✅ reports/day5_v5_metrics.json
✅ reports/swing_3d_simple.json
```

**Visualizations (PNG - committed despite size):**
```
✅ reports/regime_detection_analysis.png   # 504KB
✅ reports/final_xgboost_equity.png        # 98KB
✅ reports/xgboost_046_equity.png          # 98KB
✅ reports/xgboost_backtest_equity_curve.png # 90KB
✅ reports/backtest_enriched_v2_results.png # 134KB
✅ reports/backtest_v2_results.png         # 129KB
✅ reports/btc_full_history.png            # 72KB
✅ reports/period_analysis.png             # 41KB
✅ reports/threshold_optimization_v2.png   # 321KB
```

---

## ✅ CATEGORY 8: CONFIGURATION (configs/)

```
✅ configs/market_config.yaml              # Market configuration
```

---

## ✅ CATEGORY 9: DATA MANIFESTS (data/)

**DVC Files (Data Version Control):**
```
✅ data/raw/BTCUSDT_15m.parquet.dvc
✅ data/raw/BTCUSDT_1h.parquet.dvc
✅ data/raw/BTCUSDT_1m.parquet.dvc
✅ data/raw/ETHUSDT_15m.parquet.dvc
✅ data/raw/ETHUSDT_1h.parquet.dvc
✅ data/raw/ETHUSDT_1m.parquet.dvc
✅ data/raw/SOLUSDT_15m.parquet.dvc
✅ data/raw/SOLUSDT_1h.parquet.dvc
✅ data/raw/SOLUSDT_1m.parquet.dvc
```

**Metadata (Small JSON files):**
```
✅ data/raw/data_manifest.json
✅ data/metrics/collection_summary.json
✅ data/mhi_components/mhi_components_summary.json
✅ data/mhi_components/mhi_current_state.json
```

---

## ❌ CATEGORY 10: DATA FILES (NOT IN GIT)

**Raw Data (~10-15 GB):**
```
❌ data/raw/BTCUSDT_15m.parquet
❌ data/raw/BTCUSDT_1h.parquet
❌ data/raw/BTCUSDT_4h.parquet
❌ data/raw/BTCUSDT_1d.parquet
❌ data/raw/ETHUSDT_*.parquet (×4)
❌ data/raw/SOLUSDT_*.parquet (×4)
❌ data/raw/LINKUSDT_*.parquet (×4)
❌ data/raw/LDOUSDT_*.parquet (×4)
❌ data/raw/SUIUSDT_*.parquet (×4)
❌ data/raw/HBARUSDT_*.parquet (×4)
❌ data/raw/ENAUSDT_*.parquet (×4)
❌ data/raw/ALGOUSDT_*.parquet (×4)
❌ data/raw/AVAXUSDT_*.parquet (×4)
❌ data/raw/DOTUSDT_*.parquet (×4)
❌ data/raw/LTCUSDT_*.parquet (×4)
❌ data/raw/ONDOUSDT_*.parquet (×4)
❌ data/raw/UNIUSDT_*.parquet (×4)

TOTAL: 56 parquet files (14 assets × 4 timeframes)
```

**Features (~5-8 GB):**
```
❌ data/features/*.parquet (56 files)
❌ data/features/*.pkl
```

**Processed Data (~2-3 GB):**
```
❌ data/processed/*.parquet
❌ data/processed/*.pt
❌ data/processed/*.pkl
❌ data/processed/*.npy
❌ data/processed/btc_prepared_phase0.parquet  # Day 11 used this
```

**Prepared Data:**
```
❌ data/prepared/*.parquet
❌ data/prepared/*.pkl
```

**Other Large Data:**
```
❌ data/market_metrics/*.parquet
❌ data/metrics/*.parquet (except summaries)
❌ data/mhi_components/*.parquet (except summaries)
```

---

## 📊 DISK SPACE BREAKDOWN

```
COMMITTED TO GIT (~100 MB):
├── Documentation: 15 MB (PDFs, markdown)
├── Python code: 5 MB (all .py files)
├── Reports (summaries): 50 MB (JSONs, CSVs, small PNGs)
├── Trained models: 25 MB (PyTorch, XGBoost, scalers)
└── Config: 5 MB (YAML, JSON, requirements.txt)

NOT IN GIT (~20 GB):
├── Raw data: 10-15 GB (56 parquet files)
├── Features: 5-8 GB (56 parquet files)
├── Processed: 2-3 GB (various formats)
└── Temp/Cache: variable
```

---

## 🔍 HOW TO VERIFY FILES EXIST

### Check Git Files:
```powershell
# List all committed files
git ls-tree -r --name-only HEAD

# Count files
git ls-tree -r --name-only HEAD | wc -l
# Expected: ~80 files
```

### Check Data Files (Local):
```powershell
# Windows
dir data\raw\*.parquet
dir data\features\*.parquet

# Linux/Mac
ls -lh data/raw/*.parquet
ls -lh data/features/*.parquet
```

### Check Disk Usage:
```powershell
# Windows
dir data\raw /s

# Linux/Mac
du -sh data/raw/
du -sh data/features/
du -sh data/processed/
```

---

## 📝 NOTES

### Data Not in Git Because:
1. **Size:** 20GB would make git unusable
2. **Binary:** Parquet files are binary (not diffable)
3. **Regenerable:** Can be re-downloaded from exchange
4. **Local-only:** Each developer downloads their own copy

### Using DVC (Data Version Control):
```powershell
# DVC is configured but not fully utilized yet
# .dvc files track data file metadata without storing the data in git

# To restore data files (if DVC remote is set up):
dvc pull

# To track new data files:
dvc add data/raw/newfile.parquet
git add data/raw/newfile.parquet.dvc
```

---

## ✅ COMMIT CHECKLIST

Before committing to git:

```
☐ Check .gitignore is up to date
☐ Verify no .parquet files staged (except .dvc files)
☐ Verify no secrets in committed files (.env, API keys)
☐ Check file sizes (nothing >10MB except trained models)
☐ Test code runs (at least imports work)
☐ Update this FILE_INVENTORY.md if structure changed
```

---

**Last Updated:** Day 12 (2025-11-10)

**Total Files Tracked:** ~80 files in git + ~130 data files locally

**Repository:** https://github.com/Antihrist-star/scarlet-sails
**Branch:** claude/debug-naive-strategy-performance-011CUx4E9miJvu4k8Sn2gBkH
