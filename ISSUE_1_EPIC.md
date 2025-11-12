# Issue #1: Pipeline Restructuring - Epic

## 🎯 Goal

Restructure the Sneaker pipeline into a clean, modular system with clear separation between training and prediction, while maintaining a solid backbone that can be easily adapted for experimentation.

## 📖 Full Design Document

See `PIPELINE_DESIGN.md` for complete architectural overview.

## 🏗️ Architecture Vision

```
RAW DATA (cached)
    ↓
SHARED FEATURES (identical for training & prediction, cached)
    ↓
SPLIT
├─> TRAINING FEATURES (targets, signals, cached) → TRAIN → MODEL
└─> PREDICTION ONLY (no targets) ──────────────────────────────> PREDICT
```

## 🎯 Core Principles

1. **Separation of Concerns** - Training vs prediction clearly separated
2. **Modularity** - Easy to swap features, test variations
3. **Caching** - All intermediate results saved (fast iteration)
4. **Flexibility** - Easy to change pairs, features, experiments
5. **Rigor** - Maintain validation requirements

## 📋 Sub-Issues (8 Total)

### Phase 1: Data Collection (Issues #1.1-#1.4)

- **#1.1** - Download training Binance data (20 pairs, long-term)
- **#1.2** - Download training macro data (yfinance, long-term)
- **#1.3** - Download prediction Binance data (LINKUSDT, short-term)
- **#1.4** - Download prediction macro data (yfinance, short-term)

### Phase 2: Feature Engineering (Issues #1.5-#1.6)

- **#1.5** - Implement shared features pipeline (training & prediction)
- **#1.6** - Implement training-only features pipeline (targets, signals)

### Phase 3: Model & Prediction (Issues #1.7-#1.8)

- **#1.7** - Refactor training script with proof generation
- **#1.8** - Refactor prediction script with visualization

## 🗂️ New Directory Structure

```
data/
├── raw/                    # Raw downloads (immutable)
│   ├── training/
│   │   ├── binance_20pairs_1H.json
│   │   └── macro_training.json
│   └── prediction/
│       ├── binance_LINKUSDT_live.json
│       └── macro_prediction.json
├── features/               # Cached feature engineering
│   ├── training_shared_features.json
│   ├── prediction_shared_features.json
│   └── training_only_features.json
└── prepared/               # Final datasets
    ├── training_dataset.json
    └── prediction_dataset.json

models/
└── issue-1/
    └── model.txt

proof/
└── issue-1/
    ├── training_*.png
    ├── validation_*.png
    └── prediction_*.png
```

## 📜 New Scripts (8 Total)

1. `01_download_training_binance.py` - 20 pairs, 2021-2025
2. `02_download_training_macro.py` - SPY, VIX, DXY, etc.
3. `03_download_prediction_binance.py` - LINKUSDT, 256-512h
4. `04_download_prediction_macro.py` - Recent macro
5. `05_add_shared_features.py` - Features for both pipelines
6. `06_add_training_features.py` - Targets, signals, splits
7. `07_train_model.py` - Train with proof
8. `08_predict.py` - Predict with visualization

## 🔑 Key Code Refactoring

### Create New Modules

- `sneaker/features_shared.py` - Shared features (RSI, MACD, etc.)
- `sneaker/features_training.py` - Training-only (targets, signals)
- `sneaker/macro.py` - Macro data integration (yfinance)

### Refactor Existing

- `sneaker/data.py` - Add long-term/short-term download functions
- `sneaker/features.py` - Split into shared vs training-only
- `scripts/04_train_model.py` → `scripts/07_train_model.py`
- `scripts/05_predict.py` → `scripts/08_predict.py`

## ✅ Success Criteria

**Issue #1 complete when:**

1. ✅ All 8 sub-issues completed and validated
2. ✅ Complete pipeline runs: raw data → trained model → predictions
3. ✅ All intermediate results cached (no unnecessary recalculation)
4. ✅ Training and prediction features cleanly separated
5. ✅ Shared features IDENTICAL in both pipelines
6. ✅ Easy to swap features, pairs, or run experiments
7. ✅ Model validation passes (Signal R² ≥ 70%, etc.)
8. ✅ Prediction visualization generated (256h LINKUSDT with signals)
9. ✅ All outputs in `proof/issue-1/`
10. ✅ Documentation complete (PIPELINE_DESIGN.md updated)

## 🚨 Critical Requirements

### Data Integrity
- Raw data stays raw (never modified)
- All transformations cached
- Timestamps properly aligned

### Feature Parity
- Shared features IDENTICAL calculation
- No data leakage (training features never in prediction)
- No future peeking in prediction

### Validation
- Pass all validation criteria
- Visual inspection of proof plots
- Backtest on LINKUSDT 256h

## 🎯 Expected Outcomes

**After completion, you will have:**
- ✅ Modular, flexible pipeline
- ✅ Fast iteration (cached intermediate results)
- ✅ Easy feature testing
- ✅ Clean separation (training vs prediction)
- ✅ Macro data integrated
- ✅ Flexible pair selection
- ✅ Complete proof generation
- ✅ Solid foundation for experimentation

## 📊 Validation Plan

### Phase 1 Validation (After #1.1-#1.4)
- ✅ Raw data files exist and are readable
- ✅ Data shapes correct (timestamps, pairs, etc.)
- ✅ No missing data, gaps, or NaNs in raw downloads

### Phase 2 Validation (After #1.5-#1.6)
- ✅ Feature files generated correctly
- ✅ Shared features IDENTICAL on sample data
- ✅ Training features contain targets
- ✅ No NaN values in feature datasets

### Phase 3 Validation (After #1.7-#1.8)
- ✅ Model training completes successfully
- ✅ Validation metrics pass thresholds
- ✅ Prediction runs on 256h LINKUSDT
- ✅ Visualization generated correctly
- ✅ All proof files in `proof/issue-1/`

## 🔗 Dependencies

**Sub-issue dependencies:**
- #1.5 depends on #1.1, #1.2, #1.3, #1.4 (needs raw data)
- #1.6 depends on #1.5 (needs shared features)
- #1.7 depends on #1.6 (needs training features)
- #1.8 depends on #1.7 (needs trained model)

**Execution order:**
1. Run #1.1-#1.4 in parallel (data downloads)
2. Run #1.5 (shared features on all data)
3. Run #1.6 (training features)
4. Run #1.7 (training)
5. Run #1.8 (prediction)

## 📖 References

- **PIPELINE_DESIGN.md** - Complete architectural design
- **WORKFLOW.md** - How to execute sub-issues
- **VALIDATION.md** - Pass/fail criteria

## 🚀 Next Steps After Issue #1

Once baseline modular pipeline works:
- Issue #2: Test feature variations
- Issue #3: Test different target definitions
- Issue #4: Test different pairs
- Issue #5: Add walk-forward validation

## 💬 Discussion

**Why this restructuring?**
- Current pipeline mixes training and prediction
- Hard to test features independently
- No macro data integration
- Not modular or flexible
- Difficult to experiment

**Benefits:**
- Clean separation enables rapid experimentation
- Cached results speed up iteration
- Easy to track what works (proof folders)
- Solid foundation for future improvements

---

**This is a major refactoring but necessary for long-term success and experimentation.**
