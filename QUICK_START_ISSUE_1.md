# Quick Start: Issue #1 Implementation

## 🚀 Overview

Issue #1 restructures the Sneaker pipeline into a clean, modular system with 8 sub-issues across 3 phases.

## 📚 Documentation

1. **PIPELINE_DESIGN.md** - Complete architectural design (READ THIS FIRST)
2. **ISSUE_1_EPIC.md** - Main epic issue description
3. **ISSUE_1_COMPLETE_BREAKDOWN.md** - All 8 sub-issues detailed
4. **issues/ISSUE_1.1_*.md** - Individual sub-issue specs

## ⚡ Quick Implementation Path

### Step 1: Create GitHub Issues

```bash
# Create main epic
gh issue create --title "[EPIC] Pipeline Restructuring" \
  --body-file ISSUE_1_EPIC.md

# Create all 8 sub-issues (see ISSUE_1_COMPLETE_BREAKDOWN.md for commands)
```

### Step 2: Create Main Branch

```bash
git checkout -b issue-1-pipeline-restructuring
```

### Step 3: Execute Phases Sequentially

**Phase 1: Data Collection** (Issues #1.1-#1.4)
```bash
# Create download scripts
# Run downloads
# Verify data files exist
```

**Phase 2: Feature Engineering** (Issues #1.5-#1.6)
```bash
# Split sneaker/features.py → features_shared.py + features_training.py
# Create feature scripts
# Run feature engineering
# Verify feature files
```

**Phase 3: Training & Prediction** (Issues #1.7-#1.8)
```bash
# Refactor training script
# Refactor prediction script
# Run complete pipeline
# Validate results
```

### Step 4: Validate & Merge

```bash
# Run validation
.venv/bin/python scripts/validate_model.py --issue 1

# Run backtest
.venv/bin/python scripts/backtest.py --issue 1 --pair LINKUSDT --hours 256

# If passed, commit proof
git add proof/issue-1/
git commit -m "Add #1: Pipeline restructuring validation proof"

# Merge to main
git checkout main
git merge issue-1-pipeline-restructuring
git push origin main
```

## 📊 Expected Results

**After completion:**
- ✅ 8 new scripts (`01-08_*.py`)
- ✅ 3 new modules (`features_shared.py`, `features_training.py`, `macro.py`)
- ✅ Data cached in `data/raw/`, `data/features/`
- ✅ Model in `models/issue-1/`
- ✅ Proof in `proof/issue-1/`
- ✅ Clean, modular pipeline ready for experimentation

## 🎯 Sub-Issue Execution Order

```
#1.1 → Download training Binance (20 pairs)
#1.2 → Download training macro (yfinance)
#1.3 → Download prediction Binance (LINKUSDT)
#1.4 → Download prediction macro (yfinance)
       ↓
#1.5 → Add shared features (training & prediction)
       ↓
#1.6 → Add training-only features (targets, signals)
       ↓
#1.7 → Train model with proof
       ↓
#1.8 → Predict with visualization
```

## 💡 Key Design Principles

1. **Raw data stays raw** - Never modify downloads
2. **Shared features identical** - Same code for training & prediction
3. **Everything cached** - Save intermediate results
4. **Modular components** - Easy to swap and test
5. **Complete proof** - Visual evidence for every step

## ⚙️ New Pipeline Flow

```
RAW DATA (once) → SHARED FEATURES (cached) → SPLIT
                                              ├─> TRAINING FEATURES → TRAIN → MODEL
                                              └─> PREDICTION ONLY ──────────────> PREDICT
```

## 🔧 Files Created/Modified

**New Scripts (8):**
- `scripts/01-04_download_*.py` (data collection)
- `scripts/05-06_add_*_features.py` (feature engineering)
- `scripts/07_train_model.py` (refactored training)
- `scripts/08_predict.py` (refactored prediction)

**New Modules (3):**
- `sneaker/features_shared.py` (45 shared features)
- `sneaker/features_training.py` (40 training-only features)
- `sneaker/macro.py` (yfinance integration)

**Modified:**
- `sneaker/data.py` (add long-term/short-term download functions)

## 📏 Success Metrics

- ✅ Complete pipeline runs end-to-end
- ✅ All intermediate results cached
- ✅ Shared features produce identical output
- ✅ Training/prediction cleanly separated
- ✅ Model validation passes
- ✅ Prediction visualization generated
- ✅ All proof in `proof/issue-1/`

## 🚨 Critical Requirements

1. **Feature Parity** - Shared features MUST be identical
2. **No Data Leakage** - Training features NEVER in prediction
3. **Timestamps Aligned** - Crypto and macro data synced
4. **Caching Works** - Can re-run any step without full recompute

## ⏱️ Timeline

- **Phase 1:** 1-2 days (data collection)
- **Phase 2:** 2-3 days (feature engineering)
- **Phase 3:** 2-3 days (training & prediction)
- **Total:** 5-8 days

## 🎓 Learning Outcome

**You will gain:**
- Clean, modular pipeline architecture
- Fast iteration capability (cached results)
- Easy feature testing framework
- Solid foundation for experimentation
- Complete audit trail of changes

---

**Start with PIPELINE_DESIGN.md for complete architecture understanding, then execute sub-issues sequentially!**
