# Sneaker: Clean Extraction from Ghost

**Date:** November 12, 2025
**Purpose:** Extract only the working core from Ghost, leave behind the mess

---

## Philosophy: Start Fresh

Ghost has become a tangled mess of:
- 94+ issues worth of experiments
- Failed approaches (voting ensembles, censored models, etc.)
- V1/V2 buggy models
- Complex infrastructure that's not being used
- Confusing documentation

**Sneaker will be:**
- Clean, simple, working code only
- One approach: V3 sample weighting
- Clear documentation
- No historical baggage

---

## What to Extract (The Good Hair)

### 1. Core Ghost Signal Detection
**From:** `test/detect_ghost_signals_volnorm.py`
**What it does:** Detects indicator momentum shifts (ghost signals)
**Why:** This is the core innovation that works

### 2. Feature Engineering (83 Enhanced V3 Features)
**From:**
- `test/add_momentum_features.py` (Batch 1: 24 features)
- `test/add_even_more_features.py` (Batch 2: 35 features)
- `test/add_issue70_statistical_features.py` (Batch 3: 4 features)
**What it does:** Adds 83 technical features to raw candles
**Why:** These features give the model predictive power

### 3. V3 Training Pipeline
**From:** `scripts/train_production_model_v3.py`
**What it does:** Trains LightGBM with 5x sample weighting for signals
**Why:** This is the ONLY approach that works (5% signal rate at 4σ)

### 4. Data Collection
**From:**
- `scripts/download_1h_data.py` (Binance 1H candles)
- `ghost/data/binance_client.py` (API wrapper)
**What it does:** Fetches historical data from Binance
**Why:** Need data to train and test

### 5. Essential Utilities
**From:**
- `ghost/logging/logger.py` (Logging setup)
- Basic indicator calculations from `ghost/features/indicators.py`
**What it does:** Infrastructure for logging and calculations
**Why:** Basic necessities

### 6. Working Model
**From:** `models/production_model_v3.txt`
**What it does:** The trained V3 model (Signal R² 74%, 5% signal rate)
**Why:** The end product that actually works

### 7. Dataset
**From:** `data_artifacts/ghost_signals_volnorm_enhanced_v3_all_pairs.json`
**What it does:** 917K candles with 83 features and ghost signal targets
**Why:** The training data

---

## What to Leave Behind (The Bad Haircut)

### 1. Failed Experiments
- ❌ All archived scripts (voting ensembles, sign/magnitude split, etc.)
- ❌ V1 and V2 models (buggy/unusable)
- ❌ Censored, quantile, Huber models (have V1 bug)
- ❌ All the training logs and experimental results

### 2. Unused Infrastructure
- ❌ Chopper classes (not used in V3)
- ❌ Dual offset aggregation (not used)
- ❌ Cross-exchange features (not used)
- ❌ Futures data (not used in V3)
- ❌ Macro indicators (not used in V3)
- ❌ Complex evaluation frameworks (not used)

### 3. Confusing Documentation
- ❌ CLAUDE.md with 94 issues of history
- ❌ Multiple experimental summaries
- ❌ V1/V2 comparison docs
- ❌ All the issue tracking

### 4. Development Cruft
- ❌ .venv directories
- ❌ __pycache__
- ❌ .DS_Store files
- ❌ Old log files
- ❌ Test notebooks

---

## Clean Sneaker Structure

```
sneaker/
├── README.md                          # Fresh, simple documentation
├── requirements.txt                   # Minimal dependencies
├── setup.py                           # Package installation
│
├── sneaker/                           # Core module (renamed from ghost)
│   ├── __init__.py
│   ├── indicators.py                  # Technical indicator calculations
│   ├── signals.py                     # Ghost signal detection (volnorm)
│   ├── features.py                    # 83 Enhanced V3 features
│   ├── data.py                        # Binance data fetching
│   ├── model.py                       # Model training and prediction
│   └── logging.py                     # Logging utilities
│
├── scripts/
│   ├── 01_collect_data.py             # Step 1: Fetch Binance data
│   ├── 02_detect_signals.py           # Step 2: Find ghost signals
│   ├── 03_add_features.py             # Step 3: Add 83 features
│   ├── 04_train_model.py              # Step 4: Train V3 model
│   └── 05_predict.py                  # Step 5: Make predictions
│
├── models/
│   └── production.txt                 # V3 trained model
│
├── data/
│   └── .gitkeep                       # Data goes here
│
└── tests/
    ├── test_indicators.py
    ├── test_signals.py
    ├── test_features.py
    └── test_model.py
```

---

## Simplified Pipeline

**Ghost had:** 8+ experimental approaches, complex infrastructure, 94 issues of cruft

**Sneaker will have:** One clean pipeline that works

```
1. Collect Data → Binance 1H candles (20 pairs)
2. Detect Signals → Find indicator momentum shifts (volnorm)
3. Add Features → 83 Enhanced V3 features
4. Train Model → LightGBM with 5x sample weighting
5. Predict → Apply model, use 4σ threshold
```

That's it. Simple. Clean. Working.

---

## Key Simplifications

### Ghost Complexity → Sneaker Simplicity

| Ghost | Sneaker |
|-------|---------|
| 3 buggy model versions (V1/V2/V3) | 1 working model (V3 only) |
| Multiple training approaches | 1 approach: sample weighting |
| Complex chopper/dual-offset classes | Simple functions |
| Futures/macro data (unused) | Only what's used |
| 94 issues of history | Fresh start |
| Confusing docs | Clear README |
| Multiple experimental features | 83 working features |

### Dependencies: Minimal

```
numpy
pandas
lightgbm
binance-client
matplotlib  # for visualizations
scipy       # for statistical features
```

That's it. No xgboost, no sklearn ensemble, no complex frameworks.

---

## Implementation Steps

1. **Create Structure** ✅
   - Set up clean directory tree
   - Initialize git repo
   - Create venv

2. **Extract Core Code**
   - Copy and clean indicator calculations
   - Extract ghost signal detection (volnorm)
   - Consolidate feature engineering into one module
   - Simplify data collection
   - Extract V3 training logic

3. **Write Fresh Documentation**
   - Simple README explaining what it does
   - How to use (5 simple steps)
   - No historical baggage

4. **Test**
   - Verify each step works independently
   - Run end-to-end pipeline
   - Validate model predictions

5. **Clean Up**
   - Remove any cruft
   - Ensure code is readable
   - Add docstrings

---

## Success Criteria

Sneaker will be considered successful when:
- [ ] Can fetch data with one command
- [ ] Can detect ghost signals with one command
- [ ] Can add all 83 features with one command
- [ ] Can train V3 model with one command
- [ ] Can make predictions with one command
- [ ] Model generates 5% signals at 4σ (like V3)
- [ ] Code is clean, readable, and documented
- [ ] No historical Ghost baggage

---

## Guiding Principles

1. **KISS:** Keep It Simple, Stupid
2. **One Way:** Only include what works (V3)
3. **No History:** Fresh start, no baggage
4. **Readable:** Code that makes sense 6 months from now
5. **Working:** Every part tested and functional

---

## Next Actions

1. Create sneaker directory structure
2. Extract and clean core modules
3. Copy V3 model and dataset
4. Write fresh README
5. Test end-to-end
6. Celebrate clean codebase 🎉
