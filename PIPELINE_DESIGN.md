# Pipeline Restructuring Design - Issue #1

## 🎯 Vision: Modular, Flexible, Clean Pipeline

**Goal:** Separate training and prediction pipelines with shared core components. Make feature testing, experimentation, and adaptation easy while keeping the core intact.

---

## 🏗️ Architecture Overview

### Core Principle: Separation of Concerns

```
RAW DATA → SHARED FEATURES → SPLIT
                              ├─> TRAINING FEATURES → TRAIN → MODEL
                              └─> PREDICTION ONLY ────────────> PREDICT
```

### Key Design Decisions

1. **Raw data stays raw** - Download once, reuse forever
2. **Shared features identical** - Same calculation for training and prediction
3. **Training features separate** - Target calculation, turning points, validation splits
4. **Everything cached** - Save intermediate results to files
5. **Modular components** - Easy to swap, test, adapt

---

## 📁 New Directory Structure

```
data/
├── raw/                                  # Raw downloads (never modified)
│   ├── training/
│   │   ├── binance_20pairs_1H.json      # Long-term: 20 pairs, 2021-2025
│   │   └── macro_training.json          # Long-term: yfinance macro data
│   └── prediction/
│       ├── binance_LINKUSDT_live.json   # Short-term: 256-512 hours
│       └── macro_prediction.json        # Short-term: recent macro
│
├── features/                             # Feature engineering (cached)
│   ├── training_shared_features.json    # Shared features on training data
│   ├── prediction_shared_features.json  # Shared features on prediction data
│   └── training_only_features.json      # Training-only features added
│
└── prepared/                             # Ready for model (final datasets)
    ├── training_dataset.json            # Complete training data with targets
    └── prediction_dataset.json          # Complete prediction data (no targets)

models/
└── issue-X/                              # Models by issue
    └── model.txt                         # Trained LightGBM model

proof/
└── issue-X/                              # Validation evidence
    ├── training_*.png                    # Training visualizations
    ├── validation_*.png                  # Validation plots
    └── prediction_*.png                  # Prediction results
```

---

## 🔄 Complete Pipeline Flow

### Phase 1: Raw Data Collection (Run Once)

```bash
# Training data (long-term)
scripts/01_download_training_binance.py  → data/raw/training/binance_20pairs_1H.json
scripts/02_download_training_macro.py    → data/raw/training/macro_training.json

# Prediction data (short-term, refreshable)
scripts/03_download_prediction_binance.py → data/raw/prediction/binance_LINKUSDT_live.json
scripts/04_download_prediction_macro.py   → data/raw/prediction/macro_prediction.json
```

**Characteristics:**
- Downloads raw data as-is from sources
- No processing, no features, no calculations
- Saved in readable JSON format
- Can be downloaded once and reused for multiple experiments
- Flexible pair selection (LINKUSDT now, easily changeable)

### Phase 2: Shared Feature Engineering

```bash
scripts/05_add_shared_features.py
  ├─> Reads: data/raw/training/binance_20pairs_1H.json
  ├─> Reads: data/raw/training/macro_training.json
  └─> Writes: data/features/training_shared_features.json

scripts/05_add_shared_features.py (same script, different input!)
  ├─> Reads: data/raw/prediction/binance_LINKUSDT_live.json
  ├─> Reads: data/raw/prediction/macro_prediction.json
  └─> Writes: data/features/prediction_shared_features.json
```

**Shared Features (Used in Both Training & Prediction):**
- Technical indicators (RSI, MACD, BB, etc.)
- Price-based features (ROC, acceleration, etc.)
- Volume features (vol_ratio, etc.)
- Macro features (from yfinance)
- Normalization (if any)

**Critical:** EXACT SAME CALCULATION for both training and prediction!

### Phase 3: Training-Only Features

```bash
scripts/06_add_training_features.py
  ├─> Reads: data/features/training_shared_features.json
  └─> Writes: data/features/training_only_features.json
```

**Training-Only Features (NOT in Prediction):**
- Target calculation (future price changes, reversals)
- Turning point detection
- Signal labeling (ghost signals)
- Volatility normalization of targets
- Forward-looking features (for target definition only)
- Train/validation/test split indicators

**Why separate?** These features cannot exist in live prediction (no future data).

### Phase 4: Training Pipeline

```bash
scripts/07_train_model.py --issue X
  ├─> Reads: data/features/training_only_features.json
  ├─> Trains: LightGBM with V3 sample weighting
  ├─> Saves: models/issue-X/model.txt
  └─> Generates: proof/issue-X/training_*.png (regression, residuals, etc.)
```

**Outputs:**
- Trained model file
- Training metrics (Signal R², Direction accuracy, etc.)
- Validation plots (regression, residuals, feature importance)
- All saved to proof/issue-X/

### Phase 5: Prediction Pipeline

```bash
scripts/08_predict.py --issue X --pair LINKUSDT --hours 256
  ├─> Reads: models/issue-X/model.txt
  ├─> Reads: data/features/prediction_shared_features.json
  ├─> Predicts: Signal strength for each hour
  └─> Generates: proof/issue-X/prediction_*.png (price chart with signals)
```

**Outputs:**
- Predictions for recent 256 hours
- Visualization of price + signals
- Buy/sell markers on chart
- Saved to proof/issue-X/

---

## 🧩 Modularity & Flexibility

### Easy Feature Testing

**Want to test a new indicator?**
1. Modify `scripts/05_add_shared_features.py`
2. Re-run on both training and prediction data
3. Re-train model
4. Compare results (proof folders)

**Want to test different target definition?**
1. Modify `scripts/06_add_training_features.py`
2. Re-run to generate new targets
3. Re-train model
4. Compare results

### Easy Pair Switching

**Want to predict BTCUSDT instead of LINKUSDT?**
1. Edit `scripts/03_download_prediction_binance.py` (change pair)
2. Re-download prediction data
3. Re-run shared features
4. Predict using existing model
5. Compare results

### Easy Experiment Tracking

**Each experiment gets an issue number:**
```
proof/
├── issue-1/  (baseline)
├── issue-5/  (added RSI divergence feature)
├── issue-10/ (changed target definition)
└── issue-15/ (tested BTCUSDT instead of LINKUSDT)
```

**Complete history of what worked and what didn't!**

---

## 🔑 Key Code Changes Required

### 1. Split `sneaker/features.py`

**Current:** All 83 features in one place

**New:**
```python
# sneaker/features_shared.py
def add_shared_features(df):
    """Features used in BOTH training and prediction."""
    df = add_technical_indicators(df)  # RSI, MACD, etc.
    df = add_price_features(df)        # ROC, acceleration
    df = add_volume_features(df)       # Vol ratio
    df = add_macro_features(df)        # Macro data
    return df

# sneaker/features_training.py
def add_training_features(df):
    """Features ONLY used in training (targets, signals, etc.)."""
    df = calculate_targets(df)         # Future price changes
    df = detect_turning_points(df)     # Reversals
    df = normalize_targets(df)         # Volatility normalization
    df = add_split_indicators(df)      # Train/val/test
    return df
```

### 2. Create Macro Data Integration

**New file:** `sneaker/macro.py`
```python
def download_macro_data(start_date, end_date):
    """Download yfinance macro indicators."""
    # SPY, VIX, DXY, etc.
    return macro_df

def merge_macro_with_crypto(crypto_df, macro_df):
    """Merge macro data with crypto OHLCV."""
    # Align timestamps, forward-fill, etc.
    return merged_df
```

### 3. Refactor `sneaker/data.py`

**Add:**
```python
def download_training_data(pairs, start_date, end_date):
    """Download long-term data for training (20 pairs)."""
    pass

def download_prediction_data(pair, hours):
    """Download short-term data for prediction (single pair)."""
    pass

def save_raw_data(df, filepath):
    """Save raw OHLCV data to JSON."""
    pass

def load_raw_data(filepath):
    """Load raw OHLCV data from JSON."""
    pass
```

### 4. Create New Scripts

**8 new scripts to implement pipeline:**

1. `01_download_training_binance.py` - 20 pairs, long-term
2. `02_download_training_macro.py` - Macro data, long-term
3. `03_download_prediction_binance.py` - Single pair, short-term
4. `04_download_prediction_macro.py` - Macro data, short-term
5. `05_add_shared_features.py` - Shared features (run on both datasets)
6. `06_add_training_features.py` - Training-only features
7. `07_train_model.py` - Train with proof generation
8. `08_predict.py` - Predict with visualization

### 5. Update Model Training

**`scripts/07_train_model.py` needs:**
- Load from `data/features/training_only_features.json`
- Use `--issue X` parameter
- Generate complete proof visualizations
- Save model to `models/issue-X/model.txt`
- Save all plots to `proof/issue-X/`

### 6. Update Prediction

**`scripts/08_predict.py` needs:**
- Load model from `models/issue-X/model.txt`
- Load from `data/features/prediction_shared_features.json`
- Generate price chart with signals
- Mark buy/sell signals
- Save to `proof/issue-X/prediction_*.png`

---

## 📋 Implementation Sub-Issues

### Issue #1.1: Download Training Binance Data
**Create:** `scripts/01_download_training_binance.py`
**Output:** `data/raw/training/binance_20pairs_1H.json`
**Details:** Download 20 pairs, 1H candles, 2021-2025, ~50K candles each

### Issue #1.2: Download Training Macro Data
**Create:** `scripts/02_download_training_macro.py`
**Create:** `sneaker/macro.py`
**Output:** `data/raw/training/macro_training.json`
**Details:** Download SPY, VIX, DXY, etc. aligned with training period

### Issue #1.3: Download Prediction Binance Data
**Create:** `scripts/03_download_prediction_binance.py`
**Output:** `data/raw/prediction/binance_LINKUSDT_live.json`
**Details:** Download LINKUSDT, 256-512 hours, flexible pair selection

### Issue #1.4: Download Prediction Macro Data
**Create:** `scripts/04_download_prediction_macro.py`
**Output:** `data/raw/prediction/macro_prediction.json`
**Details:** Download recent macro data aligned with prediction period

### Issue #1.5: Implement Shared Features Pipeline
**Create:** `sneaker/features_shared.py`
**Create:** `scripts/05_add_shared_features.py`
**Output:** `data/features/training_shared_features.json`
**Output:** `data/features/prediction_shared_features.json`
**Details:** Extract shared features from current `features.py`, ensure identical calculation

### Issue #1.6: Implement Training-Only Features Pipeline
**Create:** `sneaker/features_training.py`
**Create:** `scripts/06_add_training_features.py`
**Output:** `data/features/training_only_features.json`
**Details:** Extract target calculation, turning points, signal detection

### Issue #1.7: Refactor Training Script
**Refactor:** `scripts/07_train_model.py`
**Output:** `models/issue-X/model.txt`
**Output:** `proof/issue-X/training_*.png`
**Details:** Load from prepared data, train with V3 weighting, generate proof

### Issue #1.8: Refactor Prediction Script
**Refactor:** `scripts/08_predict.py`
**Output:** `proof/issue-X/prediction_*.png`
**Details:** Load model, predict on prepared data, visualize results

---

## ✅ Success Criteria

**Issue #1 complete when:**

1. ✅ All 8 sub-issues completed
2. ✅ Can run complete pipeline from raw data → trained model → predictions
3. ✅ All intermediate results cached (no recalculation needed)
4. ✅ Training and prediction features cleanly separated
5. ✅ Easy to swap features, pairs, or experiments
6. ✅ All outputs saved to proof/issue-1/
7. ✅ Validation passes (Signal R² ≥ 70%, etc.)
8. ✅ Prediction visualization generated (256h LINKUSDT chart)

---

## 🚨 Critical Requirements

### Data Integrity

- **Raw data immutable** - Never modify raw downloads
- **Features reproducible** - Same code → same features
- **Timestamps aligned** - Crypto and macro data synced correctly

### Feature Parity

- **Shared features IDENTICAL** - Same calculation for training and prediction
- **No data leakage** - Training features never in prediction
- **No future peeking** - Prediction features use only past data

### Modularity

- **Independent scripts** - Each script can run standalone
- **Clear dependencies** - Document what depends on what
- **Easy testing** - Can test each component separately

---

## 📊 Expected Results

### After Issue #1 Completion

**You will have:**
- Clean, modular pipeline
- Separate training and prediction paths
- All data cached (fast iteration)
- Easy feature experimentation
- Complete proof generation
- Flexible pair selection
- Macro data integrated
- Everything documented and tracked

**You will be able to:**
- Test new features in minutes (not hours)
- Compare experiments easily (proof folders)
- Switch pairs without retraining
- Iterate rapidly on improvements
- Trust the validation system

---

## 🎯 Next Steps After Issue #1

**Once baseline pipeline works:**

- Issue #2: Test feature variations (add/remove features)
- Issue #3: Test different target definitions
- Issue #4: Test different pairs (BTCUSDT, ETHUSDT, etc.)
- Issue #5: Add walk-forward validation
- Issue #6: Optimize hyperparameters
- Issue #7: Ensemble methods

**Clean pipeline = Fast experimentation = Better models**

---

## 📖 References

- **WORKFLOW.md** - How to execute each sub-issue
- **VALIDATION.md** - Pass/fail criteria
- **CLAUDE.md** - Development guidelines
- Current `sneaker/features.py` - 83 features to split
- Current `scripts/04_train_model.py` - Training logic to refactor

---

**This design creates a flexible, modular foundation for rapid experimentation while maintaining rigor and reproducibility.**
