# Issue #8: Training Script Refactoring - Implementation Complete

## Status: ✅ IMPLEMENTATION COMPLETE, AWAITING TESTING

## What We Built

### Critical Architectural Insight
**The windowing step was missing!** Training on single candles loses temporal context. The model needs to see trends developing over multiple consecutive candles.

### Two-Script Architecture

#### Script 07: Create Sliding Windows
**File:** `scripts/07_create_windows.py` (340 lines)

**Purpose:** Transform individual candles into sliding windows with temporal context

**Process:**
```python
# Input: Individual candles
{
  timestamp: 1609459200000,
  rsi: 65.2,
  bb_position: 0.5,
  macd_hist: 0.2,
  ...  # 93 shared features
  target: 4.5  # Ghost signal
}

# Output: Sliding windows
{
  # t0 (oldest in window)
  rsi_t0: 30.1,
  bb_position_t0: -0.8,
  macd_hist_t0: -0.5,

  # t1
  rsi_t1: 32.4,
  bb_position_t1: -0.7,
  macd_hist_t1: -0.4,

  # ... t2 through t10 ...

  # t11 (most recent, "now")
  rsi_t11: 65.2,
  bb_position_t11: 0.5,
  macd_hist_t11: 0.2,

  # Target from last candle
  target: 4.5
}
```

**Key Features:**
- **Sliding window approach** - Maximum data utilization (stride=1)
- **Configurable window size** - Default 12 candles, user can adjust
- **Per-pair windowing** - No cross-pair leakage
- **Feature flattening** - 93 features × 12 candles = 1,116 features
- **Target preservation** - Uses target from last candle in window
- **Validation checks** - NaN detection, duplicate detection, distribution analysis

**Data Flow:**
```
Input:  791,044 candles with 97 features (93 shared + 4 training-only)
Output: ~779,000 windows with 1,116 features (93 × 12)
Loss:   ~12,000 candles lost to window boundaries (expected)
```

**Usage:**
```bash
# Default: 12-candle windows
.venv/bin/python scripts/07_create_windows.py

# Custom window size
.venv/bin/python scripts/07_create_windows.py --window-size 16

# Custom paths
.venv/bin/python scripts/07_create_windows.py \
  --input data/features/training_complete_features.json \
  --output data/features/windowed_training_data.json \
  --window-size 12
```

#### Script 08: Train Model on Windowed Data
**File:** `scripts/08_train_model.py` (684 lines)

**Purpose:** Train LightGBM model with V3 sample weighting on windowed data

**Process:**
```python
# 1. Load windowed data
df = pd.read_json('data/features/windowed_training_data.json')

# 2. Extract 1,116 windowed features
feature_cols = [f'{feat}_t{t}' for feat in SHARED_FEATURES for t in range(12)]
X = df[feature_cols].values  # (779K, 1116)
y = df['target'].values       # (779K,)

# 3. Compute V3 sample weights
sample_weights = np.ones(len(y))
sample_weights[y != 0] = 5.0  # Ghost signals weighted 5×
sample_weights[y == 0] = 1.0  # Normal candles weighted 1×

# 4. Train/test split (90/10)
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# 5. Train LightGBM with sample weighting
model = lgb.LGBMRegressor(
    num_leaves=255,
    max_depth=8,
    learning_rate=0.01,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    eval_set=[(X_test, y_test)],
    eval_sample_weight=[sw_test],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# 6. Generate proof visualizations (5 types)
# 7. Save model to models/issue-1/model.txt
```

**Key Features:**
- **Windowed feature extraction** - Automatically handles 1,116 features
- **V3 sample weighting** - 5× for signals, 1× for normal candles
- **Comprehensive metrics** - Overall + signal-only + direction accuracy
- **5 proof visualizations:**
  1. Regression plot (train vs test)
  2. Residual analysis (4 plots)
  3. Feature importance (top 30 features)
  4. Signal distribution
  5. Metrics report (text file)
- **Issue-specific outputs** - Models and proof organized by issue
- **Validation checks** - Automatic success/failure detection

**Metrics Tracked:**
- **Signal R²** - CRITICAL metric (target: ≥70%)
- **Direction accuracy** - Sign prediction (target: ≥95%)
- **RMSE/MAE** - For signals only
- **Effective signal ratio** - After sample weighting

**Usage:**
```bash
# Default (issue-1 folder)
.venv/bin/python scripts/08_train_model.py

# Custom issue folder
.venv/bin/python scripts/08_train_model.py --issue issue-8

# Custom paths
.venv/bin/python scripts/08_train_model.py \
  --input data/features/windowed_training_data.json \
  --output models/issue-8/model.txt \
  --issue issue-8 \
  --window-size 12
```

## Why Windowing Matters

### Problem: Single-Candle Training Loses Context

Without windowing, the model sees isolated snapshots:
```python
# Single candle - no context
features = {
    'rsi': 65,
    'bb_position': 0.5,
    'macd_hist': 0.2
}
target = +4.5σ

# Model learns: "When RSI=65, predict +4.5σ"
# But RSI=65 could mean:
#  - Rising from 30 (bullish momentum) → Should BUY
#  - Falling from 80 (losing steam) → Should SELL
# No way to tell without context!
```

### Solution: Windowing Provides Temporal Context

With 12-candle windows, the model sees the full trend:
```python
# 12-candle window - full trend visible
features = {
    # t0 (12 hours ago)
    'rsi_t0': 30, 'bb_position_t0': -0.8, 'macd_hist_t0': -0.5,

    # t1
    'rsi_t1': 32, 'bb_position_t1': -0.7, 'macd_hist_t1': -0.4,

    # ... t2 through t10 showing gradual rise ...

    # t11 (now)
    'rsi_t11': 65, 'bb_position_t11': 0.5, 'macd_hist_t11': 0.2
}
target = +4.5σ

# Model sees the TREND:
# RSI rising 30→65, BB crossing from negative to positive,
# MACD turning positive - this is BULLISH momentum!
# Prediction of +4.5σ makes sense in this context.
```

**The windowing gives the model "memory" of recent history, allowing it to:**
- See trends developing (up, down, consolidating)
- Detect momentum building (accelerating or decelerating)
- Identify pattern formations (breakouts, reversals)
- Understand market regime (volatile, calm, trending)

## Expected Performance Impact

### Pros of Windowing
✅ **Temporal context** - Model sees trends, not just snapshots
✅ **Better signal detection** - Can identify momentum building
✅ **Improved accuracy** - Likely +5-10% on signal R²
✅ **More robust** - Less sensitive to noise in single candles

### Cons of Windowing
❌ **Larger feature space** - 1,116 features vs 93
❌ **Longer training time** - ~3-5× slower
❌ **Larger model file** - ~50-60 MB vs ~30 MB
❌ **More complex inference** - Need 12 candles of history

### Expected Metrics (with windowing)
- **Signal R²:** 78-82% (up from 74%)
- **Direction accuracy:** 98-99% (up from 98%)
- **Training time:** 8-12 minutes (up from 2 minutes)
- **Model size:** 55 MB (up from 34 MB)

## File Structure After Completion

```
scripts/
├── 01-05: Data collection & shared features ✅
├── 06_add_training_features.py           # ✅ COMPLETE (Issue #7)
├── 07_create_windows.py                  # ✅ COMPLETE (Issue #8)
└── 08_train_model.py                     # ✅ COMPLETE (Issue #8)

data/features/
├── training_shared_features.json         # 2.7 GB (791K candles, 93 shared)
├── training_complete_features.json       # 3.5 GB (791K candles, 97 total)
└── windowed_training_data.json           # 7-8 GB (779K windows, 1,116 features)

models/
└── issue-1/  (or issue-8, configurable)
    └── model.txt                          # Trained model (~50-60 MB)

proof/
└── issue-1/  (or issue-8, configurable)
    ├── training_regression_*.png          # Train vs test regression
    ├── training_residuals_*.png           # Residual analysis
    ├── training_feature_importance_*.png  # Top 30 features
    ├── training_signal_dist_*.png         # Signal distribution
    └── training_report_*.txt              # Metrics report
```

## Testing Plan

### Phase 1: Test Script 07 (Windowing) ⏳ PENDING
```bash
# Wait for script 06 to complete (adding training features)
# Check: data/features/training_complete_features.json exists

# Run windowing script
.venv/bin/python scripts/07_create_windows.py

# Expected output:
# - data/features/windowed_training_data.json (~7-8 GB)
# - ~779,000 windows
# - 1,116 features per window
# - Processing time: ~5-10 minutes

# Validation checks (automatic):
# ✓ Feature count: 1,116 (93 × 12)
# ✓ No NaN values in features
# ✓ Target distribution: 5-10% signals
# ✓ No duplicate windows
```

### Phase 2: Test Script 08 (Training) ⏳ PENDING
```bash
# After script 07 completes successfully

# Run training script
.venv/bin/python scripts/08_train_model.py --issue issue-8

# Expected output:
# - models/issue-8/model.txt (~50-60 MB)
# - proof/issue-8/ with 5 visualization files
# - Processing time: ~8-12 minutes

# Success criteria:
# ✅ Signal R² ≥ 70% (target: 78-82%)
# ✅ Direction accuracy ≥ 95% (target: 98-99%)
# ✅ Model file created and valid
# ✅ All 5 proof visualizations generated
```

### Phase 3: Visual Inspection of Proof ⏳ PENDING

**Review proof/issue-8/ outputs:**

1. **training_regression_*.png**
   - ✓ Points cluster around diagonal (good prediction)
   - ✓ Test R² close to train R² (no overfitting)
   - ✓ No systematic bias (points evenly distributed)

2. **training_residuals_*.png**
   - ✓ Residuals centered around 0
   - ✓ No patterns in residuals (random scatter)
   - ✓ Similar distribution in train and test

3. **training_feature_importance_*.png**
   - ✓ Multiple features important (not just 1-2)
   - ✓ Windowed features from different time steps
   - ✓ No single feature dominates (>40%)

4. **training_signal_dist_*.png**
   - ✓ Signal distribution looks reasonable
   - ✓ Mean close to 0 (balanced buy/sell)
   - ✓ Similar distribution in train and test

5. **training_report_*.txt**
   - ✓ All metrics documented
   - ✓ Signal R² ≥ 70%
   - ✓ Direction accuracy ≥ 95%

## Current Status

### ✅ Complete
1. **ISSUE_8_ANALYSIS.md** - Ultra-deep architectural analysis
2. **scripts/07_create_windows.py** - Windowing implementation (340 lines)
3. **scripts/08_train_model.py** - Training implementation (684 lines)
4. **Git commits** - Both scripts committed to issue-8-train-model branch

### ⏳ Pending
1. **Script 06 completion** - Currently running (adding training features)
2. **Test script 07** - Create sliding windows
3. **Test script 08** - Train model on windowed data
4. **Visual proof inspection** - Verify quality of proof outputs
5. **Merge to main** - After successful testing

### 🔄 In Progress
- **Script 06:** Adding training features with ghost signal detection
  - Status: RUNNING (13+ minutes elapsed)
  - Input: 791,044 candles
  - Processing: Ghost signal detection across 20 pairs
  - Expected completion: 3-5 more minutes

## Next Steps

1. **Wait for script 06 to complete**
   - Monitor: `BashOutput e2a0d0`
   - Check output: `data/features/training_complete_features.json`
   - Expected size: ~3.5 GB

2. **Test script 07 (windowing)**
   ```bash
   .venv/bin/python scripts/07_create_windows.py
   ```

3. **Test script 08 (training)**
   ```bash
   .venv/bin/python scripts/08_train_model.py --issue issue-8
   ```

4. **Review proof visualizations**
   - Check `proof/issue-8/` for all 5 outputs
   - Visual inspection using criteria above
   - Verify metrics meet success criteria

5. **If all tests pass:**
   ```bash
   # Commit any updates (if needed)
   git add -A
   git commit -m "Test #8: Validate windowing and training pipeline"

   # Push branch and create PR
   git push origin issue-8-train-model
   gh pr create --title "Implement #8: Training Pipeline with Windowing" \
     --body "Implements Issue #8 with two-script architecture..."

   # Merge to main
   gh pr merge --squash

   # Clean up
   git checkout main
   git pull
   git branch -d issue-8-train-model
   ```

## Key Implementation Details

### Window Size Trade-offs

**8 candles (8 hours):**
- Pros: Faster training, smaller model, less context needed
- Cons: Less temporal context, may miss longer trends
- Use case: Fast-moving markets, short-term trading

**12 candles (12 hours) - DEFAULT:**
- Pros: Good balance of context and complexity
- Cons: Moderate training time and model size
- Use case: General purpose, half-day context

**16-24 candles (16-24 hours):**
- Pros: Maximum temporal context, full day of history
- Cons: Longer training time, larger model, more features
- Use case: Longer-term patterns, full market cycle

### Sample Weighting Analysis

Without sample weighting:
```
Signals:        5% of data (39,500 windows)
Normal candles: 95% of data (739,500 windows)

Model learns to predict 0 (safest bet) → 0% signal rate
```

With 5× sample weighting:
```
Signal weight sum:     39,500 × 5.0 = 197,500
Normal weight sum:     739,500 × 1.0 = 739,500
Total weight:          937,000

Effective signal ratio: 197,500 / 937,000 = 21.1%
Effective normal ratio: 739,500 / 937,000 = 78.9%

Model learns both patterns → optimal signal rate (~5%)
```

### Feature Naming Convention

Features are named with time step suffix:
```
rsi_t0    # RSI at oldest candle in window
rsi_t1    # RSI at 2nd oldest candle
...
rsi_t11   # RSI at most recent candle

bb_position_t0, bb_position_t1, ..., bb_position_t11
macd_hist_t0, macd_hist_t1, ..., macd_hist_t11
...
```

This allows the model to learn temporal patterns like:
- Momentum: `rsi_t11 - rsi_t0` (is RSI rising or falling?)
- Acceleration: `(rsi_t11 - rsi_t6) - (rsi_t5 - rsi_t0)` (is momentum accelerating?)
- Crossovers: Detection when indicators flip signs between time steps

## Success Criteria

### Script 07 (Windowing)
- ✅ Creates ~779,000 windows from 791,044 candles
- ✅ Each window has 1,116 features (93 × 12)
- ✅ No NaN values in features
- ✅ No duplicate windows (per pair-timestamp)
- ✅ Target distribution: 5-10% signals, 90-95% zeros
- ✅ Per-pair windowing (no cross-pair leakage)
- ✅ Processing time: <15 minutes

### Script 08 (Training)
- ✅ Loads windowed data successfully
- ✅ Extracts 1,116 features correctly
- ✅ Applies V3 sample weighting (5× for signals)
- ✅ Trains LightGBM with early stopping
- ✅ **Signal R² ≥ 70%** (target: 78-82%)
- ✅ **Direction accuracy ≥ 95%** (target: 98-99%)
- ✅ Generates all 5 proof visualizations
- ✅ Saves model to correct location
- ✅ Processing time: <20 minutes

### Overall Pipeline
- ✅ End-to-end execution without errors
- ✅ Metrics improve over single-candle baseline
- ✅ Proof visualizations show no red flags
- ✅ Model file is valid and loadable
- ✅ Ready for integration with prediction pipeline (Issue #9)

## Red Flags to Watch For

**During Windowing (Script 07):**
- 🚩 Too many NaN values (>1% of features)
- 🚩 Signal rate too low (<3%) or too high (>15%)
- 🚩 Huge file size (>10 GB) - may indicate duplicate features
- 🚩 Processing time >30 minutes - may indicate inefficient code

**During Training (Script 08):**
- 🚩 Signal R² < 70% (worse than expected)
- 🚩 Train R² >> Test R² (>15% gap = overfitting)
- 🚩 Direction accuracy < 95%
- 🚩 Single feature dominates importance (>40%)
- 🚩 Model file >100 MB (too complex)
- 🚩 Residuals show patterns (not random scatter)

**If Any Red Flags Appear:**
1. Document the issue in GitHub issue #8
2. Investigate root cause
3. Adjust hyperparameters or window size
4. Re-run and re-validate
5. Only merge when all criteria met

## Conclusion

✅ **Issue #8 implementation is complete!**

We've successfully:
- Identified the missing windowing step
- Created a two-script architecture (07 + 08)
- Implemented configurable window size
- Applied V3 sample weighting correctly
- Added comprehensive proof generation
- Documented everything thoroughly

**Next:** Wait for script 06 to complete, then test the complete pipeline end-to-end.

---

**Branch:** `issue-8-train-model`
**Files Created:** 2 scripts, 2 analysis docs
**Lines of Code:** 1,024 lines (340 + 684)
**Ready for Testing:** Yes (pending script 06 completion)
