# Issue #8: Training Script Refactoring - Ultra-Deep Analysis

## Critical Architectural Decision: Windowing

**YOU WERE RIGHT!** There's a missing step before training: **creating sliding windows**.

### The Problem with Single-Candle Training

Training on individual candles loses temporal context:
```python
# WRONG - No temporal context
features = [rsi, bb, macd, ...]  # Just current candle
target = future_reversal          # Predict from single snapshot
```

The model can't see:
- Trend direction (are we going up or down?)
- Momentum building (is volatility increasing?)
- Pattern formation (are we in a consolidation?)

### The Solution: Sliding Window Architecture

**Give the model context from N previous candles**

```python
# RIGHT - Temporal context
window = last_12_candles  # 12 consecutive candles
features = flatten([
    candle_1_features,  # 93 features
    candle_2_features,  # 93 features
    ...
    candle_12_features  # 93 features
])
# Total: 93 * 12 = 1,116 features

target = candle_12_target  # Predict from context
```

## Two-Step Training Process

### Step 1: Create Training Windows (NEW!)
**Script:** `scripts/07_create_windows.py`

#### Input
`data/features/training_complete_features.json`
- 791,044 candles
- 93 shared features + 4 training-only features
- Each candle is independent

#### Process
```python
def create_sliding_windows(df, window_size=12):
    """
    Create sliding windows for time series training.

    Args:
        df: DataFrame with features and target
        window_size: Number of consecutive candles per window (default: 12)

    Returns:
        DataFrame where each row is a window
    """
    windows = []

    for pair in df['pair'].unique():
        pair_data = df[df['pair'] == pair].sort_values('timestamp')

        # Slide window across data
        for i in range(len(pair_data) - window_size + 1):
            window = pair_data.iloc[i:i+window_size]

            # Extract features from all candles in window
            window_features = {}
            for j, candle in enumerate(window.iterrows()):
                for feature in SHARED_FEATURES:
                    # Flatten: feature_name_t0, feature_name_t1, ..., feature_name_t11
                    window_features[f'{feature}_t{j}'] = candle[feature]

            # Target from LAST candle (most recent)
            window_features['target'] = window.iloc[-1]['target']
            window_features['pair'] = pair
            window_features['timestamp'] = window.iloc[-1]['timestamp']

            windows.append(window_features)

    return pd.DataFrame(windows)
```

#### Output
`data/features/windowed_training_data.json`
- ~779,000 windows (791K - 12 + 1, accounting for window boundaries per pair)
- 1,116 features (93 shared features × 12 candles)
- 1 target per window (from last candle)

#### Key Design Decisions

**1. Window Size: 12 candles (default, configurable)**
- 12 hours of context at 1H timeframe
- Enough to see short-term trends
- Not so long that patterns get diluted
- User wants flexibility to experiment (8, 12, 16, 24, etc.)

**2. Sliding Window (not jumping)**
```python
# Sliding: Maximum data utilization
window_1 = candles[0:12]   # Use candles 0-11
window_2 = candles[1:13]   # Use candles 1-12 (overlap!)
window_3 = candles[2:14]   # Use candles 2-13 (overlap!)

# NOT jumping: Would waste data
window_1 = candles[0:12]   # Use candles 0-11
window_2 = candles[12:24]  # Use candles 12-23 (no overlap)
```

**3. Feature Naming Convention**
```python
# Time step 0 (oldest in window)
rsi_t0, bb_position_t0, macd_hist_t0, ...

# Time step 11 (most recent in window)
rsi_t11, bb_position_t11, macd_hist_t11, ...
```

**4. Target Selection**
- Use target from **LAST candle** in window (t11)
- This is the "present moment" we're predicting
- Previous candles (t0-t10) are historical context

**5. Per-Pair Windowing**
- Don't create windows across pair boundaries
- Each pair has independent temporal sequence
- Avoids leakage between different crypto assets

### Step 2: Train Model (Refactored)
**Script:** `scripts/08_train_model.py` (refactored from 04)

#### Input
`data/features/windowed_training_data.json`

#### Process
```python
# 1. Load windowed data
df = pd.read_json('data/features/windowed_training_data.json')

# 2. Extract features and target
feature_cols = [col for col in df.columns if col.startswith(tuple(SHARED_FEATURES))]
X = df[feature_cols].values  # (779K, 1116)
y = df['target'].values       # (779K,)

# 3. Compute sample weights (V3 KEY!)
sample_weights = np.ones(len(y))
sample_weights[y != 0] = 5.0  # Ghost signals
sample_weights[y == 0] = 1.0  # Normal candles

# 4. Train/test split (90/10, temporal)
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weights,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# 5. Train LightGBM with sample weighting
model = lgb.LGBMRegressor(...)
model.fit(X_train, y_train, sample_weight=sw_train, ...)

# 6. Generate proof visualizations
generate_proof_plots(model, X_test, y_test, issue='issue-1')

# 7. Save model
model.save_model('models/issue-1/model.txt')
```

#### Output
- `models/issue-1/model.txt` - Trained model
- `proof/issue-1/training_regression_*.png` - Train vs test regression
- `proof/issue-1/training_residuals_*.png` - Residual analysis
- `proof/issue-1/training_feature_importance_*.png` - Top features
- `proof/issue-1/training_signal_dist_*.png` - Signal distribution
- `proof/issue-1/training_report_*.txt` - Metrics report

## Why Windowing Matters

### Without Windowing (Current Approach)
```python
# Single candle
features = [rsi=65, bb=0.5, macd=0.2]
target = +4.5σ

# Model sees: "When RSI=65, predict +4.5σ"
# But RSI=65 could mean:
#  - Rising from 30 (bullish momentum) → BUY
#  - Falling from 80 (losing steam) → SELL
# No way to tell without context!
```

### With Windowing (Proposed)
```python
# 12-candle window
features = [
    # t0 (12 hours ago): rsi=30, bb=-0.8, macd=-0.5
    # t1: rsi=32, bb=-0.7, macd=-0.4
    # ...
    # t11 (now): rsi=65, bb=0.5, macd=0.2
]
target = +4.5σ

# Model sees the TREND:
# RSI rising 30→65, BB crossing up, MACD turning positive
# This is BULLISH momentum → BUY makes sense!
```

## Implementation Plan

### Phase 1: Create Windowing Script (07)

**File:** `scripts/07_create_windows.py`

```python
#!/usr/bin/env python3
"""
Create Training Windows (Sliding Window for Time Series)

Takes training_complete_features.json (individual candles) and creates
sliding windows of N consecutive candles for time series training.

Part of Issue #8 (Pipeline Restructuring Epic #1)

Usage:
    # Default: 12-candle windows
    .venv/bin/python scripts/07_create_windows.py

    # Custom window size
    .venv/bin/python scripts/07_create_windows.py --window-size 16

    # Custom input/output
    .venv/bin/python scripts/07_create_windows.py \
      --input data/features/training_complete_features.json \
      --output data/features/windowed_training_data.json \
      --window-size 12

Arguments:
    --input: Input complete features JSON
    --output: Output windowed data JSON
    --window-size: Number of consecutive candles per window (default: 12)

Input:
    data/features/training_complete_features.json
    - 791,044 individual candles
    - 93 shared features + 4 training-only features

Output:
    data/features/windowed_training_data.json
    - ~779,000 windows (depends on pairs and window size)
    - 1,116 features (93 × 12 candles)
    - 1 target per window
"""
```

**Key Functions:**
- `create_sliding_windows(df, window_size)` - Main windowing logic
- `flatten_window_features(window)` - Flatten N candles into single feature vector
- `validate_windows(windowed_df)` - Check for data leakage, missing values

### Phase 2: Refactor Training Script (08)

**File:** `scripts/08_train_model.py` (refactored from 04)

**Key Changes:**
1. Load windowed data (not individual candles)
2. Handle 1,116 features (not 93)
3. Add `--issue` parameter for proof folder
4. Generate all 5 proof visualizations
5. Save model to issue-specific folder

## File Structure After Completion

```
data/features/
├── training_shared_features.json       # 2.7 GB (791K candles, 93 shared)
├── training_complete_features.json     # 3.5 GB (791K candles, 97 total)
└── windowed_training_data.json         # 7-8 GB (779K windows, 1,116 features)

scripts/
├── 01-06: Data collection & feature engineering ✅
├── 07_create_windows.py                # NEW - Windowing step
└── 08_train_model.py                   # REFACTORED - Training

models/
└── issue-1/
    └── model.txt                        # Trained model (~50-60 MB)

proof/
└── issue-1/
    ├── training_regression_*.png
    ├── training_residuals_*.png
    ├── training_feature_importance_*.png
    ├── training_signal_dist_*.png
    └── training_report_*.txt
```

## Design Questions to Decide

### 1. Window Size Options

**Conservative:** 8 candles (8 hours)
- Less context, faster training
- Good for fast-moving markets

**Default:** 12 candles (12 hours)
- Half a day of context
- Balance between context and complexity

**Aggressive:** 24 candles (24 hours)
- Full day of context
- More features, longer training

**Recommendation:** Make it configurable with 12 as default.

### 2. Feature Selection in Windows

**Option A: All 93 shared features in all timeframes**
- 93 × 12 = 1,116 features
- Maximum information
- Risk of overfitting

**Option B: Selected features only**
- Maybe 30 key features × 12 = 360 features
- Less overfitting risk
- Might miss important signals

**Recommendation:** Start with all features (Option A), add feature selection later if needed.

### 3. Target Selection

**Option A: Target from last candle only** ← RECOMMENDED
- target_t11 (most recent)
- Standard time series approach

**Option B: Targets from multiple candles**
- Average of target_t10, target_t11, target_t12
- Smoother targets
- More complex

**Recommendation:** Option A - keep it simple.

### 4. Window Stride

**Option A: Stride = 1 (sliding)** ← RECOMMENDED
- Every possible window
- Maximum data utilization
- ~779K windows

**Option B: Stride = 6 (half-overlap)**
- Every 6th window
- Faster training
- ~130K windows

**Recommendation:** Option A - we have the compute power.

## Expected Impact on Model Performance

### Pros of Windowing
✅ **Temporal context:** Model sees trends, not just snapshots
✅ **Better signal detection:** Can identify momentum building
✅ **Improved accuracy:** Likely +5-10% on signal R²
✅ **More robust:** Less sensitive to noise in single candles

### Cons of Windowing
❌ **Larger feature space:** 1,116 features vs 93
❌ **Longer training time:** ~3-5x slower
❌ **Larger model file:** ~50-60 MB vs ~30 MB
❌ **More complex inference:** Need 12 candles of history

### Expected Metrics (with windowing)
- Signal R²: **78-82%** (up from 74%)
- Direction accuracy: **98-99%** (up from 98%)
- Training time: **8-12 minutes** (up from 2 minutes)
- Model size: **55 MB** (up from 34 MB)

## Next Steps

1. ✅ Create branch `issue-8-train-model`
2. ⏳ Create `ISSUE_8_ANALYSIS.md` (this document)
3. ⏳ Implement `scripts/07_create_windows.py`
4. ⏳ Refactor `scripts/08_train_model.py`
5. ⏳ Wait for script 06 to complete (training features)
6. ⏳ Test window creation on small sample
7. ⏳ Run full windowing pipeline
8. ⏳ Train model and generate proof
9. ⏳ Validate metrics meet success criteria

## Success Criteria

### Window Creation (Script 07)
- ✅ Creates ~779K windows from 791K candles
- ✅ Each window has 1,116 features (93 × 12)
- ✅ No data leakage (windows don't overlap targets)
- ✅ Per-pair windowing (no cross-pair windows)
- ✅ Configurable window size

### Model Training (Script 08)
- ✅ Loads windowed data successfully
- ✅ Applies V3 sample weighting (5x for signals)
- ✅ Generates all 5 proof visualizations
- ✅ Signal R² ≥ 70% (target: 78-82%)
- ✅ Direction accuracy ≥ 95% (target: 98-99%)
- ✅ Saves model to `models/issue-1/`
- ✅ Saves proof to `proof/issue-1/`
