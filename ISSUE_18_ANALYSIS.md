# Issue #18 Analysis: Adapting Prediction Pipeline for Windowed Data

## Overview

The training pipeline (Issue #8) now uses windowed data with 1,116 features (93 × 12 candles). The prediction pipeline must be updated to match this architecture **exactly**.

## Critical Requirements

### 1. **Feature Alignment (ABSOLUTE MUST)**

Training expects **1,116 features** in **exact order:**
```
[feature1_t0, feature1_t1, ..., feature1_t11,  # 12 time steps for feature 1
 feature2_t0, feature2_t1, ..., feature2_t11,  # 12 time steps for feature 2
 ...
 feature93_t0, feature93_t1, ..., feature93_t11] # 12 time steps for feature 93
```

**Feature order comes from:** `sneaker/features_shared.py::SHARED_FEATURE_LIST` (93 features)

**Time ordering:** t0 = oldest candle, t11 = newest candle in window

### 2. **Normalization Consistency (ABSOLUTE MUST)**

**Exactly 15 features must be normalized relative to t0:**

```python
NORMALIZE_FEATURES = [
    # Macro close prices (4)
    'macro_GOLD_close', 'macro_BNB_close',
    'macro_BTC_PREMIUM_close', 'macro_ETH_PREMIUM_close',

    # Macro velocities (4)
    'macro_GOLD_vel', 'macro_BNB_vel',
    'macro_BTC_PREMIUM_vel', 'macro_ETH_PREMIUM_vel',

    # VWAP (1)
    'vwap_20',

    # ATR (2)
    'atr_14', 'atr_vel',

    # MACD histogram (4)
    'macd_hist', 'macd_hist_vel',
    'macd_hist_2x', 'macd_hist_4x'
]
```

**Normalization logic:**
```python
if abs(t0_value) > 1e-10:  # Avoid division by zero
    normalized_value = row[feature] / t0_value
else:
    # If t0 is zero, use 1.0 or 0.0 based on current value
    normalized_value = 1.0 if abs(row[feature]) < 1e-10 else 0.0
```

**This MUST match training exactly** - any difference will produce incorrect predictions.

### 3. **Window Structure (ABSOLUTE MUST)**

- **Window size:** 12 consecutive candles (same as training)
- **Time ordering:** Oldest to newest (t0 → t11)
- **Window sliding:** Moves 1 candle at a time
- **Minimum candles:** Need at least 12 candles per pair to create ANY windows

## Differences: Training vs Prediction

| Aspect | Training (Issue #8) | Prediction (Issue #18) |
|--------|---------------------|------------------------|
| **Input file** | `training_complete_features.json` | `prediction_complete_features.json` |
| **Input size** | 791,044 candles (historical) | ~2,000-5,000 candles (recent) |
| **Target column** | ✅ Has `target` column | ❌ NO target column |
| **Training-only features** | ✅ Has 4 training-only features | ❌ NO training-only features |
| **Shared features** | 93 features | 93 features (same) |
| **Output file** | `windowed_training_data.json` | `windowed_prediction_data.json` |
| **Output columns** | 1,116 features + target + metadata | 1,116 features + metadata (NO target) |
| **Cold start risk** | Low (lots of historical data) | High (recent data may have < 12 candles) |
| **Purpose** | Model training | Generate predictions |

## Implementation Strategy

### Script 09: Create Prediction Windows

**Base template:** Copy `scripts/07_create_windows.py`

**Required changes:**

1. **Update paths:**
   ```python
   # Change input default
   '--input', default='data/features/prediction_complete_features.json'

   # Change output default
   '--output', default='data/features/windowed_prediction_data.json'
   ```

2. **Remove target handling:**
   ```python
   # DELETE this line (around line 177 in script 07):
   window_features['target'] = last_candle['target']

   # KEEP these lines:
   window_features['pair'] = pair
   window_features['timestamp'] = last_candle['timestamp']
   ```

3. **Handle cold start gracefully:**
   ```python
   # Already exists in script 07 (lines 139-141):
   if num_windows <= 0:
       logger.warning(f"  ⚠️  Skipping {pair}: only {pair_len} candles (need {window_size})")
       continue
   ```

4. **Update documentation strings:**
   - Change "training" to "prediction" in docstrings
   - Remove references to target column
   - Note that this creates windows for prediction, not training

**What to keep EXACTLY the same:**
- ✅ `NORMALIZE_FEATURES` list (all 15 features)
- ✅ Normalization logic (divide by t0, 1e-10 threshold)
- ✅ Feature iteration order (loop through `SHARED_FEATURE_LIST`)
- ✅ Time step ordering (t0 to t11)
- ✅ Window sliding logic (consecutive candles)

### Script 10: Generate Predictions

**Base template:** Borrow logic from `scripts/08_train_model.py`

**Required components:**

1. **Feature extraction function** (copy from script 08, lines 259-277):
   ```python
   def extract_windowed_features(df: pd.DataFrame, window_size: int, logger):
       """Extract windowed feature columns from DataFrame."""
       feature_cols = []
       for feature in SHARED_FEATURE_LIST:
           for t in range(window_size):
               col_name = f"{feature}_t{t}"
               if col_name in df.columns:
                   feature_cols.append(col_name)

       logger.info(f"Expected features: {len(SHARED_FEATURE_LIST)} × {window_size} = {len(SHARED_FEATURE_LIST) * window_size}")
       logger.info(f"Found features: {len(feature_cols)}")

       X = df[feature_cols].values
       return feature_cols, X
   ```

2. **Load windowed prediction data:**
   ```python
   # Load prediction windows
   with open('data/features/windowed_prediction_data.json', 'r') as f:
       data = json.load(f)

   df = pd.DataFrame(data)

   # Extract features (NO target)
   feature_cols, X = extract_windowed_features(df, window_size=12, logger)
   ```

3. **Load trained model:**
   ```python
   import lightgbm as lgb

   model_path = 'models/issue-1/model.txt'
   logger.info(f"Loading model from {model_path}...")
   model = lgb.Booster(model_file=model_path)
   ```

4. **Generate predictions:**
   ```python
   logger.info("Generating predictions...")
   predictions = model.predict(X)

   # Combine with metadata
   results = pd.DataFrame({
       'pair': df['pair'],
       'timestamp': df['timestamp'],
       'prediction': predictions
   })
   ```

5. **Optional: Filter by pair and apply threshold:**
   ```python
   parser.add_argument('--pair', type=str, help='Filter by specific pair')
   parser.add_argument('--threshold', type=float, default=4.0,
                       help='Signal threshold in σ (default: 4.0)')

   # Filter by pair if specified
   if args.pair:
       results = results[results['pair'] == args.pair]

   # Classify signals
   results['signal'] = 'HOLD'
   results.loc[results['prediction'] > args.threshold, 'signal'] = 'BUY'
   results.loc[results['prediction'] < -args.threshold, 'signal'] = 'SELL'

   # Show strong signals only
   strong_signals = results[results['signal'] != 'HOLD']
   logger.info(f"Found {len(strong_signals)} strong signals (±{args.threshold}σ)")
   ```

6. **Output results:**
   ```python
   # Save to JSON
   output_path = 'data/predictions/latest.json'
   results.to_json(output_path, orient='records', date_format='iso')

   # Show summary
   logger.info("\nRecent predictions:")
   for _, row in results.tail(10).iterrows():
       logger.info(f"  {row['pair']}: {row['prediction']:+.2f}σ ({row['signal']})")
   ```

## Validation Checklist

Before considering script complete, verify:

- [ ] Script 09 creates windows without errors
- [ ] Output has 1,116 feature columns
- [ ] Normalized features are in range [0, 2] typically (relative to t0=1.0)
- [ ] Non-normalized features keep original ranges
- [ ] Window count matches: (num_candles - window_size + 1) per pair
- [ ] Cold start handled: Pairs with < 12 candles are skipped
- [ ] No target column in output (prediction data doesn't have targets)
- [ ] Script 10 loads model successfully
- [ ] Feature extraction produces 1,116 features
- [ ] Model.predict() runs without errors
- [ ] Predictions are not all NaN or zero
- [ ] Predictions have reasonable range (typically -10σ to +10σ)
- [ ] Can filter by pair successfully
- [ ] Output JSON is well-formed

## Common Pitfalls to Avoid

### 1. **Feature Order Mismatch**

❌ **Wrong:** Iterating features in different order than training
```python
# BAD - random order
for feature in ['rsi', 'macd_hist', 'bb_position']:  # Wrong!
```

✅ **Correct:** Use exact same order as training
```python
# GOOD - use SHARED_FEATURE_LIST
from sneaker.features_shared import SHARED_FEATURE_LIST
for feature in SHARED_FEATURE_LIST:  # Correct!
```

### 2. **Normalization Differences**

❌ **Wrong:** Different normalization logic
```python
# BAD - different threshold or logic
if t0_value != 0:  # Wrong threshold!
    normalized = current / t0_value
```

✅ **Correct:** Copy exact logic from script 07
```python
# GOOD - exact same logic
if abs(t0_value) > 1e-10:  # Correct threshold!
    normalized = current / t0_value
else:
    normalized = 1.0 if abs(current) < 1e-10 else 0.0
```

### 3. **Including Target Column**

❌ **Wrong:** Trying to add target to prediction windows
```python
# BAD - prediction data doesn't have targets!
window_features['target'] = last_candle['target']  # KeyError!
```

✅ **Correct:** Omit target completely
```python
# GOOD - no target for prediction data
window_features['pair'] = pair
window_features['timestamp'] = last_candle['timestamp']
# No target!
```

### 4. **Wrong Window Size**

❌ **Wrong:** Different window size than training
```python
# BAD - training used 12!
window_size = 10  # Wrong!
```

✅ **Correct:** Same window size as training
```python
# GOOD - match training
window_size = 12  # Correct!
```

## Testing Strategy

### 1. **Test Script 09 First**

```bash
# Check if prediction features exist
ls -lh data/features/prediction_complete_features.json

# Run windowing (should complete without errors)
.venv/bin/python scripts/09_create_prediction_windows.py

# Verify output
ls -lh data/features/windowed_prediction_data.json
```

**Expected output:**
- File created successfully
- Size: Several MB (depending on recent data)
- Log shows: "Created X windows from Y candles"
- Log shows: "Normalizing 15 features per window (relative to t0)"

### 2. **Validate Window Structure**

```python
# Quick validation script
import json
with open('data/features/windowed_prediction_data.json', 'r') as f:
    data = json.load(f)

window = data[0]
feature_cols = [k for k in window.keys() if k not in ['pair', 'timestamp']]

print(f"Total features: {len(feature_cols)}")  # Should be 1116
print(f"Has pair: {'pair' in window}")  # Should be True
print(f"Has timestamp: {'timestamp' in window}")  # Should be True
print(f"Has target: {'target' in window}")  # Should be False

# Check normalized features
for feat in ['macro_GOLD_close_t0', 'vwap_20_t0', 'atr_14_t0']:
    print(f"{feat}: {window.get(feat, 'MISSING')}")  # Should all be close to 1.0
```

### 3. **Test Script 10**

```bash
# Generate predictions
.venv/bin/python scripts/10_generate_predictions.py --pair LINKUSDT

# Check output
ls -lh data/predictions/latest.json
```

**Expected output:**
- Model loads successfully
- Predictions generated for all windows
- No NaN or inf values
- Predictions in reasonable range (-10σ to +10σ)
- Console shows summary of predictions

### 4. **Validate Predictions**

```python
# Check predictions
import json
with open('data/predictions/latest.json', 'r') as f:
    preds = json.load(f)

import pandas as pd
df = pd.DataFrame(preds)

print(f"Total predictions: {len(df)}")
print(f"Pairs: {df['pair'].unique()}")
print(f"Prediction range: {df['prediction'].min():.2f}σ to {df['prediction'].max():.2f}σ")
print(f"Mean: {df['prediction'].mean():.2f}σ")
print(f"Std: {df['prediction'].std():.2f}σ")

# Check for issues
print(f"NaN predictions: {df['prediction'].isna().sum()}")
print(f"Inf predictions: {(df['prediction'] == float('inf')).sum()}")
```

## Success Criteria

✅ **Script 09 succeeds when:**
- Creates windowed_prediction_data.json without errors
- Features count = 1,116 per window
- Normalization applied to exactly 15 features
- Cold start handled (pairs with < 12 candles skipped)
- Output format matches training windows (minus target)

✅ **Script 10 succeeds when:**
- Loads model successfully
- Extracts 1,116 features from each window
- Generates predictions without errors
- Predictions are non-NaN and in reasonable range
- Can filter by pair and apply thresholds
- Outputs JSON and console summary

## Next Steps After Implementation

1. **Run complete pipeline:**
   ```bash
   # Download → Features → Windows → Predictions
   .venv/bin/python scripts/03_download_prediction_binance.py
   .venv/bin/python scripts/04_download_prediction_macro_binance.py
   .venv/bin/python scripts/05_add_shared_features.py --mode prediction
   .venv/bin/python scripts/09_create_prediction_windows.py
   .venv/bin/python scripts/10_generate_predictions.py --pair LINKUSDT
   ```

2. **Validate end-to-end:**
   - Check each stage completes
   - Verify data flows correctly
   - Inspect final predictions

3. **Document any issues:**
   - Model performance (likely poor, as seen in issue #8)
   - Prediction quality
   - Edge cases discovered

---

**Key Insight:** This issue is about **infrastructure correctness**, not model performance. Even though the model from issue #8 has poor performance (19% R²), we still need the prediction pipeline to work correctly for future model improvements.
