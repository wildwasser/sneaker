# Issue #7: Training-Only Features Pipeline - Ultra-Deep Analysis

## Critical Distinction

**TRAINING-ONLY = USES FUTURE DATA**

These features CANNOT be calculated on live prediction data because they require future information. They are ONLY for training the model to learn patterns.

## Core Training-Only Features

### 1. Target Variable (`target`)

**The Primary Training-Only Feature**

```python
def calculate_target(df: pd.DataFrame, lookahead_periods: int = 4) -> pd.DataFrame:
    """
    Calculate volatility-normalized future price change (THE TARGET).

    This is the MAIN training-only feature - what we're teaching the model to predict.
    Uses FUTURE data (lookahead), so CANNOT be used in live prediction.

    Args:
        df: DataFrame with OHLCV data
        lookahead_periods: Hours into future to look (default: 4H)

    Returns:
        DataFrame with 'target' column added (in σ units)
    """
    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        # Calculate future price change (LOOK-AHEAD!)
        future_change = np.zeros(len(closes))
        for i in range(len(closes) - lookahead_periods):
            future_close = closes[i + lookahead_periods]
            current_close = closes[i]
            pct_change = (future_close - current_close) / current_close * 100
            future_change[i] = pct_change

        # Volatility normalization (convert to σ)
        # Uses 20-period rolling std of returns
        returns = pd.Series(closes).pct_change() * 100
        volatility = returns.rolling(20).std()

        # Normalize: target = future_change / volatility
        normalized = future_change / volatility.values
        normalized = np.nan_to_num(normalized, 0)

        df.loc[mask, 'target'] = normalized

    return df
```

**Why This Is Training-Only:**
- Uses `closes[i + lookahead_periods]` - FUTURE DATA
- In live prediction, we don't know future prices
- This is what the model learns to predict

**Target Characteristics:**
- Units: σ (sigma, volatility-normalized)
- Range: Typically -10σ to +10σ
- Zero values: Normal candles (no significant reversal)
- Non-zero: Ghost signals (indicator momentum shifts preceding price reversals)
- Distribution: ~26% signals, ~74% zeros (before filtering)

### 2. Statistical Features (3 features)

**These were excluded from shared features because they may be unstable on live data.**

#### 2a. Hurst Exponent
```python
def calculate_hurst_exponent(series, max_lag=40):
    """
    Calculate Hurst exponent (trend persistence vs mean reversion).

    H > 0.5: Trending (momentum)
    H < 0.5: Mean reverting
    H = 0.5: Random walk

    Uses 40-period rolling window - may be unstable on live data.
    """
    # Complex calculation using rescaled range analysis
    # See implementation in features.py
    pass
```

#### 2b. Permutation Entropy
```python
def calculate_permutation_entropy(series):
    """
    Measure predictability of time series.

    Lower entropy = more predictable
    Higher entropy = more random

    Uses 10-period lookback - pattern analysis may overfit.
    """
    # Ordinal pattern analysis
    # See implementation in features.py
    pass
```

#### 2c. CUSUM Signal
```python
def calculate_cusum(df):
    """
    Cumulative sum for change detection.

    Grows indefinitely over time - NOT suitable for live prediction
    where you don't have a fixed baseline.
    """
    returns = df.groupby('pair')['close'].pct_change()
    mean_return = returns.rolling(50).mean()
    df['cusum_signal'] = (returns - mean_return).groupby(df['pair']).cumsum()
    return df
```

### 3. Signal Detection Metadata (Optional)

**Additional training-only features that help understand the data:**

```python
# Signal strength markers
df['is_signal'] = (df['target'] != 0).astype(int)
df['signal_strength'] = np.abs(df['target'])
df['signal_direction'] = np.sign(df['target'])

# Reversal markers
df['is_strong_signal'] = (np.abs(df['target']) > 4.0).astype(int)
df['is_extreme_signal'] = (np.abs(df['target']) > 6.0).astype(int)
```

## Total Training-Only Features

**Minimum (Core):**
- 1 target variable (`target`)
- 3 statistical features (`hurst_exponent`, `permutation_entropy`, `cusum_signal`)
- **Total: 4 features**

**Extended (With Metadata):**
- 1 target
- 3 statistical
- 5 signal metadata features
- **Total: 9 features**

**Note:** Issue description says "~40 features" but that may have been an overestimate. The CRITICAL feature is the target variable. Everything else is supplementary.

## Input/Output Structure

### Input
**From Issue #6:** `data/features/training_shared_features.json`

```json
{
  "timestamp": 1609459200000,
  "open": 28923.63,
  "high": 29031.34,
  "low": 28690.17,
  "close": 28995.13,
  "volume": 2311.811445,
  "trades": 58389,
  "pair": "BTCUSDT",
  "macro_GOLD_close": 1931.72,
  "macro_BNB_close": 37.85,
  "... (93 shared features)"
}
```

### Output
**Issue #7:** `data/features/training_complete_features.json`

```json
{
  "... (all shared features + OHLCV)",
  "target": 4.2,  // ← PRIMARY TRAINING TARGET
  "hurst_exponent": 0.62,
  "permutation_entropy": 0.45,
  "cusum_signal": -2.3,
  "is_signal": 1,
  "signal_strength": 4.2,
  "signal_direction": 1
}
```

## Implementation Architecture

### Module: `sneaker/features_training.py`

```python
"""Training-only feature engineering.

CRITICAL: USES FUTURE DATA - CANNOT BE USED IN LIVE PREDICTION

These features are ONLY for training the model. They use future information
that is not available during live prediction.
"""

import numpy as np
import pandas as pd

TRAINING_ONLY_FEATURE_LIST = [
    'target',  # Primary target (future price change, σ normalized)
    'hurst_exponent',  # Trend persistence
    'permutation_entropy',  # Predictability
    'cusum_signal',  # Change detection
]

def calculate_target(df: pd.DataFrame, lookahead_periods: int = 4) -> pd.DataFrame:
    """Calculate volatility-normalized future price change (PRIMARY TARGET)."""
    pass

def calculate_hurst_exponent(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Hurst exponent (trend persistence indicator)."""
    pass

def calculate_permutation_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate permutation entropy (predictability measure)."""
    pass

def calculate_cusum(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CUSUM signal (cumulative sum for change detection)."""
    pass

def add_all_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all training-only features.

    REQUIRES: df must already have shared features (93 features from issue #6)

    Returns:
        DataFrame with 4 additional training-only features
    """
    df = calculate_target(df, lookahead_periods=4)
    df = calculate_hurst_exponent(df)
    df = calculate_permutation_entropy(df)
    df = calculate_cusum(df)

    # Fill NaNs
    df = df.fillna(0)

    return df
```

### Script: `scripts/06_add_training_features.py`

```python
#!/usr/bin/env python3
"""
Add Training-Only Features Pipeline

Adds features that USE FUTURE DATA and cannot be used in live prediction:
- Target calculation (4H lookahead, σ normalized)
- Statistical features (Hurst, entropy, CUSUM)

Part of Issue #7 (sub-issue #1.6 of Pipeline Restructuring Epic #1)

Usage:
    .venv/bin/python scripts/06_add_training_features.py

Input:
    data/features/training_shared_features.json (from issue #6)

Output:
    data/features/training_complete_features.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.features_training import add_all_training_features
import pandas as pd


def main():
    logger = setup_logger('add_training_features')

    # Paths
    input_path = 'data/features/training_shared_features.json'
    output_path = 'data/features/training_complete_features.json'

    logger.info("="*80)
    logger.info("ADD TRAINING-ONLY FEATURES")
    logger.info("="*80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info("")
    logger.info("⚠️  WARNING: USES FUTURE DATA - TRAINING ONLY!")
    logger.info("")

    # Load shared features
    logger.info("Loading shared features...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"✓ Loaded {len(df):,} records with {len(df.columns)} columns")

    # Add training-only features
    logger.info("")
    logger.info("Adding training-only features...")
    logger.info("  1. Target calculation (4H lookahead, σ normalized)")
    logger.info("  2. Hurst exponent (trend persistence)")
    logger.info("  3. Permutation entropy (predictability)")
    logger.info("  4. CUSUM signal (change detection)")

    df = add_all_training_features(df)

    logger.info(f"✓ Features added, shape: {df.shape}")

    # Analyze target distribution
    logger.info("")
    logger.info("Target distribution:")
    signals = (df['target'] != 0).sum()
    zeros = (df['target'] == 0).sum()
    logger.info(f"  Signals: {signals:,} ({signals/len(df)*100:.1f}%)")
    logger.info(f"  Zeros:   {zeros:,} ({zeros/len(df)*100:.1f}%)")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("")
    logger.info("Saving complete training features...")
    output_data = df.to_dict('records')

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    logger.info(f"✓ Saved {len(output_data):,} records")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Total features: {len(df.columns)} (93 shared + 4 training-only + OHLCV)")

    logger.info("")
    logger.info("="*80)
    logger.info("✅ TRAINING FEATURES COMPLETE")
    logger.info("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
```

## Design Decisions

### 1. Lookahead Period: 4 Hours

**Why 4 hours?**
- Short enough to be actionable for trading
- Long enough to capture meaningful reversals
- Matches the "ghost signal" concept (indicators flip before price)
- Tested and validated in original Ghost Trader V3

### 2. Volatility Normalization

**Why normalize by volatility?**
- Makes targets comparable across different market conditions
- High volatility periods → larger raw price changes, but same σ
- Low volatility periods → smaller raw price changes, but same σ
- Model learns RELATIVE significance, not absolute price changes

**Example:**
```
BTC @ $30K, volatility=2%: 10% move = 5σ signal
BTC @ $30K, volatility=10%: 10% move = 1σ signal (not significant)
```

### 3. Statistical Features: Optional but Useful

**These 3 features were excluded from shared pipeline because:**
- May be unstable on live data (Hurst, entropy)
- Grows indefinitely (CUSUM)
- Complex calculations may overfit

**But they're useful for training because:**
- Provide additional context about market regime
- Help model learn when to trust patterns
- Historical analysis is stable with full dataset

## Validation Strategy

### 1. Target Distribution Check
```python
# After adding training features
signals = (df['target'] != 0).sum()
zeros = (df['target'] == 0).sum()
print(f"Signals: {signals:,} ({signals/len(df)*100:.1f}%)")
print(f"Zeros: {zeros:,} ({zeros/len(df)*100:.1f}%)")

# Expected: ~20-30% signals, ~70-80% zeros
```

### 2. Target Value Sanity Check
```python
# Check for reasonable target values
print(f"Target mean: {df['target'].mean():.4f}")  # Should be ~0
print(f"Target std: {df['target'].std():.4f}")    # Should be ~3-5σ
print(f"Target max: {df['target'].max():.4f}")    # Should be <20σ
print(f"Target min: {df['target'].min():.4f}")    # Should be >-20σ
```

### 3. NaN Check
```python
# Ensure no NaN values in critical columns
assert df['target'].isna().sum() == 0, "Target has NaN values!"
assert df[TRAINING_ONLY_FEATURE_LIST].isna().sum().sum() == 0
```

## Integration with Pipeline

**Complete Pipeline Flow:**

```bash
# Phase 1: Data Collection (Issues #2-5)
.venv/bin/python scripts/01_download_training_binance.py
.venv/bin/python scripts/02_download_training_macro_binance.py

# Phase 2: Feature Engineering
.venv/bin/python scripts/05_add_shared_features.py --mode training
.venv/bin/python scripts/06_add_training_features.py  # ← Issue #7

# Phase 3: Training (Issue #8)
.venv/bin/python scripts/07_train_model.py
```

**Data Flow:**
1. Raw data → `data/raw/training/`
2. + Shared features → `data/features/training_shared_features.json` (Issue #6)
3. + Training features → `data/features/training_complete_features.json` (Issue #7) ← YOU ARE HERE
4. → Model training (Issue #8)

## Success Criteria

✅ **Module created:** `sneaker/features_training.py`
✅ **Script created:** `scripts/06_add_training_features.py`
✅ **Output file:** `data/features/training_complete_features.json`
✅ **Target column present:** `target` in σ units
✅ **Statistical features added:** 3 features
✅ **Target distribution reasonable:** 20-30% signals
✅ **No NaN values:** All features filled
✅ **File size reasonable:** ~3-4 GB for 791K records
✅ **Cannot be used for prediction:** Verified by code review

## Next Steps (Issue #8)

After completing Issue #7, Issue #8 will refactor the training script to:
1. Load `training_complete_features.json`
2. Extract feature list (93 shared features only, NO training-only)
3. Use `target` column as y
4. Train LightGBM with V3 sample weighting
5. Save model for use in prediction (Issue #9)

**Key Insight:** Model is trained using ALL features but prediction uses ONLY shared features. The training-only features are just for creating the target and providing training context.
