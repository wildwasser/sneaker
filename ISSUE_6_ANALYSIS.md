# Issue #6: Shared Features Pipeline - Ultra-Deep Analysis

## Critical Constraint

**NO LOOK-FORWARD FUNCTIONALITY**

The shared features pipeline must produce IDENTICAL features for both training and prediction:
- NO future data access
- EXACT same calculation logic
- Consistent NaN handling
- Deterministic results

## Data Structure Understanding

### Binance Crypto Data
```json
{
  "timestamp": 1609459200000,
  "open": 28923.63,
  "high": 29031.34,
  "low": 28690.17,
  "close": 28995.13,
  "volume": 2311.811445,
  "trades": 58389,
  "pair": "BTCUSDT"
}
```
- **Format**: Flat list of dicts
- **Training**: 791,044 records (20 pairs × ~39,552 candles each)
- **Prediction**: 256 records (LINKUSDT only)

### Binance Macro Data
```json
{
  "timestamp": 1609459200000,
  "open": 1935.82,
  "high": 1940.71,
  "low": 1931.36,
  "close": 1931.72,
  "volume": 11.541535,
  "ticker": "GOLD"
}
```
- **Format**: Flat list of dicts
- **Training**: 170,566 records (4 indicators × ~42,641 candles each)
- **Prediction**: 1,058 records (4 indicators × ~264 candles each)
- **Indicators**: GOLD (PAXGUSDT), BNB (BNBUSDT), BTC_PREMIUM, ETH_PREMIUM

## Feature Classification

### All 83 Current Features Analyzed

#### SHARED FEATURES (80 features - NO look-forward)

**Core Indicators (20)** - ALL SHARED ✅
1. rsi (14-period RSI)
2. rsi_vel (RSI velocity)
3. rsi_7 (7-period RSI)
4. rsi_7_vel
5. bb_position (Bollinger Band position)
6. bb_position_vel
7. macd_hist (MACD histogram)
8. macd_hist_vel
9. stoch (Stochastic oscillator)
10. stoch_vel
11. di_diff (DI+ minus DI-)
12. di_diff_vel
13. adx (Average Directional Index)
14. adr (Advance/Decline Ratio)
15. adr_up_bars
16. adr_down_bars
17. is_up_bar
18. vol_ratio (Volume ratio)
19. vol_ratio_vel
20. vwap_20 (VWAP 20-period)

**Momentum Features (24)** - ALL SHARED ✅
21-24. price_roc_3, price_roc_5, price_roc_10, price_roc_20 (Price rate of change)
25-26. price_accel_5, price_accel_10 (Price acceleration)
27-32. rsi_accel, rsi_7_accel, bb_position_accel, macd_hist_accel, stoch_accel, di_diff_accel
33-34. volatility_regime, vol_regime_vel (Volatility momentum)
35. vol_ratio_accel
36-37. atr_14, atr_vel (Average True Range)
38-41. rsi_2x, bb_pos_2x, macd_hist_2x, price_change_2x (2x timeframe aggregations)
42-45. is_up_streak, dist_from_high_20, dist_from_low_20, dist_from_vwap

**Advanced Features (35)** - 32 SHARED ✅
46-50. rsi_4x, bb_pos_4x, macd_hist_4x, price_change_4x, vol_ratio_4x (4x aggregations)
51-56. rsi_bb_interaction, macd_vol_interaction, rsi_stoch_interaction, bb_vol_interaction, adx_di_interaction, price_rsi_momentum_align
57-61. vol_percentile, vol_regime_low, vol_regime_med, vol_regime_high, vol_zscore (Volatility regime)
62-64. is_new_high_20, is_new_low_20, price_range_position (Price extremes)
65-66. consecutive_higher_highs, consecutive_lower_lows (Trend patterns)
67-68. vwap_distance_pct, price_20_high
69-73. price_rsi_divergence, price_macd_divergence, price_stoch_divergence, rsi_divergence_strength, macd_divergence_strength (Divergences - uses 5-period lookback, NO lookahead) ✅
74-75. vol_momentum_5, is_high_volume (Volume patterns)
76. price_20_low
77-80. adx_vel, adx_accel, is_strong_trend, is_weak_trend (Trend strength)

**Statistical Features (4)** - 1 SHARED ✅
81. squeeze_duration (Bollinger Band squeeze duration) ✅

#### TRAINING-ONLY FEATURES (3 features - Complex/stateful)

82. **hurst_exponent** ❌ - 40-period rolling, complex calculation, may be unstable on live data
83. **permutation_entropy** ❌ - Complex pattern analysis, may overfit
84. **cusum_signal** ❌ - Cumulative sum grows indefinitely, not suitable for live prediction

**Decision: Exclude these 3 from shared features**

## Macro Feature Design

### Strategy: Merge and Expand

We have 4 Binance-native macro indicators:
1. **GOLD** (PAXGUSDT) - Tokenized gold, commodity/safe haven
2. **BNB** (BNBUSDT) - Exchange flow/liquidity indicator
3. **BTC_PREMIUM** (BTCUSDT premium index) - BTC futures vs spot sentiment
4. **ETH_PREMIUM** (ETHUSDT premium index) - ETH futures vs spot sentiment

### Macro Features to Add (12 features)

**Close Prices (4)**:
- `macro_GOLD_close`
- `macro_BNB_close`
- `macro_BTC_PREMIUM_close`
- `macro_ETH_PREMIUM_close`

**Velocities (4)**:
- `macro_GOLD_vel` (1-period diff of close)
- `macro_BNB_vel`
- `macro_BTC_PREMIUM_vel`
- `macro_ETH_PREMIUM_vel`

**Rate of Change (4)**:
- `macro_GOLD_roc_5` (5-period % change)
- `macro_BNB_roc_5`
- `macro_BTC_PREMIUM_roc_5`
- `macro_ETH_PREMIUM_roc_5`

## Total Shared Features Count

- 20 core indicators
- 24 momentum features
- 32 advanced features (excluding 3 training-only statistical)
- 1 statistical feature (squeeze_duration)
- 12 macro features

**Total: 89 shared features**

## Implementation Architecture

### Module: `sneaker/features_shared.py`

```python
"""Shared feature engineering for both training and prediction.

CRITICAL: NO LOOK-FORWARD FUNCTIONALITY
All features must be calculable on live data without future information.
"""

def merge_macro_features(crypto_df, macro_df):
    """Merge macro indicators into crypto dataframe.

    Args:
        crypto_df: DataFrame with crypto OHLCV (has 'pair' column)
        macro_df: DataFrame with macro OHLCV (has 'ticker' column)

    Returns:
        DataFrame with macro features added
    """
    # Pivot macro data: timestamp x ticker -> wide format
    # Add close prices, velocities, ROC
    pass

def add_shared_features(df):
    """Add all shared features (no look-forward).

    Includes:
    - 20 core indicators
    - 24 momentum features
    - 32 advanced features
    - 1 statistical feature (squeeze_duration)
    - NO hurst_exponent, permutation_entropy, cusum_signal

    Args:
        df: DataFrame with OHLCV + macro features

    Returns:
        DataFrame with all 77 shared features added
    """
    pass
```

### Script: `scripts/05_add_shared_features.py`

```bash
# Usage for training data
.venv/bin/python scripts/05_add_shared_features.py --mode training

# Usage for prediction data
.venv/bin/python scripts/05_add_shared_features.py --mode prediction

# Or specify explicit paths
.venv/bin/python scripts/05_add_shared_features.py \
  --binance data/raw/training/binance_20pairs_1H.json \
  --macro data/raw/training/macro_training_binance.json \
  --output data/features/training_shared_features.json
```

## Output Format

```json
[
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
    "macro_BTC_PREMIUM_close": 12.5,
    "macro_ETH_PREMIUM_close": 0.8,
    "macro_GOLD_vel": 0.5,
    "macro_BNB_vel": -0.2,
    "macro_BTC_PREMIUM_vel": 1.1,
    "macro_ETH_PREMIUM_vel": 0.05,
    "macro_GOLD_roc_5": 0.25,
    "macro_BNB_roc_5": -0.5,
    "macro_BTC_PREMIUM_roc_5": 8.2,
    "macro_ETH_PREMIUM_roc_5": 6.1,
    "rsi": 65.4,
    "rsi_vel": 1.2,
    "... (all 89 shared features)"
  },
  ...
]
```

## Validation Plan

### 1. Feature Parity Test
- Run on same 10-record sample for both training and prediction modes
- Verify EXACT same feature values

### 2. NaN Handling Test
- Check for NaN values after feature engineering
- Verify fillna(0) is applied consistently

### 3. Timestamp Alignment Test
- Ensure all crypto records have matching macro data
- No missing macro values

### 4. Live Prediction Simulation
- Load prediction data (256 hours)
- Run shared features pipeline
- Verify all 89 features present and valid

## Key Decisions Summary

1. ✅ **Include 80 of 83 original features** (exclude 3 training-only statistical)
2. ✅ **Add 12 macro features** (4 close + 4 vel + 4 ROC)
3. ✅ **Total: 89 shared features**
4. ✅ **Merge macro on timestamp** (both are 1H frequency)
5. ✅ **Same code for training and prediction** (mode parameter)
6. ✅ **Output format: same as input** (flat list with features added)
7. ✅ **NaN handling: fillna(0) at the end**

## Next Steps

1. Create `sneaker/features_shared.py`
2. Create `scripts/05_add_shared_features.py`
3. Test on training data (791K records)
4. Test on prediction data (256 records)
5. Verify feature parity
6. Update issue #6 with results
