# Feature Normalization Strategy - Windowing

## Critical Issue: Absolute Scale Differences

**Problem:** Different cryptocurrencies and macro indicators have vastly different absolute price scales.

### Scale Comparison Examples

| Feature | BTC Example | Altcoin Example | Scale Difference |
|---------|-------------|-----------------|------------------|
| Price (close) | $100,000 | $0.50 | **200,000×** |
| VWAP | $100,500 | $0.51 | **197,000×** |
| ATR (14-period) | $3,000 | $0.05 | **60,000×** |
| MACD histogram | 500 | 0.001 | **500,000×** |
| macro_GOLD_close | $2,000 | $2,000 | 1× (same for all) |
| macro_BTC_PREMIUM_close | $100,000 | $100,000 | 1× (same for all) |

**Impact Without Normalization:**
- Model learns different weights for BTC vs altcoins
- Features from expensive assets dominate training
- Poor generalization across different price ranges
- Macro features have inconsistent influence

## Solution: Per-Window Normalization

**Strategy:** Normalize features relative to t0 (first candle in window)

### Example Transformation

**Before (Raw Values):**
```python
# BTC window (price ~$100k)
{
  'macro_BTC_PREMIUM_close_t0': 100000,
  'macro_BTC_PREMIUM_close_t1': 100500,
  'macro_BTC_PREMIUM_close_t11': 101000,
  'vwap_20_t0': 100200,
  'vwap_20_t1': 99800,
  'vwap_20_t11': 100400
}

# Altcoin window (price ~$0.50)
{
  'macro_BTC_PREMIUM_close_t0': 100000,
  'macro_BTC_PREMIUM_close_t1': 100500,
  'macro_BTC_PREMIUM_close_t11': 101000,
  'vwap_20_t0': 0.50,
  'vwap_20_t1': 0.49,
  'vwap_20_t11': 0.52
}
```

**After (Normalized to t0):**
```python
# BTC window
{
  'macro_BTC_PREMIUM_close_t0': 1.0,      # 100000 / 100000
  'macro_BTC_PREMIUM_close_t1': 1.005,    # 100500 / 100000 = +0.5%
  'macro_BTC_PREMIUM_close_t11': 1.01,    # 101000 / 100000 = +1.0%
  'vwap_20_t0': 1.0,                       # 100200 / 100200
  'vwap_20_t1': 0.996,                     # 99800 / 100200 = -0.4%
  'vwap_20_t11': 1.002                     # 100400 / 100200 = +0.2%
}

# Altcoin window
{
  'macro_BTC_PREMIUM_close_t0': 1.0,      # 100000 / 100000
  'macro_BTC_PREMIUM_close_t1': 1.005,    # 100500 / 100000 = +0.5%
  'macro_BTC_PREMIUM_close_t11': 1.01,    # 101000 / 100000 = +1.0%
  'vwap_20_t0': 1.0,                       # 0.50 / 0.50
  'vwap_20_t1': 0.98,                      # 0.49 / 0.50 = -2.0%
  'vwap_20_t11': 1.04                      # 0.52 / 0.50 = +4.0%
}
```

**Key Insight:**
- Macro features are now identical (1.0, 1.005, 1.01) - same BTC movement affects all pairs equally
- VWAP shows relative price movement within the window
- Model learns from **percentage changes**, not absolute values

## Features Requiring Normalization (15 total)

### 1. Macro Close Prices (4 features)
```python
'macro_GOLD_close'         # $2,000 ± $100
'macro_BNB_close'          # $600 ± $50
'macro_BTC_PREMIUM_close'  # $100,000 ± $5,000
'macro_ETH_PREMIUM_close'  # $3,500 ± $300
```

**Why:** Different assets have vastly different absolute prices (50× range)

**Normalization:** Divide by t0 value
- Shows relative movement of macro indicator within window
- Same for all pairs (macro is global context)

### 2. Macro Velocities (4 features)
```python
'macro_GOLD_vel'           # ±$10 per hour
'macro_BNB_vel'            # ±$5 per hour
'macro_BTC_PREMIUM_vel'    # ±$500 per hour
'macro_ETH_PREMIUM_vel'    # ±$50 per hour
```

**Why:** Absolute velocity scales with asset price (100× range)

**Normalization:** Divide by t0 velocity
- Shows acceleration/deceleration of macro movement
- Comparable across different macro indicators

### 3. VWAP (1 feature)
```python
'vwap_20'  # Pair-specific absolute price
```

**Why:** Tied to pair's price (200,000× range from BTC to altcoins)

**Normalization:** Divide by t0 VWAP
- Shows price position relative to VWAP at window start
- t0=1.0 means price equals VWAP
- t11=1.02 means price 2% above initial VWAP
- Captures price drift relative to volume-weighted average

### 4. ATR - Average True Range (2 features)
```python
'atr_14'    # 14-period ATR in absolute price units
'atr_vel'   # Change in ATR
```

**Why:** Volatility scales with price (60,000× range)

**Normalization:** Divide by t0 ATR
- Shows relative volatility change
- t0=1.0 is baseline volatility
- t11=1.5 means volatility increased 50% during window
- t11=0.7 means volatility decreased 30%

### 5. MACD Histogram (4 features)
```python
'macd_hist'      # Raw MACD histogram
'macd_hist_vel'  # Change in MACD histogram
'macd_hist_2x'   # 2× aggregated MACD
'macd_hist_4x'   # 4× aggregated MACD
```

**Why:** MACD scales with price (500,000× range)

**Normalization:** Divide by t0 MACD histogram
- Shows relative momentum change
- t0=1.0 is baseline momentum
- t11=2.0 means momentum doubled (strengthening trend)
- t11=-0.5 means momentum flipped and is opposite of t0

## Features NOT Requiring Normalization (78 total)

### Already 0-100 Scale (18 features)
```python
# RSI family
'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel', 'rsi_accel', 'rsi_7_accel',
'rsi_2x', 'rsi_4x'

# Stochastic
'stoch', 'stoch_vel', 'stoch_accel'

# ADX
'adx', 'adx_vel', 'adx_accel'

# Other oscillators
'volatility_regime'  # Encoded 0/1/2
```

**Why:** Already bounded to 0-100 range, scale-independent

### Already -1 to +1 or Percentage Scale (35 features)
```python
# Bollinger Bands position
'bb_position', 'bb_position_vel', 'bb_position_accel',
'bb_pos_2x', 'bb_pos_4x'

# Directional Indicator
'di_diff', 'di_diff_vel', 'di_diff_accel'

# Price ROC (rate of change) - all in %
'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20',
'price_accel_5', 'price_accel_10', 'price_change_2x', 'price_change_4x'

# Macro ROC - all in %
'macro_GOLD_roc_5', 'macro_BNB_roc_5',
'macro_BTC_PREMIUM_roc_5', 'macro_ETH_PREMIUM_roc_5'

# Distance features - all in %
'dist_from_high_20', 'dist_from_low_20', 'dist_from_vwap',
'vwap_distance_pct', 'price_range_position'

# Volatility z-score and percentile
'vol_zscore', 'vol_percentile'
```

**Why:** Already expressed as percentages or normalized ratios

### Ratios (14 features)
```python
# Volume ratios
'vol_ratio', 'vol_ratio_vel', 'vol_ratio_accel', 'vol_ratio_2x', 'vol_ratio_4x'

# ADR (advance/decline ratio)
'adr'

# Interaction features (normalized ratios)
'rsi_bb_interaction', 'macd_vol_interaction', 'rsi_stoch_interaction',
'bb_vol_interaction', 'adx_di_interaction', 'price_rsi_momentum_align'

# Volume momentum
'vol_momentum_5'
```

**Why:** Ratios are inherently scale-independent

### Binary/Count Features (11 features)
```python
# Binary flags
'is_up_bar', 'is_up_streak', 'is_new_high_20', 'is_new_low_20',
'is_high_volume', 'is_strong_trend', 'is_weak_trend',
'vol_regime_low', 'vol_regime_med', 'vol_regime_high'

# Counts
'adr_up_bars', 'adr_down_bars', 'consecutive_higher_highs',
'consecutive_lower_lows', 'squeeze_duration'

# High/low levels
'price_20_high', 'price_20_low'
```

**Why:** Binary (0/1) or small integer counts, scale-independent

### Divergence Features (5 features)
```python
'price_rsi_divergence', 'price_macd_divergence', 'price_stoch_divergence',
'rsi_divergence_strength', 'macd_divergence_strength'
```

**Why:** Already normalized as strength indicators

## Implementation Details

### Division-by-Zero Handling

```python
if abs(t0_value) > 1e-10:  # Avoid division by zero
    normalized_value = row[feature] / t0_value
else:
    # If t0 is zero, use 1.0 (no change) or 0.0 based on current value
    normalized_value = 1.0 if abs(row[feature]) < 1e-10 else 0.0
```

**Edge Cases:**
1. **t0=0, current=0:** Both zero → normalized=1.0 (no change from zero baseline)
2. **t0=0, current≠0:** Can't normalize → normalized=0.0 (fallback)
3. **t0≠0, current=0:** 0/t0 = 0.0 (dropped to zero)
4. **Both non-zero:** current/t0 (standard normalization)

### Performance Impact

**Before Normalization:**
- BTC features dominate (large absolute values)
- Model learns pair-specific patterns
- Poor generalization to new pairs or different price regimes

**After Normalization:**
- All pairs treated equally
- Model learns relative patterns (% changes, momentum)
- Better generalization across price scales
- Expected improvement: +5-15% in cross-pair R²

## Validation Checks

After windowing, verify normalization:

```python
# For normalized features, t0 should always be 1.0
assert windowed_df['macro_BTC_PREMIUM_close_t0'].std() < 0.01  # All ~1.0

# For normalized features, other time steps should vary
assert windowed_df['macro_BTC_PREMIUM_close_t11'].std() > 0.01  # Varying

# For non-normalized features, t0 should vary by pair
assert windowed_df['rsi_t0'].std() > 5.0  # Varies (0-100 range)
```

## Impact on Model Training

### Before (No Normalization)
```python
# Model learns:
"When macro_BTC_PREMIUM_close = 100000, and vwap_20 = 100500, predict +4.5σ"

# Problem: Only works for BTC-sized prices!
# For altcoin at $0.50, model sees completely different numbers
```

### After (With Normalization)
```python
# Model learns:
"When macro_BTC_PREMIUM_close = 1.005 (up 0.5% from t0),
 and vwap_20 = 1.002 (up 0.2% from t0), predict +4.5σ"

# Works for ALL price scales!
# BTC at $100k and altcoin at $0.50 both show same relative pattern
```

## Summary

**15 Features Normalized** (absolute scale issues):
- Macro prices (4) + velocities (4)
- VWAP (1)
- ATR (2)
- MACD histogram (4)

**78 Features NOT Normalized** (already scale-independent):
- Oscillators (RSI, Stoch, ADX): 0-100 scale
- Percentages/ROC: % scale
- Ratios: inherently normalized
- Binary/counts: small integers
- Positions: -1 to +1 scale

**Method:** Divide by t0 (first candle in window)

**Result:**
- All features now comparable across price scales
- Model learns relative patterns, not absolute values
- Expected to significantly improve generalization

---

**Implementation:** `scripts/07_create_windows.py`
**Commit:** `900c7f4`
**Issue:** #8
