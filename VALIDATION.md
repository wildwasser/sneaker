# Validation Requirements

This document defines the mandatory validation requirements and baseline metrics for the Sneaker project.

## ⚠️ Fundamental Truth

**This codebase may contain statistical illusions.** Every change must be rigorously validated. The V3 model claims 74% signal R² and 98% direction accuracy - these claims must be continuously verified.

## Mandatory Validation Pipeline

Every change to the model, features, or training process **MUST pass both validation tests:**

### 1. Model Regression Analysis

**Script:** `scripts/validate_model.py`

**Command:**
```bash
.venv/bin/python scripts/validate_model.py
```

**What It Checks:**
- Overfitting (train vs test R² gap)
- Feature importance dominance
- Signal metrics (R², direction accuracy)
- Signal rate sanity
- Statistical red flags

**Pass Criteria:**
| Metric | Threshold | Status |
|--------|-----------|--------|
| Signal R² (test) | ≥ 70% | ✅ Must pass |
| Direction Accuracy (test) | ≥ 95% | ✅ Must pass |
| Train/Test R² Gap | ≤ 10% | ✅ Must pass |
| Max Feature Importance | ≤ 40% | ✅ Must pass |
| Signal Rate (4σ) | ≤ 20% | ✅ Must pass |

**Red Flags (Auto-Reject):**
- 🚩 Train/test gap >15% (severe overfitting)
- 🚩 Signal R² >99% (too perfect)
- 🚩 Feature importance >50% (single feature dominance)
- 🚩 Signal rate >30% (too many signals)

### 2. LINKUSDT 256-Hour Backtest

**Script:** `scripts/backtest.py`

**Command:**
```bash
export BINANCE_API='your_key'
export BINANCE_SECRET='your_secret'
.venv/bin/python scripts/backtest.py --pair LINKUSDT --hours 256
```

**What It Tests:**
- Real trading performance on recent data
- Risk-adjusted returns (Sharpe ratio)
- Win rate vs random (50%)
- Risk management (max drawdown)
- Trade frequency sanity

**Pass Criteria:**
| Metric | Threshold | Status |
|--------|-----------|--------|
| Sharpe Ratio | ≥ 1.0 | ✅ Must pass |
| Win Rate | ≥ 50% | ✅ Must pass |
| Max Drawdown | ≤ 25% | ✅ Must pass |
| Number of Trades | 2-15 (in 256h) | ✅ Must pass |

**Red Flags (Auto-Reject):**
- 🚩 Zero trades (model not generating signals)
- 🚩 100% win rate (suspiciously perfect)
- 🚩 Sharpe ratio >5.0 (unrealistic)
- 🚩 Max drawdown >50% (unacceptable risk)

## Baseline Metrics (V3 Model - Unverified Claims)

**⚠️ WARNING: These metrics are from the original Ghost Trader project and may be statistical illusions. They require independent verification.**

### Training Metrics (Claimed)
```
Dataset: 917,100 candles (20 pairs, 2021-2025)
Signals: 239,491 (26% of dataset)
Training: 90/10 split, 5x sample weighting

Signal R² (train): ~74% (claimed)
Signal R² (test): ~74% (claimed)
Direction Accuracy: ~98% (claimed)
Zero MAE: ~2.2σ (claimed)
Signal Rate (4σ): ~5% (claimed)
```

**Status:** ❌ UNVERIFIED - Requires validation

### Known Issues & Concerns

1. **Statistical Illusion Risk:**
   - High claimed metrics may be artifacts of:
     - Look-ahead bias in feature engineering
     - Data leakage between train/test
     - Spurious correlations in indicators
     - Overfitting to historical patterns

2. **Backtesting Gaps:**
   - No historical backtests provided
   - No out-of-sample validation on new data
   - No comparison to baseline strategies
   - No transaction cost modeling

3. **Feature Engineering Concerns:**
   - 83 features may be excessive (overfitting risk)
   - Multi-timeframe aggregations (2x, 4x) could introduce look-ahead
   - Statistical features (Hurst, entropy) may not be robust
   - Feature interactions not validated independently

4. **Signal Detection:**
   - "Ghost signal" concept is unproven
   - Indicator momentum shifts may be noise
   - Volatility normalization assumes stationary variance
   - No statistical test for signal validity

## Validation History

### Current Status: BASELINE ESTABLISHED
- ✅ Git repository initialized
- ✅ GitHub repo created: https://github.com/wildwasser/sneaker
- ✅ Issue templates created
- ✅ Validation scripts created
- ✅ Workflow requirements documented
- ❌ Model regression validation: NOT RUN
- ❌ Backtest validation: NOT RUN
- ❌ Baseline metrics: NOT VERIFIED

**Next Steps:**
1. Create Issue #1: Run baseline validation
2. Execute `validate_model.py` on existing model
3. Execute `backtest.py` on LINKUSDT 256h
4. Document actual results vs claimed metrics
5. Identify statistical red flags
6. Create issues for any problems found

## Development Workflow Integration

### Before Starting Work
1. Create GitHub issue with validation plan
2. Document expected impact on metrics
3. Define success criteria

### During Development
1. Work on issue-numbered branch
2. Make changes incrementally
3. Test locally before validation

### Before Merging
1. Run `validate_model.py` - MUST PASS
2. Run `backtest.py --pair LINKUSDT --hours 256` - MUST PASS
3. Document results in issue comments
4. Create PR only if both pass
5. Merge only after review + approval

### If Validation Fails
1. ❌ DO NOT MERGE
2. Document failure in issue
3. Create "Validation Failure" issue
4. Investigate root cause:
   - Overfitting?
   - Feature leakage?
   - Implementation bug?
   - Fundamental approach flaw?
5. Revise or abandon approach
6. Re-validate before retry

## Validation Scripts Usage

### Quick Validation (Both Tests)
```bash
# Run both validation tests sequentially
.venv/bin/python scripts/validate_model.py && \
.venv/bin/python scripts/backtest.py --pair LINKUSDT --hours 256
```

### Custom Validation
```bash
# Test specific model file
.venv/bin/python scripts/validate_model.py --model models/experimental.txt

# Test different pairs/periods
.venv/bin/python scripts/backtest.py --pair BTCUSDT --hours 512 --threshold 3.0

# Test with different thresholds
.venv/bin/python scripts/backtest.py --threshold 5.0  # More conservative
```

### Interpreting Results

**Regression Analysis Output:**
```
✅ VALIDATION PASSED - Model appears statistically sound
  All criteria met, no red flags

❌ VALIDATION FAILED - Statistical illusions or issues detected
  One or more criteria failed or red flags detected
  DO NOT MERGE - investigate issues
```

**Backtest Output:**
```
✅ BACKTEST PASSED - Model performance acceptable
  Meets risk-adjusted return thresholds
  Trading behavior reasonable

❌ BACKTEST FAILED - Performance issues detected
  Poor risk-adjusted returns OR
  Suspicious trading patterns OR
  Red flags detected
  DO NOT MERGE - investigate issues
```

## Statistical Illusion Detection Guide

### Common Red Flags

1. **Overfitting:**
   - Train R² much higher than test R² (>15% gap)
   - Perfect in-sample metrics, poor out-of-sample
   - Backtest fails but validation metrics excellent

2. **Data Leakage:**
   - Suspiciously perfect predictions (>99% accuracy)
   - Model "predicts" events it shouldn't know about
   - Features using future information

3. **Spurious Correlations:**
   - Single feature dominates (>50% importance)
   - High R² with nonsensical features
   - Correlations that don't make economic sense

4. **Selection Bias:**
   - Cherry-picked time periods
   - Optimized thresholds on test set
   - Best metrics from multiple runs

5. **P-Hacking:**
   - Many features tested, few retained
   - Parameters tuned extensively
   - Multiple model variations tried

### How to Investigate

If validation fails:

1. **Check Feature Importance:**
   - Are top features sensible?
   - Any single feature >40%?
   - Do features align with theory?

2. **Examine Residuals:**
   - Plot predicted vs actual
   - Look for patterns in errors
   - Check for heteroscedasticity

3. **Test on Different Data:**
   - Try different pairs
   - Try different time periods
   - Try walk-forward validation

4. **Simplify Model:**
   - Remove suspicious features
   - Reduce feature count
   - Test with linear model

5. **Check Assumptions:**
   - Is data stationary?
   - Are features independent?
   - Is signal definition valid?

## Future Enhancements (Optional)

**Not required for merge, but valuable for rigor:**

1. **Walk-Forward Validation:**
   - Train on period 1, test on period 2
   - Retrain, test on period 3, etc.
   - Ensures no look-ahead bias

2. **Multiple Pair Validation:**
   - Test on all 20 pairs
   - Aggregate statistics
   - Check consistency

3. **Permutation Tests:**
   - Shuffle targets, retrain
   - Compare metrics to random
   - Establish statistical significance

4. **Feature Ablation:**
   - Remove features one by one
   - Test impact on performance
   - Identify truly important features

5. **Regime Analysis:**
   - Test in different market conditions
   - Bull, bear, sideways markets
   - High vs low volatility

## Summary

**Validation is MANDATORY. No exceptions.**

Every change must prove it doesn't introduce statistical illusions. The burden of proof is on the developer to show the change is valid.

**Trust nothing. Validate everything.**

---

**Last Updated:** 2025-11-12
**Status:** Baseline established, validation pending
**Next Issue:** #1 - Run baseline validation on existing model
