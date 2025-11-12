# Sneaker: Cryptocurrency Reversal Prediction with Rigorous Validation

**A clean implementation extracted from Ghost Trader - now with mandatory proof-based validation to prevent statistical illusions.**

---

## ⚠️ CRITICAL: Validation Status

**CLAIMS UNVERIFIED UNTIL ISSUE #1 COMPLETE**

This codebase was extracted from the Ghost Trader project, which became an unmaintainable mess. While it claims excellent performance (74% R² on signals, 98% direction accuracy), **these claims have not been independently verified.**

**The model may contain statistical illusions.**

### Current Status

- ✅ **Proof-based validation system implemented**
- ✅ **Rigorous workflow established**
- ✅ **GitHub issue tracking active**
- ✅ **Visual evidence system operational**
- ❌ **Baseline validation NOT YET RUN**
- ❌ **Claims UNVERIFIED**

**See Issue #1 for baseline validation (pending).**

---

## What is Sneaker?

Sneaker predicts cryptocurrency price reversals using LightGBM regression with sample weighting. It detects "ghost signals" - indicator momentum shifts that precede price reversals.

**Key Features:**
- 83 technical features (RSI, MACD, Bollinger Bands, etc.)
- V3 sample weighting (5x for signals)
- Volatility-normalized targets
- 1H timeframe, Binance data
- **Claimed:** 5% signal rate at 4σ, 74% R² on signals *(unverified)*

**What makes this different:** We assume nothing works until proven with visual evidence.

---

## 🔬 Proof-Based Validation System

**Every change requires rigorous validation with visual proof.**

### The Proof Folder

Every validation creates timestamped evidence:

```
proof/
└── issue-X/
    ├── validation_report_{timestamp}.txt
    ├── regression_analysis_{timestamp}.png (4 subplots)
    ├── residual_analysis_{timestamp}.png (4 subplots)
    ├── feature_importance_{timestamp}.png (color-coded)
    └── signal_distribution_{timestamp}.png (4 subplots)
```

**Key principle:** If there's no proof, it didn't happen.

### Validation Visualizations

**Auto-generated on every validation:**

1. **Regression Analysis** - Predicted vs actual (train/test, all/signals)
2. **Residual Analysis** - Pattern detection, normality checks
3. **Feature Importance** - Top 10 features, dominance warnings (red >40%)
4. **Signal Distribution** - Actual vs predicted, signal rate curves

**Red flags are color-coded and automatically detected.**

### Pass Criteria

**All must pass before merging:**

| Check | Threshold | Status |
|-------|-----------|--------|
| Signal R² | ≥ 70% | ✅ Required |
| Direction Accuracy | ≥ 95% | ✅ Required |
| Train/Test Gap | ≤ 10% | ✅ Required |
| Feature Dominance | ≤ 40% | ✅ Required |
| Signal Rate (4σ) | ≤ 20% | ✅ Required |
| Backtest Sharpe | ≥ 1.0 | ✅ Required |
| Win Rate | ≥ 50% | ✅ Required |
| Max Drawdown | ≤ 25% | ✅ Required |

**If any check fails, DO NOT MERGE.**

---

## 🚀 Quick Start

### 1. Installation

```bash
cd sneaker
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt

# Set up Binance API credentials (for data collection)
export BINANCE_API='your_api_key'
export BINANCE_SECRET='your_secret_key'
```

### 2. Run Baseline Validation (Issue #1)

**FIRST STEP: Validate existing model claims**

```bash
# Create issue
gh issue create --title "[VALIDATION] Baseline model validation"

# Run validation (creates proof/issue-1/)
.venv/bin/python scripts/validate_model.py --issue 1

# Review visual evidence
open proof/issue-1/*.png  # macOS
# or
xdg-open proof/issue-1/*.png  # Linux

# Commit proof
git add proof/issue-1/
git commit -m "Add #1: Baseline validation proof"
git push origin main

# Close issue with findings
gh issue close 1 --comment "Results documented in proof/issue-1/"
```

**This will reveal if the claimed 74% R² is real or a statistical illusion.**

### 3. Development Workflow (All Future Work)

**NO work without GitHub issue. NO merges without validation proof.**

```bash
# 1. Create issue
gh issue create --title "[FEATURE] Add new indicator"
# Note: Issue #42

# 2. Create branch
git checkout -b issue-42-add-indicator

# 3. Make changes
# ... edit code ...
git add -A
git commit -m "Add #42: Implement new indicator"

# 4. MANDATORY: Validate with proof generation
.venv/bin/python scripts/validate_model.py --issue 42

# 5. MANDATORY: Backtest (if model changed)
.venv/bin/python scripts/backtest.py --issue 42 --pair LINKUSDT --hours 256

# 6. Review proof/issue-42/ visualizations
# Look for red flags!

# 7. If PASSED: Commit proof
git add proof/issue-42/
git commit -m "Add #42: Validation proof - PASSED"
git push origin issue-42-add-indicator

# 8. Create PR (only if validation passed)
gh pr create --title "Fix #42: Add new indicator"

# 9. Merge
gh pr merge 42 --squash

# 10. Cleanup
git checkout main && git pull
git branch -d issue-42-add-indicator
```

**See `WORKFLOW.md` for complete step-by-step guide (400+ lines).**

---

## 📖 Documentation

**Comprehensive workflow documentation:**

- **[WORKFLOW.md](WORKFLOW.md)** - Complete step-by-step workflow (400+ lines)
  - Phase 1: Planning (issue creation, validation plan)
  - Phase 2: Development (branch, commit, push)
  - Phase 3: Validation (run tests, review proof)
  - Phase 4: Commit proof (document results)
  - Phase 5: PR & merge (only if passed)
  - Phase 6: Failed validation handling
  - Troubleshooting, special cases, checklists

- **[VALIDATION.md](VALIDATION.md)** - Pass/fail criteria, red flag detection
- **[proof/README.md](proof/README.md)** - Proof folder interpretation guide
- **[PROOF_SYSTEM.md](PROOF_SYSTEM.md)** - Implementation summary
- **[CLAUDE.md](CLAUDE.md)** - AI assistant instructions

**Start with WORKFLOW.md for complete usage instructions.**

---

## 🎯 The Concept: "Ghost Signals"

**Ghost signals** are indicator momentum shifts that precede price reversals:

1. Multiple technical indicators flip direction simultaneously
2. This creates an "echo" or "ghost" of the coming price move
3. The model learns to detect patterns and predict reversal magnitude

**Example:** RSI crosses 50, BB position flips, MACD histogram changes sign, Stochastic reverses - all at once. This synchronized flip often happens BEFORE the price actually reverses.

**Status:** Concept is interesting but **unproven**. Validation will reveal if this is real or confirmation bias.

---

## 🧮 The V3 Innovation: Sample Weighting

**Claimed innovation** (requires validation):

```python
# Problem: 26% signals, 74% zeros in training data
# Without weighting: model predicts zero (0% signals)
# With filtering: model never learns "normal" (40% signals)

# Solution: Weight signals 5x more
sample_weights[y != 0] = 5.0  # Ghost signals
sample_weights[y == 0] = 1.0  # Normal candles

# Result: 64% effective influence for signals
```

**Claimed outcome:**
- Signal R²: 74% *(unverified)*
- Direction accuracy: 98% *(unverified)*
- Signal rate: 5% at 4σ *(unverified)*

**Validation required to confirm these claims are not statistical illusions.**

---

## 📊 83 Enhanced V3 Features

**Core Indicators (20):**
RSI, RSI-7, Bollinger Bands, MACD, Stochastic, Directional Indicators, ADX, Advance/Decline Ratio, Volume Ratio, VWAP

**Momentum Features (24):**
Price ROC, acceleration, indicator acceleration, volatility momentum, multi-timeframe 2x, price action metrics

**Advanced Features (35):**
Multi-timeframe 4x, indicator interactions, volatility regime, price extremes, trend patterns, divergences, volume patterns

**Statistical Features (4):**
Hurst exponent, permutation entropy, CUSUM signal, squeeze duration

**Total: 83 features**

**Concern:** 83 features is a lot. Overfitting risk. Feature importance analysis required.

---

## 🔴 Red Flags We Watch For

**Validation automatically detects these problems:**

### Model Issues
- 🚩 Train R² >> Test R² (>15% gap) - **OVERFITTING**
- 🚩 Feature >50% importance - **EXTREME DOMINANCE**
- 🚩 Signal R² >99% - **TOO PERFECT (suspicious)**
- 🚩 Signal rate >30% - **TOO MANY SIGNALS**

### Visual Red Flags
- 🚩 Regression plots: Points far from diagonal
- 🚩 Residual plots: Patterns (heteroscedasticity)
- 🚩 Feature importance: Red bars (>40%)
- 🚩 Distribution: Predicted ≠ actual shape

### Backtest Red Flags
- 🚩 100% win rate - **TOO PERFECT**
- 🚩 Sharpe >5.0 - **UNREALISTIC**
- 🚩 Zero trades - **MODEL NOT WORKING**
- 🚩 Drawdown >50% - **UNACCEPTABLE RISK**

**All red flags are color-coded in visualizations.**

---

## 📁 Project Structure

```
sneaker/
├── README.md                    # This file
├── WORKFLOW.md                  # Complete workflow guide (400+ lines)
├── VALIDATION.md                # Pass/fail criteria
├── PROOF_SYSTEM.md              # Implementation summary
├── CLAUDE.md                    # AI assistant instructions
├── requirements.txt             # Dependencies
│
├── sneaker/                     # Core module
│   ├── __init__.py
│   ├── logging.py              # Logging utilities
│   ├── data.py                 # Binance API integration
│   ├── indicators.py           # 20 technical indicators
│   ├── features.py             # 83-feature engineering
│   └── model.py                # Model loading & prediction
│
├── scripts/                     # Pipeline scripts
│   ├── 01_collect_data.py      # Fetch Binance data
│   ├── 04_train_model.py       # Train with V3 weighting
│   ├── 05_predict.py           # Generate predictions
│   ├── validate_model.py       # ✅ Validation with proof generation
│   └── backtest.py             # ✅ Backtest with proof generation
│
├── proof/                       # ✅ VALIDATION EVIDENCE
│   ├── README.md               # Interpretation guide
│   └── issue-X/                # One folder per issue
│       ├── validation_report_{timestamp}.txt
│       ├── regression_analysis_{timestamp}.png
│       ├── residual_analysis_{timestamp}.png
│       ├── feature_importance_{timestamp}.png
│       └── signal_distribution_{timestamp}.png
│
├── .github/
│   └── ISSUE_TEMPLATE/          # Standardized issue templates
│       ├── bug_report.md
│       ├── feature_request.md
│       └── validation_failure.md
│
├── models/
│   └── production.txt          # Trained V3 model (34MB)
│
└── data/
    └── enhanced_v3_dataset.json  # Training data (2.9GB, 917K candles)
```

---

## 🎨 What Validation Looks Like

**When you run validation:**

```bash
$ .venv/bin/python scripts/validate_model.py --issue 42

MODEL VALIDATION - Issue #42
Statistical Illusion Check
================================================================================
...

GENERATING PROOF VISUALIZATIONS
================================================================================
Creating regression analysis plot...
  Saved: proof/issue-42/regression_analysis_2025-11-12_16-30-00.png
Creating residual analysis plot...
  Saved: proof/issue-42/residual_analysis_2025-11-12_16-30-00.png
Creating feature importance plot...
  Saved: proof/issue-42/feature_importance_2025-11-12_16-30-00.png
Creating signal distribution plot...
  Saved: proof/issue-42/signal_distribution_2025-11-12_16-30-00.png
Saving validation report...
  Saved: proof/issue-42/validation_report_2025-11-12_16-30-00.txt

================================================================================
PASS/FAIL CRITERIA
================================================================================
✅ PASS - Signal R² ≥ 70%
✅ PASS - Direction Acc ≥ 95%
✅ PASS - Train/Test Gap ≤ 10%
✅ PASS - No Feature Dominance (>40%)
✅ PASS - Signal Rate ≤ 20%

RED FLAG CHECKS
================================================================================
✅ No red flags detected

================================================================================
✅ VALIDATION PASSED - Model appears statistically sound
================================================================================

📁 Proof saved to: proof/issue-42

Next steps:
  1. Review visualizations in proof/issue-42
  2. git add proof/issue-42
  3. git commit -m 'Add #42: validation proof'
  4. Continue with backtest: .venv/bin/python scripts/backtest.py --issue 42
```

**Then you visually inspect the 4 PNG files to confirm no hidden issues.**

---

## 📈 Claimed Performance (UNVERIFIED)

**Test Set (91,710 candles):**
- Signal R²: 74.03% *(claimed, unverified)*
- Direction Accuracy: 98.33% *(claimed, unverified)*
- Overall R²: 9.29% *(not meaningful - mostly zeros)*
- Zero MAE: 2.22σ *(claimed, unverified)*

**Live Testing (LINKUSDT, 179 candles):**
- 4σ threshold: 5.0% signals *(claimed, unverified)*
- 3σ threshold: 14.5% signals *(claimed, unverified)*
- 5σ threshold: 1.1% signals *(claimed, unverified)*

**Training Data:**
- 917,100 candles from 20 pairs (2021-2025)
- 239,491 ghost signals detected
- 90/10 train/test split

**⚠️ ALL CLAIMS REQUIRE VERIFICATION VIA ISSUE #1**

---

## 🚫 What We Left Behind (Ghost Project)

**Failed experiments NOT included:**
- ❌ Voting ensembles (R² negative!)
- ❌ Sign/magnitude split
- ❌ Censored regression
- ❌ Quantile regression
- ❌ V1 (40% signal rate - too many)
- ❌ V2 (0% signal rate - too few)

**Removed complexity:**
- ❌ 94+ unresolved issues
- ❌ Complex chopper classes
- ❌ Dual offset aggregation
- ❌ Confusing multi-version docs
- ❌ Experimental code everywhere

**Sneaker = Only what (supposedly) works, plus rigorous validation**

---

## 💡 Philosophy

### KISS (Keep It Simple, Stupid)
- One approach (V3 sample weighting)
- Clean, readable code
- Clear documentation
- **Rigorous validation before merge**

### Trust Nothing, Validate Everything
- Visual evidence required
- Pass/fail criteria enforced
- Red flags auto-detected
- Complete audit trail

### Issue-Driven Development
- No work without GitHub issue
- Issue → Branch → Proof folder (aligned)
- Complete traceability
- Failed validations preserved

**If it doesn't pass validation, it doesn't merge.**

---

## 🔧 Dependencies

**Core:**
- Python 3.12.9
- LightGBM (primary model)
- XGBoost (available)

**Data & ML:**
- numpy, pandas, scipy
- scikit-learn
- python-binance

**Visualization:**
- matplotlib, seaborn, plotly

**See `requirements.txt` for complete list.**

---

## ❓ FAQ

**Q: Can I trust the claimed 74% R²?**
**A:** Not until Issue #1 (baseline validation) completes. It could be a statistical illusion.

**Q: Why all the validation overhead?**
**A:** Ghost Trader had 94+ issues because claims weren't rigorously validated. We're preventing that.

**Q: What if validation fails?**
**A:** Good! That means we caught a statistical illusion before it went to production. Document it, investigate, fix or abandon.

**Q: Can I skip validation for small changes?**
**A:** No. Small changes can have big impacts. Every code change requires validation.

**Q: Why use scripts instead of an agent for workflow?**
**A:** Scripts are deterministic, debuggable, fast, and reliable. Agents are unpredictable and opaque.

**Q: What's a "statistical illusion"?**
**A:** When your model appears to work in-sample but fails out-of-sample due to overfitting, data leakage, or spurious correlations.

**Q: Can I use this for live trading?**
**A:** Not until baseline validation passes AND you've done extensive backtesting. This is experimental.

---

## 🚀 Next Steps

### Immediate (First Time Setup)

1. **Install dependencies** (see Quick Start above)

2. **Run Issue #1: Baseline Validation**
   ```bash
   .venv/bin/python scripts/validate_model.py --issue 1
   ```

3. **Review proof/issue-1/** - Are claims valid?

4. **Document findings** - Update this README with actual results

### Future Development

**Every new feature:**
1. Create GitHub issue
2. Create `issue-X-description` branch
3. Make changes
4. **MANDATORY:** Run `validate_model.py --issue X`
5. **MANDATORY:** Run `backtest.py --issue X` (if model changed)
6. Review `proof/issue-X/` visualizations
7. Commit proof
8. Create PR (only if validation passed)
9. Merge to main

**See `WORKFLOW.md` for complete details.**

---

## 📧 GitHub Repository

**https://github.com/wildwasser/sneaker**

**Issues:**
- Issue #1: Baseline validation (pending)
- Future issues: TBD based on validation results

**Branches:**
- `main` - Production code (protected, requires validation)
- `issue-X-description` - Feature branches (one per issue)

**Proof folder tracked in main branch for complete transparency.**

---

## ⚖️ License & Disclaimer

**MIT License** - Use at your own risk.

**DISCLAIMER:**
- This is a machine learning experiment, **NOT financial advice**
- Cryptocurrency trading is extremely risky
- Claimed performance is **UNVERIFIED**
- Model may contain **statistical illusions**
- Past performance (if real) does not guarantee future results
- You can lose all your capital

**USE AT YOUR OWN RISK**

---

## 🎯 Summary

Sneaker is a cryptocurrency reversal prediction system extracted from Ghost Trader, now with **mandatory proof-based validation** to prevent statistical illusions.

**Key features:**
- LightGBM with V3 sample weighting
- 83 technical features
- "Ghost signal" detection
- **Rigorous validation with visual proof**
- **Complete GitHub audit trail**
- **No merges without validation**

**Current status:**
- ✅ Proof system operational
- ✅ Workflow documented
- ❌ Claims unverified (Issue #1 pending)

**Philosophy:** Trust nothing. Validate everything. If there's no proof, it didn't happen.

**Read `WORKFLOW.md` for complete usage instructions.**

---

**Generated:** November 2025
**Status:** Baseline validation pending
**Next:** Issue #1 - Verify claimed metrics are real
