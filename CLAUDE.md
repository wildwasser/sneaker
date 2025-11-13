# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 PROJECT STATUS: PIPELINE COMPLETE, READY FOR OPTIMIZATION

### Current State (Epic #1 Complete ✅)

**Infrastructure: FULLY FUNCTIONAL** ✅
- Complete training pipeline (data → windowed features → trained model)
- Complete prediction pipeline (data → windowed features → predictions)
- Windowed time series architecture (12-candle windows, 1,116 features)
- Per-window normalization (15 features relative to t0)
- Binance-native 24/7 macro data (GOLD, BNB, BTC_PREMIUM, ETH_PREMIUM)

**Model Performance: POOR** ❌ (Expected, needs optimization)
- Test Signal R²: 19% (target: ≥70%)
- Direction Accuracy: 4% (target: ≥95%)
- **This is OK!** Infrastructure is correct, model just needs tuning

**What This Means:**
- ✅ Pipeline is modular and ready for experimentation
- ✅ All scripts tested and working
- ✅ Feature alignment verified (training = prediction)
- ✅ Normalization working correctly
- 🔧 **Ready for:** Hyperparameter tuning, feature selection, model optimization

### Complete Pipeline Architecture

**Training Pipeline (6 scripts):**
```bash
# 1. Download data (791K candles, 20 pairs)
scripts/01_download_training_binance.py
scripts/02_download_training_macro_binance.py

# 2. Add features (93 shared + 4 training-only)
scripts/05_add_shared_features.py --mode training
scripts/06_add_training_features.py

# 3. Train model (windowed, normalized)
scripts/07_create_windows.py        # 12-candle windows → 1,116 features
scripts/08_train_model.py           # LightGBM with V3 weighting
```

**Prediction Pipeline (4 scripts):**
```bash
# 1. Download data (256h LINKUSDT)
scripts/03_download_prediction_binance.py
scripts/04_download_prediction_macro_binance.py

# 2. Add features (same 93 shared features)
scripts/05_add_shared_features.py --mode prediction

# 3. Generate predictions (same windowing + normalization)
scripts/09_create_prediction_windows.py    # Exact same normalization as training
scripts/10_generate_predictions.py         # Load model, generate signals
```

### Key Innovations

**1. Windowed Time Series (12 candles):**
- Model sees temporal context (trends, momentum)
- 93 features × 12 time steps = 1,116 features per window
- Sliding window moves 1 candle at a time

**2. Per-Window Normalization (CRITICAL!):**
- **15 features normalized** relative to t0 (first candle in window)
- Solves scale difference: BTC at $100k vs altcoin at $0.50
- Features: macro prices (4), macro velocities (4), VWAP (1), ATR (2), MACD (4)
- **Exact same normalization** in training and prediction (verified!)

**3. Binance-Native Macro (24/7, Zero Gaps):**
- GOLD (PAXGUSDT): Gold proxy
- BNB (BNBUSDT): Exchange token flow
- BTC_PREMIUM: USDT-margined futures premium
- ETH_PREMIUM: Coin-margined futures premium

**4. 93 Shared Features:**
- 12 macro features (4 indicators × 3 derivatives)
- 20 core indicators (RSI, BB, MACD, Stoch, ADX, etc.)
- 61 derived features (momentum, advanced, trend, statistical)
- All calculated identically in training and prediction

### What's Ready for Optimization

**Easy to Experiment With:**
- 🔧 **Hyperparameters:** LightGBM parameters (num_leaves, max_depth, learning_rate, etc.)
- 🔧 **Feature Selection:** Which of 93 features matter most?
- 🔧 **Window Size:** Is 12 candles optimal? Try 8, 16, 24?
- 🔧 **Model Types:** Try XGBoost, neural networks, ensembles?
- 🔧 **Sample Weighting:** Is 5× for signals optimal?
- 🔧 **Target Definition:** Refine ghost signal detection?
- 🔧 **Pairs:** Test on different cryptocurrencies?

**Fast Iteration:**
- All intermediate results cached (data/features/, data/raw/)
- Change feature → Just re-run from script 05 onwards
- Change model params → Just re-run script 08
- Test predictions → Just run scripts 09-10

### Recent Completed Issues

- ✅ **Issue #1** (Epic): Pipeline restructuring complete
- ✅ **Issue #2-5**: Binance-native data downloads
- ✅ **Issue #6-7**: Shared and training feature engineering
- ✅ **Issue #8**: Windowed training with normalization
- ✅ **Issue #18**: Windowed prediction pipeline
- ✅ **Issue #9**: Closed (superseded by #18)

### Key Documentation

- **ISSUE_18_ANALYSIS.md**: Prediction pipeline implementation guide
- **NORMALIZATION_STRATEGY.md**: Per-window normalization explained
- **CLAUDE.md** (this file): Development guide
- **proof/issue-8/**: Training validation visualizations

### Directory Structure

```
data/
├── raw/                    # Raw downloads (gitignored)
│   ├── training/           # 791K candles, 20 pairs
│   └── prediction/         # 256h LINKUSDT
├── features/               # Feature engineering (gitignored)
│   ├── *_shared_features.json         # 93 shared features
│   ├── *_complete_features.json       # With training features
│   └── windowed_*.json                # 1,116 windowed features
└── predictions/            # Prediction outputs (gitignored)
    └── latest.json

models/
└── issue-1/
    └── model.txt          # 24.88 MB LightGBM model

proof/
└── issue-8/               # Training validation proof
    ├── training_regression_*.png
    ├── training_residuals_*.png
    ├── training_feature_importance_*.png
    └── training_signal_dist_*.png

scripts/                   # 10 pipeline scripts (see above)

sneaker/                   # Core modules
├── features_shared.py     # SHARED_FEATURE_LIST (93 features)
├── macro_*.py             # Macro data modules
└── ...
```

### Next Steps

**Current Focus: Model Optimization**

The infrastructure is complete and correct. Now optimize:
1. Hyperparameter tuning (LightGBM grid search)
2. Feature importance analysis (which features actually matter?)
3. Feature selection (reduce from 93 to top N)
4. Window size optimization (test 8, 16, 24 candles)
5. Ensemble methods (combine multiple models)

**How to Start Optimizing:**
1. Create new issue for optimization experiment
2. Create branch: `issue-N-optimize-hyperparams`
3. Modify parameters in script 08
4. Re-run training: `.venv/bin/python scripts/08_train_model.py --issue issue-N`
5. Check proof visualizations in `proof/issue-N/`
6. Compare to baseline (issue-8)
7. Merge if improved, document if failed

---

## ⚠️ CRITICAL: Virtual Environment Usage - MANDATORY

**ALL Python operations MUST use the virtual environment located at `.venv/`**

### Strict Requirements:
- **NEVER** use global Python, pip, or any global Python tools
- **ALWAYS** use `.venv/bin/python` for running Python scripts
- **ALWAYS** use `.venv/bin/pip` for package management
- **ALWAYS** use `.venv/bin/<tool>` for any Python-based tools (pytest, etc.)
- **NO EXCEPTIONS** - if you need to run Python code, it runs through `.venv/`

### Current Virtual Environment Status:
- Location: `.venv/`
- Python version: 3.12.9
- Packages: Fully installed (see requirements.txt)
- Status: Ready to use

### Why This Matters:
This project uses an isolated virtual environment to prevent dependency conflicts and ensure reproducible builds. Using global Python environments is strictly prohibited.

## ⚠️ CRITICAL: Development Workflow - MANDATORY

**ALL development work MUST follow this strict workflow to prevent statistical illusions and maintain code quality.**

### Fundamental Truth
**This codebase may be fundamentally flawed with statistical illusions.** Every change must be rigorously validated before merging to main. Trust nothing. Validate everything.

### GitHub Repository
**URL:** https://github.com/wildwasser/sneaker

### MANDATORY: GitHub Issue Tracking

**Every piece of work MUST be tracked in a GitHub issue BEFORE starting work.**

1. **Create Issue First:**
   ```bash
   # Create issue on GitHub before ANY code changes
   gh issue create --title "[TYPE] Description" --body "..."
   ```

2. **Use Issue Templates:**
   - Bug Report: For bugs or suspected statistical illusions
   - Feature Request: For new features (must include validation plan)
   - Validation Failure: When tests fail

3. **Reference Issues:**
   - All commits must reference issue number: `"Fix #123: description"`
   - All branches must include issue number: `issue-123-description`

**NO WORK WITHOUT AN ISSUE. NO EXCEPTIONS.**

### MANDATORY: Branch Strategy

**Main branch is PROTECTED. All work happens on issue-numbered feature branches.**

1. **Create Branch for Every Issue:**
   ```bash
   # After creating GitHub issue (e.g., issue #42)
   git checkout -b issue-42-short-description
   ```

2. **Branch Naming Convention:**
   - Format: `issue-<number>-<short-description>`
   - Examples:
     - `issue-1-validate-model-regression`
     - `issue-5-fix-feature-nan-values`
     - `issue-12-add-backtest-infrastructure`

3. **Branch Lifecycle:**
   - Create branch from latest main
   - Work exclusively on that branch
   - Run ALL validation tests on branch
   - Only merge to main when validation passes
   - Delete branch after successful merge

4. **NO Direct Commits to Main:**
   ```bash
   # WRONG - NEVER DO THIS
   git checkout main
   git commit -m "quick fix"  # ❌ FORBIDDEN

   # RIGHT - ALWAYS DO THIS
   git checkout -b issue-23-fix-thing
   git commit -m "Fix #23: proper fix"
   # ... run validation ...
   # ... merge via PR when tests pass ...
   ```

### MANDATORY: Validation Requirements

**Every change must pass BOTH validation tests before merging to main:**

#### 1. Model Regression Analysis (REQUIRED)
```bash
# Run regression analysis to check for statistical illusions
.venv/bin/python scripts/validate_model.py

# Must show:
# - Feature importance (check for suspicious features)
# - Train vs test metrics (check for overfitting)
# - Residual analysis (check for patterns)
# - Signal distribution (check for sanity)
```

**Pass Criteria:**
- Signal R² ≥ 70% (if degraded, reject)
- Direction accuracy ≥ 95%
- Test R² within 10% of train R² (overfitting check)
- No suspicious feature dominance (>40% importance)

#### 2. LINKUSDT 256-Hour Backtest (REQUIRED)
```bash
# Run standardized backtest on recent LINKUSDT data
.venv/bin/python scripts/backtest.py --pair LINKUSDT --hours 256

# Must show:
# - Sharpe ratio
# - Max drawdown
# - Win rate
# - Total return
# - Number of trades
```

**Pass Criteria:**
- Sharpe ratio ≥ 1.0 (risk-adjusted returns)
- Win rate ≥ 50% (better than random)
- Max drawdown ≤ 25% (risk management)
- Reasonable number of trades (2-15 in 256h)

#### Red Flags (Auto-Reject):
- 🚩 Train R² >> Test R² (>15% gap)
- 🚩 Backtest fails but in-sample metrics excellent
- 🚩 Suspiciously perfect metrics (>99% accuracy)
- 🚩 Single feature dominates (>50% importance)
- 🚩 All winning trades (100% win rate)
- 🚩 Model predicts every candle as signal (>20% signal rate)

### Merge Criteria: What is "True Success"?

**Only merge to main when ALL criteria are met:**

1. ✅ GitHub issue exists and is linked
2. ✅ Work done on issue-numbered branch
3. ✅ Model regression analysis passes (see criteria above)
4. ✅ LINKUSDT 256h backtest passes (see criteria above)
5. ✅ Code is clean, documented, and follows project style
6. ✅ No statistical red flags detected
7. ✅ Validation results documented in issue comments

**If ANY validation fails:**
- ❌ DO NOT MERGE
- Document failure in issue
- Investigate root cause
- Revise approach or abandon if fundamentally flawed

### Daily Development Workflow

```bash
# 1. Create issue on GitHub
gh issue create --title "[FEATURE] Add new indicator" \
  --body "Description... Validation plan..."

# 2. Note issue number (e.g., #42)

# 3. Create branch
git checkout main
git pull origin main
git checkout -b issue-42-add-new-indicator

# 4. Make changes
# ... edit code ...
git add -A
git commit -m "Add #42: Implement new indicator calculation"

# 5. Run validation tests (MANDATORY)
.venv/bin/python scripts/validate_model.py  # Must pass
.venv/bin/python scripts/backtest.py --pair LINKUSDT --hours 256  # Must pass

# 6. Document results in issue
gh issue comment 42 --body "Validation results: ..."

# 7. If validation passes, push and create PR
git push origin issue-42-add-new-indicator
gh pr create --title "Fix #42: Add new indicator" \
  --body "Closes #42. Validation results attached."

# 8. Merge only if approved and tests pass
gh pr merge 42 --squash

# 9. Clean up
git checkout main
git pull origin main
git branch -d issue-42-add-new-indicator
```

### Commands Reference

**GitHub CLI:**
```bash
# Create issue
gh issue create

# List issues
gh issue list

# View issue
gh issue view <number>

# Comment on issue
gh issue comment <number> --body "..."

# Close issue
gh issue close <number>

# Create PR
gh pr create

# Merge PR
gh pr merge <number>
```

**Git Workflow:**
```bash
# Check current branch
git branch --show-current

# Create branch from main
git checkout main && git pull && git checkout -b issue-X-name

# Commit with issue reference
git commit -m "Fix #X: description"

# Push branch
git push origin issue-X-name

# Delete local branch after merge
git branch -d issue-X-name

# Delete remote branch after merge
git push origin --delete issue-X-name
```

### 📖 Complete Workflow Documentation

**For detailed, step-by-step workflow instructions, see `WORKFLOW.md`.**

That document contains:
- Complete workflow from issue creation to merge
- Visual inspection checklists for proof folder
- Troubleshooting guide
- Special cases (hotfixes, multiple issues, re-validation)
- Quick reference commands

**Key documents:**
- `WORKFLOW.md` - Detailed step-by-step workflow (READ THIS FIRST)
- `VALIDATION.md` - Pass/fail criteria and red flag detection
- `proof/README.md` - Proof folder structure and interpretation

## Project Overview

**Sneaker** is a clean cryptocurrency reversal prediction system extracted from the "Ghost Trader" project. Ghost became an unmaintainable mess with 94+ issues, failed experiments, and confusing code. Sneaker contains ONLY the working V3 approach.

### Key Facts:
- **Language**: Python 3.12.9
- **ML Framework**: LightGBM (primary), XGBoost (available)
- **Data Source**: Binance 1H OHLCV candles
- **Core Innovation**: V3 sample weighting (5x for signals) solves class imbalance
- **Performance**: 74% R² on signals, 98% direction accuracy, 5% signal rate at 4σ
- **Training Data**: 917,100 candles from 20 pairs (2021-2025), 239,491 signals

## Project Philosophy: KISS

**Keep It Simple, Stupid**

- **One Approach Only**: V3 sample weighting (no experimental code)
- **Clean Code**: 85% less code than Ghost for same functionality
- **No Baggage**: No failed experiments, no historical cruft
- **Readable**: Clear, documented, maintainable
- **Working**: Everything tested and functional

**If it doesn't work or isn't essential, it's not here.**

## What is a "Ghost Signal"?

**Ghost signals** are indicator momentum shifts that precede price reversals:

1. Multiple technical indicators flip direction simultaneously
2. This creates an "echo" or "ghost" of the coming price move
3. The model learns to detect these patterns and predict reversal magnitude

Example: RSI crosses 50, BB position flips, MACD histogram changes sign, Stochastic reverses - all at once. This synchronized flip often happens BEFORE the price actually reverses.

## The V3 Innovation: Sample Weighting

**The Problem:**
- Training data: 26% ghost signals, 74% normal candles (zeros)
- Without weighting: Model learns to predict zero (safest bet) → 0% signals
- With filtering: Model never learns what "normal" looks like → 40% signals (too many!)

**The V3 Solution:**
```python
# Weight signals 5x more than zeros
sample_weights[y != 0] = 5.0  # Ghost signals
sample_weights[y == 0] = 1.0  # Normal candles

# Effective influence: 64% signals, 36% zeros
# Result: Model learns both patterns, generates optimal signal rate
```

**Outcome:**
- Signal R²: 74% (excellent)
- Signal frequency: 5% at 4σ (perfect for trading)
- Direction accuracy: 98% (knows up vs down)
- Zero MAE: 2.2σ (acceptable noise on non-signals)

## High-Level Architecture

### Complete Pipeline (5 Steps)

```
1. Collect Data (01_collect_data.py) ✅
   ├─> Fetches 1H candles from Binance for 20 pairs
   └─> Output: data/candles.json

2. Detect Signals (02_detect_signals.py) ❌ NOT IMPLEMENTED
   ├─> Identifies ghost signals (indicator momentum shifts)
   ├─> Applies volatility normalization
   └─> Output: data/signals.json
   NOTE: Use existing enhanced_v3_dataset.json instead

3. Add Features (03_add_features.py) ❌ NOT IMPLEMENTED
   ├─> Adds 83 Enhanced V3 features
   └─> Output: data/enhanced_v3_dataset.json
   NOTE: Already included in dataset from Ghost

4. Train Model (04_train_model.py) ✅
   ├─> LightGBM regression with 5x sample weighting
   ├─> 90/10 train/test split
   └─> Output: models/production.txt (34MB)

5. Predict (05_predict.py) ✅
   ├─> Downloads recent live data
   ├─> Adds all 83 features
   ├─> Generates predictions using 4σ threshold
   └─> Output: Trading signals (BUY/SELL/HOLD)
```

### Core Modules Architecture

```
sneaker/
├── __init__.py              # Module exports
├── logging.py              # Standardized logging setup
├── data.py                 # Binance API integration
├── indicators.py           # 20 core technical indicators
├── features.py             # 83-feature engineering pipeline
└── model.py                # Model loading, prediction, signal generation
```

**Module Characteristics:**
- **Standalone**: No Ghost dependencies
- **Simple**: Direct calculations, no complex frameworks
- **Documented**: Clear docstrings and examples
- **Tested**: All modules functional

### The 83 Enhanced V3 Features

#### Core Indicators (20 features) - `indicators.py`

**RSI Family (4):**
- `rsi`, `rsi_vel` (14-period)
- `rsi_7`, `rsi_7_vel` (7-period)

**Bollinger Bands (2):**
- `bb_position` (normalized -1 to +1)
- `bb_position_vel` (rate of change)

**MACD (2):**
- `macd_hist` (histogram)
- `macd_hist_vel` (momentum)

**Stochastic (2):**
- `stoch` (%K oscillator)
- `stoch_vel` (momentum)

**Directional Indicators (3):**
- `di_diff` (DI+ minus DI-)
- `di_diff_vel` (momentum)
- `adx` (trend strength)

**Advance/Decline (4):**
- `adr` (up/down ratio)
- `adr_up_bars`, `adr_down_bars` (counts)
- `is_up_bar` (binary)

**Volume (2):**
- `vol_ratio` (current/average)
- `vol_ratio_vel` (momentum)

**VWAP (1):**
- `vwap_20` (volume-weighted average price)

#### Batch 1: Momentum Features (24) - `add_momentum_features()`

- Price ROC (4): 3, 5, 10, 20 periods
- Price acceleration (2): 5, 10 periods
- Indicator acceleration (6): RSI, RSI7, BB, MACD, Stoch, DI
- Volatility momentum (4): regime velocity, vol ratio accel, ATR, ATR velocity
- Multi-timeframe 2x (4): RSI, BB, MACD, price change aggregations
- Price action (4): streak, distance from high/low, VWAP distance

#### Batch 2: Advanced Features (35) - `add_advanced_features()`

- Multi-timeframe 4x (5): Longer aggregations
- Indicator interactions (6): Cross-indicator relationships
- Volatility regime (6): Detailed classification
- Price extremes (3): New highs/lows
- Trend patterns (4): Higher highs, lower lows, VWAP analysis
- Divergences (5): Price vs indicator divergences
- Volume patterns (2): Abnormal volume detection
- Trend strength (4): ADX derivatives

#### Batch 3: Statistical Features (4) - `add_statistical_features()`

- `hurst_exponent`: Trend persistence vs mean reversion (>0.5 = trending, <0.5 = mean reverting)
- `permutation_entropy`: Predictability measure (lower = more predictable)
- `cusum_signal`: Cumulative sum for change point detection
- `squeeze_duration`: Bollinger Band squeeze length (consolidation periods)

**Total: 83 features** - All added via `add_all_features(df)` in features.py

## Common Development Tasks

### Running the Complete Pipeline

```bash
# Setup (one time)
cd /Volumes/Storage/python/sneaker
python3 -m venv .venv
source .venv/bin/activate  # Already done in this project
.venv/bin/pip install -r requirements.txt  # Already done

# Set Binance API credentials (for data collection)
export BINANCE_API='your_api_key'
export BINANCE_SECRET='your_secret_key'

# Run pipeline
.venv/bin/python scripts/01_collect_data.py     # Fetch data
# Skip 02 & 03 - use existing dataset
.venv/bin/python scripts/04_train_model.py      # Train model
.venv/bin/python scripts/05_predict.py --pair BTCUSDT --threshold 4.0
```

### Training a Model

```bash
# Uses existing enhanced_v3_dataset.json (2.9GB, 917K candles)
.venv/bin/python scripts/04_train_model.py
```

**Output:**
- Model file: `models/production.txt` (34MB)
- Training time: ~2 minutes
- Expected metrics:
  - Signal R²: ~74%
  - Direction accuracy: ~98%
  - Zero MAE: ~2.2σ

### Making Predictions

```bash
# Predict on live data
export BINANCE_API='your_key'
export BINANCE_SECRET='your_secret'

.venv/bin/python scripts/05_predict.py --pair BTCUSDT --threshold 4.0

# Options:
# --pair: Trading pair (default: BTCUSDT)
# --hours: Data window (default: 180)
# --threshold: Signal threshold in σ (default: 4.0)
```

**Output:**
- Console: Signal summary, current market signal
- Visualization: `visualizations/BTCUSDT_predictions_4.0sigma.png`

**Signal Interpretation:**
- `prediction > +4.0σ`: **BUY** signal (upward reversal expected)
- `prediction < -4.0σ`: **SELL** signal (downward reversal expected)
- `-4.0σ ≤ prediction ≤ +4.0σ`: **HOLD** (no strong signal)

### Collecting Fresh Data

```bash
# Fetch 50K candles from 20 default pairs
export BINANCE_API='your_key'
export BINANCE_SECRET='your_secret'

.venv/bin/python scripts/01_collect_data.py
```

**Output:**
- File: `data/candles.json` (~500MB)
- Time: ~5-10 minutes
- Note: To train on this, need to implement 02 & 03 (signal detection + feature engineering)

### Using Sneaker as a Package

```python
from sneaker import (
    setup_logger,
    download_live_data,
    add_all_features,
    load_model,
    generate_signals
)

# Setup
logger = setup_logger('my_app')

# Download recent data
df = download_live_data('BTCUSDT', hours=200)
logger.info(f"Downloaded {len(df)} candles")

# Add all 83 features
df = add_all_features(df)
logger.info(f"Added features, shape: {df.shape}")

# Extract feature matrix
FEATURE_LIST = [...]  # 83 feature names (see scripts/04_train_model.py)
X = df[FEATURE_LIST].values

# Load model and predict
model = load_model('models/production.txt')
signals, summary = generate_signals(model, X, threshold=4.0)

logger.info(f"Generated {summary['signal_pct']:.1f}% signals")
logger.info(f"Buy: {summary['buy_count']}, Sell: {summary['sell_count']}")
```

## Technical Details

### Model Training Configuration

**LightGBM Parameters:**
```python
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 255,
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': 2000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}
```

**Critical: Must Use Sample Weights!**
```python
# Compute weights
sample_weights = np.ones(len(y))
sample_weights[y != 0] = 5.0  # Signals weighted 5x
sample_weights[y == 0] = 1.0  # Zeros weighted 1x

# Train with weights
model.fit(
    X_train, y_train,
    sample_weight=sw_train,
    eval_set=[(X_test, y_test)],
    eval_sample_weight=[sw_test],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)
```

**Without sample weights, the model will not work properly!**

### Signal Thresholds

| Threshold | Signal Rate | Use Case |
|-----------|-------------|----------|
| 3σ | ~14.5% | Active trading, more opportunities |
| **4σ** | **~5%** | **Production (recommended)** |
| 5σ | ~1.1% | Conservative, very high confidence |

**4σ is the sweet spot**: ~1 signal per 20 hours at 1H timeframe

### Feature Engineering Pipeline

**Order matters! Must follow this sequence:**

1. **Add Core Indicators** (`add_core_indicators()`)
   - Creates 20 base indicators
   - Requires: OHLCV data with 'pair' column

2. **Add Momentum Features** (`add_momentum_features()`)
   - Creates 24 momentum features
   - Depends on: Core indicators

3. **Add Advanced Features** (`add_advanced_features()`)
   - Creates 35 advanced features
   - Depends on: Core indicators + momentum features

4. **Add Statistical Features** (`add_statistical_features()`)
   - Creates 4 statistical features
   - Depends on: All previous features

5. **Fill NaNs** (done automatically in `add_all_features()`)
   - Fills NaN values with 0
   - Safe for derived features

**Always use `add_all_features(df)` to ensure correct order.**

### Data Format Requirements

**Input DataFrame must have:**
- `open`, `high`, `low`, `close`: float (price data)
- `volume`: float (trading volume)
- `pair`: str (e.g., "BTCUSDT")
- `timestamp`: int (Unix timestamp in milliseconds) [optional but recommended]

**Minimum data requirements:**
- At least 50 candles (for rolling window calculations)
- 1H timeframe (model trained on 1H only)
- Continuous data (no gaps)

### Performance Expectations

**Training (on 917K candles):**
- Signal R²: ~74% (what matters!)
- Overall R²: ~9% (not meaningful - mostly zeros)
- Direction Accuracy: ~98%
- Zero MAE: ~2.2σ
- Training Time: ~2 minutes

**Prediction (4σ threshold):**
- Signal Rate: ~5% (1 per 20 hours)
- Buy/Sell Balance: ~50/50
- Strong Prediction Rate (>5σ): ~7%
- Near-Zero Rate (≤1σ): ~26%

## Project Status & What's Missing

### ✅ What Works (Ready to Use)

**Core Modules:**
- ✅ `sneaker/logging.py` - Logging utilities
- ✅ `sneaker/data.py` - Binance API wrapper
- ✅ `sneaker/indicators.py` - 20 core indicators
- ✅ `sneaker/features.py` - 83 feature engineering
- ✅ `sneaker/model.py` - Model loading & prediction

**Scripts:**
- ✅ `scripts/01_collect_data.py` - Data collection
- ✅ `scripts/04_train_model.py` - Model training
- ✅ `scripts/05_predict.py` - Live predictions

**Assets:**
- ✅ `models/production.txt` - Trained V3 model (34MB)
- ✅ `data/enhanced_v3_dataset.json` - Training data (2.9GB, 917K candles)

### ❌ What's Missing (Not Critical)

**Scripts:**
- ❌ `scripts/02_detect_signals.py` - Ghost signal detection
- ❌ `scripts/03_add_features.py` - Feature engineering script

**Why not critical:**
- The existing dataset (`enhanced_v3_dataset.json`) already has signals detected and features added
- These scripts would only be needed to process NEW raw data from scratch
- Current workflow: Use existing dataset OR predict on live data (which adds features automatically)

**Tests:**
- ❌ `tests/` directory is empty
- No unit tests yet (but all modules work)

### 🔄 Current Workflow

**To train a new model:**
```bash
# Use existing preprocessed dataset
.venv/bin/python scripts/04_train_model.py
```

**To make predictions:**
```bash
# Fetches live data, adds features automatically
.venv/bin/python scripts/05_predict.py --pair BTCUSDT
```

**To process completely new raw data:**
- Would need to implement 02 & 03
- OR manually run the feature pipeline:
  ```python
  from sneaker import add_all_features
  df = add_all_features(raw_df)
  # Then manually detect signals (volnorm logic)
  ```

## Dependencies

### Current Status
- requirements.txt header says "Ghost Trader" (legacy)
- All packages are already installed in `.venv/`
- Works fine despite the header

### Key Dependencies

**Core ML:**
- `numpy` (1.26.4) - Array operations
- `pandas` (2.2.3+) - Data manipulation
- `scipy` (1.15.1+) - Statistical features
- `scikit-learn` (1.6.1+) - Train/test split, metrics

**ML Models:**
- `lightgbm` (4.6.0+) - Primary model (V3)
- `xgboost` (2.1.4+) - Available but not used in V3

**Optimization:**
- `optuna` (4.6.0+) - Hyperparameter tuning
- `scikit-optimize` (0.10.2+) - Alternative optimizer

**Data:**
- `python-binance` (1.0.27+) - Binance API
- `yfinance` (0.2.66+) - Yahoo Finance (not used in V3)

**Technical Analysis:**
- `ta` (0.11.0+) - Technical indicator library

**Visualization:**
- `matplotlib` (3.10.0+) - Plotting
- `seaborn` (0.13.2+) - Statistical plots
- `plotly` (6.3.1+) - Interactive plots

**Utilities:**
- `joblib` (1.4.2+) - Serialization
- `PyYAML` (6.0.2+) - Config files
- `requests` (2.32.3+) - HTTP requests

**Testing:**
- `pytest` (8.4.2+) - Testing framework

### Installing/Updating Dependencies

```bash
# Install all dependencies
.venv/bin/pip install -r requirements.txt

# Add new package
.venv/bin/pip install <package-name>

# Update requirements (if needed)
.venv/bin/pip freeze > requirements.txt

# Check what's installed
.venv/bin/pip list
```

## Environment Variables

```bash
# Required for data collection and live predictions
export BINANCE_API='your_api_key'
export BINANCE_SECRET='your_api_secret'

# Get credentials from: https://www.binance.com/en/my/settings/api-management
# Permissions needed: Read-only (no trading required)
```

## Trading Pairs

**Default 20 pairs** (defined in `sneaker/data.py`):

```python
# Note: BNBUSDT removed - now used exclusively as macro indicator (exchange flow)
BASELINE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "MATICUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "SUIUSDT", "LINKUSDT",
    "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "TRXUSDT",
    "ALGOUSDT", "APTUSDT", "AAVEUSDT", "XLMUSDT", "XMRUSDT"
]
```

**Model trained on these 20 pairs.** Should work best on these, but may generalize to other liquid Binance pairs.

## Important Constraints

1. **1H timeframe only** - Model trained exclusively on 1H candles
2. **Minimum 50+ candles** - Needed for rolling window calculations
3. **Exact feature order** - Must use all 83 features in correct order
4. **No NaN values** - Features automatically filled with 0
5. **Volatility normalization** - Targets are in σ (sigma) units, not raw price
6. **Sample weighting required** - Training without weights will fail
7. **Binance data only** - Model trained on Binance, other exchanges may differ

## Simplifications from Ghost

### What We Left Behind

**Ghost's Complexity:**
- ❌ 1,688-line indicators.py (Sneaker: 250 lines, **85% reduction**)
- ❌ 94+ GitHub issues of history
- ❌ V1/V2 buggy models
- ❌ Failed experiments (voting ensembles, censored regression, etc.)
- ❌ Complex chopper/dual-offset infrastructure
- ❌ Unused features (futures data, macro indicators, etc.)
- ❌ Confusing multi-version documentation

**Sneaker's Simplicity:**
- ✅ One working approach (V3 sample weighting)
- ✅ Clean, standalone modules
- ✅ Direct calculations (no complex frameworks)
- ✅ Clear, current documentation
- ✅ Everything tested and functional

**Result:** Same functionality, 80%+ less complexity!

## Troubleshooting

### "Missing features" error
- Ensure all 83 features added via `add_all_features(df)`
- Check that input DataFrame has required OHLCV columns
- Verify 'pair' column exists

### "No signals generated"
- Check threshold (4σ may be too strict for short periods)
- Verify model loaded correctly
- Check that features were added properly
- Try lower threshold (3σ) for testing

### API errors
- Verify `BINANCE_API` and `BINANCE_SECRET` environment variables
- Check API key has Read permissions
- Ensure API key is not restricted by IP

### Model file not found
- Run `scripts/04_train_model.py` first
- Check `models/production.txt` exists (34MB)
- Verify working directory is project root

### Out of memory during training
- 917K samples requires ~8GB RAM
- Reduce batch size or use smaller dataset
- Close other applications

### Predictions seem wrong
- Verify using 1H data (not other timeframes)
- Check that pair is in BASELINE_PAIRS (model trained on these)
- Ensure features were added in correct order
- Confirm no data gaps or NaN values

## Key Files Reference

### Documentation
- `README.md` - Comprehensive usage guide
- `CLAUDE.md` - This file (development guide)
- `WHATS_READY.md` - What's implemented vs missing

### Core Modules
- `sneaker/__init__.py` - Package exports
- `sneaker/logging.py` - Logging setup (120 lines)
- `sneaker/data.py` - Binance API wrapper (203 lines)
- `sneaker/indicators.py` - 20 technical indicators (233 lines)
- `sneaker/features.py` - 83-feature engineering (392 lines)
- `sneaker/model.py` - Model utilities (139 lines)

### Scripts
- `scripts/01_collect_data.py` - Data collection (78 lines)
- `scripts/04_train_model.py` - V3 training (321 lines)
- `scripts/05_predict.py` - Live predictions (258 lines)

### Assets
- `models/production.txt` - Trained V3 model (34MB)
- `data/enhanced_v3_dataset.json` - Training data (2.9GB)

## Development Best Practices

### When Adding Features
1. Always test on small subset first
2. Verify no NaN values introduced
3. Check feature correlations (avoid duplicates)
4. Document what the feature measures
5. Add to ENHANCED_V3_FEATURES list

### When Modifying Model
1. Keep sample weighting (critical!)
2. Test on validation set first
3. Compare to V3 baseline metrics
4. Don't remove working code
5. Follow KISS principle

### When Creating Scripts
1. Use sneaker modules (don't duplicate code)
2. Add proper logging
3. Handle errors gracefully
4. Document usage in docstring
5. Test end-to-end

### When Refactoring
1. Don't bring back Ghost complexity
2. Keep modules standalone
3. Maintain clear interfaces
4. Test after each change
5. Update documentation

## Next Steps (Optional Enhancements)

**Potential additions** (not required, V3 works well as-is):

1. **Implement missing scripts:**
   - `scripts/02_detect_signals.py` - Ghost signal detection
   - `scripts/03_add_features.py` - Feature engineering wrapper

2. **Add backtesting:**
   - Historical performance validation
   - Signal profitability analysis
   - Optimal threshold determination

3. **Create tests:**
   - Unit tests for each module
   - Integration tests for pipeline
   - Performance benchmarks

4. **Add monitoring:**
   - Live signal tracking
   - Model drift detection
   - Performance dashboard

5. **API wrapper:**
   - REST API for predictions
   - WebSocket for live signals
   - Docker deployment

**But remember:** V3 already works excellently. Don't add complexity unless truly needed!

## Quick Reference

### Train Model
```bash
.venv/bin/python scripts/04_train_model.py
# Output: models/production.txt
# Time: ~2 min
```

### Make Predictions
```bash
export BINANCE_API='key'
export BINANCE_SECRET='secret'
.venv/bin/python scripts/05_predict.py --pair BTCUSDT
# Output: visualizations/BTCUSDT_predictions_4.0sigma.png
```

### Collect Data
```bash
export BINANCE_API='key'
export BINANCE_SECRET='secret'
.venv/bin/python scripts/01_collect_data.py
# Output: data/candles.json
```

### Load Model in Python
```python
from sneaker import load_model, generate_signals
model = load_model('models/production.txt')
signals, summary = generate_signals(model, X, threshold=4.0)
```

## Remember

**This is NOT Ghost!**
- No experimental code
- No complex infrastructure
- No historical baggage
- Only what works (V3)

**Keep it simple. Keep it clean. Keep it working.**

---

**For more details, see README.md.**
