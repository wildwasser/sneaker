# Sneaker: Simple Cryptocurrency Reversal Prediction

**A clean, minimal implementation extracted from the Ghost project mess.**

---

## What is This?

Sneaker predicts cryptocurrency price reversals using machine learning. It was extracted from the "Ghost" project after that codebase became an unmaintainable mess of 94+ issues and failed experiments.

**This is the ONE approach that actually works:**
- LightGBM regression with sample weighting
- 83 technical features
- Volatility-normalized targets
- **Result:** 5% signal rate at 4σ threshold, 74% R² on signals

---

## Why "Sneaker"?

Ghost was like a bad haircut - too many failed experiments, tangled code, nothing working. Sneaker is starting fresh with only the good parts. We're "sneaking" away from the mess with just what works.

---

## Quick Start

### 1. Install

```bash
cd /Volumes/Storage/python/sneaker
python3 -m venv .venv
source .venv/bin/activate  # or: .venv/bin/activate on some systems
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Step 1: Collect data from Binance (20 trading pairs, 1H candles)
.venv/bin/python scripts/01_collect_data.py

# Step 2: Detect ghost signals (indicator momentum shifts)
.venv/bin/python scripts/02_detect_signals.py

# Step 3: Add 83 technical features
.venv/bin/python scripts/03_add_features.py

# Step 4: Train model with sample weighting
.venv/bin/python scripts/04_train_model.py

# Step 5: Make predictions
.venv/bin/python scripts/05_predict.py --pair BTCUSDT
```

---

## What It Does

### Ghost Signals

**"Ghost signals"** are indicator momentum shifts that precede price reversals:

1. Multiple technical indicators flip direction simultaneously
2. This creates an "echo" or "ghost" of the coming price move
3. The model learns to detect these patterns and predict the magnitude

### The Pipeline

```
Raw 1H Candles
    ↓
Detect Ghost Signals (volnorm approach)
    ↓
Add 83 Technical Features
    ↓
Train LightGBM (5x sample weighting for signals)
    ↓
Predictions (use 4σ threshold)
```

### 83 Features (Enhanced V3)

**Original (20):** RSI, Bollinger Bands, MACD, Stochastic, ADX, ATR, Volume, VWAP

**Batch 1 - Momentum (24):** Price ROC, acceleration, multi-timeframe indicators, position metrics

**Batch 2 - Advanced (35):** Indicator interactions, regime detection, divergences, trend strength

**Batch 3 - Statistical (4):** Hurst exponent, permutation entropy, CUSUM, squeeze detection

---

## Key Innovation: Sample Weighting

**The Problem:**
- Training data: 26% ghost signals, 74% normal candles (zeros)
- Without weighting: model learns to predict zero (safest bet)
- With filtering: model never learns what "normal" looks like

**The Solution (V3):**
```python
# Weight signals 5x more than zeros
sample_weights[y != 0] = 5.0  # Ghost signals
sample_weights[y == 0] = 1.0  # Normal candles

# Result: 64% effective influence for signals despite being 26% of data
```

**Outcome:**
- Signal R²: 74%
- Signal frequency: 5% at 4σ (perfect for trading)
- Direction accuracy: 98%

---

## Trading Signals

**Use 4σ threshold for production:**

```python
if prediction > +4.0:
    # BUY signal (predicted upward reversal)
    pass
elif prediction < -4.0:
    # SELL signal (predicted downward reversal)
    pass
else:
    # HOLD (no strong signal)
    pass
```

**Expected:** ~1 signal per 20 hours at 1H timeframe (5% signal rate)

---

## File Structure

```
sneaker/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── sneaker/                     # Core module
│   ├── __init__.py
│   ├── logging.py              # Logging utilities
│   ├── data.py                 # Binance data fetching
│   ├── indicators.py           # Technical indicators
│   ├── signals.py              # Ghost signal detection
│   ├── features.py             # 83 feature engineering
│   └── model.py                # Training and prediction
├── scripts/                     # Pipeline scripts
│   ├── 01_collect_data.py
│   ├── 02_detect_signals.py
│   ├── 03_add_features.py
│   ├── 04_train_model.py
│   └── 05_predict.py
├── models/
│   └── production.txt          # Trained V3 model
└── data/                        # Data goes here
    └── candles.json
    └── signals.json
    └── features.json
```

---

## What Was Left Behind in Ghost

❌ **Failed Experiments:**
- Voting ensembles (R² negative!)
- Sign/magnitude split
- Censored regression
- Quantile regression
- V1 (trained only on signals - generated 40% signals)
- V2 (equal weighting - generated 0% signals)

❌ **Unused Infrastructure:**
- Complex chopper classes
- Dual offset aggregation
- Futures data integration
- Macro indicators
- Cross-exchange features

❌ **The Mess:**
- 94+ GitHub issues
- 3 model versions (2 broken)
- Confusing documentation
- Experimental scripts everywhere
- Nothing working together

---

## Philosophy

**Sneaker follows KISS:**
- **Keep It Simple, Stupid**
- One approach (V3 sample weighting)
- Clean, readable code
- Clear documentation
- Everything works

**If it's not essential, it's not here.**

---

## Dependencies

```
numpy          # Array operations
pandas         # Data manipulation
lightgbm       # Gradient boosting
python-binance # Binance API
matplotlib     # Visualization
scipy          # Statistical features
```

That's it. No complex frameworks.

---

## Performance

**Test Set (91,710 candles, 10%):**
- Overall R²: 9.29% (not meaningful - mostly zeros)
- **Signal R²: 74.03%** (what matters!)
- Direction Accuracy: 98.33%
- Zero MAE: 2.22σ (acceptable)

**Live Testing (LINKUSDT, 179 candles):**
- 4σ threshold: 5.0% signals (perfect!)
- 3σ threshold: 14.5% signals (more active)
- 5σ threshold: 1.1% signals (too conservative)

---

## FAQ

**Q: Why "sample weighting" instead of filtering?**
A: Filtering creates training/deployment mismatch. The model needs to see both signals and non-signals during training.

**Q: Why 4σ threshold?**
A: Balances opportunity (enough signals) vs noise (high precision). Generates ~5% signals which is ideal for selective trading.

**Q: Can I use this for other assets?**
A: The approach works for any cryptocurrency with liquid 1H data on Binance. Modify the pair list in `01_collect_data.py`.

**Q: Why LightGBM instead of XGBoost?**
A: LightGBM is faster, uses less memory, and performed equally well in testing.

**Q: What's the training data?**
A: 917,100 1H candles from 20 Binance trading pairs (2021-2025), with 239,491 ghost signals detected.

---

## Next Steps

1. Run the 5-step pipeline
2. Validate predictions on recent data
3. Paper trade with 4σ threshold
4. Backtest performance
5. Deploy if successful

---

## Credits

Extracted from the Ghost Trader project (November 2025).

Key insight: Sample weighting solves the class imbalance problem elegantly.

---

## License

MIT License - Use at your own risk. This is for educational purposes.

**DISCLAIMER:** Cryptocurrency trading is risky. This is a machine learning experiment, not financial advice.
