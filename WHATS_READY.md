# What's Ready in Sneaker

**Date:** November 12, 2025
**Status:** Core extraction complete, needs full pipeline scripts

---

## ✅ What's Done

### 1. Clean Structure Created
```
sneaker/
├── README.md                          ✅ Fresh, clear documentation
├── requirements.txt                   ✅ Minimal dependencies
├── EXTRACTION_PLAN.md                 ✅ Detailed extraction plan
├── sneaker/                           ✅ Core module directory
│   ├── __init__.py                    ✅ Module init
│   └── logging.py                     ✅ Logging utilities
├── scripts/
│   └── 04_train_model.py              ✅ V3 training script (simplified)
├── models/
│   └── production.txt                 ✅ V3 trained model (34MB)
└── data/
    └── enhanced_v3_dataset.json       ✅ Full dataset (2.9GB)
```

### 2. Key Files Extracted

**✅ V3 Model:**
- File: `models/production.txt` (34MB)
- Source: `ghost/models/production_model_v3.txt`
- Performance: Signal R² 74%, 5% signal rate at 4σ

**✅ Enhanced V3 Dataset:**
- File: `data/enhanced_v3_dataset.json` (2.9GB)
- Source: `ghost/data_artifacts/ghost_signals_volnorm_enhanced_v3_all_pairs.json`
- Contains: 917,100 candles with 83 features and targets

**✅ V3 Training Script:**
- File: `scripts/04_train_model.py`
- Simplified from `ghost/scripts/train_production_model_v3.py`
- Standalone, no Ghost dependencies

**✅ Core Utilities:**
- `sneaker/logging.py` - Clean logging setup
- `sneaker/__init__.py` - Module initialization

**✅ Documentation:**
- `README.md` - Complete usage guide
- `EXTRACTION_PLAN.md` - What was extracted and why
- `requirements.txt` - Minimal dependencies

---

## ⏳ What's Still Needed

### Missing Pipeline Scripts

These need to be created (can copy from Ghost and simplify):

**1. Data Collection (`scripts/01_collect_data.py`)**
- Source: `ghost/scripts/download_1h_data.py`
- What it does: Fetch 1H candles from Binance (20 pairs)
- Dependencies: python-binance, pandas

**2. Signal Detection (`scripts/02_detect_signals.py`)**
- Source: `ghost/test/detect_ghost_signals_volnorm.py`
- What it does: Detect indicator momentum shifts (ghost signals)
- Creates: Volnorm-normalized targets

**3. Feature Engineering (`scripts/03_add_features.py`)**
- Sources:
  - `ghost/test/add_momentum_features.py` (Batch 1: 24 features)
  - `ghost/test/add_even_more_features.py` (Batch 2: 35 features)
  - `ghost/test/add_issue70_statistical_features.py` (Batch 3: 4 features)
- What it does: Add 83 Enhanced V3 features to candles

**4. Prediction (`scripts/05_predict.py`)**
- Source: `ghost/scripts/test_live_predictions_thresholds.py`
- What it does: Load model, make predictions, apply threshold

### Missing Core Modules

**`sneaker/data.py`:**
- Binance API wrapper
- Data preprocessing
- Simple, clean implementation

**`sneaker/indicators.py`:**
- Technical indicator calculations (RSI, BB, MACD, etc.)
- Core 20 indicators needed for features

**`sneaker/signals.py`:**
- Ghost signal detection logic
- Volatility normalization

**`sneaker/features.py`:**
- 83 feature engineering functions
- Consolidated from Ghost's scattered code

**`sneaker/model.py`:**
- Model loading/prediction utilities
- Threshold application

---

## 🎯 Quick Start Options

### Option 1: Just Train (Data Already There)

Since we have the dataset, you can train immediately:

```bash
cd /Volumes/Storage/python/sneaker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/04_train_model.py
```

**Result:** Trains a new model from the Enhanced V3 dataset

### Option 2: Full Pipeline (Needs Missing Scripts)

To collect fresh data and run end-to-end:

```bash
# Step 1-3: Need to create these scripts first
python scripts/01_collect_data.py     # ⏳ TODO
python scripts/02_detect_signals.py   # ⏳ TODO
python scripts/03_add_features.py     # ⏳ TODO

# Step 4: Ready!
python scripts/04_train_model.py      # ✅ DONE

# Step 5: Need to create
python scripts/05_predict.py          # ⏳ TODO
```

### Option 3: Use Existing Model

The V3 model is already there, just need prediction script:

```bash
# Load model and predict on new data
python scripts/05_predict.py --pair BTCUSDT  # ⏳ TODO
```

---

## 📋 Next Steps to Complete Extraction

### Priority 1: Create Prediction Script (Most Useful)

**File:** `scripts/05_predict.py`
**Why:** Lets you use the existing V3 model right away
**Copy from:** `ghost/scripts/test_live_predictions_thresholds.py`
**Simplify:** Remove Ghost dependencies, make standalone

### Priority 2: Create Core Modules

These enable the full pipeline:

1. **`sneaker/indicators.py`**
   - Extract indicator calculations from `ghost/features/indicators.py`
   - Just the 20 core indicators used in Enhanced V3

2. **`sneaker/signals.py`**
   - Extract from `ghost/test/detect_ghost_signals_volnorm.py`
   - Ghost signal detection logic

3. **`sneaker/features.py`**
   - Consolidate from Ghost's scattered feature scripts
   - All 83 features in one place

4. **`sneaker/data.py`**
   - Extract from `ghost/data/binance_client.py` and `download.py`
   - Simple Binance API wrapper

### Priority 3: Create Pipeline Scripts

Once modules are done, create the pipeline scripts:

1. `scripts/01_collect_data.py` - Fetch from Binance
2. `scripts/02_detect_signals.py` - Find ghost signals
3. `scripts/03_add_features.py` - Add 83 features

### Priority 4: Testing

Create basic tests to verify everything works:
- `tests/test_indicators.py`
- `tests/test_signals.py`
- `tests/test_features.py`
- `tests/test_model.py`

---

## 🎨 Philosophy Check

**Ghost:** Complex, tangled, full of failed experiments, confusing

**Sneaker:** Simple, clean, only what works, clear

**Current Status:** We've started fresh with clean structure and documentation. Now we need to extract and simplify the working code without bringing over the mess.

---

## 💡 Recommendations

### For Quick Results:

1. **Create `scripts/05_predict.py` first**
   - Lets you use the existing V3 model immediately
   - Copy from Ghost's test_live_predictions_thresholds.py
   - Simplify for Binance data fetching and model loading

2. **Test with the existing model**
   - Validate predictions on recent BTCUSDT data
   - Confirm 5% signal rate at 4σ threshold

### For Full Pipeline:

1. **Extract core modules systematically**
   - Start with indicators.py (needed by everything)
   - Then signals.py (depends on indicators)
   - Then features.py (depends on both)
   - Finally data.py (simple wrapper)

2. **Test each module independently**
   - Don't wait until the end
   - Verify each piece works before moving on

3. **Create pipeline scripts last**
   - They're just wrappers around the modules
   - Should be simple once modules are done

---

## ✅ What's Working Right Now

**You can already train a model:**
```bash
cd /Volumes/Storage/python/sneaker
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas lightgbm scikit-learn matplotlib scipy
python scripts/04_train_model.py
```

This will retrain from the Enhanced V3 dataset and should give you:
- Signal R² ~74%
- Direction accuracy ~98%
- Near-zero predictions ~26%
- Strong predictions ~7%

**The model file is there:**
- `models/production.txt` (34MB)
- Can be loaded with LightGBM immediately
- Just needs a prediction script to use it

---

## 🚀 Ready to Continue?

**What's done:**
- ✅ Clean structure
- ✅ V3 model copied
- ✅ Dataset copied
- ✅ Training script simplified
- ✅ Documentation written

**What's next:**
- Create prediction script (Priority 1)
- Extract core modules (Priority 2)
- Create pipeline scripts (Priority 3)
- Add tests (Priority 4)

**Current state:** Foundation is solid. Need to fill in the missing pieces systematically.

The "new barber" (Sneaker) has the space set up, the good equipment (V3 model + dataset), and a plan. Now we need to finish unpacking the tools and set up the full shop!
