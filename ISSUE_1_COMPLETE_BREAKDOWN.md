# Issue #1: Complete Sub-Issue Breakdown

## GitHub Issue Creation Commands

```bash
# Create main epic
gh issue create --title "[EPIC] Issue #1: Pipeline Restructuring" \
  --body-file ISSUE_1_EPIC.md \
  --label "epic,enhancement"

# Create sub-issues
gh issue create --title "[#1.1] Download Training Binance Data" \
  --body-file issues/ISSUE_1.1_download_training_binance.md \
  --label "data,phase-1"

gh issue create --title "[#1.2] Download Training Macro Data" \
  --body "$(cat issues/ISSUE_1.2_download_training_macro.md)" \
  --label "data,phase-1"

gh issue create --title "[#1.3] Download Prediction Binance Data" \
  --body "$(cat issues/ISSUE_1.3_download_prediction_binance.md)" \
  --label "data,phase-1"

gh issue create --title "[#1.4] Download Prediction Macro Data" \
  --body "$(cat issues/ISSUE_1.4_download_prediction_macro.md)" \
  --label "data,phase-1"

gh issue create --title "[#1.5] Implement Shared Features Pipeline" \
  --body "$(cat issues/ISSUE_1.5_shared_features.md)" \
  --label "features,phase-2"

gh issue create --title "[#1.6] Implement Training-Only Features Pipeline" \
  --body "$(cat issues/ISSUE_1.6_training_features.md)" \
  --label "features,phase-2"

gh issue create --title "[#1.7] Refactor Training Script" \
  --body "$(cat issues/ISSUE_1.7_train_model.md)" \
  --label "training,phase-3"

gh issue create --title "[#1.8] Refactor Prediction Script" \
  --body "$(cat issues/ISSUE_1.8_predict.md)" \
  --label "prediction,phase-3"
```

---

## Sub-Issue #1.2: Download Training Macro Data

**Objective:** Download long-term macro economic data from yfinance.

**Create:**
- `scripts/02_download_training_macro.py`
- `sneaker/macro.py` (new module)

**Output:** `data/raw/training/macro_training.json`

**Macro Indicators:**
- SPY (S&P 500)
- VIX (Volatility Index)
- DXY (Dollar Index)
- GLD (Gold)
- TLT (20-Year Treasury)

**Key Functions:**
```python
# sneaker/macro.py
def download_macro_indicators(start_date, end_date, indicators=['SPY', 'VIX', 'DXY', 'GLD', 'TLT']):
    """Download macro data from yfinance."""
    pass

def resample_to_hourly(daily_df):
    """Convert daily macro data to hourly (forward-fill)."""
    pass
```

**Success Criteria:**
- All 5 indicators downloaded
- Aligned with training date range (2021-present)
- Resampled to 1H frequency
- Saved to JSON

---

## Sub-Issue #1.3: Download Prediction Binance Data

**Objective:** Download short-term 1H candle data for a single pair (LINKUSDT) for prediction.

**Create:** `scripts/03_download_prediction_binance.py`

**Output:** `data/raw/prediction/binance_LINKUSDT_live.json`

**Requirements:**
- Download recent 256-512 hours
- Single pair (configurable, default: LINKUSDT)
- Same format as training data
- Refreshable (can re-run to update)

**Key Features:**
```python
def download_prediction_data(pair='LINKUSDT', hours=512):
    """Download recent data for prediction."""
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=hours)
    return download_historical_data(pair, start_date, end_date, '1h')
```

**Success Criteria:**
- 256-512 hours of data downloaded
- LINKUSDT candles present
- Recent data (up to current hour)
- JSON format matches training data

---

## Sub-Issue #1.4: Download Prediction Macro Data

**Objective:** Download recent macro data aligned with prediction period.

**Create:** `scripts/04_download_prediction_macro.py`

**Output:** `data/raw/prediction/macro_prediction.json`

**Requirements:**
- Same indicators as training (SPY, VIX, DXY, GLD, TLT)
- Recent 256-512 hours
- Resampled to 1H
- Aligned with prediction Binance data timestamps

**Success Criteria:**
- Macro data covers prediction period
- Aligned with LINKUSDT timestamps
- Forward-filled for hourly frequency
- JSON format

---

## Sub-Issue #1.5: Implement Shared Features Pipeline

**Objective:** Create feature engineering pipeline for features used in BOTH training and prediction.

**Create:**
- `sneaker/features_shared.py`
- `scripts/05_add_shared_features.py`

**Output:**
- `data/features/training_shared_features.json`
- `data/features/prediction_shared_features.json`

**Shared Features (from current 83):**

**Technical Indicators (20):**
- RSI, RSI-7, BB position, MACD, Stochastic
- Directional indicators, ADX
- Volume ratio, VWAP

**Momentum Features (~20):**
- Price ROC (3, 5, 10, 20)
- Price acceleration
- Indicator accelerations
- ATR, volatility regime

**Macro Features (5):**
- SPY, VIX, DXY, GLD, TLT (merged)

**Total: ~45 shared features**

**Critical Requirement:**
- EXACT SAME calculation for training and prediction
- No future peeking
- Handles NaN consistently

**Success Criteria:**
- Script runs on both training and prediction data
- Produces identical features (same calculation)
- All 45 features present
- No NaN values (or handled consistently)
- Test: Same input → same output

---

## Sub-Issue #1.6: Implement Training-Only Features Pipeline

**Objective:** Create feature engineering for training-exclusive features (targets, signals).

**Create:**
- `sneaker/features_training.py`
- `scripts/06_add_training_features.py`

**Output:** `data/features/training_only_features.json`

**Training-Only Features:**

**Target Calculation:**
- Future price changes (1h, 4h, 12h ahead)
- Reversal magnitude
- Volatility normalization (convert to σ)

**Signal Detection:**
- Indicator momentum shifts (ghost signals)
- Turning point identification
- Signal strength quantification

**Data Splitting:**
- Train/validation/test indicators
- Temporal split markers

**Additional (~35 features):**
- Multi-timeframe aggregations (use future data for labels)
- Divergences (calculated with hindsight)
- Statistical features (Hurst, entropy, etc.)

**Total: ~40 training-only features**

**Success Criteria:**
- Target column present (`target` in σ units)
- Signal indicators present
- Split indicators present
- Cannot be calculated on live prediction data
- Saved to JSON

---

## Sub-Issue #1.7: Refactor Training Script

**Objective:** Refactor training to use prepared features and generate comprehensive proof.

**Refactor:** `scripts/07_train_model.py` (from `04_train_model.py`)

**Input:** `data/features/training_only_features.json`

**Output:**
- `models/issue-1/model.txt`
- `proof/issue-1/training_regression_*.png`
- `proof/issue-1/training_residuals_*.png`
- `proof/issue-1/training_feature_importance_*.png`
- `proof/issue-1/training_signal_dist_*.png`
- `proof/issue-1/training_report_*.txt`

**Key Changes:**
1. Load from `training_only_features.json`
2. Use `--issue` parameter for proof folder
3. Separate shared vs training-only features
4. Generate all 5 validation plots
5. Save model to issue-specific folder

**Training Process:**
- Load features (shared + training-only)
- Split train/test (90/10, temporal)
- Apply V3 sample weighting (5x for signals)
- Train LightGBM
- Generate proof visualizations
- Save model

**Success Criteria:**
- Model trains successfully
- Signal R² ≥ 70%
- Direction accuracy ≥ 95%
- All proof files generated
- Saved to `proof/issue-1/`

---

## Sub-Issue #1.8: Refactor Prediction Script

**Objective:** Refactor prediction to use trained model and generate visualization.

**Refactor:** `scripts/08_predict.py` (from `05_predict.py`)

**Input:**
- `models/issue-1/model.txt`
- `data/features/prediction_shared_features.json`

**Output:**
- `proof/issue-1/prediction_LINKUSDT_256h_*.png`
- `proof/issue-1/prediction_signals_*.txt`

**Key Changes:**
1. Load model from issue-specific folder
2. Load ONLY shared features (no training features)
3. Generate predictions
4. Create price chart with buy/sell signals
5. Save visualization to proof folder

**Visualization:**
- 256-hour price chart (LINKUSDT)
- Overlay predictions (color-coded by strength)
- Mark buy signals (prediction > +4σ, green)
- Mark sell signals (prediction < -4σ, red)
- Show signal strength
- Legend and annotations

**Success Criteria:**
- Predictions generated for all 256 hours
- Visualization created
- Signals marked correctly
- No errors (no training features used)
- Saved to `proof/issue-1/`

---

## Execution Order

```bash
# Phase 1: Data Collection (can run in parallel)
.venv/bin/python scripts/01_download_training_binance.py &
.venv/bin/python scripts/02_download_training_macro.py &
.venv/bin/python scripts/03_download_prediction_binance.py &
.venv/bin/python scripts/04_download_prediction_macro.py &
wait

# Phase 2: Feature Engineering
.venv/bin/python scripts/05_add_shared_features.py --mode training
.venv/bin/python scripts/05_add_shared_features.py --mode prediction
.venv/bin/python scripts/06_add_training_features.py

# Phase 3: Training & Prediction
.venv/bin/python scripts/07_train_model.py --issue 1
.venv/bin/python scripts/08_predict.py --issue 1 --pair LINKUSDT --hours 256
```

---

## File Structure After Completion

```
data/
├── raw/
│   ├── training/
│   │   ├── binance_20pairs_1H.json         # 500-1000 MB
│   │   └── macro_training.json             # ~1 MB
│   └── prediction/
│       ├── binance_LINKUSDT_live.json      # ~1 MB
│       └── macro_prediction.json           # ~10 KB
├── features/
│   ├── training_shared_features.json       # ~800 MB
│   ├── prediction_shared_features.json     # ~1 MB
│   └── training_only_features.json         # ~1.2 GB
└── prepared/ (optional, if needed)
    ├── training_dataset.json
    └── prediction_dataset.json

models/
└── issue-1/
    └── model.txt                            # 30-40 MB

proof/
└── issue-1/
    ├── training_regression_*.png
    ├── training_residuals_*.png
    ├── training_feature_importance_*.png
    ├── training_signal_dist_*.png
    ├── training_report_*.txt
    ├── prediction_LINKUSDT_256h_*.png
    └── prediction_signals_*.txt

scripts/ (new)
├── 01_download_training_binance.py
├── 02_download_training_macro.py
├── 03_download_prediction_binance.py
├── 04_download_prediction_macro.py
├── 05_add_shared_features.py
├── 06_add_training_features.py
├── 07_train_model.py
└── 08_predict.py

sneaker/ (modified/new)
├── features_shared.py                       # NEW
├── features_training.py                     # NEW
├── macro.py                                 # NEW
├── data.py                                  # MODIFIED
└── ... (existing modules)
```

---

## Validation Checklist

### Phase 1 Complete (Data Collection)
- [ ] All 4 raw data files exist
- [ ] Training: 20 pairs, ~1M candles
- [ ] Training: 5 macro indicators
- [ ] Prediction: LINKUSDT, 256-512 hours
- [ ] Prediction: 5 macro indicators
- [ ] All JSON files valid and readable

### Phase 2 Complete (Feature Engineering)
- [ ] Shared features on training data
- [ ] Shared features on prediction data
- [ ] Training-only features added
- [ ] Feature counts match expectations
- [ ] No NaN values (or handled)
- [ ] Timestamps aligned

### Phase 3 Complete (Training & Prediction)
- [ ] Model trains successfully
- [ ] Validation metrics pass
- [ ] Predictions generated
- [ ] Visualization created
- [ ] All proof files in `proof/issue-1/`

### Overall Success
- [ ] Complete pipeline runs end-to-end
- [ ] All intermediate results cached
- [ ] Can re-run any step without errors
- [ ] Shared features identical calculation verified
- [ ] Training/prediction separation maintained
- [ ] Proof folder complete

---

## Timeline Estimate

**Phase 1 (Data Collection):** 1-2 days
- #1.1: 4 hours
- #1.2: 2 hours
- #1.3: 2 hours
- #1.4: 1 hour

**Phase 2 (Features):** 2-3 days
- #1.5: 8 hours (split existing features, test)
- #1.6: 6 hours (extract training features, targets)

**Phase 3 (Training & Prediction):** 2-3 days
- #1.7: 8 hours (refactor training, proof)
- #1.8: 6 hours (refactor prediction, viz)

**Total:** 5-8 days for complete restructuring

---

## Risk Mitigation

**Risk: Data download failures**
- Mitigation: Retry logic, save progress, resume capability

**Risk: Feature parity not maintained**
- Mitigation: Unit tests, integration tests, sample comparison

**Risk: Performance degradation**
- Mitigation: Benchmark before/after, optimize if needed

**Risk: Validation failures**
- Mitigation: Incremental testing, fallback to original

**Risk: Data leakage in shared features**
- Mitigation: Code review, explicit checks, documentation

---

**This restructuring creates a solid, modular foundation for all future experimentation.**
