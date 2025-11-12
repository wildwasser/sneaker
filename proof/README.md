# Proof Folder - Validation Evidence Repository

This folder contains **irrefutable evidence** of validation results for every issue worked on in this project.

## Purpose

**Trust nothing. Validate everything.** This folder exists to prevent statistical illusions and ensure rigorous testing of all model changes.

##Structure

```
proof/
├── README.md (this file)
├── issue-1/
│   ├── validation_report_2025-11-12_14-30-00.txt
│   ├── regression_analysis_2025-11-12_14-30-00.png
│   ├── residual_analysis_2025-11-12_14-30-00.png
│   ├── feature_importance_2025-11-12_14-30-00.png
│   ├── signal_distribution_2025-11-12_14-30-00.png
│   ├── backtest_report_2025-11-12_14-35-00.txt
│   ├── backtest_trades_2025-11-12_14-35-00.png
│   ├── backtest_equity_curve_2025-11-12_14-35-00.png
│   └── summary.md
├── issue-2/
│   └── ...
└── issue-N/
    └── ...
```

## File Types

### Validation Files (from `validate_model.py`)

**validation_report_{timestamp}.txt**
- Complete text report of all metrics
- Pass/fail status for each criterion
- Red flag warnings
- Feature importance rankings

**regression_analysis_{timestamp}.png**
- 4 subplots showing predicted vs actual
- Train and test sets
- All samples and signals-only views
- Checks for regression quality

**residual_analysis_{timestamp}.png**
- 4 subplots of residual diagnostics
- Residuals vs predicted (pattern detection)
- Residual histogram (normality check)
- Q-Q plot (normality verification)
- Residuals vs actual

**feature_importance_{timestamp}.png**
- Top 10 features by importance
- Color-coded warnings (red >40%, orange >30%)
- Dominance threshold visualization

**signal_distribution_{timestamp}.png**
- 4 subplots of signal analysis
- Actual vs predicted distributions
- Zeros vs signals comparison
- Signal rate vs threshold curve

### Backtest Files (from `backtest.py`)

**backtest_report_{timestamp}.txt**
- Complete backtest results
- Trade statistics (win rate, profit factor, etc.)
- Risk metrics (Sharpe, drawdown)
- Pass/fail status

**backtest_trades_{timestamp}.png**
- Price chart with buy/sell signals marked
- Entry and exit points visualized
- Prediction strength color-coded
- Complete trade history overlay

**backtest_equity_curve_{timestamp}.png**
- Equity curve over time
- Drawdown visualization
- Trade markers
- Performance statistics overlay

### Summary File

**summary.md**
- Human-readable summary of issue
- Links to GitHub issue
- Final verdict (PASS/FAIL)
- Key findings
- Next actions

## Workflow Integration

### Step 1: Create Issue
```bash
gh issue create --title "[TYPE] Description"
# Note the issue number (e.g., #42)
```

### Step 2: Create Branch
```bash
git checkout -b issue-42-description
```

### Step 3: Make Changes
```bash
# Edit code...
git add -A
git commit -m "Add #42: description"
```

### Step 4: Run Validation
```bash
.venv/bin/python scripts/validate_model.py --issue 42
# Creates proof/issue-42/ with validation results
```

### Step 5: Run Backtest
```bash
.venv/bin/python scripts/backtest.py --issue 42
# Adds backtest results to proof/issue-42/
```

### Step 6: Commit Proof
```bash
git add proof/issue-42/
git commit -m "Add #42: validation proof"
git push origin issue-42-description
```

### Step 7: Review & Merge
```bash
# Review all visualizations in proof/issue-42/
# If PASS: create PR and merge
# If FAIL: investigate, revise, re-validate
```

## Why This Matters

1. **Accountability:** Every change has documented evidence
2. **Transparency:** GitHub tracks all proof alongside code
3. **Reproducibility:** Timestamps and exact results preserved
4. **Red Flag Detection:** Visual inspection catches statistical illusions
5. **Audit Trail:** Complete history of all validation attempts

## Key Principles

- **Never delete proof:** Even failed validations stay in history
- **Always commit proof:** Proof goes to main branch, code stays on feature branch until validated
- **Visual inspection mandatory:** Automated metrics can lie, plots reveal truth
- **Timestamps matter:** Multiple validation runs show iteration process

## Red Flags to Look For

### In Regression Plots
- 🚩 Points far from diagonal line (poor predictions)
- 🚩 Non-linear patterns (model missing relationships)
- 🚩 Different train vs test patterns (overfitting)

### In Residual Plots
- 🚩 Patterns in residuals vs predicted (heteroscedasticity)
- 🚩 Non-normal residual distribution (model assumptions violated)
- 🚩 Q-Q plot deviating from line (non-normality)

### In Feature Importance
- 🚩 Single feature >40% importance (dominance)
- 🚩 Nonsensical top features (data leakage)
- 🚩 Unstable importance across runs (overfitting)

### In Signal Distribution
- 🚩 Predicted distribution vastly different from actual
- 🚩 Signal rate >20% at threshold
- 🚩 All predictions near zero or all at extremes

### In Backtest Trades
- 🚩 Perfect win rate (100%)
- 🚩 Suspiciously high Sharpe (>5.0)
- 🚩 All trades same direction (market regime dependency)
- 🚩 Trades clustered in time (not robust)

### In Equity Curve
- 🚩 Smooth monotonic growth (unrealistic)
- 🚩 Sudden jumps (lucky trades, not skill)
- 🚩 Massive drawdown (>50%)
- 🚩 Recent losses after earlier gains (regime change)

## Example: Good vs Bad

### ✅ Good Validation (Pass)
```
proof/issue-10/
  validation_report_*.txt → Signal R² 73%, Dir Acc 96%, No red flags
  regression_analysis_*.png → Points cluster near diagonal, similar train/test
  feature_importance_*.png → Top feature 25%, diverse importance
  backtest_report_*.txt → Sharpe 1.8, Win rate 58%, Drawdown 18%
  backtest_trades_*.png → Mix of wins/losses, distributed over time
```

**Verdict:** Statistical evidence solid, backtest confirms, MERGE APPROVED

### ❌ Bad Validation (Fail)
```
proof/issue-15/
  validation_report_*.txt → Signal R² 45%, Dir Acc 82%, Train/test gap 22%
  regression_analysis_*.png → Scattered points, train good but test poor
  feature_importance_*.png → Single feature 67% importance (DOMINANCE!)
  backtest_report_*.txt → Sharpe 0.3, Win rate 42%, Drawdown 38%
  backtest_trades_*.png → Losing trades cluster at end, regime dependent
```

**Verdict:** SEVERE OVERFITTING + DOMINANCE + POOR BACKTEST → DO NOT MERGE

## Maintenance

- **Regular cleanup:** NOT RECOMMENDED - keep all history
- **Large files:** Proof folder may grow large, but that's okay
- **GitHub LFS:** Consider if proof folder exceeds 1GB
- **Archiving:** Old issues can be archived but never deleted

## Questions?

See `VALIDATION.md` for detailed pass/fail criteria and `CLAUDE.md` for complete workflow documentation.

**Remember: If there's no proof, it didn't happen. If validation failed, don't merge. Trust nothing. Validate everything.**
