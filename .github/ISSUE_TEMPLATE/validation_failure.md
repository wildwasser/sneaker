---
name: Validation Failure
about: Report when model/feature fails validation tests
title: '[VALIDATION] '
labels: validation-failure, critical
assignees: ''
---

## What Failed Validation?
- [ ] Model training (regression metrics below threshold)
- [ ] Backtest performance (LINKUSDT 256h failed)
- [ ] Statistical test failure (illusion detected)
- [ ] Feature addition (degraded performance)
- [ ] Other

## Test Details
**Command used:**
```bash
# Paste exact command used for validation
```

**Expected Results:**
```
What metrics/outcomes were you expecting?
```

**Actual Results:**
```
Paste actual results here (full output)
```

## Regression Metrics
```
Signal R²:
Overall R²:
Direction Accuracy:
Zero MAE:
Signal Rate:
```

## Backtest Results (if applicable)
```
Sharpe Ratio:
Max Drawdown:
Win Rate:
Total Return:
Number of Trades:
```

## Statistical Analysis
Is this a statistical illusion? Evidence:
- [ ] High training R², low test R²
- [ ] Suspiciously high metrics
- [ ] Backtest failure vs in-sample success
- [ ] Other red flags

## Root Cause Hypothesis
What do you think caused this failure?

## Recommended Action
- [ ] Revert changes
- [ ] Adjust parameters
- [ ] Redesign approach
- [ ] Investigate further

## Branch
Branch where failure occurred: `issue-<number>-<description>`

## Additional Context
Attach any relevant logs, visualizations, or analysis.
