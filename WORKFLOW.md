# Complete Development Workflow

**Strict, repeatable, verifiable workflow for all development on Sneaker project.**

## ⚠️ Core Philosophy

- **NO work without a GitHub issue**
- **NO merges without validation proof**
- **Trust nothing, validate everything**
- **Proof folder is your audit trail**

## Quick Reference Card

```bash
# 1. Start new work
gh issue create
git checkout -b issue-X-description

# 2. Make changes
# ... edit code ...
git add -A && git commit -m "Add #X: description"

# 3. Validate (MANDATORY)
.venv/bin/python scripts/validate_model.py --issue X
# Review: proof/issue-X/*

# 4. Backtest (MANDATORY - if model changed)
.venv/bin/python scripts/backtest.py --issue X
# Review: proof/issue-X/*

# 5. Commit proof
git add proof/issue-X/
git commit -m "Add #X: validation proof"

# 6. Push branch
git push origin issue-X-description

# 7. Create PR (only if validation PASSED)
gh pr create --title "Fix #X: description"

# 8. Merge & cleanup
gh pr merge X --squash
git checkout main && git pull
git branch -d issue-X-description
```

## Detailed Step-by-Step Workflow

### Phase 1: Planning

#### 1.1 Create GitHub Issue

**Before any code changes**, create an issue describing what you want to do.

```bash
# Interactive creation
gh issue create

# Or with parameters
gh issue create \
  --title "[FEATURE] Add new technical indicator" \
  --body "Description of feature, validation plan, success criteria"
```

**Issue templates available:**
- `bug_report.md` - For bugs or statistical illusions
- `feature_request.md` - For new features (requires validation plan)
- `validation_failure.md` - When validation tests fail

**Note the issue number** (e.g., #42) - you'll use this everywhere.

#### 1.2 Plan Your Validation

**Before writing code**, define in the issue:
- What metrics should improve?
- What could go wrong (red flags)?
- How will you know it works?

Example:
```
## Validation Plan

Expected improvements:
- Signal R² should stay ≥70%
- Direction accuracy should improve by 2-3%
- No new red flags

Red flags to watch:
- Feature dominance (new feature >40% importance)
- Backtest degradation
- Overfitting (train/test gap increases)

Success criteria:
- All validation checks pass
- Backtest Sharpe ≥ 1.0
- No statistical illusions detected
```

### Phase 2: Development

#### 2.1 Create Feature Branch

```bash
# Ensure you're on latest main
git checkout main
git pull origin main

# Create issue-numbered branch
git checkout -b issue-42-add-new-indicator
```

**Branch naming format:** `issue-<number>-<short-description>`

**Examples:**
- `issue-1-baseline-validation`
- `issue-5-fix-nan-values`
- `issue-12-add-rsi-divergence`

#### 2.2 Make Your Changes

```bash
# Edit files...
vim sneaker/indicators.py

# Test locally (optional but recommended)
.venv/bin/python -c "from sneaker.indicators import new_function; print(new_function.__doc__)"
```

#### 2.3 Commit Changes

```bash
# Stage changes
git add -A

# Commit with issue reference
git commit -m "Add #42: Implement RSI divergence indicator

- Added divergence detection to indicators.py
- Calculates hidden and regular divergences
- Returns binary signal (1=bullish, -1=bearish, 0=none)

Next: Run validation to check for statistical issues"
```

**Commit message format:**
```
Add #<issue>: <Short description>

<Detailed explanation>
- Bullet points for key changes
- What was added/modified/fixed
- Why these changes were made

Next: <What happens next>
```

#### 2.4 Push Branch Regularly

```bash
# Push to GitHub for visibility
git push origin issue-42-add-new-indicator

# Push again after each significant change
git push origin issue-42-add-new-indicator
```

**Why push frequently?**
- Backup your work
- Make progress visible
- Enable collaboration
- GitHub tracks everything

### Phase 3: Validation (MANDATORY)

#### 3.1 Run Model Validation

```bash
# REQUIRED: Validate model regression
.venv/bin/python scripts/validate_model.py --issue 42
```

**This creates:** `proof/issue-42/` with:
- `validation_report_{timestamp}.txt`
- `regression_analysis_{timestamp}.png`
- `residual_analysis_{timestamp}.png`
- `feature_importance_{timestamp}.png`
- `signal_distribution_{timestamp}.png`

**What it checks:**
- ✅ Signal R² ≥ 70%
- ✅ Direction accuracy ≥ 95%
- ✅ Train/test gap ≤ 10%
- ✅ No feature dominance (>40%)
- ✅ Signal rate ≤ 20%
- 🚩 Red flag detection

#### 3.2 Review Validation Results

```bash
# View text report
cat proof/issue-42/validation_report_*.txt

# Open visualizations
open proof/issue-42/*.png  # macOS
xdg-open proof/issue-42/*.png  # Linux
```

**Visual inspection checklist:**

**Regression plots:**
- [ ] Points cluster near diagonal line
- [ ] Train and test plots look similar
- [ ] Signals-only plot shows good fit
- [ ] No obvious outlier patterns

**Residual plots:**
- [ ] Residuals scattered randomly (no patterns)
- [ ] Histogram centered at zero
- [ ] Q-Q plot follows red line
- [ ] No heteroscedasticity (funnel shape)

**Feature importance:**
- [ ] Top feature <40% importance
- [ ] Diverse feature set (not dominated)
- [ ] Top features make sense
- [ ] No suspicious rankings

**Signal distribution:**
- [ ] Predicted matches actual shape roughly
- [ ] Signal rate reasonable (<20% at 4σ)
- [ ] Not all zeros or all signals
- [ ] Smooth signal rate curve

#### 3.3 Run Backtest (if model changed)

```bash
# REQUIRED for model changes: Backtest on LINKUSDT 256h
.venv/bin/python scripts/backtest.py --issue 42 --pair LINKUSDT --hours 256
```

**Note:** Backtest script needs updating for full proof integration (currently being developed)

**Pass criteria:**
- ✅ Sharpe ratio ≥ 1.0
- ✅ Win rate ≥ 50%
- ✅ Max drawdown ≤ 25%
- ✅ Trade count 2-15 (reasonable)
- 🚩 Red flag detection

#### 3.4 Interpret Results

**If PASSED:**
```
✅ VALIDATION PASSED - Model appears statistically sound
✅ BACKTEST PASSED - Model performance acceptable

→ Proceed to Phase 4 (Commit Proof)
```

**If FAILED:**
```
❌ VALIDATION FAILED - Statistical illusions or issues detected
OR
❌ BACKTEST FAILED - Performance issues detected

→ DO NOT PROCEED
→ Review proof/ visualizations
→ Investigate root cause
→ Fix issues or abandon approach
→ Re-run validation after fixes
```

**Common failures and fixes:**

| Failure | Likely Cause | Fix |
|---------|--------------|-----|
| Train R² >> Test R² | Overfitting | Reduce features, increase regularization |
| Feature >40% importance | Dominance | Remove or combine correlated features |
| Backtest fails, validation passes | Look-ahead bias | Check feature engineering for future data |
| Signal rate >20% | Threshold too low | Adjust threshold or fix signal detection |
| Win rate <50% | Random predictions | Model not learning, check target definition |

### Phase 4: Commit Proof

#### 4.1 Commit Validation Results

```bash
# Add proof folder
git add proof/issue-42/

# Commit with clear message
git commit -m "Add #42: Validation proof - PASSED

Model regression validation:
- Signal R²: 72.3% ✅
- Direction accuracy: 96.1% ✅
- Train/test gap: 7.2% ✅
- Max feature importance: 28.4% ✅
- No red flags ✅

Backtest results (LINKUSDT 256h):
- Sharpe ratio: 1.45 ✅
- Win rate: 56% ✅
- Max drawdown: 19% ✅
- Trades: 8 ✅

All criteria passed. Ready for merge."
```

**Important:**
- Always include pass/fail summary
- List key metrics in commit message
- Reference timestamp if multiple runs
- Explain any deviations

#### 4.2 Push Proof to GitHub

```bash
# Push branch with proof
git push origin issue-42-add-new-indicator
```

**Now GitHub has:**
- Issue #42 (what/why)
- Branch `issue-42-add-new-indicator` (code changes)
- Proof folder `proof/issue-42/` (validation evidence)

**Complete traceability!**

### Phase 5: Pull Request & Merge

#### 5.1 Create Pull Request

**Only create PR if validation PASSED!**

```bash
gh pr create \
  --title "Fix #42: Add RSI divergence indicator" \
  --body "$(cat <<'EOF'
## Summary
Adds RSI divergence detection to improve reversal prediction.

## Changes
- Added `calculate_rsi_divergence()` to indicators.py
- Integrated divergence signal into feature pipeline
- Added tests for edge cases

## Validation Results
✅ Model validation PASSED
✅ Backtest PASSED

See `proof/issue-42/` for complete evidence.

### Key Metrics
- Signal R²: 72.3% (baseline: 74.0%) ✅
- Direction accuracy: 96.1% (baseline: 98.0%) ✅
- Backtest Sharpe: 1.45 (min: 1.0) ✅
- No red flags detected ✅

## Closes #42
EOF
)"
```

**PR checklist:**
- [ ] Title references issue (`Fix #42`)
- [ ] Summary explains what and why
- [ ] Validation results included
- [ ] Link to proof folder
- [ ] Key metrics listed
- [ ] Closes #42 in body

#### 5.2 Review Process

**Self-review:**
1. Check all proof visualizations one more time
2. Verify no red flags
3. Confirm all criteria passed
4. Read commit messages for clarity

**If working with team:**
- Request review from colleague
- Share proof/ folder location
- Discuss any marginal results
- Get explicit approval

#### 5.3 Merge to Main

**Only merge if:**
- ✅ Validation passed
- ✅ Backtest passed (if applicable)
- ✅ Proof committed
- ✅ PR approved (if team workflow)
- ✅ No red flags

```bash
# Squash merge (clean history)
gh pr merge 42 --squash --delete-branch

# Or merge via GitHub UI
```

**What happens:**
1. Code changes merge to main
2. Proof folder merges to main
3. Issue #42 closes automatically
4. Branch deleted from remote

#### 5.4 Cleanup Local Branch

```bash
# Switch to main
git checkout main

# Pull merged changes
git pull origin main

# Delete local branch
git branch -d issue-42-add-new-indicator

# Verify proof is in main
ls -la proof/issue-42/
```

**Success!** Your changes are in main with complete validation evidence.

### Phase 6: Failed Validation Handling

#### 6.1 Document Failure

If validation fails, create a "Validation Failure" issue:

```bash
gh issue create \
  --title "[VALIDATION] Issue #42 failed validation" \
  --body "$(cat <<'EOF'
## Failed Validation

Original issue: #42
Branch: issue-42-add-new-indicator

## What Failed
❌ Model validation: Feature dominance detected

## Results
- Signal R²: 68.2% ❌ (below 70% threshold)
- Max feature importance: 52.1% ❌ (above 40% threshold)
- Feature: rsi_divergence (new feature)

## Analysis
New feature dominates model, likely overfitting to noise.
Backtest also shows degraded performance (Sharpe 0.7).

## Next Steps
1. Investigate rsi_divergence calculation
2. Check for data leakage
3. Try ensemble with other features
4. Consider abandoning if fundamentally flawed

## Proof Location
proof/issue-42/validation_report_2025-11-12_15-30-00.txt
EOF
)"
```

#### 6.2 Investigate & Iterate

```bash
# Stay on same branch
git checkout issue-42-add-new-indicator

# Make fixes
# ... revise code ...

# Commit iteration
git add -A
git commit -m "Fix #42: Attempt 2 - Reduce feature dominance

- Normalized divergence signal
- Added regularization
- Combined with other indicators

Re-running validation..."

# Re-run validation
.venv/bin/python scripts/validate_model.py --issue 42

# New timestamp, same proof folder
# proof/issue-42/validation_report_2025-11-12_16-45-00.txt
```

**Multiple attempts OK:**
- Same proof folder gets multiple timestamped files
- Shows iteration process
- Last successful validation is what matters

#### 6.3 Abandon if Necessary

If after multiple attempts validation still fails:

```bash
# Document decision
gh issue comment 42 --body "After 3 validation attempts, unable to pass criteria.
Fundamental approach appears flawed. Abandoning this feature.

See proof/issue-42/ for all attempts."

# Close issue without merging
gh issue close 42

# Delete branch
git checkout main
git branch -D issue-42-add-new-indicator
git push origin --delete issue-42-add-new-indicator

# Keep proof folder for future reference
git add proof/issue-42/
git commit -m "Archive #42: Failed validation attempts

Attempted to add RSI divergence but validation failed repeatedly.
Keeping proof for future reference.

Red flags detected:
- Feature dominance
- Backtest failure
- Likely statistical illusion

DO NOT RETRY without fundamental redesign."

git push origin main
```

**Important:** Failed proof is still valuable! It documents what NOT to do.

## Special Cases

### Working on Multiple Issues Simultaneously

**NOT RECOMMENDED,** but if necessary:

```bash
# Issue 10: Feature work
git checkout -b issue-10-feature-a
# ... work ...
.venv/bin/python scripts/validate_model.py --issue 10

# Switch to issue 11
git checkout main
git checkout -b issue-11-bugfix-b
# ... work ...
.venv/bin/python scripts/validate_model.py --issue 11

# Each gets separate proof folder
ls proof/
#  issue-10/
#  issue-11/
```

### Emergency Hotfix

**Still requires validation**, but can expedite:

```bash
# Create issue
gh issue create --title "[HOTFIX] Critical bug"

# Fast branch
git checkout -b issue-99-hotfix

# Fix
# ... minimal change ...
git commit -m "Fix #99: Emergency fix for production bug"

# Validate (MANDATORY even for hotfixes)
.venv/bin/python scripts/validate_model.py --issue 99

# If PASSED, fast-track merge
git add proof/issue-99/
git commit -m "Add #99: Validation proof"
git push origin issue-99-hotfix
gh pr create --title "Fix #99: Hotfix" --body "Emergency fix. Validation passed."
gh pr merge 99 --squash
```

### Re-running Validation

If you need to re-validate same issue (e.g., after fixes):

```bash
# Just run again with same issue number
.venv/bin/python scripts/validate_model.py --issue 42

# New timestamped files added to same proof folder
# proof/issue-42/
#   validation_report_2025-11-12_14-30-00.txt  # First attempt
#   validation_report_2025-11-12_16-45-00.txt  # Second attempt (after fixes)
#   ...
```

**Latest timestamp = current status**

### Baseline Validation (No Code Changes)

To validate existing model without changes:

```bash
# Create issue
gh issue create --title "[VALIDATION] Baseline model validation"

# No branch needed
# Run validation from main
.venv/bin/python scripts/validate_model.py --issue 1

# Commit proof directly to main
git add proof/issue-1/
git commit -m "Add #1: Baseline validation proof"
git push origin main

# Close issue
gh issue close 1 --comment "Baseline validation complete. See proof/issue-1/"
```

## Troubleshooting

### "ERROR: --issue parameter is REQUIRED"

```bash
# Wrong
.venv/bin/python scripts/validate_model.py

# Right
.venv/bin/python scripts/validate_model.py --issue 42
```

### "Proof directory already exists"

**This is fine!** Multiple validation runs go to same folder with different timestamps.

### "git add proof/ not working"

Check `.gitignore`:

```bash
# Proof folder should NOT be ignored
grep proof .gitignore
# (should return nothing)
```

### "Can't push large proof files"

If proof folder exceeds GitHub limits (rare):

```bash
# Check size
du -sh proof/issue-42/

# If >100MB, consider Git LFS
git lfs track "proof/**/*.png"
```

### "Validation passed but backtest failed"

**DO NOT MERGE!** This indicates:
- Look-ahead bias in features
- Overfitting to in-sample data
- Statistical illusion

Investigate feature engineering carefully.

## Summary Checklist

Before merging any code:

- [ ] GitHub issue created (#X)
- [ ] Branch created (`issue-X-description`)
- [ ] Changes committed with issue reference
- [ ] Validation run: `.venv/bin/python scripts/validate_model.py --issue X`
- [ ] Validation PASSED (all criteria met)
- [ ] Backtest run (if model changed): `.venv/bin/python scripts/backtest.py --issue X`
- [ ] Backtest PASSED (all criteria met)
- [ ] Proof visualizations reviewed (no red flags)
- [ ] Proof committed to branch
- [ ] Branch pushed to GitHub
- [ ] Pull request created
- [ ] PR approved (if team workflow)
- [ ] Merged to main
- [ ] Local branch cleaned up

**If ANY step fails, DO NOT MERGE.**

## Quick Commands Cheat Sheet

```bash
# Start
gh issue create && git checkout -b issue-X-description

# Develop
git add -A && git commit -m "Add #X: ..." && git push

# Validate
.venv/bin/python scripts/validate_model.py --issue X
.venv/bin/python scripts/backtest.py --issue X

# Commit proof
git add proof/issue-X/ && git commit -m "Add #X: validation proof" && git push

# Merge
gh pr create && gh pr merge X --squash

# Cleanup
git checkout main && git pull && git branch -d issue-X-description
```

---

**Remember:** Trust nothing. Validate everything. Proof folder is your friend.

For more details:
- `VALIDATION.md` - Pass/fail criteria
- `CLAUDE.md` - AI assistant instructions
- `proof/README.md` - Proof folder structure

Questions? Create an issue!
