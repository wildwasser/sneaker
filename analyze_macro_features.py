"""
Analyze macro feature importance to identify which can be removed.

This script:
1. Loads trained model from issue-22 (8-candle baseline)
2. Extracts feature importances for all features
3. Identifies all macro features and their rankings
4. Provides recommendations on which macro features to remove
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path

# Load the trained model
model_path = Path("models/issue-1/model.txt")
print(f"Loading model from: {model_path}")
model = lgb.Booster(model_file=str(model_path))

# Get feature importances
feature_names = model.feature_name()
importances = model.feature_importance(importance_type='gain')

# Create DataFrame
df_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).reset_index(drop=True)

# Add rank
df_importance['rank'] = range(1, len(df_importance) + 1)

print(f"\nTotal features: {len(df_importance)}")
print(f"Total importance: {df_importance['importance'].sum():.2f}")

# Identify macro features
df_importance['is_macro'] = df_importance['feature'].str.startswith('macro_')
df_macro = df_importance[df_importance['is_macro']].copy()

print(f"\n{'='*80}")
print("MACRO FEATURE ANALYSIS")
print(f"{'='*80}")

print(f"\nTotal macro features: {len(df_macro)}")
print(f"Macro importance: {df_macro['importance'].sum():.2f} ({df_macro['importance'].sum() / df_importance['importance'].sum() * 100:.1f}%)")

# Parse macro features to understand structure
df_macro['macro_type'] = df_macro['feature'].str.extract(r'macro_([A-Z_]+)_')[0]
df_macro['derivative'] = df_macro['feature'].str.extract(r'_(close|vel|roc_5)_')[0]
df_macro['time_step'] = df_macro['feature'].str.extract(r'_t(\d+)$')[0]

print(f"\n{'='*80}")
print("TOP 30 MACRO FEATURES")
print(f"{'='*80}")
print(df_macro.head(30).to_string(index=False))

print(f"\n{'='*80}")
print("BOTTOM 30 MACRO FEATURES (WEAKEST)")
print(f"{'='*80}")
print(df_macro.tail(30).to_string(index=False))

print(f"\n{'='*80}")
print("BREAKDOWN BY MACRO TYPE")
print(f"{'='*80}")
for macro_type in ['GOLD', 'BNB', 'BTC_PREMIUM', 'ETH_PREMIUM']:
    subset = df_macro[df_macro['macro_type'] == macro_type]
    print(f"\n{macro_type}:")
    print(f"  Features: {len(subset)}")
    print(f"  Total importance: {subset['importance'].sum():.2f}")
    print(f"  Avg rank: {subset['rank'].mean():.0f}")
    print(f"  Best rank: {subset['rank'].min()}")
    print(f"  Worst rank: {subset['rank'].max()}")

print(f"\n{'='*80}")
print("BREAKDOWN BY DERIVATIVE TYPE")
print(f"{'='*80}")
for derivative in ['close', 'vel', 'roc_5']:
    subset = df_macro[df_macro['derivative'] == derivative]
    print(f"\n{derivative}:")
    print(f"  Features: {len(subset)}")
    print(f"  Total importance: {subset['importance'].sum():.2f}")
    print(f"  Avg rank: {subset['rank'].mean():.0f}")
    print(f"  Best rank: {subset['rank'].min()}")
    print(f"  Worst rank: {subset['rank'].max()}")

print(f"\n{'='*80}")
print("BREAKDOWN BY TIME STEP")
print(f"{'='*80}")
for t in range(8):  # 8-candle window (t0-t7)
    subset = df_macro[df_macro['time_step'] == str(t)]
    print(f"\nt{t}:")
    print(f"  Features: {len(subset)}")
    print(f"  Total importance: {subset['importance'].sum():.2f}")
    print(f"  Avg rank: {subset['rank'].mean():.0f}")

print(f"\n{'='*80}")
print("RECOMMENDATIONS")
print(f"{'='*80}")

# Calculate importance thresholds
total_importance = df_importance['importance'].sum()
macro_importance = df_macro['importance'].sum()
avg_importance = total_importance / len(df_importance)

# Find weak macro features
weak_threshold = avg_importance * 0.1  # Less than 10% of average
df_weak = df_macro[df_macro['importance'] < weak_threshold]

print(f"\nAverage feature importance: {avg_importance:.2f}")
print(f"Weak threshold (10% of avg): {weak_threshold:.2f}")
print(f"\nWeak macro features (< {weak_threshold:.2f}): {len(df_weak)}")
print(f"Weak macro importance: {df_weak['importance'].sum():.2f} ({df_weak['importance'].sum() / total_importance * 100:.2f}% of total)")

# Group weak features by derivative type
print("\nWeak features by derivative type:")
for derivative in ['close', 'vel', 'roc_5']:
    count = len(df_weak[df_weak['derivative'] == derivative])
    print(f"  {derivative}: {count}")

# Recommendation: Remove all "close" features since velocities/ROC can be calculated from them
close_features = df_macro[df_macro['derivative'] == 'close']
print(f"\n{'='*80}")
print("RECOMMENDATION: Remove 'close' features")
print(f"{'='*80}")
print(f"  Total 'close' features: {len(close_features)}")
print(f"  Total importance: {close_features['importance'].sum():.2f} ({close_features['importance'].sum() / total_importance * 100:.2f}% of total)")
print(f"  Average rank: {close_features['rank'].mean():.0f}")
print(f"  Best rank: {close_features['rank'].min()}")
print(f"\nRationale:")
print("  - Raw close prices have very low importance")
print("  - Velocities (vel) and ROC capture the same information in derivative form")
print("  - Removing close will reduce features by 4 per candle (4 macro types)")
print("  - In 8-candle window: 32 features removed (4 × 8)")
print("  - In 12-candle window: 48 features removed (4 × 12)")
print(f"  - Impact on total importance: {close_features['importance'].sum() / total_importance * 100:.2f}% loss")

print(f"\n{'='*80}")
print("FINAL RECOMMENDATION")
print(f"{'='*80}")
print("\nRemove from SHARED_FEATURE_LIST:")
print("  - macro_GOLD_close")
print("  - macro_BNB_close")
print("  - macro_BTC_PREMIUM_close")
print("  - macro_ETH_PREMIUM_close")
print("\nKeep in merge_macro_features() for calculation:")
print("  - Calculate close prices (needed for vel and roc_5)")
print("  - Calculate vel (important!)")
print("  - Calculate roc_5 (moderately important)")
print("  - But don't include 'close' in final feature list")
print("\nExpected impact:")
print(f"  - Feature reduction: {len(close_features)} features ({len(close_features) / len(df_macro) * 100:.1f}% of macro features)")
print(f"  - Importance loss: {close_features['importance'].sum() / total_importance * 100:.2f}% of total")
print(f"  - Speed improvement: ~{len(close_features) / len(df_importance) * 100:.1f}% fewer features to process")
print("  - Model performance: Minimal impact (close prices redundant with vel/roc)")

print(f"\n{'='*80}")
print("DONE")
print(f"{'='*80}")
