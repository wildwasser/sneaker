#!/usr/bin/env python3
"""
Model Validation Script - Checks for Statistical Illusions

This script performs comprehensive regression analysis to detect:
- Overfitting (train vs test performance gap)
- Feature dominance (single features with >40% importance)
- Suspicious metrics (too perfect to be real)
- Signal distribution sanity checks

MANDATORY: Run this before merging any model changes to main.

Usage:
    .venv/bin/python scripts/validate_model.py --issue <NUMBER> [--model MODEL_PATH] [--data DATA_PATH]

Pass Criteria:
    - Signal R² ≥ 70%
    - Direction accuracy ≥ 95%
    - Train/test R² gap ≤ 10%
    - No feature >40% importance
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger

# Set plotting style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Feature list (83 Enhanced V3 features)
ENHANCED_V3_FEATURES = [
    # Core indicators (20)
    'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel', 'bb_position', 'bb_position_vel',
    'macd_hist', 'macd_hist_vel', 'stoch', 'stoch_vel', 'di_diff', 'di_diff_vel',
    'adx', 'adr', 'adr_up_bars', 'adr_down_bars', 'is_up_bar', 'vol_ratio',
    'vol_ratio_vel', 'vwap_20',
    # Momentum features (24)
    'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20', 'price_accel_5',
    'price_accel_10', 'rsi_accel', 'rsi_7_accel', 'bb_pos_accel', 'macd_hist_accel',
    'stoch_accel', 'di_diff_accel', 'vol_regime_vel', 'vol_ratio_accel', 'atr',
    'atr_vel', 'rsi_2x', 'bb_pos_2x', 'macd_hist_2x', 'price_chg_2x', 'streak',
    'dist_from_high', 'dist_from_low', 'vwap_dist',
    # Advanced features (35)
    'rsi_4x', 'bb_pos_4x', 'macd_hist_4x', 'price_chg_4x', 'vol_ratio_4x',
    'rsi_bb_cross', 'macd_stoch_align', 'rsi_di_align', 'bb_macd_cross',
    'stoch_di_align', 'bb_stoch_cross', 'vol_regime_stable', 'vol_regime_rising',
    'vol_regime_falling', 'vol_regime_extreme', 'vol_spike', 'vol_drought',
    'new_high_20', 'new_low_20', 'range_position', 'higher_high', 'lower_low',
    'higher_low', 'lower_high', 'above_vwap_20', 'trend_consistent', 'adx_rising',
    'adx_falling', 'adx_stable', 'rsi_price_div', 'macd_price_div', 'stoch_price_div',
    'bb_price_div', 'di_price_div', 'vol_spike_extreme', 'vol_drought_extreme',
    # Statistical features (4)
    'hurst_exponent', 'permutation_entropy', 'cusum_signal', 'squeeze_duration'
]

# Validation thresholds
THRESHOLDS = {
    'signal_r2_min': 0.70,
    'direction_acc_min': 0.95,
    'train_test_gap_max': 0.10,
    'feature_importance_max': 0.40,
    'signal_rate_max': 0.20
}


def load_data(data_path):
    """Load and prepare dataset."""
    logger = setup_logger('validate')
    logger.info(f"Loading data from {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} samples")

    return df


def prepare_features(df):
    """Extract features and target."""
    logger = setup_logger('validate')

    # Check all features present
    missing = [f for f in ENHANCED_V3_FEATURES if f not in df.columns]
    if missing:
        logger.error(f"Missing features: {missing}")
        sys.exit(1)

    X = df[ENHANCED_V3_FEATURES].values
    y = df['target'].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    return X, y


def compute_sample_weights(y):
    """Compute V3 sample weights (5x for signals)."""
    weights = np.ones(len(y))
    weights[y != 0] = 5.0
    return weights


def train_model(X_train, y_train, sw_train, X_test, y_test, sw_test):
    """Train LightGBM model with V3 configuration."""
    logger = setup_logger('validate')
    logger.info("Training model...")

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
        'n_jobs': -1,
        'verbose': -1
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_test, y_test)],
        eval_sample_weight=[sw_test],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    logger.info(f"Training completed. Best iteration: {model.best_iteration_}")

    return model


def evaluate_metrics(model, X_train, y_train, X_test, y_test, threshold=4.0):
    """Compute comprehensive evaluation metrics."""
    logger = setup_logger('validate')

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Overall metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Signal-only metrics
    train_signal_mask = y_train != 0
    test_signal_mask = y_test != 0

    if train_signal_mask.sum() > 0:
        train_signal_r2 = r2_score(y_train[train_signal_mask], y_train_pred[train_signal_mask])
        train_signal_mae = mean_absolute_error(y_train[train_signal_mask], y_train_pred[train_signal_mask])
    else:
        train_signal_r2 = 0
        train_signal_mae = 0

    if test_signal_mask.sum() > 0:
        test_signal_r2 = r2_score(y_test[test_signal_mask], y_test_pred[test_signal_mask])
        test_signal_mae = mean_absolute_error(y_test[test_signal_mask], y_test_pred[test_signal_mask])
    else:
        test_signal_r2 = 0
        test_signal_mae = 0

    # Direction accuracy (signals only)
    train_correct_dir = np.sum(np.sign(y_train[train_signal_mask]) == np.sign(y_train_pred[train_signal_mask]))
    train_dir_acc = train_correct_dir / train_signal_mask.sum() if train_signal_mask.sum() > 0 else 0

    test_correct_dir = np.sum(np.sign(y_test[test_signal_mask]) == np.sign(y_test_pred[test_signal_mask]))
    test_dir_acc = test_correct_dir / test_signal_mask.sum() if test_signal_mask.sum() > 0 else 0

    # Zero metrics
    train_zero_mask = y_train == 0
    test_zero_mask = y_test == 0

    train_zero_mae = mean_absolute_error(y_train[train_zero_mask], y_train_pred[train_zero_mask]) if train_zero_mask.sum() > 0 else 0
    test_zero_mae = mean_absolute_error(y_test[test_zero_mask], y_test_pred[test_zero_mask]) if test_zero_mask.sum() > 0 else 0

    # Signal rate at threshold
    test_signal_rate = np.sum(np.abs(y_test_pred) >= threshold) / len(y_test_pred)

    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_signal_r2': train_signal_r2,
        'test_signal_r2': test_signal_r2,
        'train_signal_mae': train_signal_mae,
        'test_signal_mae': test_signal_mae,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'train_zero_mae': train_zero_mae,
        'test_zero_mae': test_zero_mae,
        'test_signal_rate': test_signal_rate,
        'train_test_gap': abs(train_r2 - test_r2)
    }

    return metrics


def analyze_feature_importance(model):
    """Analyze feature importance for dominance."""
    importance = model.feature_importances_
    total_importance = importance.sum()
    normalized_importance = importance / total_importance

    # Get top 10 features
    top_indices = np.argsort(importance)[-10:][::-1]
    top_features = [(ENHANCED_V3_FEATURES[i], normalized_importance[i]) for i in top_indices]

    # Check for dominance
    max_importance = normalized_importance.max()
    max_feature = ENHANCED_V3_FEATURES[normalized_importance.argmax()]

    return {
        'top_features': top_features,
        'max_feature': max_feature,
        'max_importance': max_importance
    }


def create_proof_directory(issue_number):
    """Create proof directory for issue."""
    proof_dir = Path(__file__).parent.parent / 'proof' / f'issue-{issue_number}'
    proof_dir.mkdir(parents=True, exist_ok=True)
    return proof_dir


def plot_regression_analysis(y_train, y_train_pred, y_test, y_test_pred, proof_dir, timestamp):
    """Plot regression: predicted vs actual."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Train: All samples
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.3, s=1)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Target')
    axes[0, 0].set_ylabel('Predicted Target')
    axes[0, 0].set_title('Train: Predicted vs Actual (All Samples)')
    axes[0, 0].grid(True, alpha=0.3)

    # Test: All samples
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.3, s=1)
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('Actual Target')
    axes[0, 1].set_ylabel('Predicted Target')
    axes[0, 1].set_title('Test: Predicted vs Actual (All Samples)')
    axes[0, 1].grid(True, alpha=0.3)

    # Train: Signals only
    train_signal_mask = y_train != 0
    if train_signal_mask.sum() > 0:
        axes[1, 0].scatter(y_train[train_signal_mask], y_train_pred[train_signal_mask], alpha=0.5, s=2)
        min_val = min(y_train[train_signal_mask].min(), y_train_pred[train_signal_mask].min())
        max_val = max(y_train[train_signal_mask].max(), y_train_pred[train_signal_mask].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Target')
        axes[1, 0].set_ylabel('Predicted Target')
        axes[1, 0].set_title('Train: Predicted vs Actual (Signals Only)')
        axes[1, 0].grid(True, alpha=0.3)

    # Test: Signals only
    test_signal_mask = y_test != 0
    if test_signal_mask.sum() > 0:
        axes[1, 1].scatter(y_test[test_signal_mask], y_test_pred[test_signal_mask], alpha=0.5, s=2)
        min_val = min(y_test[test_signal_mask].min(), y_test_pred[test_signal_mask].min())
        max_val = max(y_test[test_signal_mask].max(), y_test_pred[test_signal_mask].max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Target')
        axes[1, 1].set_ylabel('Predicted Target')
        axes[1, 1].set_title('Test: Predicted vs Actual (Signals Only)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = proof_dir / f'regression_analysis_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(save_path)


def plot_residual_analysis(y_test, y_test_pred, proof_dir, timestamp):
    """Plot residual analysis."""
    residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Residuals vs Predicted
    axes[0, 0].scatter(y_test_pred, residuals, alpha=0.3, s=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Predicted Value')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title('Residuals vs Predicted (Look for patterns = bad)')
    axes[0, 0].grid(True, alpha=0.3)

    # Residual histogram
    axes[0, 1].hist(residuals, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residual Distribution (Should be centered at 0)')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Should follow red line for normality)')
    axes[1, 0].grid(True, alpha=0.3)

    # Residuals vs Actual
    axes[1, 1].scatter(y_test, residuals, alpha=0.3, s=1)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Actual Value')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].set_title('Residuals vs Actual')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = proof_dir / f'residual_analysis_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(save_path)


def plot_feature_importance(importance_analysis, proof_dir, timestamp):
    """Plot feature importance."""
    top_features = importance_analysis['top_features']

    fig, ax = plt.subplots(figsize=(12, 8))

    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    y_pos = np.arange(len(features))
    colors = ['red' if imp > 0.40 else 'orange' if imp > 0.30 else 'green' for imp in importances]

    ax.barh(y_pos, importances, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Top 10 Feature Importance (Red >40% = Dominance Warning)')
    ax.axvline(x=0.40, color='red', linestyle='--', lw=2, label='40% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = proof_dir / f'feature_importance_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(save_path)


def plot_signal_distribution(y_test, y_test_pred, threshold, proof_dir, timestamp):
    """Plot signal distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Actual target distribution
    axes[0, 0].hist(y_test, bins=100, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=threshold, color='red', linestyle='--', lw=2, label=f'{threshold}σ threshold')
    axes[0, 0].axvline(x=-threshold, color='red', linestyle='--', lw=2)
    axes[0, 0].set_xlabel('Target Value (σ)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Actual Target Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Predicted distribution
    axes[0, 1].hist(y_test_pred, bins=100, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(x=threshold, color='red', linestyle='--', lw=2, label=f'{threshold}σ threshold')
    axes[0, 1].axvline(x=-threshold, color='red', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Value (σ)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Predicted Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Signal vs zero comparison (actual)
    signal_mask = y_test != 0
    axes[1, 0].hist([y_test[~signal_mask], y_test[signal_mask]],
                    bins=100, label=['Zeros', 'Signals'], alpha=0.7, stacked=True)
    axes[1, 0].set_xlabel('Target Value (σ)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Actual: Zeros vs Signals')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Predicted signal rate by threshold
    thresholds = np.linspace(0, 10, 100)
    signal_rates = [np.mean(np.abs(y_test_pred) >= t) * 100 for t in thresholds]
    axes[1, 1].plot(thresholds, signal_rates, lw=2)
    axes[1, 1].axvline(x=threshold, color='red', linestyle='--', lw=2, label=f'{threshold}σ (current)')
    axes[1, 1].axhline(y=20, color='orange', linestyle='--', lw=2, label='20% (max acceptable)')
    axes[1, 1].set_xlabel('Threshold (σ)')
    axes[1, 1].set_ylabel('Signal Rate (%)')
    axes[1, 1].set_title('Signal Rate vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = proof_dir / f'signal_distribution_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return str(save_path)


def save_validation_report(metrics, importance_analysis, checks, red_flags, proof_dir, timestamp, issue_number):
    """Save comprehensive validation report."""
    report_path = proof_dir / f'validation_report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL VALIDATION REPORT - Issue #{issue_number}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train R²:           {metrics['train_r2']:.4f}\n")
        f.write(f"Test R²:            {metrics['test_r2']:.4f}\n")
        f.write(f"Train/Test Gap:     {metrics['train_test_gap']:.4f}\n")
        f.write(f"Train RMSE:         {metrics['train_rmse']:.4f}\n")
        f.write(f"Test RMSE:          {metrics['test_rmse']:.4f}\n\n")

        f.write("SIGNAL METRICS (What Matters!)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train Signal R²:    {metrics['train_signal_r2']:.4f}\n")
        f.write(f"Test Signal R²:     {metrics['test_signal_r2']:.4f}\n")
        f.write(f"Train Signal MAE:   {metrics['train_signal_mae']:.4f}σ\n")
        f.write(f"Test Signal MAE:    {metrics['test_signal_mae']:.4f}σ\n")
        f.write(f"Train Dir Acc:      {metrics['train_dir_acc']:.4f}\n")
        f.write(f"Test Dir Acc:       {metrics['test_dir_acc']:.4f}\n\n")

        f.write("ZERO METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train Zero MAE:     {metrics['train_zero_mae']:.4f}σ\n")
        f.write(f"Test Zero MAE:      {metrics['test_zero_mae']:.4f}σ\n\n")

        f.write("SIGNAL RATE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Signal Rate:   {metrics['test_signal_rate']:.4f}\n\n")

        f.write("FEATURE IMPORTANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Max Feature:        {importance_analysis['max_feature']}\n")
        f.write(f"Max Importance:     {importance_analysis['max_importance']:.4f}\n\n")
        f.write("Top 10 Features:\n")
        for i, (feat, imp) in enumerate(importance_analysis['top_features'], 1):
            f.write(f"  {i:2d}. {feat:30s} {imp:.4f}\n")
        f.write("\n")

        f.write("PASS/FAIL CRITERIA\n")
        f.write("=" * 80 + "\n")
        for check, passed in checks.items():
            status = 'PASS' if passed else 'FAIL'
            f.write(f"[{status}] {check}\n")
        f.write("\n")

        f.write("RED FLAGS\n")
        f.write("=" * 80 + "\n")
        if red_flags:
            for flag in red_flags:
                f.write(f"{flag}\n")
        else:
            f.write("No red flags detected\n")
        f.write("\n")

        f.write("FINAL VERDICT\n")
        f.write("=" * 80 + "\n")
        if all(checks.values()) and not red_flags:
            f.write("✅ VALIDATION PASSED - Model appears statistically sound\n")
        else:
            f.write("❌ VALIDATION FAILED - Statistical illusions or issues detected\n")
            f.write("DO NOT MERGE TO MAIN\n")

    return str(report_path)


def validate(model_path=None, data_path=None, issue_number=None):
    """Run full validation pipeline."""
    logger = setup_logger('validate')

    # Require issue number
    if issue_number is None:
        logger.error("ERROR: --issue parameter is REQUIRED")
        logger.error("Usage: .venv/bin/python scripts/validate_model.py --issue <NUMBER>")
        return 1

    # Default paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'data' / 'enhanced_v3_dataset.json'

    # Create proof directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    proof_dir = create_proof_directory(issue_number)
    logger.info(f"Proof directory: {proof_dir}")

    logger.info("=" * 80)
    logger.info(f"MODEL VALIDATION - Issue #{issue_number}")
    logger.info("Statistical Illusion Check")
    logger.info("=" * 80)

    # Load data
    df = load_data(data_path)
    X, y = prepare_features(df)

    # Split data
    logger.info("Splitting data (90/10 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Compute sample weights
    sw_train = compute_sample_weights(y_train)
    sw_test = compute_sample_weights(y_test)

    # Train model (or load if provided)
    if model_path and Path(model_path).exists():
        logger.info(f"Loading model from {model_path}")
        model = lgb.Booster(model_file=str(model_path))
    else:
        model = train_model(X_train, y_train, sw_train, X_test, y_test, sw_test)

    # Evaluate
    logger.info("\nEvaluating metrics...")
    metrics = evaluate_metrics(model, X_train, y_train, X_test, y_test)

    # Feature importance
    logger.info("\nAnalyzing feature importance...")
    importance_analysis = analyze_feature_importance(model)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)

    logger.info("\n--- Overall Metrics ---")
    logger.info(f"Train R²:      {metrics['train_r2']:.4f}")
    logger.info(f"Test R²:       {metrics['test_r2']:.4f}")
    logger.info(f"Train/Test Gap: {metrics['train_test_gap']:.4f} ({'PASS' if metrics['train_test_gap'] <= THRESHOLDS['train_test_gap_max'] else 'FAIL'})")

    logger.info("\n--- Signal Metrics (What Matters!) ---")
    logger.info(f"Train Signal R²: {metrics['train_signal_r2']:.4f}")
    logger.info(f"Test Signal R²:  {metrics['test_signal_r2']:.4f} ({'PASS' if metrics['test_signal_r2'] >= THRESHOLDS['signal_r2_min'] else 'FAIL'})")
    logger.info(f"Train Dir Acc:   {metrics['train_dir_acc']:.4f}")
    logger.info(f"Test Dir Acc:    {metrics['test_dir_acc']:.4f} ({'PASS' if metrics['test_dir_acc'] >= THRESHOLDS['direction_acc_min'] else 'FAIL'})")

    logger.info("\n--- Zero Metrics ---")
    logger.info(f"Train Zero MAE: {metrics['train_zero_mae']:.4f}σ")
    logger.info(f"Test Zero MAE:  {metrics['test_zero_mae']:.4f}σ")

    logger.info("\n--- Signal Rate ---")
    logger.info(f"Test Signal Rate (4σ): {metrics['test_signal_rate']:.4f} ({'PASS' if metrics['test_signal_rate'] <= THRESHOLDS['signal_rate_max'] else 'FAIL'})")

    logger.info("\n--- Top 10 Features by Importance ---")
    for i, (feat, imp) in enumerate(importance_analysis['top_features'], 1):
        logger.info(f"{i:2d}. {feat:30s} {imp:.4f}")

    logger.info(f"\nMax Feature: {importance_analysis['max_feature']} ({importance_analysis['max_importance']:.4f})")
    logger.info(f"Feature Dominance: {'PASS' if importance_analysis['max_importance'] <= THRESHOLDS['feature_importance_max'] else 'FAIL'}")

    # Overall pass/fail
    logger.info("\n" + "=" * 80)
    logger.info("PASS/FAIL CRITERIA")
    logger.info("=" * 80)

    checks = {
        'Signal R² ≥ 70%': metrics['test_signal_r2'] >= THRESHOLDS['signal_r2_min'],
        'Direction Acc ≥ 95%': metrics['test_dir_acc'] >= THRESHOLDS['direction_acc_min'],
        'Train/Test Gap ≤ 10%': metrics['train_test_gap'] <= THRESHOLDS['train_test_gap_max'],
        'No Feature Dominance (>40%)': importance_analysis['max_importance'] <= THRESHOLDS['feature_importance_max'],
        'Signal Rate ≤ 20%': metrics['test_signal_rate'] <= THRESHOLDS['signal_rate_max']
    }

    all_passed = True
    for check, passed in checks.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        logger.info(f"{status} - {check}")
        if not passed:
            all_passed = False

    # Red flag checks
    logger.info("\n" + "=" * 80)
    logger.info("RED FLAG CHECKS")
    logger.info("=" * 80)

    red_flags = []

    if metrics['train_test_gap'] > 0.15:
        red_flags.append("🚩 SEVERE OVERFITTING: Train/test gap >15%")

    if metrics['test_signal_r2'] > 0.99:
        red_flags.append("🚩 SUSPICIOUSLY PERFECT: Signal R² >99%")

    if importance_analysis['max_importance'] > 0.50:
        red_flags.append(f"🚩 EXTREME DOMINANCE: {importance_analysis['max_feature']} >50% importance")

    if metrics['test_signal_rate'] > 0.30:
        red_flags.append("🚩 TOO MANY SIGNALS: >30% signal rate")

    if red_flags:
        for flag in red_flags:
            logger.error(flag)
        all_passed = False
    else:
        logger.info("✅ No red flags detected")

    # Generate proof visualizations
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PROOF VISUALIZATIONS")
    logger.info("=" * 80)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    logger.info("Creating regression analysis plot...")
    reg_plot = plot_regression_analysis(y_train, y_train_pred, y_test, y_test_pred, proof_dir, timestamp)
    logger.info(f"  Saved: {reg_plot}")

    logger.info("Creating residual analysis plot...")
    res_plot = plot_residual_analysis(y_test, y_test_pred, proof_dir, timestamp)
    logger.info(f"  Saved: {res_plot}")

    logger.info("Creating feature importance plot...")
    feat_plot = plot_feature_importance(importance_analysis, proof_dir, timestamp)
    logger.info(f"  Saved: {feat_plot}")

    logger.info("Creating signal distribution plot...")
    sig_plot = plot_signal_distribution(y_test, y_test_pred, 4.0, proof_dir, timestamp)
    logger.info(f"  Saved: {sig_plot}")

    logger.info("Saving validation report...")
    report = save_validation_report(metrics, importance_analysis, checks, red_flags, proof_dir, timestamp, issue_number)
    logger.info(f"  Saved: {report}")

    # Final verdict
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ VALIDATION PASSED - Model appears statistically sound")
        logger.info("=" * 80)
        logger.info(f"\n📁 Proof saved to: {proof_dir}")
        logger.info("\nNext steps:")
        logger.info(f"  1. Review visualizations in {proof_dir}")
        logger.info(f"  2. git add {proof_dir}")
        logger.info(f"  3. git commit -m 'Add #{issue_number}: validation proof'")
        logger.info(f"  4. Continue with backtest: .venv/bin/python scripts/backtest.py --issue {issue_number}")
        return 0
    else:
        logger.error("❌ VALIDATION FAILED - Statistical illusions or issues detected")
        logger.error("=" * 80)
        logger.error(f"\n📁 Proof saved to: {proof_dir}")
        logger.error("\nDO NOT MERGE TO MAIN. Investigate issues before proceeding.")
        logger.error(f"Review visualizations in {proof_dir} to diagnose problems.")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Validate model for statistical illusions')
    parser.add_argument('--issue', type=int, required=True, help='GitHub issue number (REQUIRED)')
    parser.add_argument('--model', type=str, help='Path to model file (optional, will train if not provided)')
    parser.add_argument('--data', type=str, help='Path to dataset JSON (default: data/enhanced_v3_dataset.json)')

    args = parser.parse_args()

    sys.exit(validate(model_path=args.model, data_path=args.data, issue_number=args.issue))


if __name__ == '__main__':
    main()
