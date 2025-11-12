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
    .venv/bin/python scripts/validate_model.py [--model MODEL_PATH] [--data DATA_PATH]

Pass Criteria:
    - Signal R² ≥ 70%
    - Direction accuracy ≥ 95%
    - Train/test R² gap ≤ 10%
    - No feature >40% importance
"""

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger

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


def validate(model_path=None, data_path=None):
    """Run full validation pipeline."""
    logger = setup_logger('validate')

    # Default paths
    if data_path is None:
        data_path = Path(__file__).parent.parent / 'data' / 'enhanced_v3_dataset.json'

    logger.info("=" * 80)
    logger.info("MODEL VALIDATION - Statistical Illusion Check")
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

    # Final verdict
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("✅ VALIDATION PASSED - Model appears statistically sound")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("❌ VALIDATION FAILED - Statistical illusions or issues detected")
        logger.error("=" * 80)
        logger.error("\nDO NOT MERGE TO MAIN. Investigate issues before proceeding.")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Validate model for statistical illusions')
    parser.add_argument('--model', type=str, help='Path to model file (optional, will train if not provided)')
    parser.add_argument('--data', type=str, help='Path to dataset JSON (default: data/enhanced_v3_dataset.json)')

    args = parser.parse_args()

    sys.exit(validate(model_path=args.model, data_path=args.data))


if __name__ == '__main__':
    main()
