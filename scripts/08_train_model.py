#!/usr/bin/env python3
"""
Train Model on Windowed Data

Trains LightGBM regression model using windowed time series data with
V3 sample weighting (5x for ghost signals).

Part of Issue #8 (Pipeline Restructuring Epic #1)

Usage:
    # Default: 12-candle windows, default hyperparameters
    .venv/bin/python scripts/08_train_model.py

    # Custom paths and issue folder
    .venv/bin/python scripts/08_train_model.py \
      --input data/features/windowed_training_data.json \
      --output models/issue-1/model.txt \
      --issue issue-1

    # Custom hyperparameters for optimization experiments
    .venv/bin/python scripts/08_train_model.py \
      --issue issue-24 \
      --num-leaves 127 \
      --max-depth 6 \
      --learning-rate 0.05 \
      --n-estimators 1000

Arguments:
    --input: Input windowed data JSON (from script 07)
    --output: Output model path
    --issue: Issue folder name for proof outputs (e.g., 'issue-1')
    --test-size: Test set fraction (default: 0.1 = 10%)

    Model Hyperparameters (for optimization):
    --num-leaves: Maximum leaves in one tree (default: 255)
    --max-depth: Maximum tree depth (default: 8)
    --learning-rate: Learning rate (default: 0.01)
    --n-estimators: Number of boosting iterations (default: 2000)
    --subsample: Fraction of data to sample (default: 0.8)
    --colsample-bytree: Fraction of features to sample (default: 0.8)

Input:
    data/features/windowed_training_data.json
    - ~779,000 windows
    - 1,068 features (89 shared (Issue #23) × 12 candles)
    - 1 target per window

Output:
    models/issue-1/model.txt - Trained model
    proof/issue-1/training_regression_*.png - Train vs test regression
    proof/issue-1/training_residuals_*.png - Residual analysis
    proof/issue-1/training_feature_importance_*.png - Top features
    proof/issue-1/training_signal_dist_*.png - Signal distribution
    proof/issue-1/training_report_*.txt - Metrics report
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.features_shared import SHARED_FEATURE_LIST
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def extract_windowed_features(df: pd.DataFrame, window_size: int, logger) -> tuple:
    """
    Extract windowed feature columns from DataFrame.

    Features are named like: feature_name_t0, feature_name_t1, ..., feature_name_t11

    Args:
        df: DataFrame with windowed features
        window_size: Number of time steps per window
        logger: Logger instance

    Returns:
        (feature_columns, X, y) tuple
    """
    # Build list of expected windowed feature columns
    feature_cols = []
    for feature in SHARED_FEATURE_LIST:
        for t in range(window_size):
            col_name = f"{feature}_t{t}"
            if col_name in df.columns:
                feature_cols.append(col_name)

    logger.info(f"Expected features: {len(SHARED_FEATURE_LIST)} × {window_size} = {len(SHARED_FEATURE_LIST) * window_size}")
    logger.info(f"Found features: {len(feature_cols)}")

    if len(feature_cols) != len(SHARED_FEATURE_LIST) * window_size:
        missing_count = len(SHARED_FEATURE_LIST) * window_size - len(feature_cols)
        logger.warning(f"  ⚠️  Missing {missing_count} feature columns!")

    # Extract feature matrix and target
    X = df[feature_cols].values
    y = df['target'].values

    return feature_cols, X, y


def compute_sample_weights(y: np.ndarray, signal_weight: float = 5.0) -> np.ndarray:
    """
    Compute V3 sample weights.

    Ghost signals weighted signal_weight×, normal candles weighted 1×.

    Args:
        y: Target array
        signal_weight: Weight multiplier for ghost signals (default: 5.0)

    Returns:
        Sample weights array
    """
    weights = np.ones(len(y))
    weights[y != 0] = signal_weight  # Ghost signals
    weights[y == 0] = 1.0            # Normal candles
    return weights


def train_model(X_train, y_train, sw_train, X_test, y_test, sw_test, logger,
                num_leaves=255, max_depth=8, learning_rate=0.01, n_estimators=2000,
                subsample=0.8, colsample_bytree=0.8):
    """
    Train LightGBM model with V3 sample weighting.

    Args:
        X_train, y_train: Training data
        sw_train: Training sample weights
        X_test, y_test: Test data
        sw_test: Test sample weights
        logger: Logger instance
        num_leaves: Maximum leaves in one tree (default: 255)
        max_depth: Maximum tree depth (default: 8)
        learning_rate: Learning rate (default: 0.01)
        n_estimators: Number of boosting iterations (default: 2000)
        subsample: Fraction of data to sample (default: 0.8)
        colsample_bytree: Fraction of features to sample (default: 0.8)

    Returns:
        Trained LightGBM model
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING LIGHTGBM MODEL")
    logger.info("=" * 80)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': num_leaves,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    logger.info("Model parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")

    logger.info("")
    logger.info("Training with V3 sample weighting...")
    logger.info(f"  Signal weight: 5.0× (ghost signals)")
    logger.info(f"  Normal weight: 1.0× (normal candles)")

    start_time = time.time()

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sw_train,
        eval_set=[(X_test, y_test)],
        eval_sample_weight=[sw_test],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    train_time = time.time() - start_time

    logger.info(f"✓ Training complete in {train_time:.1f}s ({train_time/60:.1f} minutes)")
    logger.info(f"  Best iteration: {model.best_iteration_}")
    logger.info(f"  Best score: {model.best_score_['valid_0']['rmse']:.4f}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, logger):
    """
    Evaluate model performance on train and test sets.

    Args:
        model: Trained model
        X_train, y_train: Training data
        X_test, y_test: Test data
        logger: Logger instance

    Returns:
        Dictionary of metrics
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Overall metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    logger.info("Overall Performance:")
    logger.info(f"  Train R²:  {train_r2:.4f}")
    logger.info(f"  Test R²:   {test_r2:.4f}")
    logger.info(f"  Train RMSE: {train_rmse:.4f}σ")
    logger.info(f"  Test RMSE:  {test_rmse:.4f}σ")

    # Signal-only metrics (what matters!)
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

    logger.info("")
    logger.info("Signal Performance (CRITICAL):")
    logger.info(f"  Train Signal R²:  {train_signal_r2:.4f}")
    logger.info(f"  Test Signal R²:   {test_signal_r2:.4f}")
    logger.info(f"  Train Signal MAE: {train_signal_mae:.4f}σ")
    logger.info(f"  Test Signal MAE:  {test_signal_mae:.4f}σ")

    # Direction accuracy (sign prediction)
    train_dir_correct = (np.sign(y_train_pred) == np.sign(y_train)).sum()
    test_dir_correct = (np.sign(y_test_pred) == np.sign(y_test)).sum()
    train_dir_acc = train_dir_correct / len(y_train) * 100
    test_dir_acc = test_dir_correct / len(y_test) * 100

    logger.info("")
    logger.info("Direction Accuracy:")
    logger.info(f"  Train: {train_dir_acc:.2f}%")
    logger.info(f"  Test:  {test_dir_acc:.2f}%")

    return {
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
        'y_train': y_train,
        'y_train_pred': y_train_pred,
        'y_test': y_test,
        'y_test_pred': y_test_pred
    }


def generate_proof_visualizations(model, metrics, feature_cols, issue_folder, logger):
    """
    Generate proof visualizations for model validation.

    Args:
        model: Trained model
        metrics: Dictionary of evaluation metrics
        feature_cols: List of feature column names
        issue_folder: Path to issue proof folder
        logger: Logger instance
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING PROOF VISUALIZATIONS")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Regression Plot (Train vs Test)
    logger.info("1. Creating regression plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Train
    ax1.scatter(metrics['y_train'], metrics['y_train_pred'], alpha=0.3, s=1)
    ax1.plot([metrics['y_train'].min(), metrics['y_train'].max()],
             [metrics['y_train'].min(), metrics['y_train'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Target (σ)')
    ax1.set_ylabel('Predicted Target (σ)')
    ax1.set_title(f'Train Set (R² = {metrics["train_r2"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test
    ax2.scatter(metrics['y_test'], metrics['y_test_pred'], alpha=0.3, s=1)
    ax2.plot([metrics['y_test'].min(), metrics['y_test'].max()],
             [metrics['y_test'].min(), metrics['y_test'].max()],
             'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Target (σ)')
    ax2.set_ylabel('Predicted Target (σ)')
    ax2.set_title(f'Test Set (R² = {metrics["test_r2"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    regression_path = issue_folder / f'training_regression_{timestamp}.png'
    plt.savefig(regression_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {regression_path.name}")

    # 2. Residual Analysis
    logger.info("2. Creating residual analysis...")
    train_residuals = metrics['y_train'] - metrics['y_train_pred']
    test_residuals = metrics['y_test'] - metrics['y_test_pred']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train residuals vs predicted
    axes[0, 0].scatter(metrics['y_train_pred'], train_residuals, alpha=0.3, s=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Target (σ)')
    axes[0, 0].set_ylabel('Residual (σ)')
    axes[0, 0].set_title('Train: Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)

    # Test residuals vs predicted
    axes[0, 1].scatter(metrics['y_test_pred'], test_residuals, alpha=0.3, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Target (σ)')
    axes[0, 1].set_ylabel('Residual (σ)')
    axes[0, 1].set_title('Test: Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)

    # Train residual distribution
    axes[1, 0].hist(train_residuals, bins=100, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residual (σ)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Train: Residual Distribution (μ={train_residuals.mean():.4f})')
    axes[1, 0].grid(True, alpha=0.3)

    # Test residual distribution
    axes[1, 1].hist(test_residuals, bins=100, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Residual (σ)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Test: Residual Distribution (μ={test_residuals.mean():.4f})')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    residuals_path = issue_folder / f'training_residuals_{timestamp}.png'
    plt.savefig(residuals_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {residuals_path.name}")

    # 3. Feature Importance (Top 30)
    logger.info("3. Creating feature importance plot...")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).head(30)

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 30 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    importance_path = issue_folder / f'training_feature_importance_{timestamp}.png'
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {importance_path.name}")

    # 4. Signal Distribution
    logger.info("4. Creating signal distribution plot...")
    train_signals = metrics['y_train'][metrics['y_train'] != 0]
    test_signals = metrics['y_test'][metrics['y_test'] != 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Train
    ax1.hist(train_signals, bins=50, alpha=0.7, edgecolor='black', label='Actual')
    ax1.axvline(train_signals.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {train_signals.mean():.2f}σ')
    ax1.set_xlabel('Target (σ)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Train: Signal Distribution ({len(train_signals):,} signals)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test
    ax2.hist(test_signals, bins=50, alpha=0.7, edgecolor='black', label='Actual')
    ax2.axvline(test_signals.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {test_signals.mean():.2f}σ')
    ax2.set_xlabel('Target (σ)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Test: Signal Distribution ({len(test_signals):,} signals)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    signal_dist_path = issue_folder / f'training_signal_dist_{timestamp}.png'
    plt.savefig(signal_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   ✓ Saved: {signal_dist_path.name}")

    # 5. Metrics Report (Text)
    logger.info("5. Creating metrics report...")
    report_path = issue_folder / f'training_report_{timestamp}.txt'

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Overall Performance:\n")
        f.write(f"  Train R²:       {metrics['train_r2']:.4f}\n")
        f.write(f"  Test R²:        {metrics['test_r2']:.4f}\n")
        f.write(f"  Train RMSE:     {metrics['train_rmse']:.4f}σ\n")
        f.write(f"  Test RMSE:      {metrics['test_rmse']:.4f}σ\n\n")

        f.write("Signal Performance (CRITICAL):\n")
        f.write(f"  Train Signal R²:  {metrics['train_signal_r2']:.4f}\n")
        f.write(f"  Test Signal R²:   {metrics['test_signal_r2']:.4f}\n")
        f.write(f"  Train Signal MAE: {metrics['train_signal_mae']:.4f}σ\n")
        f.write(f"  Test Signal MAE:  {metrics['test_signal_mae']:.4f}σ\n\n")

        f.write("Direction Accuracy:\n")
        f.write(f"  Train: {metrics['train_dir_acc']:.2f}%\n")
        f.write(f"  Test:  {metrics['test_dir_acc']:.2f}%\n\n")

        f.write("Dataset Size:\n")
        f.write(f"  Train samples: {len(metrics['y_train']):,}\n")
        f.write(f"  Test samples:  {len(metrics['y_test']):,}\n")
        f.write(f"  Train signals: {(metrics['y_train'] != 0).sum():,}\n")
        f.write(f"  Test signals:  {(metrics['y_test'] != 0).sum():,}\n\n")

        f.write("=" * 80 + "\n")

    logger.info(f"   ✓ Saved: {report_path.name}")
    logger.info("")
    logger.info(f"All proof visualizations saved to: {issue_folder}/")


def main():
    parser = argparse.ArgumentParser(
        description='Train model on windowed time series data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features/windowed_training_data.json',
        help='Input windowed data JSON (from script 07)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/issue-1/model.txt',
        help='Output model path'
    )
    parser.add_argument(
        '--issue',
        type=str,
        default='issue-1',
        help='Issue folder name for proof outputs (e.g., "issue-1")'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.1,
        help='Test set fraction (default: 0.1 = 10%%)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=12,
        help='Window size used in script 07 (for feature extraction)'
    )

    # Model hyperparameters
    parser.add_argument(
        '--num-leaves',
        type=int,
        default=255,
        help='Maximum leaves in one tree (default: 255)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=8,
        help='Maximum tree depth (default: 8)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=2000,
        help='Number of boosting iterations (default: 2000)'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Fraction of data to sample (default: 0.8)'
    )
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Fraction of features to sample (default: 0.8)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('train_model')

    logger.info("=" * 80)
    logger.info("TRAIN MODEL ON WINDOWED DATA")
    logger.info("=" * 80)
    logger.info(f"Input:       {args.input}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Issue:       {args.issue}")
    logger.info(f"Test size:   {args.test_size * 100:.0f}%")
    logger.info(f"Window size: {args.window_size} candles")
    logger.info("")
    logger.info("Model Hyperparameters:")
    logger.info(f"  num_leaves:       {args.num_leaves}")
    logger.info(f"  max_depth:        {args.max_depth}")
    logger.info(f"  learning_rate:    {args.learning_rate}")
    logger.info(f"  n_estimators:     {args.n_estimators}")
    logger.info(f"  subsample:        {args.subsample}")
    logger.info(f"  colsample_bytree: {args.colsample_bytree}")
    logger.info("")

    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    proof_folder = Path('proof') / args.issue
    proof_folder.mkdir(parents=True, exist_ok=True)

    # Load windowed data
    logger.info("=" * 80)
    logger.info("LOADING WINDOWED DATA")
    logger.info("=" * 80)

    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("   Run scripts/07_create_windows.py first!")
        return 1

    logger.info(f"Loading {args.input}...")
    start_time = time.time()

    with open(args.input, 'r') as f:
        data = json.load(f)

    load_time = time.time() - start_time

    logger.info(f"✓ Loaded {len(data):,} windows in {load_time:.1f}s")

    # Convert to DataFrame
    logger.info("")
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(data)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {len(df.columns)}")

    # Extract features and target
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTING FEATURES AND TARGET")
    logger.info("=" * 80)

    feature_cols, X, y = extract_windowed_features(df, args.window_size, logger)

    logger.info(f"  Feature matrix: {X.shape}")
    logger.info(f"  Target vector:  {y.shape}")

    # Analyze target distribution
    signals = (y != 0).sum()
    zeros = (y == 0).sum()
    logger.info("")
    logger.info("Target distribution:")
    logger.info(f"  Signals: {signals:,} ({signals/len(y)*100:.1f}%)")
    logger.info(f"  Zeros:   {zeros:,} ({zeros/len(y)*100:.1f}%)")

    # Compute sample weights
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPUTING SAMPLE WEIGHTS (V3)")
    logger.info("=" * 80)

    sample_weights = compute_sample_weights(y, signal_weight=5.0)

    signal_weight_sum = sample_weights[y != 0].sum()
    zero_weight_sum = sample_weights[y == 0].sum()
    total_weight = signal_weight_sum + zero_weight_sum

    logger.info(f"Signal weight sum: {signal_weight_sum:,.0f}")
    logger.info(f"Zero weight sum:   {zero_weight_sum:,.0f}")
    logger.info(f"Total weight:      {total_weight:,.0f}")
    logger.info(f"Effective signal ratio: {signal_weight_sum/total_weight*100:.1f}%")
    logger.info("")
    logger.info("This balances the ~5-10% signal rate to ~50-60% effective influence.")

    # Train/test split
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weights,
        test_size=args.test_size,
        random_state=42,
        shuffle=True
    )

    logger.info(f"Train set: {len(y_train):,} samples ({(y_train != 0).sum():,} signals)")
    logger.info(f"Test set:  {len(y_test):,} samples ({(y_test != 0).sum():,} signals)")

    # Train model
    model = train_model(
        X_train, y_train, sw_train, X_test, y_test, sw_test, logger,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree
    )

    # Evaluate model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test, logger)

    # Generate proof visualizations
    generate_proof_visualizations(model, metrics, feature_cols, proof_folder, logger)

    # Save model
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    logger.info(f"Saving model to {args.output}...")
    model.booster_.save_model(str(output_path))

    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Model saved ({file_size_mb:.2f} MB)")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Input windows:     {len(data):,}")
    logger.info(f"Features:          {len(feature_cols)}")
    logger.info(f"Test Signal R²:    {metrics['test_signal_r2']:.4f}")
    logger.info(f"Test Direction:    {metrics['test_dir_acc']:.2f}%")
    logger.info(f"Model saved:       {args.output}")
    logger.info(f"Proof folder:      proof/{args.issue}/")

    logger.info("")
    if metrics['test_signal_r2'] >= 0.70:
        logger.info(f"✅ SUCCESS! Signal R² ≥ 70% threshold")
    else:
        logger.warning(f"⚠️  Signal R² below 70% target (got {metrics['test_signal_r2']:.2f})")

    logger.info("")
    logger.info("Next step: Validate model with backtest (if available)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
