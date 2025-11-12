#!/usr/bin/env python3
"""
Step 4: Train Model with Sample Weighting (V3 Approach)

This script trains a LightGBM model using the V3 sample weighting approach:
- Trains on ALL candles (both signals and zeros)
- Weights signals 5x more than zeros for balanced importance
- Result: 74% R² on signals, 5% signal rate at 4σ threshold

Usage:
    python scripts/04_train_model.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sneaker.logging import setup_logger


# Define the 83 Enhanced V3 features
ENHANCED_V3_FEATURES = [
    # Original 20
    'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel',
    'bb_position', 'bb_position_vel',
    'macd_hist', 'macd_hist_vel',
    'stoch', 'stoch_vel',
    'di_diff', 'di_diff_vel',
    'adx',
    'adr', 'adr_up_bars', 'adr_down_bars',
    'is_up_bar',
    'vol_ratio', 'vol_ratio_vel',
    'vwap_20',
    # Batch 1: 24 momentum features
    'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20',
    'price_accel_5', 'price_accel_10',
    'rsi_accel', 'rsi_7_accel', 'bb_position_accel',
    'macd_hist_accel', 'stoch_accel', 'di_diff_accel',
    'vol_regime_vel', 'vol_ratio_accel', 'atr_14', 'atr_vel',
    'rsi_2x', 'bb_pos_2x', 'macd_hist_2x', 'price_change_2x',
    'is_up_streak', 'dist_from_high_20', 'dist_from_low_20', 'dist_from_vwap',
    # Batch 2: 35 advanced features
    'rsi_4x', 'bb_pos_4x', 'macd_hist_4x', 'price_change_4x', 'vol_ratio_4x',
    'rsi_bb_interaction', 'macd_vol_interaction', 'rsi_stoch_interaction',
    'bb_vol_interaction', 'adx_di_interaction', 'price_rsi_momentum_align',
    'vol_percentile', 'vol_regime_low', 'vol_regime_med', 'vol_regime_high', 'vol_zscore',
    'is_new_high_20', 'is_new_low_20', 'price_range_position',
    'consecutive_higher_highs', 'consecutive_lower_lows', 'vwap_distance_pct',
    'price_20_high',
    'price_rsi_divergence', 'price_macd_divergence', 'price_stoch_divergence',
    'rsi_divergence_strength', 'macd_divergence_strength',
    'vol_momentum_5', 'is_high_volume', 'price_20_low',
    'adx_vel', 'adx_accel', 'is_strong_trend', 'is_weak_trend',
    # Batch 3: 4 statistical features
    'hurst_exponent',
    'permutation_entropy',
    'cusum_signal',
    'squeeze_duration'
]


def load_data(logger):
    """Load the Enhanced V3 dataset."""
    logger.info("="*80)
    logger.info("V3 Model Training - Sample Weighting Approach")
    logger.info("="*80)

    logger.info("\nLoading Enhanced V3 dataset...")
    with open('data/enhanced_v3_dataset.json', 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    logger.info(f"  Loaded {len(df):,} candles from {df['pair'].nunique()} pairs")

    return df


def prepare_features(df, logger):
    """Prepare feature matrix and target."""
    logger.info("\nPreparing features...")

    # Validate features exist
    missing = [f for f in ENHANCED_V3_FEATURES if f not in df.columns]
    if missing:
        logger.error(f"  Missing features: {missing}")
        raise ValueError(f"Missing {len(missing)} features")

    logger.info(f"  Using {len(ENHANCED_V3_FEATURES)} Enhanced V3 features")

    # Count signals vs zeros
    zero_count = (df['ghost_tp_volnorm'] == 0).sum()
    signal_count = (df['ghost_tp_volnorm'] != 0).sum()

    logger.info(f"\n  Dataset composition:")
    logger.info(f"    Total candles:  {len(df):,}")
    logger.info(f"    Ghost signals:  {signal_count:,} ({signal_count/len(df)*100:.1f}%)")
    logger.info(f"    Normal candles: {zero_count:,} ({zero_count/len(df)*100:.1f}%)")

    X = df[ENHANCED_V3_FEATURES].values
    y = df['ghost_tp_volnorm'].values

    return X, y


def compute_sample_weights(y, logger, signal_weight=5.0):
    """
    Compute sample weights for balanced training.

    V3 KEY INNOVATION: Weight signals more heavily than zeros.

    Args:
        y: Target values
        signal_weight: Weight multiplier for non-zero targets (default: 5.0)

    Returns:
        sample_weights: Array of weights (signals=5.0, zeros=1.0)
    """
    logger.info(f"\nComputing sample weights (signal_weight={signal_weight}):")

    sample_weights = np.ones(len(y))
    signal_mask = y != 0

    sample_weights[signal_mask] = signal_weight
    sample_weights[~signal_mask] = 1.0

    signal_count = signal_mask.sum()
    zero_count = (~signal_mask).sum()

    # Effective weight contribution
    signal_effective = signal_count * signal_weight
    zero_effective = zero_count * 1.0
    total_effective = signal_effective + zero_effective

    logger.info(f"  Signals: {signal_count:,} × {signal_weight} = {signal_effective:,.0f} ({signal_effective/total_effective*100:.1f}%)")
    logger.info(f"  Zeros:   {zero_count:,} × 1.0 = {zero_effective:,.0f} ({zero_effective/total_effective*100:.1f}%)")
    logger.info(f"  ✓ Balanced: Signals have {signal_effective/total_effective*100:.0f}% influence")

    return sample_weights


def train_model(X_train, y_train, X_test, y_test, sw_train, sw_test, logger):
    """Train LightGBM with sample weighting."""
    logger.info("\nTraining LightGBM with sample weighting...")

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
        sample_weight=sw_train,  # V3 KEY: Sample weighting
        eval_set=[(X_test, y_test)],
        eval_sample_weight=[sw_test],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    logger.info(f"  ✓ Training complete (iterations: {model.best_iteration_})")

    return model


def evaluate_model(model, X_test, y_test, logger):
    """Evaluate model performance."""
    logger.info("\nEvaluating model...")

    y_pred = model.predict(X_test)

    # Overall metrics
    r2_overall = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Signal-only metrics
    signal_mask = y_test != 0
    if signal_mask.sum() > 0:
        r2_signals = r2_score(y_test[signal_mask], y_pred[signal_mask])
        rmse_signals = np.sqrt(mean_squared_error(y_test[signal_mask], y_pred[signal_mask]))
        direction_acc = (((y_test[signal_mask] > 0) & (y_pred[signal_mask] > 0)) |
                        ((y_test[signal_mask] < 0) & (y_pred[signal_mask] < 0))).mean()
    else:
        r2_signals = None
        rmse_signals = None
        direction_acc = None

    # Zero-only metrics
    zero_mask = ~signal_mask
    if zero_mask.sum() > 0:
        zero_mae = np.abs(y_pred[zero_mask] - y_test[zero_mask]).mean()
        within_1sigma = (np.abs(y_pred[zero_mask]) <= 1.0).mean() * 100
    else:
        zero_mae = None
        within_1sigma = None

    # Prediction distribution
    near_zero = (np.abs(y_pred) <= 1.0).mean() * 100
    strong = (np.abs(y_pred) > 5.0).mean() * 100

    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*80)
    logger.info(f"\nOverall:")
    logger.info(f"  R²:              {r2_overall:.4f} ({r2_overall*100:.2f}%)")
    logger.info(f"  RMSE:            {rmse:.4f}σ")
    logger.info(f"  MAE:             {mae:.4f}σ")

    if r2_signals is not None:
        logger.info(f"\nSignals (what matters):")
        logger.info(f"  R²:              {r2_signals:.4f} ({r2_signals*100:.2f}%)")
        logger.info(f"  RMSE:            {rmse_signals:.4f}σ")
        logger.info(f"  Direction Acc:   {direction_acc:.4f} ({direction_acc*100:.2f}%)")

    if zero_mae is not None:
        logger.info(f"\nZeros:")
        logger.info(f"  MAE:             {zero_mae:.4f}σ")
        logger.info(f"  Within ±1σ:      {within_1sigma:.1f}%")

    logger.info(f"\nPrediction Distribution:")
    logger.info(f"  Near zero (≤1σ): {near_zero:.1f}%")
    logger.info(f"  Strong (>5σ):    {strong:.1f}%")
    logger.info("="*80)

    return {
        'r2_overall': r2_overall,
        'r2_signals': r2_signals,
        'direction_acc': direction_acc,
        'zero_mae': zero_mae,
        'near_zero_pct': near_zero,
        'strong_pct': strong
    }


def save_model(model, logger):
    """Save trained model."""
    logger.info("\nSaving model...")

    os.makedirs('models', exist_ok=True)
    model_path = 'models/production.txt'
    model.booster_.save_model(model_path)

    logger.info(f"  ✓ Saved to: {model_path}")


def main():
    """Main training pipeline."""
    logger = setup_logger('train_model')

    try:
        # 1. Load data
        df = load_data(logger)

        # 2. Prepare features
        X, y = prepare_features(df, logger)

        # 3. Compute sample weights (V3 KEY)
        sample_weights = compute_sample_weights(y, logger, signal_weight=5.0)

        # 4. Train/test split
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weights,
            test_size=0.1,
            random_state=42,
            shuffle=True
        )

        logger.info(f"\nTrain/Test Split:")
        logger.info(f"  Train: {len(X_train):,} samples (90%)")
        logger.info(f"  Test:  {len(X_test):,} samples (10%)")

        # 5. Train model
        model = train_model(X_train, y_train, X_test, y_test, sw_train, sw_test, logger)

        # 6. Evaluate
        metrics = evaluate_model(model, X_test, y_test, logger)

        # 7. Save
        save_model(model, logger)

        # 8. Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Model: models/production.txt")
        if metrics['r2_signals']:
            logger.info(f"Signal R²: {metrics['r2_signals']:.4f} ({metrics['r2_signals']*100:.2f}%)")
        if metrics['direction_acc']:
            logger.info(f"Direction Accuracy: {metrics['direction_acc']:.4f} ({metrics['direction_acc']*100:.2f}%)")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
