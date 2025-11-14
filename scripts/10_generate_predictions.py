#!/usr/bin/env python3
"""
Generate Predictions on Windowed Data

Loads windowed prediction data and generates predictions using the trained model.

Part of Issue #18 (Prediction Pipeline)

Usage:
    # Generate predictions for all pairs
    .venv/bin/python scripts/10_generate_predictions.py

    # Generate predictions for specific pair
    .venv/bin/python scripts/10_generate_predictions.py --pair LINKUSDT

    # Custom threshold
    .venv/bin/python scripts/10_generate_predictions.py --threshold 5.0

    # Custom paths
    .venv/bin/python scripts/10_generate_predictions.py \
      --input data/features/windowed_prediction_data.json \
      --model models/issue-1/model.txt \
      --output data/predictions/latest.json

Arguments:
    --input: Input windowed data JSON (from script 09)
    --model: Trained model path
    --output: Output predictions JSON
    --pair: Filter predictions for specific pair (optional)
    --threshold: Signal threshold in σ (default: 4.0)

Input:
    data/features/windowed_prediction_data.json
    - Windows from recent data
    - 1,068 features (89 shared (Issue #23) × 12 candles)
    - NO target column

Output:
    data/predictions/latest.json
    - Predictions for each window
    - Includes pair, timestamp, prediction, signal classification
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


def extract_windowed_features(df: pd.DataFrame, window_size: int, logger) -> tuple:
    """
    Extract windowed feature columns from DataFrame.

    Features are named like: feature_name_t0, feature_name_t1, ..., feature_name_t11

    CRITICAL: This MUST use EXACT same feature order as training!

    Args:
        df: DataFrame with windowed features (NO target for prediction)
        window_size: Number of time steps per window
        logger: Logger instance

    Returns:
        (feature_columns, X) tuple
    """
    # Build list of expected windowed feature columns
    # CRITICAL: Must iterate in EXACT same order as training!
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

    # Extract feature matrix (NO target for prediction)
    X = df[feature_cols].values

    return feature_cols, X


def classify_signals(predictions: np.ndarray, threshold: float) -> list:
    """
    Classify predictions into BUY/SELL/HOLD signals.

    Args:
        predictions: Array of predictions in σ units
        threshold: Threshold for strong signals

    Returns:
        List of signal classifications
    """
    signals = []
    for pred in predictions:
        if pred > threshold:
            signals.append('BUY')
        elif pred < -threshold:
            signals.append('SELL')
        else:
            signals.append('HOLD')
    return signals


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions on windowed data'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features/windowed_prediction_data.json',
        help='Input windowed data JSON (from script 09)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/issue-1/model.txt',
        help='Trained model path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/predictions/latest.json',
        help='Output predictions JSON'
    )
    parser.add_argument(
        '--pair',
        type=str,
        help='Filter by specific pair (optional)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=4.0,
        help='Signal threshold in σ (default: 4.0)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=12,
        help='Window size (must match training, default: 12)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('generate_predictions')

    logger.info("=" * 80)
    logger.info("GENERATE PREDICTIONS ON WINDOWED DATA")
    logger.info("=" * 80)
    logger.info(f"Input:       {args.input}")
    logger.info(f"Model:       {args.model}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Window size: {args.window_size} candles")
    logger.info(f"Threshold:   ±{args.threshold}σ")
    if args.pair:
        logger.info(f"Filter pair: {args.pair}")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load windowed data
    logger.info("=" * 80)
    logger.info("LOADING WINDOWED DATA")
    logger.info("=" * 80)

    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("   Run scripts/09_create_prediction_windows.py first!")
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
    logger.info(f"  Unique pairs: {df['pair'].nunique()}")

    # Filter by pair if specified
    if args.pair:
        logger.info("")
        logger.info(f"Filtering for pair: {args.pair}")
        df = df[df['pair'] == args.pair]
        logger.info(f"  Filtered to {len(df):,} windows")

        if len(df) == 0:
            logger.error(f"❌ No data found for pair: {args.pair}")
            return 1

    # Extract features
    logger.info("")
    logger.info("=" * 80)
    logger.info("EXTRACTING FEATURES")
    logger.info("=" * 80)

    feature_cols, X = extract_windowed_features(df, args.window_size, logger)

    logger.info(f"✓ Feature matrix shape: {X.shape}")
    logger.info(f"  Windows: {X.shape[0]:,}")
    logger.info(f"  Features: {X.shape[1]:,}")

    # Check for NaN or inf values
    if np.isnan(X).any():
        logger.warning(f"  ⚠️  Found {np.isnan(X).sum():,} NaN values in feature matrix")
    if np.isinf(X).any():
        logger.warning(f"  ⚠️  Found {np.isinf(X).sum():,} inf values in feature matrix")

    # Load model
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)

    if not Path(args.model).exists():
        logger.error(f"❌ Model file not found: {args.model}")
        logger.error("   Run scripts/08_train_model.py first!")
        return 1

    logger.info(f"Loading model from {args.model}...")
    start_time = time.time()

    model = lgb.Booster(model_file=args.model)

    load_time = time.time() - start_time

    logger.info(f"✓ Model loaded in {load_time:.3f}s")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Num features: {model.num_feature()}")

    # Verify feature count matches
    if model.num_feature() != X.shape[1]:
        logger.error(f"❌ Feature count mismatch!")
        logger.error(f"   Model expects: {model.num_feature()} features")
        logger.error(f"   Data has: {X.shape[1]} features")
        return 1

    # Generate predictions
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING PREDICTIONS")
    logger.info("=" * 80)

    logger.info(f"Predicting {X.shape[0]:,} windows...")
    start_time = time.time()

    predictions = model.predict(X)

    pred_time = time.time() - start_time

    logger.info(f"✓ Predictions generated in {pred_time:.3f}s")
    logger.info(f"  Speed: {X.shape[0] / pred_time:.0f} windows/sec")

    # Classify signals
    logger.info("")
    logger.info("Classifying signals...")
    signals = classify_signals(predictions, args.threshold)

    # Combine with metadata
    results = pd.DataFrame({
        'pair': df['pair'].values,
        'timestamp': df['timestamp'].values,
        'prediction': predictions,
        'signal': signals
    })

    # Summary statistics
    logger.info("")
    logger.info("=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)

    logger.info(f"Total predictions: {len(results):,}")
    logger.info(f"Unique pairs: {results['pair'].nunique()}")
    logger.info("")

    logger.info("Prediction range:")
    logger.info(f"  Min:  {predictions.min():+.2f}σ")
    logger.info(f"  Max:  {predictions.max():+.2f}σ")
    logger.info(f"  Mean: {predictions.mean():+.2f}σ")
    logger.info(f"  Std:  {predictions.std():.2f}σ")
    logger.info("")

    buy_count = (results['signal'] == 'BUY').sum()
    sell_count = (results['signal'] == 'SELL').sum()
    hold_count = (results['signal'] == 'HOLD').sum()

    logger.info(f"Signal classification (threshold: ±{args.threshold}σ):")
    logger.info(f"  BUY:  {buy_count:,} ({buy_count/len(results)*100:.1f}%)")
    logger.info(f"  SELL: {sell_count:,} ({sell_count/len(results)*100:.1f}%)")
    logger.info(f"  HOLD: {hold_count:,} ({hold_count/len(results)*100:.1f}%)")

    # Show strong signals
    strong_signals = results[results['signal'] != 'HOLD']
    if len(strong_signals) > 0:
        logger.info("")
        logger.info(f"Strong signals (±{args.threshold}σ): {len(strong_signals):,}")
        logger.info("")
        logger.info("Recent strong signals:")
        for _, row in strong_signals.tail(10).iterrows():
            timestamp_str = pd.to_datetime(row['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M')
            logger.info(f"  {timestamp_str} | {row['pair']:10s} | {row['prediction']:+6.2f}σ | {row['signal']}")
    else:
        logger.info("")
        logger.info(f"No strong signals above threshold (±{args.threshold}σ)")

    # Show most recent prediction
    logger.info("")
    logger.info("=" * 80)
    logger.info("MOST RECENT PREDICTION")
    logger.info("=" * 80)

    latest = results.iloc[-1]
    timestamp_str = pd.to_datetime(latest['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M:%S')

    logger.info(f"Pair:       {latest['pair']}")
    logger.info(f"Timestamp:  {timestamp_str}")
    logger.info(f"Prediction: {latest['prediction']:+.2f}σ")
    logger.info(f"Signal:     {latest['signal']}")

    if latest['signal'] == 'BUY':
        logger.info("⬆️  STRONG BUY SIGNAL - Upward reversal expected")
    elif latest['signal'] == 'SELL':
        logger.info("⬇️  STRONG SELL SIGNAL - Downward reversal expected")
    else:
        logger.info("➡️  NO STRONG SIGNAL - Hold or wait")

    # Save results
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    logger.info(f"Converting to JSON records...")
    output_data = results.to_dict('records')

    logger.info(f"Writing {len(output_data):,} predictions to {args.output}...")
    start_time = time.time()

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    save_time = time.time() - start_time
    file_size_kb = Path(args.output).stat().st_size / 1024

    logger.info(f"✓ Saved in {save_time:.3f}s")
    logger.info(f"  File size: {file_size_kb:.2f} KB")

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total predictions:  {len(results):,}")
    logger.info(f"Strong signals:     {len(strong_signals):,} ({len(strong_signals)/len(results)*100:.1f}%)")
    logger.info(f"Output file:        {args.output}")
    logger.info("")
    logger.info("✅ Prediction generation complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
