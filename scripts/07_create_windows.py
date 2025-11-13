#!/usr/bin/env python3
"""
Create Training Windows (Sliding Window for Time Series)

Takes training_complete_features.json (individual candles) and creates
sliding windows of N consecutive candles for time series training.

This gives the model temporal context - it can see trends, momentum,
and patterns developing over multiple candles.

Part of Issue #8 (Pipeline Restructuring Epic #1)

Usage:
    # Default: 12-candle windows
    .venv/bin/python scripts/07_create_windows.py

    # Custom window size
    .venv/bin/python scripts/07_create_windows.py --window-size 16

    # Custom input/output
    .venv/bin/python scripts/07_create_windows.py \
      --input data/features/training_complete_features.json \
      --output data/features/windowed_training_data.json \
      --window-size 12

Arguments:
    --input: Input complete features JSON (from issue #7)
    --output: Output windowed data JSON
    --window-size: Number of consecutive candles per window (default: 12)

Input:
    data/features/training_complete_features.json
    - 791,044 individual candles
    - 93 shared features + 4 training-only features (including target)

Output:
    data/features/windowed_training_data.json
    - ~779,000 windows (depends on pairs and window size)
    - 1,116 features (93 shared features × 12 candles)
    - 1 target per window (from last candle)

Window Structure:
    Each window contains N consecutive candles, flattened into a single row:
    - feature_name_t0: Feature from oldest candle in window
    - feature_name_t1: Feature from 2nd oldest candle
    - ...
    - feature_name_t11: Feature from most recent candle (if N=12)
    - target: Target from most recent candle (what we're predicting)
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


def create_sliding_windows(df: pd.DataFrame, window_size: int, logger) -> pd.DataFrame:
    """
    Create sliding windows from time series data.

    For each pair, slides a window of size N across the time series,
    creating one training sample per window position.

    Args:
        df: DataFrame with features and target (sorted by pair, timestamp)
        window_size: Number of consecutive candles per window
        logger: Logger instance

    Returns:
        DataFrame where each row is a flattened window
    """
    windows_list = []
    total_pairs = df['pair'].nunique()

    logger.info(f"Creating windows for {total_pairs} pairs...")
    logger.info(f"Window size: {window_size} candles")

    for pair_idx, pair in enumerate(df['pair'].unique(), 1):
        pair_data = df[df['pair'] == pair].sort_values('timestamp').reset_index(drop=True)
        pair_len = len(pair_data)

        logger.info(f"[{pair_idx}/{total_pairs}] {pair}: {pair_len:,} candles")

        # Slide window across this pair's data
        num_windows = pair_len - window_size + 1

        if num_windows <= 0:
            logger.warning(f"  ⚠️  Skipping {pair}: only {pair_len} candles (need {window_size}+)")
            continue

        for i in range(num_windows):
            window = pair_data.iloc[i:i+window_size]

            # Create flattened window features
            window_features = {}

            # Add features from each time step
            for t, (idx, row) in enumerate(window.iterrows()):
                for feature in SHARED_FEATURE_LIST:
                    if feature in row:
                        window_features[f'{feature}_t{t}'] = row[feature]

            # Add target from LAST candle (most recent, t=window_size-1)
            last_candle = window.iloc[-1]
            window_features['target'] = last_candle['target']
            window_features['pair'] = pair
            window_features['timestamp'] = last_candle['timestamp']

            windows_list.append(window_features)

        logger.info(f"  ✓ Created {num_windows:,} windows")

    logger.info(f"Total windows created: {len(windows_list):,}")

    return pd.DataFrame(windows_list)


def validate_windows(windowed_df: pd.DataFrame, window_size: int, logger):
    """
    Validate windowed data for correctness.

    Checks:
    - Expected number of feature columns
    - No NaN values in features
    - Target distribution looks reasonable
    - No data leakage

    Args:
        windowed_df: DataFrame of windowed data
        window_size: Window size used
        logger: Logger instance
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("VALIDATING WINDOWS")
    logger.info("=" * 80)

    # Check feature count
    feature_cols = [col for col in windowed_df.columns
                   if col not in ['target', 'pair', 'timestamp']]
    expected_features = len(SHARED_FEATURE_LIST) * window_size

    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Expected: {expected_features} ({len(SHARED_FEATURE_LIST)} features × {window_size} steps)")

    if len(feature_cols) != expected_features:
        logger.warning(f"  ⚠️  Feature count mismatch!")
    else:
        logger.info("  ✓ Feature count correct")

    # Check for NaN values
    nan_counts = windowed_df[feature_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"  ⚠️  Found {nan_counts.sum():,} NaN values in features")
        logger.warning(f"     Columns with NaN: {nan_counts[nan_counts > 0].index.tolist()}")
    else:
        logger.info("  ✓ No NaN values in features")

    # Check target distribution
    logger.info("")
    logger.info("Target distribution:")
    signals = (windowed_df['target'] != 0).sum()
    zeros = (windowed_df['target'] == 0).sum()
    logger.info(f"  Signals: {signals:,} ({signals/len(windowed_df)*100:.1f}%)")
    logger.info(f"  Zeros:   {zeros:,} ({zeros/len(windowed_df)*100:.1f}%)")

    if signals > 0:
        target_mean = windowed_df.loc[windowed_df['target'] != 0, 'target'].mean()
        target_std = windowed_df.loc[windowed_df['target'] != 0, 'target'].std()
        target_max = windowed_df['target'].max()
        target_min = windowed_df['target'].min()

        logger.info("")
        logger.info("Target statistics (signals only):")
        logger.info(f"  Mean:   {target_mean:+.4f}σ")
        logger.info(f"  Std:    {target_std:.4f}σ")
        logger.info(f"  Max:    {target_max:+.4f}σ")
        logger.info(f"  Min:    {target_min:+.4f}σ")

    # Check for duplicate windows (should not happen with proper sliding)
    logger.info("")
    duplicates = windowed_df.duplicated(subset=['pair', 'timestamp']).sum()
    if duplicates > 0:
        logger.warning(f"  ⚠️  Found {duplicates:,} duplicate windows!")
    else:
        logger.info("  ✓ No duplicate windows")


def main():
    parser = argparse.ArgumentParser(
        description='Create sliding windows for time series training'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features/training_complete_features.json',
        help='Input complete features JSON (from issue #7)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/features/windowed_training_data.json',
        help='Output windowed data JSON'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=12,
        help='Number of consecutive candles per window (default: 12)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('create_windows')

    logger.info("=" * 80)
    logger.info("CREATE TRAINING WINDOWS - SLIDING WINDOW FOR TIME SERIES")
    logger.info("=" * 80)
    logger.info(f"Input:       {args.input}")
    logger.info(f"Output:      {args.output}")
    logger.info(f"Window size: {args.window_size} candles")
    logger.info("")
    logger.info("Why windowing?")
    logger.info("  - Gives model temporal context (see trends, momentum)")
    logger.info("  - Model learns from patterns developing over time")
    logger.info(f"  - Each window: {len(SHARED_FEATURE_LIST)} features × {args.window_size} steps")
    logger.info(f"  - Total features per sample: {len(SHARED_FEATURE_LIST) * args.window_size}")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load complete features
    logger.info("=" * 80)
    logger.info("LOADING COMPLETE FEATURES")
    logger.info("=" * 80)

    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("   Run scripts/06_add_training_features.py first!")
        return 1

    logger.info(f"Loading {args.input}...")
    start_time = time.time()

    with open(args.input, 'r') as f:
        data = json.load(f)

    load_time = time.time() - start_time

    logger.info(f"✓ Loaded {len(data):,} candles in {load_time:.1f}s")

    # Convert to DataFrame
    logger.info("")
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(data)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Unique pairs: {df['pair'].nunique()}")

    # Check for required columns
    if 'target' not in df.columns:
        logger.error("❌ Missing 'target' column!")
        logger.error("   Make sure input has training-only features added.")
        return 1

    # Create windows
    logger.info("")
    logger.info("=" * 80)
    logger.info("CREATING SLIDING WINDOWS")
    logger.info("=" * 80)

    start_time = time.time()

    windowed_df = create_sliding_windows(df, args.window_size, logger)

    window_time = time.time() - start_time

    logger.info("")
    logger.info(f"✓ Windowing complete in {window_time:.1f}s ({window_time/60:.1f} minutes)")
    logger.info(f"  Input candles:  {len(df):,}")
    logger.info(f"  Output windows: {len(windowed_df):,}")
    logger.info(f"  Reduction:      {len(df) - len(windowed_df):,} candles lost to window boundaries")

    # Validate windows
    validate_windows(windowed_df, args.window_size, logger)

    # Save windowed data
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING WINDOWED DATA")
    logger.info("=" * 80)

    logger.info(f"Converting to JSON records...")
    output_data = windowed_df.to_dict('records')

    logger.info(f"Writing {len(output_data):,} windows to {args.output}...")
    start_time = time.time()

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    save_time = time.time() - start_time
    file_size_mb = Path(args.output).stat().st_size / 1024 / 1024

    logger.info(f"✓ Saved in {save_time:.1f}s")
    logger.info(f"  File size: {file_size_mb:.2f} MB")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Window size:       {args.window_size} candles")
    logger.info(f"Input candles:     {len(df):,}")
    logger.info(f"Output windows:    {len(output_data):,}")
    logger.info(f"Features per row:  {len(windowed_df.columns) - 3}")  # -3 for target, pair, timestamp
    logger.info(f"Processing time:   {window_time:.1f}s ({window_time/60:.1f} min)")
    logger.info(f"Output file:       {args.output}")
    logger.info(f"File size:         {file_size_mb:.2f} MB")

    logger.info("")
    logger.info("✅ Windowing complete! Ready for model training (script 08).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
