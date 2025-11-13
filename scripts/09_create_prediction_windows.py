#!/usr/bin/env python3
"""
Create Prediction Windows (Sliding Window for Time Series Prediction)

Takes prediction_complete_features.json (individual candles) and creates
sliding windows of N consecutive candles for generating predictions.

This gives the model temporal context - it can see trends, momentum,
and patterns developing over multiple candles.

CRITICAL: Per-Window Normalization
- 15 features with absolute scale issues (prices, ATR, MACD) are normalized
  relative to t0 (first candle in window)
- Example: BTC at $100k and altcoin at $0.50 both normalize to t0=1.0
- Makes features comparable across different price ranges
- Other 78 features (RSI, BB, ratios, %) are already scale-independent

IMPORTANT: This uses EXACT same normalization as training (script 07)
- Same 15 NORMALIZE_FEATURES
- Same t0 normalization logic
- Same 1e-10 threshold
- Same feature iteration order

Part of Issue #18 (Prediction Pipeline)

Usage:
    # Default: 12-candle windows with normalization
    .venv/bin/python scripts/09_create_prediction_windows.py

    # Custom window size (must match training!)
    .venv/bin/python scripts/09_create_prediction_windows.py --window-size 12

    # Custom input/output
    .venv/bin/python scripts/09_create_prediction_windows.py \
      --input data/features/prediction_complete_features.json \
      --output data/features/windowed_prediction_data.json \
      --window-size 12

Arguments:
    --input: Input complete features JSON (from issue #5)
    --output: Output windowed data JSON
    --window-size: Number of consecutive candles per window (default: 12, MUST match training!)

Input:
    data/features/prediction_complete_features.json
    - Recent candles (~2,000-5,000 per pair)
    - 93 shared features (NO training-only features)

Output:
    data/features/windowed_prediction_data.json
    - Windows from recent data
    - 1,116 features (93 shared features × 12 candles)
    - NO target column (prediction data doesn't have targets)
    - Metadata: pair, timestamp (from last candle in window)

Window Structure:
    Each window contains N consecutive candles, flattened into a single row:
    - feature_name_t0: Feature from oldest candle in window (normalized if needed)
    - feature_name_t1: Feature from 2nd oldest candle (normalized if needed)
    - ...
    - feature_name_t11: Feature from most recent candle (if N=12)
    - pair: Trading pair
    - timestamp: Timestamp from most recent candle
    - NO target (prediction data doesn't have targets)

Normalized Features (15 total - EXACT SAME AS TRAINING):
    - Macro close prices (4): macro_GOLD_close, macro_BNB_close, etc.
    - Macro velocities (4): macro_GOLD_vel, macro_BNB_vel, etc.
    - VWAP (1): vwap_20
    - ATR (2): atr_14, atr_vel
    - MACD histogram (4): macd_hist, macd_hist_vel, macd_hist_2x, macd_hist_4x
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
    Create sliding windows from time series data with per-window normalization.

    For each pair, slides a window of size N across the time series,
    creating one prediction sample per window position.

    CRITICAL: Normalizes features with absolute scale (prices, ATR, MACD)
    relative to t0 to make them comparable across different price ranges.

    This uses EXACT SAME normalization logic as training (script 07).

    Example: BTC at $100k and altcoin at $0.50 both normalize to t0=1.0,
    making relative changes comparable.

    Args:
        df: DataFrame with features (NO target column for prediction)
        window_size: Number of consecutive candles per window
        logger: Logger instance

    Returns:
        DataFrame where each row is a flattened window with normalized features
    """
    # Features requiring per-window normalization (absolute scale issues)
    # CRITICAL: This list MUST match training exactly!
    NORMALIZE_FEATURES = [
        # Macro close prices (4) - different assets have vastly different prices
        'macro_GOLD_close', 'macro_BNB_close',
        'macro_BTC_PREMIUM_close', 'macro_ETH_PREMIUM_close',

        # Macro velocities (4) - absolute $ changes vary by asset
        'macro_GOLD_vel', 'macro_BNB_vel',
        'macro_BTC_PREMIUM_vel', 'macro_ETH_PREMIUM_vel',

        # VWAP (1) - pair-specific absolute price
        'vwap_20',

        # ATR (2) - BTC ATR ~$3000, altcoin ATR ~$0.05
        'atr_14', 'atr_vel',

        # MACD histogram (4) - scale varies by price
        'macd_hist', 'macd_hist_vel',
        'macd_hist_2x', 'macd_hist_4x'
    ]

    windows_list = []
    total_pairs = df['pair'].nunique()

    logger.info(f"Creating windows for {total_pairs} pairs...")
    logger.info(f"Window size: {window_size} candles")
    logger.info(f"Normalizing {len(NORMALIZE_FEATURES)} features per window (relative to t0)")

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

            # Get t0 (first candle) values for normalization
            t0_row = window.iloc[0]
            t0_values = {feat: t0_row[feat] for feat in NORMALIZE_FEATURES if feat in t0_row}

            # Add features from each time step
            for t, (idx, row) in enumerate(window.iterrows()):
                for feature in SHARED_FEATURE_LIST:
                    if feature not in row:
                        continue

                    # Check if this feature needs normalization
                    if feature in NORMALIZE_FEATURES:
                        # Normalize relative to t0 value
                        t0_value = t0_values.get(feature, 0)
                        if abs(t0_value) > 1e-10:  # Avoid division by zero
                            normalized_value = row[feature] / t0_value
                        else:
                            # If t0 is zero, use 1.0 (no change from baseline)
                            # This handles edge cases like zero MACD or zero velocity
                            normalized_value = 1.0 if abs(row[feature]) < 1e-10 else 0.0

                        window_features[f'{feature}_t{t}'] = normalized_value
                    else:
                        # Use raw value (already normalized or scale-independent)
                        window_features[f'{feature}_t{t}'] = row[feature]

            # Add metadata from LAST candle (most recent, t=window_size-1)
            # NOTE: NO target column for prediction data!
            last_candle = window.iloc[-1]
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
    - No data leakage
    - Proper metadata

    NOTE: Does NOT check target distribution (prediction data has no targets)

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
                   if col not in ['pair', 'timestamp']]
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

    # Check metadata
    logger.info("")
    logger.info("Metadata check:")
    if 'pair' in windowed_df.columns:
        logger.info(f"  ✓ Has 'pair' column ({windowed_df['pair'].nunique()} unique pairs)")
    else:
        logger.warning("  ⚠️  Missing 'pair' column!")

    if 'timestamp' in windowed_df.columns:
        logger.info(f"  ✓ Has 'timestamp' column")
    else:
        logger.warning("  ⚠️  Missing 'timestamp' column!")

    # Check for NO target column (correct for prediction)
    if 'target' in windowed_df.columns:
        logger.warning("  ⚠️  Found 'target' column (should not exist in prediction data)!")
    else:
        logger.info("  ✓ No 'target' column (correct for prediction data)")

    # Check for duplicate windows (should not happen with proper sliding)
    logger.info("")
    duplicates = windowed_df.duplicated(subset=['pair', 'timestamp']).sum()
    if duplicates > 0:
        logger.warning(f"  ⚠️  Found {duplicates:,} duplicate windows!")
    else:
        logger.info("  ✓ No duplicate windows")


def main():
    parser = argparse.ArgumentParser(
        description='Create sliding windows for time series prediction'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features/prediction_complete_features.json',
        help='Input complete features JSON (from issue #5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/features/windowed_prediction_data.json',
        help='Output windowed data JSON'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=12,
        help='Number of consecutive candles per window (default: 12, MUST match training!)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('create_prediction_windows')

    logger.info("=" * 80)
    logger.info("CREATE PREDICTION WINDOWS - SLIDING WINDOW FOR TIME SERIES")
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
    logger.info("CRITICAL: Uses EXACT same normalization as training!")
    logger.info("  - Same 15 NORMALIZE_FEATURES list")
    logger.info("  - Same t0 normalization logic")
    logger.info("  - Same feature iteration order (SHARED_FEATURE_LIST)")
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
        logger.error("   Run scripts/05_add_shared_features.py --mode prediction first!")
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

    # Check that target column does NOT exist (correct for prediction)
    if 'target' in df.columns:
        logger.warning("⚠️  Found 'target' column in prediction data!")
        logger.warning("   This should not exist for prediction data.")
        logger.warning("   Proceeding anyway, but will not include target in windows.")

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
    logger.info(f"Features per row:  {len(windowed_df.columns) - 2}")  # -2 for pair, timestamp (no target)
    logger.info(f"Processing time:   {window_time:.1f}s ({window_time/60:.1f} min)")
    logger.info(f"Output file:       {args.output}")
    logger.info(f"File size:         {file_size_mb:.2f} MB")

    logger.info("")
    logger.info("✅ Windowing complete! Ready for prediction generation (script 10).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
