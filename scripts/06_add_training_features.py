#!/usr/bin/env python3
"""
Add Training-Only Features Pipeline

Adds features that USE FUTURE DATA and cannot be used in live prediction:
- Target calculation (4H lookahead, σ normalized)
- Statistical features (Hurst, entropy, CUSUM)

⚠️  WARNING: TRAINING ONLY - USES FUTURE DATA!

Part of Issue #7 (sub-issue #1.6 of Pipeline Restructuring Epic #1)

Usage:
    # Default paths
    .venv/bin/python scripts/06_add_training_features.py

    # Custom paths
    .venv/bin/python scripts/06_add_training_features.py \
      --input data/features/training_shared_features.json \
      --output data/features/training_complete_features.json \
      --lookahead 4

Arguments:
    --input: Path to shared features JSON (from issue #6)
    --output: Output path for complete training features JSON
    --lookahead: Lookahead periods for target calculation (default: 4H)

Input:
    data/features/training_shared_features.json (from issue #6)
    - OHLCV data
    - 93 shared features
    - ~791,044 records

Output:
    data/features/training_complete_features.json
    - All input columns
    - 4 additional training-only features:
      * target (PRIMARY - future price change, σ normalized)
      * hurst_exponent
      * permutation_entropy
      * cusum_signal
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
from sneaker.features_training import add_all_training_features, TRAINING_ONLY_FEATURE_LIST
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Add training-only features (uses future data)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features/training_shared_features.json',
        help='Path to shared features JSON (from issue #6)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/features/training_complete_features.json',
        help='Output path for complete training features JSON'
    )
    parser.add_argument(
        '--lookahead',
        type=int,
        default=4,
        help='Lookahead periods for target calculation (default: 4H)'
    )
    parser.add_argument(
        '--min-flips',
        type=int,
        default=3,
        help='Minimum simultaneous indicator flips for ghost signal (default: 3)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('add_training_features')

    logger.info("=" * 80)
    logger.info("ADD TRAINING-ONLY FEATURES")
    logger.info("=" * 80)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Lookahead: {args.lookahead} hours")
    logger.info(f"Min flips: {args.min_flips} (for ghost signal detection)")
    logger.info("")
    logger.info("⚠️  WARNING: USES FUTURE DATA - TRAINING ONLY!")
    logger.info("")
    logger.info("Features to add:")
    logger.info(f"  1. target (PRIMARY - {args.lookahead}H lookahead, σ normalized)")
    logger.info(f"     - Ghost signal detection: {args.min_flips}+ indicators flip simultaneously")
    logger.info(f"     - Target = 0 for normal candles")
    logger.info(f"     - Target = reversal magnitude for ghost signals")
    logger.info("  2. hurst_exponent (trend persistence)")
    logger.info("  3. permutation_entropy (predictability)")
    logger.info("  4. cusum_signal (change detection)")
    logger.info("")

    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load shared features
    logger.info("=" * 80)
    logger.info("LOADING SHARED FEATURES")
    logger.info("=" * 80)

    if not Path(args.input).exists():
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error("   Run scripts/05_add_shared_features.py first!")
        return 1

    logger.info(f"Loading {args.input}...")
    start_time = time.time()

    with open(args.input, 'r') as f:
        data = json.load(f)

    load_time = time.time() - start_time

    logger.info(f"✓ Loaded {len(data):,} records in {load_time:.1f}s")

    # Convert to DataFrame
    logger.info("")
    logger.info("Converting to DataFrame...")
    df = pd.DataFrame(data)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {len(df.columns)}")

    # Analyze pairs
    if 'pair' in df.columns:
        unique_pairs = df['pair'].nunique()
        logger.info(f"  Unique pairs: {unique_pairs}")

    # Add training-only features
    logger.info("")
    logger.info("=" * 80)
    logger.info("ADDING TRAINING-ONLY FEATURES")
    logger.info("=" * 80)

    logger.info("Running feature engineering pipeline...")
    logger.info("This may take several minutes for large datasets...")
    start_time = time.time()

    try:
        result_df = add_all_training_features(
            df,
            lookahead_periods=args.lookahead,
            min_flips=args.min_flips
        )
        feature_time = time.time() - start_time

        logger.info(f"✓ Features added in {feature_time:.1f}s ({feature_time/60:.1f} minutes)")
        logger.info(f"  Output shape: {result_df.shape}")
        logger.info(f"  Total columns: {len(result_df.columns)}")

        # Verify features
        added_features = [col for col in result_df.columns if col in TRAINING_ONLY_FEATURE_LIST]
        missing_features = [col for col in TRAINING_ONLY_FEATURE_LIST if col not in result_df.columns]

        logger.info("")
        logger.info("Feature verification:")
        logger.info(f"  Expected: {len(TRAINING_ONLY_FEATURE_LIST)} features")
        logger.info(f"  Added:    {len(added_features)} features")

        if missing_features:
            logger.error(f"  ❌ Missing: {len(missing_features)} features")
            for feat in missing_features:
                logger.error(f"    - {feat}")
            return 1
        else:
            logger.info("  ✓ All expected features present")

        # Analyze target distribution
        logger.info("")
        logger.info("=" * 80)
        logger.info("TARGET DISTRIBUTION ANALYSIS")
        logger.info("=" * 80)

        target_values = result_df['target'].values
        signals = (target_values != 0).sum()
        zeros = (target_values == 0).sum()

        logger.info(f"Signals: {signals:,} ({signals/len(result_df)*100:.1f}%)")
        logger.info(f"Zeros:   {zeros:,} ({zeros/len(result_df)*100:.1f}%)")

        if signals > 0:
            target_mean = target_values[target_values != 0].mean()
            target_std = target_values[target_values != 0].std()
            target_max = target_values.max()
            target_min = target_values.min()

            logger.info("")
            logger.info("Target statistics (signals only):")
            logger.info(f"  Mean:   {target_mean:+.4f}σ")
            logger.info(f"  Std:    {target_std:.4f}σ")
            logger.info(f"  Max:    {target_max:+.4f}σ")
            logger.info(f"  Min:    {target_min:+.4f}σ")

            # Signal strength distribution
            strong_signals = (np.abs(target_values) > 4.0).sum()
            extreme_signals = (np.abs(target_values) > 6.0).sum()

            logger.info("")
            logger.info("Signal strength distribution:")
            logger.info(f"  Strong (>4σ):   {strong_signals:,} ({strong_signals/len(result_df)*100:.1f}%)")
            logger.info(f"  Extreme (>6σ):  {extreme_signals:,} ({extreme_signals/len(result_df)*100:.1f}%)")

        # Check for NaN values
        logger.info("")
        logger.info("NaN check:")
        nan_cols = result_df.columns[result_df.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"  ⚠️  Columns with NaN: {len(nan_cols)}")
            for col in nan_cols[:10]:
                nan_count = result_df[col].isna().sum()
                logger.warning(f"    {col}: {nan_count:,} NaN values")
            if len(nan_cols) > 10:
                logger.warning(f"    ... and {len(nan_cols) - 10} more")
        else:
            logger.info("  ✓ No NaN values found")

    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    logger.info(f"Converting to JSON records...")
    output_data = result_df.to_dict('records')

    logger.info(f"Writing {len(output_data):,} records to {args.output}...")
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
    logger.info(f"Input records:     {len(data):,}")
    logger.info(f"Output records:    {len(output_data):,}")
    logger.info(f"Features added:    {len(added_features)}/{len(TRAINING_ONLY_FEATURE_LIST)}")
    logger.info(f"Total columns:     {len(result_df.columns)}")
    logger.info(f"Processing time:   {feature_time:.1f}s ({feature_time/60:.1f} min)")
    logger.info(f"Output file:       {args.output}")
    logger.info(f"File size:         {file_size_mb:.2f} MB")

    if signals > 0:
        logger.info("")
        logger.info("Target Analysis:")
        logger.info(f"  Total signals:     {signals:,} ({signals/len(result_df)*100:.1f}%)")
        logger.info(f"  Strong (>4σ):      {strong_signals:,} ({strong_signals/len(result_df)*100:.1f}%)")
        logger.info(f"  Target range:      {target_min:+.2f}σ to {target_max:+.2f}σ")

    # Per-pair analysis
    if 'pair' in result_df.columns:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PER-PAIR ANALYSIS")
        logger.info("=" * 80)

        pair_counts = result_df['pair'].value_counts().sort_index()
        for pair in pair_counts.index[:10]:
            pair_mask = result_df['pair'] == pair
            pair_signals = (result_df.loc[pair_mask, 'target'] != 0).sum()
            logger.info(f"{pair:15s} {pair_counts[pair]:,} records  ({pair_signals:,} signals)")
        if len(pair_counts) > 10:
            logger.info(f"... and {len(pair_counts) - 10} more pairs")

    # Success
    logger.info("")
    if missing_features:
        logger.error(f"❌ Completed with {len(missing_features)} missing features")
        return 1
    else:
        logger.info("✅ All training features added successfully!")
        logger.info("")
        logger.info("Next step: Issue #8 - Train model using this complete dataset")
        return 0


if __name__ == '__main__':
    import numpy as np  # For target analysis
    sys.exit(main())
