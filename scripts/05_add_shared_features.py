#!/usr/bin/env python3
"""
Add Shared Features Pipeline (training & prediction)

Adds shared features (NO look-forward) for both training and prediction:
- 12 macro features (GOLD, BNB, BTC_PREMIUM, ETH_PREMIUM: close + vel + ROC)
- 20 core indicators
- 24 momentum features
- 32 advanced features (excluding training-only statistical)
- 4 trend strength features
- 1 statistical feature (squeeze_duration)

Total: 93 shared features

Part of Issue #6 (sub-issue #1.5 of Pipeline Restructuring Epic #1)

Usage:
    # Training data (default)
    .venv/bin/python scripts/05_add_shared_features.py --mode training

    # Prediction data
    .venv/bin/python scripts/05_add_shared_features.py --mode prediction

    # Custom paths
    .venv/bin/python scripts/05_add_shared_features.py \
      --binance data/raw/training/binance_20pairs_1H.json \
      --macro data/raw/training/macro_training_binance.json \
      --output data/features/training_shared_features.json

Arguments:
    --mode: Mode (training/prediction), sets default paths
    --binance: Path to Binance crypto data JSON
    --macro: Path to macro data JSON
    --output: Output path for features JSON

Output:
    - data/features/training_shared_features.json (training mode)
    - data/features/prediction_shared_features.json (prediction mode)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.features_shared import add_all_shared_features, SHARED_FEATURE_LIST
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='Add shared features for training or prediction data'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['training', 'prediction'],
        default='training',
        help='Mode: training or prediction (sets default paths)'
    )
    parser.add_argument(
        '--binance',
        type=str,
        default=None,
        help='Path to Binance crypto data JSON (overrides --mode default)'
    )
    parser.add_argument(
        '--macro',
        type=str,
        default=None,
        help='Path to macro data JSON (overrides --mode default)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for features JSON (overrides --mode default)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('add_shared_features')

    # Set default paths based on mode
    if args.mode == 'training':
        binance_path = args.binance or 'data/raw/training/binance_20pairs_1H.json'
        macro_path = args.macro or 'data/raw/training/macro_training_binance.json'
        output_path = args.output or 'data/features/training_shared_features.json'
    else:  # prediction
        binance_path = args.binance or 'data/raw/prediction/binance_LINKUSDT_live.json'
        macro_path = args.macro or 'data/raw/prediction/macro_prediction_binance.json'
        output_path = args.output or 'data/features/prediction_shared_features.json'

    logger.info("=" * 80)
    logger.info(f"ADD SHARED FEATURES - {args.mode.upper()} MODE")
    logger.info("=" * 80)
    logger.info(f"Binance data: {binance_path}")
    logger.info(f"Macro data:   {macro_path}")
    logger.info(f"Output:       {output_path}")
    logger.info("")
    logger.info(f"Features to add: {len(SHARED_FEATURE_LIST)} shared features")
    logger.info("  - 12 macro features (GOLD, BNB, BTC_PREMIUM, ETH_PREMIUM)")
    logger.info("  - 20 core indicators")
    logger.info("  - 24 momentum features")
    logger.info("  - 32 advanced features")
    logger.info("  - 4 trend strength features")
    logger.info("  - 1 statistical feature (squeeze_duration)")
    logger.info("")
    logger.info("✅ NO LOOK-FORWARD - Safe for live prediction!")
    logger.info("")

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Binance crypto data
    logger.info("=" * 80)
    logger.info("LOADING BINANCE CRYPTO DATA")
    logger.info("=" * 80)

    if not Path(binance_path).exists():
        logger.error(f"❌ Binance data not found: {binance_path}")
        return 1

    logger.info(f"Loading {binance_path}...")
    start_time = time.time()

    with open(binance_path, 'r') as f:
        binance_data = json.load(f)

    load_time = time.time() - start_time

    logger.info(f"✓ Loaded {len(binance_data):,} records in {load_time:.1f}s")

    # Analyze pairs
    pairs = [item['pair'] for item in binance_data]
    unique_pairs = sorted(set(pairs))
    logger.info(f"Pairs: {len(unique_pairs)} unique")
    for pair in unique_pairs[:5]:
        count = pairs.count(pair)
        logger.info(f"  {pair}: {count:,} records")
    if len(unique_pairs) > 5:
        logger.info(f"  ... and {len(unique_pairs) - 5} more")

    # Load macro data
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING MACRO DATA")
    logger.info("=" * 80)

    if not Path(macro_path).exists():
        logger.error(f"❌ Macro data not found: {macro_path}")
        return 1

    logger.info(f"Loading {macro_path}...")
    start_time = time.time()

    with open(macro_path, 'r') as f:
        macro_data = json.load(f)

    load_time = time.time() - start_time

    logger.info(f"✓ Loaded {len(macro_data):,} records in {load_time:.1f}s")

    # Analyze macro indicators
    tickers = [item['ticker'] for item in macro_data]
    unique_tickers = sorted(set(tickers))
    logger.info(f"Indicators: {len(unique_tickers)}")
    for ticker in unique_tickers:
        count = tickers.count(ticker)
        logger.info(f"  {ticker}: {count:,} records")

    # Convert to DataFrames
    logger.info("")
    logger.info("=" * 80)
    logger.info("CONVERTING TO DATAFRAMES")
    logger.info("=" * 80)

    logger.info("Creating crypto DataFrame...")
    crypto_df = pd.DataFrame(binance_data)
    logger.info(f"  Shape: {crypto_df.shape}")
    logger.info(f"  Columns: {list(crypto_df.columns)}")

    logger.info("Creating macro DataFrame...")
    macro_df = pd.DataFrame(macro_data)
    logger.info(f"  Shape: {macro_df.shape}")
    logger.info(f"  Columns: {list(macro_df.columns)}")

    # Check timestamp alignment
    crypto_timestamps = set(crypto_df['timestamp'].unique())
    macro_timestamps = set(macro_df['timestamp'].unique())
    common_timestamps = crypto_timestamps & macro_timestamps

    logger.info("")
    logger.info("Timestamp alignment:")
    logger.info(f"  Crypto timestamps: {len(crypto_timestamps):,}")
    logger.info(f"  Macro timestamps:  {len(macro_timestamps):,}")
    logger.info(f"  Common:            {len(common_timestamps):,}")
    logger.info(f"  Coverage:          {len(common_timestamps)/len(crypto_timestamps)*100:.1f}%")

    # Add shared features
    logger.info("")
    logger.info("=" * 80)
    logger.info("ADDING SHARED FEATURES")
    logger.info("=" * 80)

    logger.info("Running feature engineering pipeline...")
    logger.info("This may take several minutes for large datasets...")
    start_time = time.time()

    try:
        result_df = add_all_shared_features(crypto_df, macro_df)
        feature_time = time.time() - start_time

        logger.info(f"✓ Features added in {feature_time:.1f}s ({feature_time/60:.1f} minutes)")
        logger.info(f"  Output shape: {result_df.shape}")
        logger.info(f"  Total columns: {len(result_df.columns)}")

        # Verify features
        added_features = [col for col in result_df.columns if col in SHARED_FEATURE_LIST]
        missing_features = [col for col in SHARED_FEATURE_LIST if col not in result_df.columns]

        logger.info("")
        logger.info("Feature verification:")
        logger.info(f"  Expected: {len(SHARED_FEATURE_LIST)} features")
        logger.info(f"  Added:    {len(added_features)} features")

        if missing_features:
            logger.warning(f"  ⚠️  Missing: {len(missing_features)} features")
            for feat in missing_features[:10]:
                logger.warning(f"    - {feat}")
            if len(missing_features) > 10:
                logger.warning(f"    ... and {len(missing_features) - 10} more")
        else:
            logger.info("  ✓ All expected features present")

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

    logger.info(f"Writing {len(output_data):,} records to {output_path}...")
    start_time = time.time()

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    save_time = time.time() - start_time
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024

    logger.info(f"✓ Saved in {save_time:.1f}s")
    logger.info(f"  File size: {file_size_mb:.2f} MB")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Mode:              {args.mode}")
    logger.info(f"Input records:     {len(binance_data):,}")
    logger.info(f"Output records:    {len(output_data):,}")
    logger.info(f"Features added:    {len(added_features)}/{len(SHARED_FEATURE_LIST)}")
    logger.info(f"Total columns:     {len(result_df.columns)}")
    logger.info(f"Processing time:   {feature_time:.1f}s ({feature_time/60:.1f} min)")
    logger.info(f"Output file:       {output_path}")
    logger.info(f"File size:         {file_size_mb:.2f} MB")

    # Per-pair analysis (training mode only)
    if args.mode == 'training' and 'pair' in result_df.columns:
        logger.info("")
        logger.info("=" * 80)
        logger.info("PER-PAIR ANALYSIS")
        logger.info("=" * 80)

        pair_counts = result_df['pair'].value_counts().sort_index()
        for pair in pair_counts.index[:10]:
            logger.info(f"{pair:15s} {pair_counts[pair]:,} records")
        if len(pair_counts) > 10:
            logger.info(f"... and {len(pair_counts) - 10} more pairs")

    # Success
    logger.info("")
    if missing_features:
        logger.warning(f"⚠️  Completed with {len(missing_features)} missing features")
        return 1
    else:
        logger.info("✅ All shared features added successfully!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
