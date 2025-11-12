#!/usr/bin/env python3
"""
Download Training Binance Data - 20 Pairs, Long-Term

Downloads 1H OHLCV candles for 20 trading pairs from Binance.
Saves raw data to data/raw/training/binance_20pairs_1H.json.

Part of Issue #2 (sub-issue #1.1 of Pipeline Restructuring Epic #1)

Usage:
    .venv/bin/python scripts/01_download_training_binance.py [--start START] [--end END] [--pairs PAIRS]

Arguments:
    --start: Start date (YYYY-MM-DD), default: 2021-01-01
    --end: End date (YYYY-MM-DD), default: today
    --pairs: Comma-separated list, default: 20 baseline pairs

Requirements:
    - BINANCE_API environment variable set
    - BINANCE_SECRET environment variable set

Output:
    - data/raw/training/binance_20pairs_1H.json
    - Expected size: 500-1000 MB
    - Expected records: ~1,000,000 candles (20 pairs × ~50,000 candles)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger, BASELINE_PAIRS, download_historical_data


def main():
    parser = argparse.ArgumentParser(
        description='Download training Binance data (20 pairs, long-term)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2021-01-01',
        help='Start date (YYYY-MM-DD), default: 2021-01-01'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--pairs',
        type=str,
        default=None,
        help='Comma-separated pairs, default: 20 baseline pairs'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('download_training_binance')

    # Parse dates
    start_date = args.start
    end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')

    # Parse pairs
    if args.pairs:
        pairs = [p.strip() for p in args.pairs.split(',')]
    else:
        pairs = BASELINE_PAIRS

    logger.info("=" * 80)
    logger.info("DOWNLOAD TRAINING BINANCE DATA - Issue #2")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Pairs: {len(pairs)}")
    logger.info(f"Interval: 1H")
    logger.info("")

    # Create output directory
    output_path = Path('data/raw/training')
    output_path.mkdir(parents=True, exist_ok=True)

    # Download all pairs
    all_data = []
    failed_pairs = []
    total_candles = 0

    for i, pair in enumerate(pairs, 1):
        logger.info(f"[{i:2d}/{len(pairs)}] Downloading {pair}...")

        try:
            # Download with date range
            df = download_historical_data(
                pair,
                start_date=start_date,
                end_date=end_date,
                interval='1h'
            )

            if df is None or df.empty:
                logger.error(f"  ✗ No data returned for {pair}")
                failed_pairs.append(pair)
                continue

            # Convert to records (list of dicts)
            records = df.to_dict('records')
            all_data.extend(records)
            total_candles += len(records)

            logger.info(f"  ✓ {len(records):,} candles")
            logger.info(f"    First: {datetime.fromtimestamp(records[0]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"    Last:  {datetime.fromtimestamp(records[-1]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")

        except Exception as e:
            logger.error(f"  ✗ Error downloading {pair}: {e}")
            failed_pairs.append(pair)
            continue

        # Rate limit protection (Binance limits: 1200 req/min)
        time.sleep(0.5)

    # Save to file
    output_file = output_path / 'binance_20pairs_1H.json'

    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING DATA")
    logger.info("=" * 80)
    logger.info(f"Writing {len(all_data):,} candles to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

    file_size_mb = output_file.stat().st_size / 1024 / 1024

    logger.info(f"✓ Saved successfully")
    logger.info(f"  File size: {file_size_mb:.2f} MB")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pairs requested:    {len(pairs)}")
    logger.info(f"Pairs downloaded:   {len(pairs) - len(failed_pairs)}")
    logger.info(f"Pairs failed:       {len(failed_pairs)}")
    if failed_pairs:
        logger.info(f"  Failed: {', '.join(failed_pairs)}")
    logger.info(f"Total candles:      {total_candles:,}")
    logger.info(f"Date range:         {start_date} to {end_date}")
    logger.info(f"Output file:        {output_file}")
    logger.info(f"File size:          {file_size_mb:.2f} MB")

    # Validation checks
    logger.info("")
    logger.info("VALIDATION")
    logger.info("-" * 80)

    # Check minimum candles per pair (expect ~35,000 for 4 years)
    avg_candles_per_pair = total_candles / max(len(pairs) - len(failed_pairs), 1)
    logger.info(f"Avg candles/pair:   {avg_candles_per_pair:,.0f}")

    if avg_candles_per_pair < 30000:
        logger.warning("⚠ Warning: Average candles per pair is low (<30,000)")
        logger.warning("  Expected ~35,000 for 2021-2025 (4 years)")

    # Check all pairs present
    pairs_in_data = set(d['pair'] for d in all_data)
    missing_pairs = set(pairs) - pairs_in_data
    if missing_pairs:
        logger.warning(f"⚠ Warning: Missing pairs in data: {missing_pairs}")

    # Success/failure
    if len(failed_pairs) == 0:
        logger.info("✅ All pairs downloaded successfully")
        return 0
    elif len(failed_pairs) < len(pairs):
        logger.warning(f"⚠ Partial success: {len(failed_pairs)} pairs failed")
        return 1
    else:
        logger.error("❌ All pairs failed to download")
        return 1


if __name__ == '__main__':
    sys.exit(main())
