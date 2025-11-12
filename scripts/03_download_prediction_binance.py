#!/usr/bin/env python3
"""
Download Prediction Binance Data - LINKUSDT (short-term, live)

Downloads recent 1H OHLCV candles from Binance for a single pair.
Default: LINKUSDT, 256 hours (configurable).
Saves raw data to data/raw/prediction/binance_{PAIR}_live.json.

Part of Issue #4 (sub-issue #1.3 of Pipeline Restructuring Epic #1)

Usage:
    .venv/bin/python scripts/03_download_prediction_binance.py [--pair PAIR] [--hours HOURS]

Arguments:
    --pair: Trading pair (default: LINKUSDT)
    --hours: Hours of recent data (default: 256)

Requirements:
    - BINANCE_API environment variable set
    - BINANCE_SECRET environment variable set

Output:
    - data/raw/prediction/binance_{PAIR}_live.json
    - Expected size: 50-200 KB (for 256-512 hours)
    - Expected records: 256-512 candles
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger, download_live_data


def main():
    parser = argparse.ArgumentParser(
        description='Download prediction Binance data (single pair, short-term)'
    )
    parser.add_argument(
        '--pair',
        type=str,
        default='LINKUSDT',
        help='Trading pair (default: LINKUSDT)'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=256,
        help='Hours of recent data to fetch (default: 256)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('download_prediction_binance')

    logger.info("=" * 80)
    logger.info("DOWNLOAD PREDICTION BINANCE DATA - Issue #4")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Hours: {args.hours}")
    logger.info(f"Interval: 1H")
    logger.info("")

    # Create output directory
    output_path = Path('data/raw/prediction')
    output_path.mkdir(parents=True, exist_ok=True)

    # Download live data
    logger.info(f"Downloading recent {args.hours} hours of {args.pair}...")

    try:
        df = download_live_data(args.pair, hours=args.hours)

        if df is None or df.empty:
            logger.error(f"✗ No data returned for {args.pair}")
            return 1

        # Convert to records (list of dicts)
        records = df.to_dict('records')

        logger.info(f"✓ {len(records):,} candles")
        logger.info(f"  First: {datetime.fromtimestamp(records[0]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"  Last:  {datetime.fromtimestamp(records[-1]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        logger.error(f"✗ Error downloading {args.pair}: {e}")
        return 1

    # Save to file
    output_file = output_path / f'binance_{args.pair}_live.json'

    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING DATA")
    logger.info("=" * 80)
    logger.info(f"Writing {len(records):,} candles to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2, default=str)

    file_size_kb = output_file.stat().st_size / 1024

    logger.info(f"✓ Saved successfully")
    logger.info(f"  File size: {file_size_kb:.2f} KB")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pair:           {args.pair}")
    logger.info(f"Candles:        {len(records):,}")
    logger.info(f"Hours covered:  {len(records)}")
    logger.info(f"Date range:     {datetime.fromtimestamp(records[0]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')} to {datetime.fromtimestamp(records[-1]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"Output file:    {output_file}")
    logger.info(f"File size:      {file_size_kb:.2f} KB")

    # Validation checks
    logger.info("")
    logger.info("VALIDATION")
    logger.info("-" * 80)

    # Check minimum candles (expect at least requested hours)
    if len(records) < args.hours * 0.9:  # Allow 10% tolerance
        logger.warning(f"⚠ Warning: Got {len(records)} candles, expected ~{args.hours}")
    else:
        logger.info(f"✓ Got {len(records)} candles (expected {args.hours})")

    # Check data is recent (last candle within 2 hours of now)
    last_timestamp = records[-1]['timestamp'] / 1000
    now = datetime.now().timestamp()
    hours_old = (now - last_timestamp) / 3600

    if hours_old > 2:
        logger.warning(f"⚠ Warning: Last candle is {hours_old:.1f} hours old")
    else:
        logger.info(f"✓ Data is recent (last candle {hours_old:.1f} hours old)")

    # Success
    logger.info("✅ Download completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
