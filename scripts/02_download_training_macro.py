#!/usr/bin/env python3
"""
Download Training Macro Data - yfinance (SPY, VIX, DXY, GLD, TLT)

Downloads macro economic indicators from yfinance.
Resamples daily data to 1H frequency (forward-fill).
Saves raw data to data/raw/training/macro_training.json.

Part of Issue #3 (sub-issue #1.2 of Pipeline Restructuring Epic #1)

Usage:
    .venv/bin/python scripts/02_download_training_macro.py [--start START] [--end END] [--tickers TICKERS]

Arguments:
    --start: Start date (YYYY-MM-DD), default: 2021-01-01
    --end: End date (YYYY-MM-DD), default: today
    --tickers: Comma-separated list, default: SPY,^VIX,DX-Y.NYB,GLD,TLT

Output:
    - data/raw/training/macro_training.json
    - Expected size: 5-20 MB
    - Expected records: ~35,000 per ticker (resampled to 1H)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.macro import download_macro_data, MACRO_TICKERS


def main():
    parser = argparse.ArgumentParser(
        description='Download training macro data (yfinance indicators)'
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
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated tickers, default: SPY,^VIX,DX-Y.NYB,GLD,TLT'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('download_training_macro')

    # Parse dates
    start_date = args.start
    end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')

    # Parse tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        tickers = MACRO_TICKERS

    logger.info("=" * 80)
    logger.info("DOWNLOAD TRAINING MACRO DATA - Issue #3")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Resample: Daily -> 1H (forward-fill)")
    logger.info("")

    # Create output directory
    output_path = Path('data/raw/training')
    output_path.mkdir(parents=True, exist_ok=True)

    # Download all tickers
    logger.info("Downloading macro data...")
    data = download_macro_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        resample_1h=True
    )

    # Report results
    total_candles = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        if ticker in data:
            records = data[ticker]
            total_candles += len(records)
            logger.info(f"[{i}/{len(tickers)}] {ticker:12s} ✓ {len(records):,} candles")
            logger.info(f"     First: {datetime.fromtimestamp(records[0]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"     Last:  {datetime.fromtimestamp(records[-1]['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")
        else:
            logger.error(f"[{i}/{len(tickers)}] {ticker:12s} ✗ Failed to download")
            failed_tickers.append(ticker)

    # Flatten data structure (combine all tickers into single list)
    all_data = []
    for ticker, records in data.items():
        all_data.extend(records)

    # Save to file
    output_file = output_path / 'macro_training.json'

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
    logger.info(f"Tickers requested:  {len(tickers)}")
    logger.info(f"Tickers downloaded: {len(data)}")
    logger.info(f"Tickers failed:     {len(failed_tickers)}")
    if failed_tickers:
        logger.info(f"  Failed: {', '.join(failed_tickers)}")
    logger.info(f"Total candles:      {total_candles:,}")
    logger.info(f"Date range:         {start_date} to {end_date}")
    logger.info(f"Output file:        {output_file}")
    logger.info(f"File size:          {file_size_mb:.2f} MB")

    # Validation checks
    logger.info("")
    logger.info("VALIDATION")
    logger.info("-" * 80)

    # Check minimum candles per ticker (expect ~35,000 for 4 years at 1H)
    avg_candles_per_ticker = total_candles / max(len(data), 1)
    logger.info(f"Avg candles/ticker: {avg_candles_per_ticker:,.0f}")

    if avg_candles_per_ticker < 30000:
        logger.warning("⚠ Warning: Average candles per ticker is low (<30,000)")
        logger.warning("  Expected ~35,000 for 2021-2025 (4 years, 1H resampled)")

    # Check all tickers present
    tickers_in_data = set(data.keys())
    missing_tickers = set(tickers) - tickers_in_data
    if missing_tickers:
        logger.warning(f"⚠ Warning: Missing tickers in data: {missing_tickers}")

    # Success/failure
    if len(failed_tickers) == 0:
        logger.info("✅ All tickers downloaded successfully")
        return 0
    elif len(failed_tickers) < len(tickers):
        logger.warning(f"⚠ Partial success: {len(failed_tickers)} tickers failed")
        return 1
    else:
        logger.error("❌ All tickers failed to download")
        return 1


if __name__ == '__main__':
    sys.exit(main())
