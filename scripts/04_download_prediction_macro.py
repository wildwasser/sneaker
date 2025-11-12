#!/usr/bin/env python3
"""
Download Prediction Macro Data - yfinance (recent, short-term)

Downloads recent macro economic indicators from yfinance (SPY, VIX, DXY, GLD, TLT).
Resamples daily data to 1H frequency (forward-fill).
Saves raw data to data/raw/prediction/macro_prediction.json.

Part of Issue #5 (sub-issue #1.4 of Pipeline Restructuring Epic #1)

Usage:
    .venv/bin/python scripts/04_download_prediction_macro.py [--hours HOURS] [--tickers TICKERS]

Arguments:
    --hours: Hours of recent data to fetch (default: 256)
    --tickers: Comma-separated list, default: SPY,^VIX,DX-Y.NYB,GLD,TLT

Output:
    - data/raw/prediction/macro_prediction.json
    - Expected size: 100-500 KB
    - Expected records: ~1,280 (256 hours × 5 tickers)
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.macro import download_macro_data, MACRO_TICKERS


def main():
    parser = argparse.ArgumentParser(
        description='Download prediction macro data (yfinance indicators, recent)'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=256,
        help='Hours of recent data to fetch (default: 256)'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        default=None,
        help='Comma-separated tickers, default: SPY,^VIX,DX-Y.NYB,GLD,TLT'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('download_prediction_macro')

    # Parse tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        tickers = MACRO_TICKERS

    # Calculate date range to cover requested hours
    # Since macro data is daily, we need to download enough days to cover the hours
    # Add buffer to ensure full coverage after resampling
    days_needed = (args.hours // 24) + 5  # +5 days buffer
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_needed)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    logger.info("=" * 80)
    logger.info("DOWNLOAD PREDICTION MACRO DATA - Issue #5")
    logger.info("=" * 80)
    logger.info(f"Target hours: {args.hours}")
    logger.info(f"Date range: {start_str} to {end_str} (~{days_needed} days)")
    logger.info(f"Tickers: {len(tickers)}")
    logger.info(f"Resample: Daily -> 1H (forward-fill)")
    logger.info("")

    # Create output directory
    output_path = Path('data/raw/prediction')
    output_path.mkdir(parents=True, exist_ok=True)

    # Download all tickers
    logger.info("Downloading macro data...")
    data = download_macro_data(
        tickers=tickers,
        start_date=start_str,
        end_date=end_str,
        resample_1h=True
    )

    # Report results
    total_candles = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        if ticker in data:
            records = data[ticker]

            # Trim to requested hours (keep most recent N hours)
            if len(records) > args.hours:
                records = records[-args.hours:]
                data[ticker] = records

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
    output_file = output_path / 'macro_prediction.json'

    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING DATA")
    logger.info("=" * 80)
    logger.info(f"Writing {len(all_data):,} candles to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)

    file_size_kb = output_file.stat().st_size / 1024

    logger.info(f"✓ Saved successfully")
    logger.info(f"  File size: {file_size_kb:.2f} KB")

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
    logger.info(f"Target hours:       {args.hours}")
    logger.info(f"Output file:        {output_file}")
    logger.info(f"File size:          {file_size_kb:.2f} KB")

    # Validation checks
    logger.info("")
    logger.info("VALIDATION")
    logger.info("-" * 80)

    # Check candles per ticker
    avg_candles_per_ticker = total_candles / max(len(data), 1)
    logger.info(f"Avg candles/ticker: {avg_candles_per_ticker:,.0f}")

    if avg_candles_per_ticker < args.hours * 0.9:  # Allow 10% tolerance
        logger.warning(f"⚠ Warning: Avg candles per ticker ({avg_candles_per_ticker:.0f}) is less than requested hours ({args.hours})")
    else:
        logger.info(f"✓ Got ~{avg_candles_per_ticker:.0f} candles per ticker (expected {args.hours})")

    # Check data is recent
    if all_data:
        last_timestamp = max(d['timestamp'] for d in all_data) / 1000
        now = datetime.now().timestamp()
        hours_old = (now - last_timestamp) / 3600

        if hours_old > 48:  # Macro data updates daily, allow 2 days
            logger.warning(f"⚠ Warning: Last data point is {hours_old:.1f} hours old")
        else:
            logger.info(f"✓ Data is recent (last point {hours_old:.1f} hours old)")

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
