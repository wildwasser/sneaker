#!/usr/bin/env python3
"""
Step 1: Collect Data from Binance

Downloads 1H OHLCV candles for 20 trading pairs from Binance.

Usage:
    export BINANCE_API='your_key'
    export BINANCE_SECRET='your_secret'
    python scripts/01_collect_data.py
"""

import sys
import os
import json

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sneaker.data import download_multiple_pairs, BASELINE_PAIRS
from sneaker.logging import setup_logger


def main():
    """Download 1H data for all baseline pairs."""
    logger = setup_logger('collect_data')

    logger.info("="*80)
    logger.info("Step 1: Data Collection")
    logger.info("="*80)

    logger.info(f"\nDownloading 1H data for {len(BASELINE_PAIRS)} pairs...")
    logger.info(f"Pairs: {', '.join(BASELINE_PAIRS)}")
    logger.info(f"Target: ~50,000 candles per pair")
    logger.info("")

    try:
        # Download data
        data = download_multiple_pairs(
            pairs=BASELINE_PAIRS,
            max_candles=50000,
            verbose=True
        )

        if not data:
            logger.error("No data downloaded")
            return 1

        # Save to file
        os.makedirs('data', exist_ok=True)
        output_path = 'data/candles.json'

        with open(output_path, 'w') as f:
            json.dump(data, f)

        # Summary
        total_candles = sum(len(records) for records in data.values())
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        logger.info("")
        logger.info("="*80)
        logger.info("✅ Download Complete!")
        logger.info(f"   Pairs: {len(data)}")
        logger.info(f"   Total candles: {total_candles:,}")
        logger.info(f"   File size: {file_size_mb:.1f} MB")
        logger.info(f"   Saved to: {output_path}")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"❌ Download failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
