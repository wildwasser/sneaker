#!/usr/bin/env python3
"""
Download Prediction Macro Data - Binance Native Indicators (GOLD, BNB, Premiums)

Downloads recent 24/7 macro indicators from Binance for prediction:
- PAXGUSDT: Tokenized gold (commodity, safe haven)
- BNBUSDT: Exchange liquidity/flow indicator
- BTCUSDT premium index: USDT-margined sentiment (speculators)
- BTCUSD_PERP premium index: Coin-margined sentiment (BTC holders)

Part of Issue #5 (sub-issue #1.4 of Pipeline Restructuring Epic #1)

✅ TRUE 24/7 TRADING - ZERO DATA GAPS!
   - All from Binance (single API source)
   - Perfect alignment with Binance crypto data
   - Fast download (<10 seconds for 256 hours)

Macro Coverage:
   - GOLD (PAXGUSDT spot): Commodity/safe haven
   - BNB (BNBUSDT spot): Exchange health (money flow in/out of Binance)
   - BTC_PREMIUM (BTCUSDT USDT-M): Crypto sentiment (futures vs spot, 1H)
   - BTC_CM_PREMIUM (BTCUSD_PERP Coin-M): BTC holder sentiment (1H)

Usage:
    # Default: Last 256 hours
    .venv/bin/python scripts/04_download_prediction_macro_binance.py

    # Custom hours
    .venv/bin/python scripts/04_download_prediction_macro_binance.py --hours 180

    # Custom date range
    .venv/bin/python scripts/04_download_prediction_macro_binance.py --start 2024-11-01 --end 2024-11-12

Arguments:
    --hours: Hours of recent data (default: 256)
    --start: Start date (YYYY-MM-DD), overrides --hours if provided
    --end: End date (YYYY-MM-DD), default: now

Output:
    - data/raw/prediction/macro_prediction_binance.json
    - Expected size: ~0.2 MB (for 256 hours)
    - Expected records: ~1,024 (256 hours × 4 indicators)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sneaker import setup_logger
from sneaker.data import get_binance_client
import pandas as pd


# Binance-native macro indicators (same as training)
BINANCE_MACRO_INDICATORS = {
    'GOLD': 'PAXGUSDT',    # Tokenized gold (commodity)
    'BNB': 'BNBUSDT',      # Exchange flow/liquidity
}

PREMIUM_INDEX_SYMBOLS = {
    'BTC_PREMIUM': 'BTCUSDT',  # USDT-margined premium (speculators)
}

COIN_MARGINED_PREMIUM_SYMBOLS = {
    'BTC_CM_PREMIUM': 'BTCUSD_PERP',  # Coin-margined premium (BTC holders)
}


def download_premium_index_klines(client, symbol: str, start_date: str, end_date: str, ticker_name: str, logger) -> list:
    """
    Download historical premium index klines from Binance Futures.

    Premium Index = Perpetual Futures Price - Spot Index Price
    - Positive: Futures trading at premium (bullish sentiment)
    - Negative: Futures trading at discount (bearish sentiment)
    - Updates every hour (native 1H data, no interpolation needed!)

    Args:
        client: Binance client instance
        symbol: Futures symbol (e.g., 'BTCUSDT')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        ticker_name: Name for ticker field (e.g., 'BTC_PREMIUM')
        logger: Logger instance

    Returns:
        List of 1-hour premium index records
    """
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    logger.info(f"  Downloading premium index klines for {symbol}...")
    logger.info(f"    Start: {start_date}")
    logger.info(f"    End: {end_date}")

    all_klines = []
    current_start = start_ts

    batch_count = 0
    while current_start < end_ts:
        try:
            # Binance limits to 1500 klines per request
            batch = client.futures_premium_index_klines(
                symbol=symbol,
                interval='1h',
                startTime=current_start,
                endTime=end_ts,
                limit=1500
            )

            if not batch:
                break

            all_klines.extend(batch)
            batch_count += 1

            # Update start time for next batch
            current_start = batch[-1][0] + 3600000  # Add 1 hour in ms

            # Rate limit respect
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"    Error downloading batch: {e}")
            break

    logger.info(f"  ✓ Downloaded {len(all_klines):,} premium index klines")

    # Convert to standard format
    records = []
    for kline in all_klines:
        records.append({
            'timestamp': int(kline[0]),
            'open': float(kline[1]),      # Premium index open
            'high': float(kline[2]),      # Premium index high
            'low': float(kline[3]),       # Premium index low
            'close': float(kline[4]),     # Premium index close
            'volume': 0.0,  # Premium index has no volume
            'ticker': ticker_name
        })

    return records


def download_coin_margined_premium_klines(client, symbol: str, start_date: str, end_date: str, ticker_name: str, logger) -> list:
    """
    Download historical coin-margined premium index klines from Binance Futures.

    Coin-Margined Premium = BTCUSD_PERP (settled in BTC) premium
    - Different from USDT-margined (different trader base)
    - BTC holders vs USDT speculators
    - Spread shows positioning differences

    Args:
        client: Binance client instance
        symbol: Coin futures symbol (e.g., 'BTCUSD_PERP')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        ticker_name: Name for ticker field (e.g., 'BTC_CM_PREMIUM')
        logger: Logger instance

    Returns:
        List of 1-hour coin-margined premium index records
    """
    # Convert dates to timestamps
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    logger.info(f"  Downloading coin-margined premium index for {symbol}...")
    logger.info(f"    Start: {start_date}")
    logger.info(f"    End: {end_date}")

    all_klines = []
    current_start = start_ts

    batch_count = 0
    while current_start < end_ts:
        try:
            # Binance coin futures API
            batch = client.futures_coin_premium_index_klines(
                symbol=symbol,
                interval='1h',
                startTime=current_start,
                endTime=end_ts,
                limit=1500
            )

            if not batch:
                break

            all_klines.extend(batch)
            batch_count += 1

            # Update start time for next batch
            current_start = batch[-1][0] + 3600000  # Add 1 hour in ms

            # Rate limit respect
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"    Error downloading batch: {e}")
            break

    logger.info(f"  ✓ Downloaded {len(all_klines):,} coin-margined premium klines")

    # Convert to standard format
    records = []
    for kline in all_klines:
        records.append({
            'timestamp': int(kline[0]),
            'open': float(kline[1]),      # Premium index open
            'high': float(kline[2]),      # Premium index high
            'low': float(kline[3]),       # Premium index low
            'close': float(kline[4]),     # Premium index close
            'volume': 0.0,  # Premium index has no volume
            'ticker': ticker_name
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description='Download Binance-native macro indicators for prediction (recent data)'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=256,
        help='Hours of recent data to fetch (default: 256)'
    )
    parser.add_argument(
        '--start',
        type=str,
        default=None,
        help='Start date (YYYY-MM-DD), overrides --hours if provided'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD), default: now'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger('download_prediction_macro_binance')

    # Parse dates
    if args.start:
        # Use explicit date range
        start_date = args.start
        end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')

        # Calculate hours for display
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        hours = int((end_dt - start_dt).total_seconds() / 3600)
    else:
        # Use hours parameter (default: 256)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(hours=args.hours)).strftime('%Y-%m-%d')
        hours = args.hours

    logger.info("=" * 80)
    logger.info("DOWNLOAD PREDICTION MACRO DATA - Binance Native Indicators")
    logger.info("=" * 80)
    logger.info(f"Source: Binance API (spot + futures)")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Approximate hours: {hours}")
    total_indicators = len(BINANCE_MACRO_INDICATORS) + len(PREMIUM_INDEX_SYMBOLS) + len(COIN_MARGINED_PREMIUM_SYMBOLS)
    logger.info(f"Total indicators: {total_indicators}")
    logger.info(f"  Spot pairs: {len(BINANCE_MACRO_INDICATORS)} ({', '.join(BINANCE_MACRO_INDICATORS.keys())})")
    logger.info(f"  USDT-margined premium: {len(PREMIUM_INDEX_SYMBOLS)} ({', '.join(PREMIUM_INDEX_SYMBOLS.keys())})")
    logger.info(f"  Coin-margined premium: {len(COIN_MARGINED_PREMIUM_SYMBOLS)} ({', '.join(COIN_MARGINED_PREMIUM_SYMBOLS.keys())})")
    logger.info("")
    logger.info("✅ TRUE 24/7 DATA - ZERO GAPS!")
    logger.info("   - All from Binance (single API)")
    logger.info("   - Perfect alignment with crypto trading")
    logger.info("   - Fast download (<10 seconds)")
    logger.info("")

    # Create output directory
    output_path = Path('data/raw/prediction')
    output_path.mkdir(parents=True, exist_ok=True)

    # Get Binance client
    try:
        client = get_binance_client()
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        return 1

    all_data = []

    # Download spot pairs (GOLD, BNB)
    logger.info("=" * 80)
    logger.info("DOWNLOADING SPOT PAIRS")
    logger.info("=" * 80)

    for name, symbol in BINANCE_MACRO_INDICATORS.items():
        try:
            logger.info(f"\n{name} ({symbol}):")

            logger.info(f"  Downloading from {start_date} to {end_date}...")

            # Use get_historical_klines directly with correct date range
            klines = client.get_historical_klines(
                symbol,
                '1h',  # 1-hour interval
                start_date,
                end_date
            )

            if klines:
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])

                # Keep only what we need
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

                # Convert types
                df['timestamp'] = df['timestamp'].astype(int)
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['ticker'] = name

                records = df.to_dict('records')
                all_data.extend(records)

                logger.info(f"  ✓ Downloaded {len(records):,} candles")
                first_dt = datetime.fromtimestamp(records[0]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                last_dt = datetime.fromtimestamp(records[-1]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                logger.info(f"  Range: {first_dt} to {last_dt}")
            else:
                logger.warning(f"  ⚠️  No data returned for {symbol}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Download premium index klines
    logger.info("")
    logger.info("=" * 80)
    logger.info("DOWNLOADING PREMIUM INDEX (Futures vs Spot)")
    logger.info("=" * 80)

    for name, symbol in PREMIUM_INDEX_SYMBOLS.items():
        try:
            logger.info(f"\n{name} ({symbol}):")

            # Download premium index klines (native 1H data!)
            records = download_premium_index_klines(
                client, symbol, start_date, end_date, name, logger
            )

            if records:
                all_data.extend(records)

                logger.info(f"  ✓ Downloaded {len(records):,} hourly records")
                if records:
                    first_dt = datetime.fromtimestamp(records[0]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                    last_dt = datetime.fromtimestamp(records[-1]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                    logger.info(f"  Range: {first_dt} to {last_dt}")
            else:
                logger.warning(f"  ⚠️  No premium index data returned for {symbol}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    # Download coin-margined premium index klines
    logger.info("")
    logger.info("=" * 80)
    logger.info("DOWNLOADING COIN-MARGINED PREMIUM INDEX")
    logger.info("=" * 80)

    for name, symbol in COIN_MARGINED_PREMIUM_SYMBOLS.items():
        try:
            logger.info(f"\n{name} ({symbol}):")

            # Download coin-margined premium index klines (native 1H data!)
            records = download_coin_margined_premium_klines(
                client, symbol, start_date, end_date, name, logger
            )

            if records:
                all_data.extend(records)

                logger.info(f"  ✓ Downloaded {len(records):,} hourly records")
                if records:
                    first_dt = datetime.fromtimestamp(records[0]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                    last_dt = datetime.fromtimestamp(records[-1]['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M')
                    logger.info(f"  Range: {first_dt} to {last_dt}")
            else:
                logger.warning(f"  ⚠️  No coin-margined premium data returned for {symbol}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    if not all_data:
        logger.error("❌ No data downloaded!")
        return 1

    # Analyze by ticker
    logger.info("")
    logger.info("=" * 80)
    logger.info("PER-INDICATOR ANALYSIS")
    logger.info("=" * 80)

    ticker_data = defaultdict(list)
    for record in all_data:
        ticker_data[record['ticker']].append(record)

    for ticker in sorted(ticker_data.keys()):
        records = ticker_data[ticker]
        first_ts = records[0]['timestamp']
        last_ts = records[-1]['timestamp']
        first_dt = datetime.fromtimestamp(first_ts / 1000).strftime('%Y-%m-%d %H:%M')
        last_dt = datetime.fromtimestamp(last_ts / 1000).strftime('%Y-%m-%d %H:%M')

        logger.info(f"{ticker:15s} {len(records):,} records  ({first_dt} to {last_dt})")

    # Save combined data
    output_file = output_path / 'macro_prediction_binance.json'

    logger.info("")
    logger.info("=" * 80)
    logger.info("SAVING DATA")
    logger.info("=" * 80)
    logger.info(f"Writing {len(all_data):,} records to {output_file}")

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
    logger.info(f"Indicators requested:  {total_indicators}")
    logger.info(f"Indicators downloaded: {len(ticker_data)}")
    logger.info(f"Total records:         {len(all_data):,}")
    logger.info(f"Approximate hours:     {hours}")
    logger.info(f"Date range:            {start_date} to {end_date}")
    logger.info(f"Output file:           {output_file}")
    logger.info(f"File size:             {file_size_kb:.2f} KB")

    # Validation
    logger.info("")
    logger.info("VALIDATION")
    logger.info("-" * 80)

    avg_records_per_indicator = len(all_data) / max(len(ticker_data), 1)
    logger.info(f"Avg records/indicator: {avg_records_per_indicator:,.0f}")

    expected_per_indicator = hours  # ~1 record per hour
    logger.info(f"Expected per indicator: ~{expected_per_indicator:,} (24/7 for ~{hours} hours)")

    if avg_records_per_indicator < expected_per_indicator * 0.9:
        logger.warning(f"⚠️  Warning: Record count is low (<90% of expected)")
        logger.warning(f"   Expected ~{expected_per_indicator:,}, got {avg_records_per_indicator:,.0f}")
    else:
        logger.info("✓ Record count looks good for 24/7 data")

    # Success
    if len(ticker_data) == total_indicators:
        logger.info("✅ All indicators downloaded successfully (TRUE 24/7 Binance data!)")
        return 0
    else:
        logger.warning(f"⚠️  Partial success: {total_indicators - len(ticker_data)} indicators failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
