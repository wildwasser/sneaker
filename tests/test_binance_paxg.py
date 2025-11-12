#!/usr/bin/env python3
"""
Test PAXGUSDT availability on Binance with historical data.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from sneaker.data import get_binance_client
from binance.exceptions import BinanceAPIException

# Check environment
api_key = os.environ.get("BINANCE_API")
api_secret = os.environ.get("BINANCE_SECRET")

if not api_key or not api_secret:
    print("❌ BINANCE_API and BINANCE_SECRET not set")
    sys.exit(1)

print("Testing PAXGUSDT on Binance...")
print("=" * 60)

try:
    client = get_binance_client()

    # Test 1: Check if symbol exists
    print("\n1. Checking if PAXGUSDT exists...")
    exchange_info = client.get_exchange_info()
    paxg_symbol = None
    for symbol in exchange_info['symbols']:
        if symbol['symbol'] == 'PAXGUSDT':
            paxg_symbol = symbol
            break

    if paxg_symbol:
        print(f"   ✅ PAXGUSDT found on Binance")
        print(f"   Status: {paxg_symbol['status']}")
        print(f"   Base asset: {paxg_symbol['baseAsset']}")
        print(f"   Quote asset: {paxg_symbol['quoteAsset']}")
    else:
        print("   ❌ PAXGUSDT not found on Binance")
        sys.exit(1)

    # Test 2: Get recent data (last 24 hours)
    print("\n2. Testing recent data (last 24 hours)...")
    klines_recent = client.get_klines(
        symbol='PAXGUSDT',
        interval='1h',
        limit=24
    )
    print(f"   ✅ Got {len(klines_recent)} recent 1H candles")

    # Test 3: Try historical data (2021)
    print("\n3. Testing historical data (Jan 2021)...")
    try:
        klines_2021 = client.get_historical_klines(
            symbol='PAXGUSDT',
            interval='1h',
            start_str='1 Jan 2021',
            end_str='31 Jan 2021'
        )

        if klines_2021:
            print(f"   ✅ Got {len(klines_2021)} candles from Jan 2021")
            print(f"   First candle timestamp: {klines_2021[0][0]}")
            print(f"   Last candle timestamp: {klines_2021[-1][0]}")
        else:
            print("   ⚠️  No data returned for Jan 2021")

    except BinanceAPIException as e:
        print(f"   ⚠️  API error for Jan 2021: {e}")

    # Test 4: Find earliest available data
    print("\n4. Finding earliest available data...")
    test_years = [2019, 2020, 2021, 2022, 2023]
    earliest_year = None

    for year in test_years:
        try:
            klines = client.get_historical_klines(
                symbol='PAXGUSDT',
                interval='1h',
                start_str=f'1 Jan {year}',
                end_str=f'7 Jan {year}',
                limit=100
            )

            if klines:
                print(f"   ✅ Data available for {year}")
                if earliest_year is None:
                    earliest_year = year
            else:
                print(f"   ❌ No data for {year}")

        except:
            print(f"   ❌ Error checking {year}")

    if earliest_year:
        print(f"\n   Earliest data: {earliest_year}")
    else:
        print(f"\n   ⚠️  Could not determine earliest data year")

    # Test 5: Check funding rate (futures)
    print("\n5. Checking if PAXGUSDT has futures/funding rate...")
    try:
        funding = client.futures_funding_rate(symbol='PAXGUSDT', limit=10)
        print(f"   ✅ Funding rate available: {len(funding)} records")
    except Exception as e:
        print(f"   ❌ No futures data: {str(e)[:50]}")

    print("\n" + "=" * 60)
    print("✅ PAXGUSDT TEST COMPLETE")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
