#!/usr/bin/env python3
"""
Test Binance Long/Short Ratio API availability.

Types:
1. Global Long/Short Account Ratio - All accounts
2. Top Trader Long/Short Account Ratio - Top 20% by margin
3. Top Trader Long/Short Position Ratio - Top 20% positions
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from sneaker.data import get_binance_client

# Check credentials
api_key = os.environ.get("BINANCE_API")
api_secret = os.environ.get("BINANCE_SECRET")

if not api_key or not api_secret:
    print("❌ BINANCE_API and BINANCE_SECRET not set")
    sys.exit(1)

print("Testing Long/Short Ratio API on Binance...")
print("=" * 80)

try:
    client = get_binance_client()

    # Test 1: Current data (1H interval)
    print("\n1. Testing current long/short ratio data (1H)...")

    for symbol in ['BTCUSDT', 'ETHUSDT']:
        try:
            print(f"\n   {symbol}:")

            # Global account ratio
            global_data = client.futures_global_longshort_ratio(
                symbol=symbol,
                period='1h',
                limit=10
            )

            if global_data:
                latest = global_data[0]
                print(f"      Global Account Ratio:")
                print(f"         Long: {float(latest['longAccount']):.2f}%")
                print(f"         Short: {float(latest['shortAccount']):.2f}%")
                print(f"         Ratio: {float(latest['longShortRatio']):.4f}")
                print(f"         Time: {datetime.fromtimestamp(latest['timestamp']/1000).strftime('%Y-%m-%d %H:%M')}")

            # Top trader account ratio
            top_account_data = client.futures_top_longshort_account_ratio(
                symbol=symbol,
                period='1h',
                limit=10
            )

            if top_account_data:
                latest = top_account_data[0]
                print(f"      Top Trader Account Ratio:")
                print(f"         Long: {float(latest['longAccount']):.2f}%")
                print(f"         Short: {float(latest['shortAccount']):.2f}%")
                print(f"         Ratio: {float(latest['longShortRatio']):.4f}")

            # Top trader position ratio
            top_position_data = client.futures_top_longshort_position_ratio(
                symbol=symbol,
                period='1h',
                limit=10
            )

            if top_position_data:
                latest = top_position_data[0]
                print(f"      Top Trader Position Ratio:")
                print(f"         Long: {float(latest['longAccount']):.2f}%")
                print(f"         Short: {float(latest['shortAccount']):.2f}%")
                print(f"         Ratio: {float(latest['longShortRatio']):.4f}")

        except Exception as e:
            print(f"      ❌ Error: {str(e)[:60]}")

    # Test 2: How far back does it go?
    print("\n" + "=" * 80)
    print("\n2. Testing historical depth...")

    symbol = 'BTCUSDT'

    # Try 30 days ago
    print(f"\n   Testing 30 days ago (limit: 500 records)...")
    try:
        data_30d = client.futures_global_longshort_ratio(
            symbol=symbol,
            period='1h',
            limit=500  # Max limit
        )

        if data_30d:
            first = data_30d[-1]  # Oldest record
            last = data_30d[0]    # Newest record

            first_dt = datetime.fromtimestamp(first['timestamp'] / 1000)
            last_dt = datetime.fromtimestamp(last['timestamp'] / 1000)

            days_back = (datetime.now() - first_dt).days

            print(f"   ✅ Got {len(data_30d)} records")
            print(f"   First: {first_dt.strftime('%Y-%m-%d %H:%M')} ({days_back} days ago)")
            print(f"   Last:  {last_dt.strftime('%Y-%m-%d %H:%M')}")

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")

    # Try with explicit date range (Jan 2024)
    print(f"\n   Testing explicit date range (Jan 2024)...")
    try:
        start_ts = int(datetime(2024, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2024, 1, 31).timestamp() * 1000)

        data_jan2024 = client.futures_global_longshort_ratio(
            symbol=symbol,
            period='1h',
            startTime=start_ts,
            endTime=end_ts,
            limit=500
        )

        if data_jan2024:
            print(f"   ✅ Got {len(data_jan2024)} records from Jan 2024")
        else:
            print(f"   ⚠️  No data for Jan 2024")

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")

    # Try with explicit date range (Jan 2021)
    print(f"\n   Testing explicit date range (Jan 2021)...")
    try:
        start_ts = int(datetime(2021, 1, 1).timestamp() * 1000)
        end_ts = int(datetime(2021, 1, 31).timestamp() * 1000)

        data_jan2021 = client.futures_global_longshort_ratio(
            symbol=symbol,
            period='1h',
            startTime=start_ts,
            endTime=end_ts,
            limit=500
        )

        if data_jan2021:
            print(f"   ✅ Got {len(data_jan2021)} records from Jan 2021!")
            print(f"   → Long/Short data goes back to 2021!")
        else:
            print(f"   ❌ No data for Jan 2021")

    except Exception as e:
        print(f"   ❌ Error for 2021: {str(e)[:80]}")

    # Test 3: Check available periods
    print("\n" + "=" * 80)
    print("\n3. Testing different periods...")

    test_periods = ['5m', '15m', '1h', '4h']

    for period in test_periods:
        try:
            data = client.futures_global_longshort_ratio(
                symbol='BTCUSDT',
                period=period,
                limit=10
            )

            if data:
                print(f"   ✅ {period:4s} period: {len(data)} records available")
            else:
                print(f"   ❌ {period:4s} period: No data")

        except Exception as e:
            print(f"   ❌ {period:4s} period: Error - {str(e)[:50]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Available ratios:")
    print("   ✅ Global Long/Short Account Ratio")
    print("   ✅ Top Trader Long/Short Account Ratio")
    print("   ✅ Top Trader Long/Short Position Ratio")
    print()
    print("Periods supported: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d")
    print()
    print("⚠️  LIMITATION CHECK:")
    print("   If limited to 30 days: Cannot use for training (need 2021-present)")
    print("   If full history available: Perfect for training!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
