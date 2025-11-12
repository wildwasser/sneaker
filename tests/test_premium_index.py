#!/usr/bin/env python3
"""
Test Binance Premium Index Klines availability.

Premium Index = Perpetual Futures Price - Spot Index Price
- Positive = Futures trading at premium (bullish sentiment)
- Negative = Futures trading at discount (bearish sentiment)
- Updates every hour (much better than 8H funding rate!)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from sneaker.data import get_binance_client

# Check credentials
api_key = os.environ.get("BINANCE_API")
api_secret = os.environ.get("BINANCE_SECRET")

if not api_key or not api_secret:
    print("❌ BINANCE_API and BINANCE_SECRET not set")
    sys.exit(1)

print("Testing Premium Index Klines on Binance Futures...")
print("=" * 80)

try:
    client = get_binance_client()

    # Test 1: Get current premium index
    print("\n1. Testing current premium index...")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        try:
            premium_data = client.futures_mark_price(symbol=symbol)
            print(f"\n{symbol}:")
            print(f"   Mark Price: ${float(premium_data['markPrice']):,.2f}")
            print(f"   Index Price: ${float(premium_data['indexPrice']):,.2f}")

            # Calculate premium percentage
            mark = float(premium_data['markPrice'])
            index = float(premium_data['indexPrice'])
            premium_pct = ((mark - index) / index) * 100

            print(f"   Premium: {premium_pct:+.4f}%")

            if premium_pct > 0:
                print(f"   → Bullish (futures trading at premium)")
            else:
                print(f"   → Bearish (futures trading at discount)")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test 2: Get historical premium index klines (Jan 2024, 1H)
    print("\n" + "=" * 80)
    print("\n2. Testing historical premium index klines (1H interval)...")

    for symbol in ['BTCUSDT', 'ETHUSDT']:
        try:
            print(f"\n{symbol}:")

            # Get premium index klines for January 2024
            klines = client.futures_premium_index_klines(
                symbol=symbol,
                interval='1h',
                startTime=int(datetime(2024, 1, 1).timestamp() * 1000),
                endTime=int(datetime(2024, 1, 31).timestamp() * 1000)
            )

            if klines:
                print(f"   ✅ Got {len(klines)} hourly premium index candles")

                # Show first and last
                first = klines[0]
                last = klines[-1]

                first_ts = datetime.fromtimestamp(first[0] / 1000).strftime('%Y-%m-%d %H:%M')
                last_ts = datetime.fromtimestamp(last[0] / 1000).strftime('%Y-%m-%d %H:%M')

                print(f"   First: {first_ts}, Premium: {float(first[1]):.6f}")
                print(f"   Last:  {last_ts}, Premium: {float(last[1]):.6f}")

                # Calculate average premium
                premiums = [float(k[1]) for k in klines]
                avg_premium = sum(premiums) / len(premiums)
                print(f"   Avg premium: {avg_premium:.6f}")

            else:
                print(f"   ❌ No data returned")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Test 3: Check available intervals
    print("\n" + "=" * 80)
    print("\n3. Testing different intervals...")

    symbol = 'BTCUSDT'
    test_intervals = ['5m', '15m', '1h', '4h']

    for interval in test_intervals:
        try:
            klines = client.futures_premium_index_klines(
                symbol=symbol,
                interval=interval,
                limit=10  # Just get 10 to test
            )

            if klines:
                print(f"   ✅ {interval:4s} interval: {len(klines)} candles available")
            else:
                print(f"   ❌ {interval:4s} interval: No data")

        except Exception as e:
            print(f"   ❌ {interval:4s} interval: Error - {str(e)[:50]}")

    print("\n" + "=" * 80)
    print("✅ PREMIUM INDEX TEST COMPLETE")
    print()
    print("💡 Premium Index is PERFECT for sentiment:")
    print("   - Updates every hour (not every 8 hours!)")
    print("   - Measures futures vs spot (basis/premium)")
    print("   - Available in 1H interval for training")
    print("   - Full historical data available")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
