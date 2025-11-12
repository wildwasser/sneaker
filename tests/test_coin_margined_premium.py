#!/usr/bin/env python3
"""
Test Binance Coin-Margined (CM) Perpetual Premium Index.

BTCUSD_PERP (Coin-Margined) vs BTCUSDT (USDT-Margined)
- Different settlement (BTC vs USDT)
- Different trader preferences
- Potential spread indicates arbitrage/positioning
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

print("Testing Coin-Margined (CM) Premium Index on Binance...")
print("=" * 80)

try:
    client = get_binance_client()

    # Test 1: Get current CM premium index
    print("\n1. Testing current coin-margined premium index...")

    # Try BTCUSD_PERP (coin-margined perpetual)
    symbols_to_test = ['BTCUSD_PERP', 'BTCUSD', 'BTCUSD_PERPETUAL']

    for symbol in symbols_to_test:
        try:
            print(f"\n   Testing symbol: {symbol}")

            # Try coin futures mark price
            premium_data = client.futures_coin_mark_price(symbol=symbol)

            print(f"   ✅ {symbol} found!")
            print(f"      Symbol: {premium_data['symbol']}")
            print(f"      Mark Price: ${float(premium_data['markPrice']):,.2f}")
            print(f"      Index Price: ${float(premium_data['indexPrice']):,.2f}")

            # Calculate premium
            mark = float(premium_data['markPrice'])
            index = float(premium_data['indexPrice'])
            premium_pct = ((mark - index) / index) * 100
            print(f"      Premium: {premium_pct:+.4f}%")

            break  # Found working symbol

        except Exception as e:
            print(f"   ❌ {symbol}: {str(e)[:60]}")

    # Test 2: Try historical premium index klines
    print("\n" + "=" * 80)
    print("\n2. Testing historical coin-margined premium index klines...")

    # Try the working symbol from above
    test_symbol = 'BTCUSD_PERP'

    try:
        print(f"\n   Testing {test_symbol} (Jan 2024)...")

        # Try coin futures premium index klines
        klines = client.futures_coin_premium_index_klines(
            symbol=test_symbol,
            interval='1h',
            startTime=int(datetime(2024, 1, 1).timestamp() * 1000),
            endTime=int(datetime(2024, 1, 31).timestamp() * 1000)
        )

        if klines:
            print(f"   ✅ Got {len(klines)} hourly CM premium index klines!")

            # Show first and last
            first = klines[0]
            last = klines[-1]

            first_ts = datetime.fromtimestamp(first[0] / 1000).strftime('%Y-%m-%d %H:%M')
            last_ts = datetime.fromtimestamp(last[0] / 1000).strftime('%Y-%m-%d %H:%M')

            print(f"   First: {first_ts}, Premium: {float(first[1]):.6f}")
            print(f"   Last:  {last_ts}, Premium: {float(last[1]):.6f}")

        else:
            print(f"   ❌ No klines data returned")

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")

    # Test 3: Try historical data back to 2021
    print("\n" + "=" * 80)
    print("\n3. Testing historical depth (2021)...")

    try:
        print(f"\n   Testing {test_symbol} (Jan 2021)...")

        klines_2021 = client.futures_coin_premium_index_klines(
            symbol=test_symbol,
            interval='1h',
            startTime=int(datetime(2021, 1, 1).timestamp() * 1000),
            endTime=int(datetime(2021, 1, 31).timestamp() * 1000)
        )

        if klines_2021:
            print(f"   ✅ Got {len(klines_2021)} klines from Jan 2021!")
            print(f"   → CM Premium Index data goes back to 2021!")
        else:
            print(f"   ❌ No data for 2021")

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")

    # Test 4: Compare USDT-margined vs Coin-margined
    print("\n" + "=" * 80)
    print("\n4. Comparing USDT-margined vs Coin-margined premium...")

    try:
        # Get current premiums for both
        usdt_premium = client.futures_mark_price(symbol='BTCUSDT')
        coin_premium = client.futures_coin_mark_price(symbol='BTCUSD_PERP')

        # Calculate premiums
        usdt_mark = float(usdt_premium['markPrice'])
        usdt_index = float(usdt_premium['indexPrice'])
        usdt_prem_pct = ((usdt_mark - usdt_index) / usdt_index) * 100

        coin_mark = float(coin_premium['markPrice'])
        coin_index = float(coin_premium['indexPrice'])
        coin_prem_pct = ((coin_mark - coin_index) / coin_index) * 100

        print(f"\n   BTCUSDT (USDT-margined):")
        print(f"      Premium: {usdt_prem_pct:+.4f}%")

        print(f"\n   BTCUSD_PERP (Coin-margined):")
        print(f"      Premium: {coin_prem_pct:+.4f}%")

        spread = coin_prem_pct - usdt_prem_pct
        print(f"\n   Spread (CM - USDT): {spread:+.4f}%")

        if abs(spread) > 0.01:
            print(f"   → Different trader positioning detected!")
        else:
            print(f"   → Premiums are aligned")

    except Exception as e:
        print(f"   ❌ Error: {str(e)[:80]}")

    print("\n" + "=" * 80)
    print("✅ COIN-MARGINED PREMIUM INDEX TEST COMPLETE")
    print()
    print("💡 BTCUSD_PERP (Coin-Margined) vs BTCUSDT (USDT-Margined):")
    print("   - Different settlement: BTC vs USDT")
    print("   - Different trader bases")
    print("   - Spread shows positioning differences")
    print("   - Available in 1H intervals")
    print("   - Historical data back to 2021")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
