#!/usr/bin/env python3
"""
Test 24/7 crypto-based macro ticker availability on yfinance.

Tests:
- BTCUSDT (Bitcoin - risk-on/off)
- ETHUSDT (Ethereum - tech sector proxy)
- PAXGUSDT (Tokenized Gold - commodity)
- BTC-USD, ETH-USD (yfinance format)
- PAX
G-USD (yfinance format for tokenized gold)
"""

import yfinance as yf
from datetime import datetime, timedelta

# Test tickers in both formats
test_tickers = {
    'Bitcoin': ['BTC-USD', 'BTCUSDT'],
    'Ethereum': ['ETH-USD', 'ETHUSDT'],
    'Tokenized Gold (PAXG)': ['PAXG-USD', 'PAXGUSD', 'PAXG'],
    'Tether Gold (XAUT)': ['XAUT-USD', 'XAUTUSD', 'XAUT'],
}

# Test date range (Jan 2024, 1 month)
start_date = '2024-01-01'
end_date = '2024-01-31'

print("=" * 80)
print("Testing 24/7 Crypto Macro Tickers - 1H Data Availability")
print("=" * 80)
print(f"Test period: {start_date} to {end_date}")
print()

results = {}

for name, symbols in test_tickers.items():
    print(f"\n{name}:")
    print("-" * 60)

    for symbol in symbols:
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start=start_date, end=end_date, interval='1h')

            if not df.empty:
                # Calculate candles per day
                days = (df.index[-1] - df.index[0]).days + 1
                candles_per_day = len(df) / days if days > 0 else 0

                first_dt = df.index[0].strftime('%Y-%m-%d %H:%M')
                last_dt = df.index[-1].strftime('%Y-%m-%d %H:%M')

                print(f"  ✅ {symbol:15s} {len(df):4d} candles ({candles_per_day:.1f}/day)")
                print(f"     Range: {first_dt} to {last_dt}")

                results[f"{name} ({symbol})"] = {
                    'symbol': symbol,
                    'candles': len(df),
                    'candles_per_day': candles_per_day,
                    'status': 'SUCCESS'
                }
            else:
                print(f"  ❌ {symbol:15s} No data")

        except Exception as e:
            print(f"  ❌ {symbol:15s} Error: {str(e)[:50]}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

successful = [k for k, v in results.items() if v['status'] == 'SUCCESS']
print(f"\nSuccessful tickers: {len(successful)}")
for name in successful:
    info = results[name]
    print(f"  - {name:30s} {info['candles']:4d} candles ({info['candles_per_day']:.1f}/day)")

print("\n24/7 Indicators (24 candles/day):")
crypto_247 = [k for k, v in results.items() if v['candles_per_day'] >= 23.0]
for name in crypto_247:
    print(f"  ✓ {name}")

print()
