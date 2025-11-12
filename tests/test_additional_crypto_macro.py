#!/usr/bin/env python3
"""
Test additional 24/7 crypto macro tickers.
"""

import yfinance as yf

# Additional test tickers
test_tickers = {
    'Volatility (BVOL)': ['BVOL-USD', 'BVOL'],
    'Silver Token (SLVT)': ['SLVT-USD', 'SLVT'],
    'BNB (Exchange Health)': ['BNB-USD'],
    'Solana (L1 Blockchain)': ['SOL-USD'],
    'Litecoin (Payment)': ['LTC-USD'],
    'Chainlink (Oracle)': ['LINK-USD'],
}

start_date = '2024-01-01'
end_date = '2024-01-31'

print("=" * 80)
print("Testing Additional Crypto Macro Tickers - 1H Data")
print("=" * 80)
print(f"Test period: {start_date} to {end_date}\n")

successful = []

for name, symbols in test_tickers.items():
    print(f"{name}:")
    for symbol in symbols:
        try:
            ticker_obj = yf.Ticker(symbol)
            df = ticker_obj.history(start=start_date, end=end_date, interval='1h')

            if not df.empty:
                days = (df.index[-1] - df.index[0]).days + 1
                candles_per_day = len(df) / days if days > 0 else 0
                first_dt = df.index[0].strftime('%Y-%m-%d %H:%M')
                last_dt = df.index[-1].strftime('%Y-%m-%d %H:%M')

                print(f"  ✅ {symbol:15s} {len(df):4d} candles ({candles_per_day:.1f}/day)")
                print(f"     Range: {first_dt} to {last_dt}")

                if candles_per_day >= 23.0:
                    successful.append((name, symbol, len(df), candles_per_day))
            else:
                print(f"  ❌ {symbol:15s} No data")

        except Exception as e:
            print(f"  ❌ {symbol:15s} Error: {str(e)[:50]}")

    print()

# Summary
print("=" * 80)
print("24/7 TICKERS (≥23 candles/day)")
print("=" * 80)
for name, symbol, candles, cpd in successful:
    print(f"✓ {name:30s} {symbol:15s} ({cpd:.1f}/day)")

print()
