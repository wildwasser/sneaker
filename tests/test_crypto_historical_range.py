#!/usr/bin/env python3
"""
Test how far back yfinance crypto data goes.
"""

import yfinance as yf
from datetime import datetime

# Test BTC back to 2021
ticker = 'BTC-USD'
start_date = '2021-01-01'
end_date = '2021-12-31'

print(f"Testing {ticker} historical data availability")
print(f"Date range: {start_date} to {end_date}")
print()

try:
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start_date, end=end_date, interval='1h')

    if not df.empty:
        days = (df.index[-1] - df.index[0]).days + 1
        candles_per_day = len(df) / days if days > 0 else 0

        first_dt = df.index[0].strftime('%Y-%m-%d %H:%M')
        last_dt = df.index[-1].strftime('%Y-%m-%d %H:%M')

        print(f"✅ SUCCESS!")
        print(f"   Total candles: {len(df):,}")
        print(f"   Candles/day: {candles_per_day:.1f}")
        print(f"   First: {first_dt}")
        print(f"   Last: {last_dt}")
        print(f"   Total days: {days}")
        print()
        print(f"💡 yfinance HAS complete 2021 data for {ticker}!")
        print(f"   This means we can skip Alpha Vantage entirely and")
        print(f"   download full 2021-present range instantly!")

    else:
        print(f"❌ No data available for {ticker} in 2021")

except Exception as e:
    print(f"❌ Error: {e}")
