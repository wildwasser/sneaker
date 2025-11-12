"""Macro economic data collection from yfinance.

Downloads macro indicators (SPY, VIX, DXY, GLD, TLT) and resamples to 1H frequency.
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict
from datetime import datetime


# Default macro indicators
MACRO_TICKERS = ['SPY', '^VIX', 'DX-Y.NYB', 'GLD', 'TLT']


def download_ticker(
    ticker: str,
    start_date: str,
    end_date: str = None
) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yfinance.

    Args:
        ticker: Ticker symbol (e.g., "SPY", "^VIX", "DX-Y.NYB")
        start_date: Start date (e.g., "2021-01-01")
        end_date: End date (default: today)

    Returns:
        DataFrame with columns:
            - timestamp: Unix timestamp (ms)
            - open, high, low, close: Price
            - volume: Trading volume
            - ticker: Ticker symbol
    """
    # Download from yfinance
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Download daily data
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(start=start_date, end=end_date, interval='1d')

    if df.empty:
        return pd.DataFrame()

    # Reset index to get datetime as column
    df = df.reset_index()

    # Rename columns to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Convert datetime to Unix timestamp (ms)
    df['timestamp'] = (df['date'].astype(int) // 10**6)  # Convert to ms

    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Convert types
    df['timestamp'] = df['timestamp'].astype(int)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Add ticker column
    df['ticker'] = ticker

    return df


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to 1H frequency using forward-fill.

    Args:
        df: DataFrame with timestamp column (Unix ms)

    Returns:
        DataFrame resampled to 1H with forward-filled values
    """
    if df.empty:
        return df

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime')

    # Resample to 1H (forward-fill for price data)
    resampled = df.resample('1h').ffill()

    # Reset index and convert back to timestamp
    resampled = resampled.reset_index()
    resampled['timestamp'] = (resampled['datetime'].astype(int) // 10**6)

    # Drop datetime column
    resampled = resampled.drop('datetime', axis=1)

    # Reorder columns
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ticker']
    resampled = resampled[cols]

    return resampled


def download_macro_data(
    tickers: List[str] = None,
    start_date: str = '2021-01-01',
    end_date: str = None,
    resample_1h: bool = True
) -> Dict[str, List[Dict]]:
    """
    Download macro economic data for multiple tickers.

    Args:
        tickers: List of ticker symbols (default: MACRO_TICKERS)
        start_date: Start date (default: 2021-01-01)
        end_date: End date (default: today)
        resample_1h: Resample to 1H frequency (default: True)

    Returns:
        Dictionary: {ticker: [candle records]}
    """
    if tickers is None:
        tickers = MACRO_TICKERS

    data = {}

    for ticker in tickers:
        # Download daily data
        df = download_ticker(ticker, start_date, end_date)

        if df.empty:
            continue

        # Resample to 1H if requested
        if resample_1h:
            df = resample_to_1h(df)

        # Convert to list of dicts for JSON serialization
        data[ticker] = df.to_dict('records')

    return data
