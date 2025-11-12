"""Data collection from Binance.

Simplified, standalone data fetching for Sneaker.
"""

import os
import pandas as pd
import numpy as np
from binance import Client
from typing import List, Dict


# 20 trading pairs used in training
BASELINE_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "SUIUSDT", "LINKUSDT",
    "AVAXUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "TRXUSDT",
    "ALGOUSDT", "APTUSDT", "AAVEUSDT", "XLMUSDT", "XMRUSDT"
]


def get_binance_client() -> Client:
    """
    Create authenticated Binance client.

    Requires environment variables:
        BINANCE_API: Your API key
        BINANCE_SECRET: Your API secret

    Returns:
        Binance client instance

    Raises:
        RuntimeError: If credentials not set
    """
    api_key = os.environ.get("BINANCE_API")
    api_secret = os.environ.get("BINANCE_SECRET")

    if not api_key or not api_secret:
        raise RuntimeError(
            "BINANCE_API and BINANCE_SECRET environment variables required.\n"
            "Set them:\n"
            "  export BINANCE_API='your_key'\n"
            "  export BINANCE_SECRET='your_secret'"
        )

    return Client(api_key, api_secret)


def download_pair(
    client: Client,
    pair: str,
    max_candles: int = 50000
) -> pd.DataFrame:
    """
    Download 1H OHLCV data for a single pair from Binance.

    Args:
        client: Authenticated Binance client
        pair: Trading pair (e.g., "BTCUSDT")
        max_candles: Maximum candles to fetch (default: 50,000)

    Returns:
        DataFrame with columns:
            - timestamp: Unix timestamp (ms)
            - open, high, low, close: Price
            - volume: Trading volume
            - trades: Number of trades
            - pair: Trading pair symbol
    """
    # Fetch historical klines (1H interval)
    klines = client.get_historical_klines(
        pair,
        Client.KLINE_INTERVAL_1HOUR,
        "1 Jan 2020",  # Start from 2020
        limit=max_candles
    )

    if not klines:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Keep only what we need
    df = df[[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'
    ]].copy()

    # Convert types
    df['timestamp'] = df['timestamp'].astype(int)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['trades'] = df['trades'].astype(int)

    # Add pair column
    df['pair'] = pair

    return df


def download_multiple_pairs(
    pairs: List[str] = None,
    max_candles: int = 50000,
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Download 1H data for multiple pairs.

    Args:
        pairs: List of pairs (default: BASELINE_PAIRS)
        max_candles: Max candles per pair (default: 50,000)
        verbose: Print progress (default: True)

    Returns:
        Dictionary: {pair: [candle records]}
    """
    if pairs is None:
        pairs = BASELINE_PAIRS

    client = get_binance_client()
    data = {}

    for i, pair in enumerate(pairs, 1):
        if verbose:
            print(f"[{i}/{len(pairs)}] Downloading {pair}...")

        try:
            df = download_pair(client, pair, max_candles)
            if len(df) > 0:
                # Convert to list of dicts for JSON serialization
                data[pair] = df.to_dict('records')
                if verbose:
                    print(f"  ✓ {len(df):,} candles")
            else:
                if verbose:
                    print(f"  ✗ No data")
        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")

    return data


def download_live_data(
    pair: str,
    hours: int = 180
) -> pd.DataFrame:
    """
    Download recent live data for a pair.

    Args:
        pair: Trading pair (e.g., "BTCUSDT")
        hours: Hours of data to fetch (default: 180)

    Returns:
        DataFrame with recent 1H candles
    """
    client = get_binance_client()

    # Fetch recent klines
    klines = client.get_klines(
        symbol=pair,
        interval=Client.KLINE_INTERVAL_1HOUR,
        limit=hours
    )

    if not klines:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Keep only what we need
    df = df[[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades'
    ]].copy()

    # Convert types
    df['timestamp'] = df['timestamp'].astype(int)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['trades'] = df['trades'].astype(int)

    # Add pair column
    df['pair'] = pair

    return df
