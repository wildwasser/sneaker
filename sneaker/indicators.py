"""Technical indicators for Sneaker.

Simplified implementations of the 20 core indicators used in Enhanced V3.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD, ADXIndicator


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    rsi = RSIIndicator(close=df['close'], window=period)
    return rsi.rsi()


def calculate_rsi_velocity(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI velocity (rate of change)."""
    rsi = calculate_rsi(df, period)
    return rsi.diff()


def calculate_bb_position(df: pd.DataFrame, period: int = 20, std: int = 2) -> pd.Series:
    """
    Calculate Bollinger Band position.

    Returns:
        Position between -1 (at lower band) and +1 (at upper band)
    """
    bb = BollingerBands(close=df['close'], window=period, window_dev=std)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    middle = bb.bollinger_mavg()

    # Position: 0 at middle, +1 at upper, -1 at lower
    bb_width = upper - lower
    position = np.where(
        bb_width > 0,
        (df['close'] - middle) / (bb_width / 2),
        0
    )

    return pd.Series(position, index=df.index)


def calculate_bb_position_velocity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate BB position velocity."""
    bb_pos = calculate_bb_position(df, period)
    return bb_pos.diff()


def calculate_macd_histogram(df: pd.DataFrame) -> pd.Series:
    """Calculate MACD histogram."""
    macd = MACD(close=df['close'])
    return macd.macd_diff()


def calculate_macd_histogram_velocity(df: pd.DataFrame) -> pd.Series:
    """Calculate MACD histogram velocity."""
    hist = calculate_macd_histogram(df)
    return hist.diff()


def calculate_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Stochastic Oscillator %K.

    Returns:
        Stochastic %K value (0-100)
    """
    high_roll = df['high'].rolling(period).max()
    low_roll = df['low'].rolling(period).min()

    stoch = np.where(
        high_roll - low_roll > 0,
        100 * (df['close'] - low_roll) / (high_roll - low_roll),
        50
    )

    return pd.Series(stoch, index=df.index)


def calculate_stochastic_velocity(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate stochastic velocity."""
    stoch = calculate_stochastic(df, period)
    return stoch.diff()


def calculate_di_diff(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Directional Indicator difference (DI+ minus DI-).

    Returns:
        DI difference (-100 to +100)
    """
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
    di_plus = adx.adx_pos()
    di_minus = adx.adx_neg()

    return di_plus - di_minus


def calculate_di_diff_velocity(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate DI difference velocity."""
    di_diff = calculate_di_diff(df, period)
    return di_diff.diff()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=period)
    return adx.adx()


def calculate_adr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Advance/Decline Ratio.

    Ratio of up bars to down bars over rolling window.
    """
    up_bars = (df['close'] > df['close'].shift(1)).astype(int)
    down_bars = (df['close'] < df['close'].shift(1)).astype(int)

    up_sum = up_bars.rolling(period).sum()
    down_sum = down_bars.rolling(period).sum()

    adr = np.where(down_sum > 0, up_sum / down_sum, 1.0)

    return pd.Series(adr, index=df.index)


def calculate_adr_up_bars(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Count of up bars in rolling window."""
    up_bars = (df['close'] > df['close'].shift(1)).astype(int)
    return up_bars.rolling(period).sum()


def calculate_adr_down_bars(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Count of down bars in rolling window."""
    down_bars = (df['close'] < df['close'].shift(1)).astype(int)
    return down_bars.rolling(period).sum()


def calculate_is_up_bar(df: pd.DataFrame) -> pd.Series:
    """Binary indicator: 1 if up bar, 0 otherwise."""
    return (df['close'] > df['close'].shift(1)).astype(int)


def calculate_vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate volume ratio.

    Current volume / average volume over period.
    """
    avg_vol = df['volume'].rolling(period).mean()
    vol_ratio = np.where(avg_vol > 0, df['volume'] / avg_vol, 1.0)

    return pd.Series(vol_ratio, index=df.index)


def calculate_vol_ratio_velocity(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate volume ratio velocity."""
    vol_ratio = calculate_vol_ratio(df, period)
    return vol_ratio.diff()


def calculate_vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Volume-Weighted Average Price.

    VWAP over rolling window.
    """
    pv = df['close'] * df['volume']
    rolling_pv = pv.rolling(period).sum()
    rolling_vol = df['volume'].rolling(period).sum()

    vwap = np.where(rolling_vol > 0, rolling_pv / rolling_vol, df['close'])

    return pd.Series(vwap, index=df.index)


def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 20 core indicators to DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # RSI family
    df['rsi'] = calculate_rsi(df, period=14)
    df['rsi_vel'] = calculate_rsi_velocity(df, period=14)
    df['rsi_7'] = calculate_rsi(df, period=7)
    df['rsi_7_vel'] = calculate_rsi_velocity(df, period=7)

    # Bollinger Bands
    df['bb_position'] = calculate_bb_position(df, period=20)
    df['bb_position_vel'] = calculate_bb_position_velocity(df, period=20)

    # MACD
    df['macd_hist'] = calculate_macd_histogram(df)
    df['macd_hist_vel'] = calculate_macd_histogram_velocity(df)

    # Stochastic
    df['stoch'] = calculate_stochastic(df, period=14)
    df['stoch_vel'] = calculate_stochastic_velocity(df, period=14)

    # Directional Indicators
    df['di_diff'] = calculate_di_diff(df, period=14)
    df['di_diff_vel'] = calculate_di_diff_velocity(df, period=14)
    df['adx'] = calculate_adx(df, period=14)

    # Advance/Decline
    df['adr'] = calculate_adr(df, period=14)
    df['adr_up_bars'] = calculate_adr_up_bars(df, period=14)
    df['adr_down_bars'] = calculate_adr_down_bars(df, period=14)
    df['is_up_bar'] = calculate_is_up_bar(df)

    # Volume
    df['vol_ratio'] = calculate_vol_ratio(df, period=20)
    df['vol_ratio_vel'] = calculate_vol_ratio_velocity(df, period=20)

    # VWAP
    df['vwap_20'] = calculate_vwap(df, period=20)

    return df
