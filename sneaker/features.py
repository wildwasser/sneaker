"""Feature engineering for Sneaker.

Adds all 83 Enhanced V3 features to raw candle data.

Feature breakdown:
- 20 core indicators (in indicators.py)
- 24 momentum features (Batch 1)
- 35 advanced features (Batch 2)
- 4 statistical features (Batch 3)
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from .indicators import add_core_indicators


# ====================================================================================
# BATCH 1: Momentum Features (24 features)
# ====================================================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 24 momentum features.

    Features:
    - Price ROC (4): 3, 5, 10, 20 periods
    - Price acceleration (2): 5, 10 periods
    - Indicator acceleration (6): RSI, RSI7, BB, MACD, Stoch, DI
    - Volatility momentum (4): regime vel, vol ratio accel, ATR, ATR vel
    - Multi-timeframe (4): 2x aggregations
    - Price action (4): streak, distance from high/low, VWAP distance
    """
    df = df.copy()

    # 1. Price momentum (6 features)
    for period in [3, 5, 10, 20]:
        df[f'price_roc_{period}'] = df.groupby('pair')['close'].pct_change(period) * 100

    df['price_accel_5'] = df.groupby('pair')['price_roc_5'].diff()
    df['price_accel_10'] = df.groupby('pair')['price_roc_10'].diff()

    # 2. Indicator acceleration (6 features)
    df['rsi_accel'] = df.groupby('pair')['rsi_vel'].diff()
    df['rsi_7_accel'] = df.groupby('pair')['rsi_7_vel'].diff()
    df['bb_position_accel'] = df.groupby('pair')['bb_position_vel'].diff()
    df['macd_hist_accel'] = df.groupby('pair')['macd_hist_vel'].diff()
    df['stoch_accel'] = df.groupby('pair')['stoch_vel'].diff()
    df['di_diff_accel'] = df.groupby('pair')['di_diff_vel'].diff()

    # 3. Volatility momentum (4 features)
    # Calculate volatility regime first
    returns = df.groupby('pair')['close'].pct_change()
    df['volatility_regime'] = returns.rolling(20).std()
    df['vol_regime_vel'] = df.groupby('pair')['volatility_regime'].diff()
    df['vol_ratio_accel'] = df.groupby('pair')['vol_ratio_vel'].diff()

    # ATR (Average True Range)
    df['atr_14'] = df.groupby('pair').apply(
        lambda x: (x['high'] - x['low']).rolling(14).mean()
    ).reset_index(level=0, drop=True)
    df['atr_vel'] = df.groupby('pair')['atr_14'].diff()

    # 4. Multi-timeframe features (4 features)
    # 2x aggregations (every 2 candles)
    df['rsi_2x'] = df.groupby('pair')['rsi'].rolling(2).mean().reset_index(level=0, drop=True)
    df['bb_pos_2x'] = df.groupby('pair')['bb_position'].rolling(2).mean().reset_index(level=0, drop=True)
    df['macd_hist_2x'] = df.groupby('pair')['macd_hist'].rolling(2).mean().reset_index(level=0, drop=True)
    df['price_change_2x'] = df.groupby('pair')['close'].pct_change(2) * 100

    # 5. Price action features (4 features)
    df['is_up_streak'] = df.groupby('pair')['is_up_bar'].rolling(3).sum().reset_index(level=0, drop=True)

    rolling_high = df.groupby('pair')['close'].rolling(20).max().reset_index(level=0, drop=True)
    rolling_low = df.groupby('pair')['close'].rolling(20).min().reset_index(level=0, drop=True)

    df['dist_from_high_20'] = (rolling_high - df['close']) / df['close'] * 100
    df['dist_from_low_20'] = (df['close'] - rolling_low) / df['close'] * 100
    df['dist_from_vwap'] = (df['close'] - df['vwap_20']) / df['close'] * 100

    return df


# ====================================================================================
# BATCH 2: Advanced Features (35 features)
# ====================================================================================

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 35 advanced features.

    Features:
    - 4x timeframe (5): Longer aggregations
    - Indicator interactions (6): Cross-indicator relationships
    - Volatility regime (6): Detailed volatility classification
    - Price extremes (3): New highs/lows
    - Trend patterns (4): Higher highs, lower lows
    - Divergences (5): Price vs indicator divergences
    - Volume patterns (2): Abnormal volume
    - Trend strength (4): ADX derivatives
    """
    df = df.copy()

    # 1. 4x timeframe aggregations (5 features)
    df['rsi_4x'] = df.groupby('pair')['rsi'].rolling(4).mean().reset_index(level=0, drop=True)
    df['bb_pos_4x'] = df.groupby('pair')['bb_position'].rolling(4).mean().reset_index(level=0, drop=True)
    df['macd_hist_4x'] = df.groupby('pair')['macd_hist'].rolling(4).mean().reset_index(level=0, drop=True)
    df['price_change_4x'] = df.groupby('pair')['close'].pct_change(4) * 100
    df['vol_ratio_4x'] = df.groupby('pair')['vol_ratio'].rolling(4).mean().reset_index(level=0, drop=True)

    # 2. Indicator interactions (6 features)
    df['rsi_bb_interaction'] = df['rsi'] * df['bb_position']
    df['macd_vol_interaction'] = df['macd_hist'] * df['vol_ratio']
    df['rsi_stoch_interaction'] = df['rsi'] * df['stoch'] / 100  # Normalize
    df['bb_vol_interaction'] = df['bb_position'] * df['vol_ratio']
    df['adx_di_interaction'] = df['adx'] * df['di_diff'] / 100  # Normalize

    # Price/momentum alignment
    price_direction = np.sign(df.groupby('pair')['close'].diff())
    rsi_direction = np.sign(df['rsi'] - 50)
    df['price_rsi_momentum_align'] = (price_direction == rsi_direction).astype(int)

    # 3. Volatility regime classification (6 features)
    vol = df['volatility_regime']
    vol_rolling_mean = df.groupby('pair')['volatility_regime'].rolling(50).mean().reset_index(level=0, drop=True)
    vol_rolling_std = df.groupby('pair')['volatility_regime'].rolling(50).std().reset_index(level=0, drop=True)

    df['vol_percentile'] = df.groupby('pair')['volatility_regime'].rank(pct=True)
    df['vol_regime_low'] = (vol < vol_rolling_mean - vol_rolling_std).astype(int)
    df['vol_regime_med'] = ((vol >= vol_rolling_mean - vol_rolling_std) &
                            (vol <= vol_rolling_mean + vol_rolling_std)).astype(int)
    df['vol_regime_high'] = (vol > vol_rolling_mean + vol_rolling_std).astype(int)
    df['vol_zscore'] = (vol - vol_rolling_mean) / (vol_rolling_std + 1e-8)

    # 4. Price extremes (3 features)
    rolling_high_20 = df.groupby('pair')['high'].rolling(20).max().reset_index(level=0, drop=True)
    rolling_low_20 = df.groupby('pair')['low'].rolling(20).min().reset_index(level=0, drop=True)

    df['is_new_high_20'] = (df['high'] >= rolling_high_20).astype(int)
    df['is_new_low_20'] = (df['low'] <= rolling_low_20).astype(int)
    df['price_range_position'] = np.where(
        rolling_high_20 - rolling_low_20 > 0,
        (df['close'] - rolling_low_20) / (rolling_high_20 - rolling_low_20),
        0.5
    )

    # 5. Trend patterns (4 features)
    # Higher highs / lower lows
    df['consecutive_higher_highs'] = 0
    df['consecutive_lower_lows'] = 0

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        highs = df.loc[mask, 'high'].values
        lows = df.loc[mask, 'low'].values

        higher_highs = np.zeros(len(highs))
        lower_lows = np.zeros(len(lows))

        for i in range(1, len(highs)):
            if highs[i] > highs[i-1]:
                higher_highs[i] = higher_highs[i-1] + 1
            if lows[i] < lows[i-1]:
                lower_lows[i] = lower_lows[i-1] + 1

        df.loc[mask, 'consecutive_higher_highs'] = higher_highs
        df.loc[mask, 'consecutive_lower_lows'] = lower_lows

    # VWAP analysis
    df['vwap_distance_pct'] = (df['close'] - df['vwap_20']) / df['vwap_20'] * 100
    df['price_20_high'] = df.groupby('pair')['close'].rolling(20).max().reset_index(level=0, drop=True)

    # 6. Price/indicator divergences (5 features)
    price_change_5 = df.groupby('pair')['close'].pct_change(5)
    rsi_change_5 = df.groupby('pair')['rsi'].diff(5)
    macd_change_5 = df.groupby('pair')['macd_hist'].diff(5)
    stoch_change_5 = df.groupby('pair')['stoch'].diff(5)

    # Divergence: price up but indicator down (or vice versa)
    df['price_rsi_divergence'] = ((price_change_5 > 0) & (rsi_change_5 < 0)) | \
                                  ((price_change_5 < 0) & (rsi_change_5 > 0))
    df['price_rsi_divergence'] = df['price_rsi_divergence'].astype(int)

    df['price_macd_divergence'] = ((price_change_5 > 0) & (macd_change_5 < 0)) | \
                                   ((price_change_5 < 0) & (macd_change_5 > 0))
    df['price_macd_divergence'] = df['price_macd_divergence'].astype(int)

    df['price_stoch_divergence'] = ((price_change_5 > 0) & (stoch_change_5 < 0)) | \
                                    ((price_change_5 < 0) & (stoch_change_5 > 0))
    df['price_stoch_divergence'] = df['price_stoch_divergence'].astype(int)

    # Divergence strength
    df['rsi_divergence_strength'] = np.abs(price_change_5 * rsi_change_5)
    df['macd_divergence_strength'] = np.abs(price_change_5 * macd_change_5)

    # 7. Volume patterns (2 features)
    df['vol_momentum_5'] = df.groupby('pair')['vol_ratio'].diff(5)
    vol_mean = df.groupby('pair')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
    vol_std = df.groupby('pair')['volume'].rolling(20).std().reset_index(level=0, drop=True)
    df['is_high_volume'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)

    # 8. Additional features
    df['price_20_low'] = df.groupby('pair')['close'].rolling(20).min().reset_index(level=0, drop=True)

    # 9. Trend strength (4 features)
    df['adx_vel'] = df.groupby('pair')['adx'].diff()
    df['adx_accel'] = df.groupby('pair')['adx_vel'].diff()
    df['is_strong_trend'] = (df['adx'] > 25).astype(int)
    df['is_weak_trend'] = (df['adx'] < 20).astype(int)

    return df


# ====================================================================================
# BATCH 3: Statistical Features (4 features)
# ====================================================================================

def calculate_hurst_exponent(series, max_lag=20):
    """Calculate Hurst exponent for trend persistence."""
    if len(series) < max_lag * 2:
        return 0.5  # Neutral

    lags = range(2, max_lag)
    tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]

    # Filter out zeros
    valid_indices = [i for i, t in enumerate(tau) if t > 0]
    if len(valid_indices) < 3:
        return 0.5

    lags_valid = [lags[i] for i in valid_indices]
    tau_valid = [tau[i] for i in valid_indices]

    # Linear regression in log space
    try:
        poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
        return poly[0]  # Slope is Hurst exponent
    except:
        return 0.5


def calculate_permutation_entropy(series, order=3, delay=1):
    """Calculate permutation entropy for predictability."""
    if len(series) < order:
        return 0.5

    series = np.array(series)

    # Create permutations
    permutations = []
    for i in range(len(series) - delay * (order - 1)):
        indices = [i + j * delay for j in range(order)]
        permutations.append(tuple(np.argsort(series[indices])))

    # Count unique permutations
    unique, counts = np.unique(permutations, return_counts=True)
    probabilities = counts / len(permutations)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    # Normalize (max entropy for order=3 is log2(6) = 2.58)
    max_entropy = np.log2(np.math.factorial(order))
    return entropy / max_entropy if max_entropy > 0 else 0


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 statistical features.

    Features:
    - Hurst exponent: Trend persistence vs mean reversion
    - Permutation entropy: Predictability measure
    - CUSUM signal: Change point detection
    - Squeeze duration: Bollinger Band squeeze length
    """
    df = df.copy()

    # 1. Hurst exponent (rolling)
    df['hurst_exponent'] = 0.5  # Default neutral

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        hurst_values = []
        for i in range(len(closes)):
            if i < 40:
                hurst_values.append(0.5)
            else:
                window = closes[max(0, i-40):i]
                hurst = calculate_hurst_exponent(window)
                hurst_values.append(hurst)

        df.loc[mask, 'hurst_exponent'] = hurst_values

    # 2. Permutation entropy (rolling)
    df['permutation_entropy'] = 0.5  # Default neutral

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        perm_ent_values = []
        for i in range(len(closes)):
            if i < 10:
                perm_ent_values.append(0.5)
            else:
                window = closes[max(0, i-10):i]
                perm_ent = calculate_permutation_entropy(window)
                perm_ent_values.append(perm_ent)

        df.loc[mask, 'permutation_entropy'] = perm_ent_values

    # 3. CUSUM signal (cumulative sum for change detection)
    returns = df.groupby('pair')['close'].pct_change()
    mean_return = returns.rolling(50).mean()
    df['cusum_signal'] = (returns - mean_return).groupby(df['pair']).cumsum()

    # 4. Squeeze duration (BB squeeze length)
    # Squeeze = when BB width is narrow
    bb_upper = df.groupby('pair')['close'].rolling(20).mean() + \
               2 * df.groupby('pair')['close'].rolling(20).std()
    bb_lower = df.groupby('pair')['close'].rolling(20).mean() - \
               2 * df.groupby('pair')['close'].rolling(20).std()

    bb_width = ((bb_upper - bb_lower) / df['close']).reset_index(level=0, drop=True)
    bb_width_ma = bb_width.rolling(100).mean()

    is_squeeze = (bb_width < bb_width_ma * 0.7).astype(int)

    # Count consecutive squeeze candles
    df['squeeze_duration'] = 0

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        squeeze_array = is_squeeze[mask].values

        duration = np.zeros(len(squeeze_array))
        counter = 0

        for i in range(len(squeeze_array)):
            if squeeze_array[i]:
                counter += 1
                duration[i] = counter
            else:
                counter = 0

        df.loc[mask, 'squeeze_duration'] = duration

    return df


# ====================================================================================
# Main Function: Add All Features
# ====================================================================================

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all 83 Enhanced V3 features to DataFrame.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume, pair)

    Returns:
        DataFrame with all 83 features added

    Features added:
    - 20 core indicators (from indicators.py)
    - 24 momentum features (Batch 1)
    - 35 advanced features (Batch 2)
    - 4 statistical features (Batch 3)
    Total: 83 features
    """
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume', 'pair']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add features in order
    df = add_core_indicators(df)  # 20 features
    df = add_momentum_features(df)  # 24 features
    df = add_advanced_features(df)  # 35 features
    df = add_statistical_features(df)  # 4 features

    # Fill NaNs with 0 (safe for derived features)
    df = df.fillna(0)

    return df
