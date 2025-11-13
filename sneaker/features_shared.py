"""Shared feature engineering for both training and prediction.

CRITICAL: NO LOOK-FORWARD FUNCTIONALITY
All features in this module must be calculable on live data without future information.

Shared features (89 total):
- 20 core indicators (from indicators.py)
- 24 momentum features (price ROC, accelerations, vol features)
- 32 advanced features (interactions, vol regime, divergences, trend strength)
- 1 statistical feature (squeeze_duration)
- 12 macro features (GOLD, BNB, BTC_PREMIUM, ETH_PREMIUM: close + vel + ROC)

EXCLUDED from shared (training-only):
- hurst_exponent (complex, unstable on live)
- permutation_entropy (complex, may overfit)
- cusum_signal (cumulative, grows indefinitely)
"""

import numpy as np
import pandas as pd

from .indicators import add_core_indicators


# ====================================================================================
# MACRO FEATURES: Merge and Expand
# ====================================================================================

def merge_macro_features(crypto_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge macro indicators into crypto dataframe.

    Adds 12 macro features per row:
    - 4 close prices (macro_GOLD_close, macro_BNB_close, etc.)
    - 4 velocities (1-period diff)
    - 4 ROC values (5-period % change)

    Args:
        crypto_df: DataFrame with crypto OHLCV (must have 'pair' and 'timestamp' columns)
        macro_df: DataFrame with macro OHLCV (must have 'ticker' and 'timestamp' columns)

    Returns:
        DataFrame with 12 macro features added
    """
    # Create pivot tables for each macro field we need
    macro_close = macro_df.pivot(index='timestamp', columns='ticker', values='close')
    macro_close.columns = [f'macro_{col}_close' for col in macro_close.columns]

    # Merge close prices
    merged = crypto_df.merge(macro_close, on='timestamp', how='left')

    # Calculate velocities (1-period diff) for each macro indicator
    for ticker in macro_df['ticker'].unique():
        close_col = f'macro_{ticker}_close'
        if close_col in merged.columns:
            merged[f'macro_{ticker}_vel'] = merged[close_col].diff()

    # Calculate ROC (5-period % change) for each macro indicator
    for ticker in macro_df['ticker'].unique():
        close_col = f'macro_{ticker}_close'
        if close_col in merged.columns:
            merged[f'macro_{ticker}_roc_5'] = merged[close_col].pct_change(5) * 100

    # Fill NaN values in macro columns with forward-fill, then 0
    # (In case of any missing macro data at start)
    macro_cols = [col for col in merged.columns if col.startswith('macro_')]
    for col in macro_cols:
        merged[col] = merged[col].ffill().fillna(0)

    return merged


# ====================================================================================
# BATCH 1: Momentum Features (24 features)
# ====================================================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 24 momentum features (ALL SHARED - no look-forward).

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
    returns = df.groupby('pair')['close'].pct_change()
    df['volatility_regime'] = returns.rolling(20).std()
    df['vol_regime_vel'] = df.groupby('pair')['volatility_regime'].diff()
    df['vol_ratio_accel'] = df.groupby('pair')['vol_ratio_vel'].diff()

    # ATR (Average True Range)
    df['high_low_range'] = df['high'] - df['low']
    df['atr_14'] = df.groupby('pair')['high_low_range'].rolling(14).mean().reset_index(level=0, drop=True)
    df = df.drop(columns=['high_low_range'])
    df['atr_vel'] = df.groupby('pair')['atr_14'].diff()

    # 4. Multi-timeframe features (4 features) - 2x aggregations
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
# BATCH 2: Advanced Features (32 features) - SHARED ONLY
# ====================================================================================

def add_advanced_features_shared(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 32 advanced SHARED features (no look-forward).

    Excludes 3 training-only statistical features:
    - hurst_exponent (excluded - complex, unstable)
    - permutation_entropy (excluded - complex, may overfit)
    - cusum_signal (excluded - cumulative, grows indefinitely)

    Features:
    - 4x timeframe (5): Longer aggregations
    - Indicator interactions (6): Cross-indicator relationships
    - Volatility regime (6): Detailed volatility classification
    - Price extremes (3): New highs/lows
    - Trend patterns (2): Higher highs, lower lows
    - VWAP analysis (2): Distance and high
    - Divergences (5): Price vs indicator divergences (uses 5-period lookback, NO lookahead)
    - Volume patterns (2): Abnormal volume
    - Additional (1): price_20_low
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
    df['vol_rolling_mean'] = df.groupby('pair')['volatility_regime'].rolling(50).mean().reset_index(level=0, drop=True)
    df['vol_rolling_std'] = df.groupby('pair')['volatility_regime'].rolling(50).std().reset_index(level=0, drop=True)

    df['vol_percentile'] = df.groupby('pair')['volatility_regime'].rank(pct=True)
    df['vol_regime_low'] = (df['volatility_regime'] < df['vol_rolling_mean'] - df['vol_rolling_std']).astype(int)
    df['vol_regime_med'] = ((df['volatility_regime'] >= df['vol_rolling_mean'] - df['vol_rolling_std']) &
                            (df['volatility_regime'] <= df['vol_rolling_mean'] + df['vol_rolling_std'])).astype(int)
    df['vol_regime_high'] = (df['volatility_regime'] > df['vol_rolling_mean'] + df['vol_rolling_std']).astype(int)
    df['vol_zscore'] = (df['volatility_regime'] - df['vol_rolling_mean']) / (df['vol_rolling_std'] + 1e-8)

    # Clean up temporary columns
    df = df.drop(columns=['vol_rolling_mean', 'vol_rolling_std'])

    # 4. Price extremes (3 features)
    df['rolling_high_20'] = df.groupby('pair')['high'].rolling(20).max().reset_index(level=0, drop=True)
    df['rolling_low_20'] = df.groupby('pair')['low'].rolling(20).min().reset_index(level=0, drop=True)

    df['is_new_high_20'] = (df['high'] >= df['rolling_high_20']).astype(int)
    df['is_new_low_20'] = (df['low'] <= df['rolling_low_20']).astype(int)
    df['price_range_position'] = np.where(
        df['rolling_high_20'] - df['rolling_low_20'] > 0,
        (df['close'] - df['rolling_low_20']) / (df['rolling_high_20'] - df['rolling_low_20']),
        0.5
    )

    # Clean up temporary columns
    df = df.drop(columns=['rolling_high_20', 'rolling_low_20'])

    # 5. Trend patterns (2 features) - Simplified for performance
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

    # 6. VWAP analysis (2 features)
    df['vwap_distance_pct'] = (df['close'] - df['vwap_20']) / df['vwap_20'] * 100
    df['price_20_high'] = df.groupby('pair')['close'].rolling(20).max().reset_index(level=0, drop=True)

    # 7. Price/indicator divergences (5 features)
    # Uses 5-period LOOKBACK (not lookahead) - compares current price/indicator to 5 periods ago
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

    # 8. Volume patterns (2 features)
    df['vol_momentum_5'] = df.groupby('pair')['vol_ratio'].diff(5)
    df['vol_mean'] = df.groupby('pair')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
    df['vol_std'] = df.groupby('pair')['volume'].rolling(20).std().reset_index(level=0, drop=True)
    df['is_high_volume'] = (df['volume'] > df['vol_mean'] + 2 * df['vol_std']).astype(int)

    # Clean up temporary columns
    df = df.drop(columns=['vol_mean', 'vol_std'])

    # 9. Additional (1 feature)
    df['price_20_low'] = df.groupby('pair')['close'].rolling(20).min().reset_index(level=0, drop=True)

    return df


# ====================================================================================
# BATCH 3: Trend Strength Features (4 features)
# ====================================================================================

def add_trend_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4 trend strength features (ALL SHARED).

    Features:
    - adx_vel: ADX velocity
    - adx_accel: ADX acceleration
    - is_strong_trend: ADX > 25
    - is_weak_trend: ADX < 20
    """
    df = df.copy()

    df['adx_vel'] = df.groupby('pair')['adx'].diff()
    df['adx_accel'] = df.groupby('pair')['adx_vel'].diff()
    df['is_strong_trend'] = (df['adx'] > 25).astype(int)
    df['is_weak_trend'] = (df['adx'] < 20).astype(int)

    return df


# ====================================================================================
# BATCH 4: Squeeze Duration (1 statistical feature - SHARED)
# ====================================================================================

def add_squeeze_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add squeeze duration feature (SHARED - no look-forward).

    Squeeze = when Bollinger Band width is narrow.
    Counts consecutive squeeze candles.
    """
    df = df.copy()

    # Calculate BB width per pair
    df['bb_mean'] = df.groupby('pair')['close'].rolling(20).mean().reset_index(level=0, drop=True)
    df['bb_std'] = df.groupby('pair')['close'].rolling(20).std().reset_index(level=0, drop=True)

    df['bb_upper'] = df['bb_mean'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mean'] - 2 * df['bb_std']

    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_width_ma'] = df.groupby('pair')['bb_width'].rolling(100).mean().reset_index(level=0, drop=True)

    # Determine if in squeeze (width < 70% of MA)
    df['is_squeeze'] = (df['bb_width'] < df['bb_width_ma'] * 0.7).fillna(False).astype(int)

    # Count consecutive squeeze candles
    df['squeeze_duration'] = 0

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        squeeze_array = df.loc[mask, 'is_squeeze'].values

        duration = np.zeros(len(squeeze_array))
        counter = 0

        for i in range(len(squeeze_array)):
            if squeeze_array[i]:
                counter += 1
                duration[i] = counter
            else:
                counter = 0

        df.loc[mask, 'squeeze_duration'] = duration

    # Clean up temporary columns
    df = df.drop(columns=['bb_mean', 'bb_std', 'bb_upper', 'bb_lower', 'bb_width', 'bb_width_ma', 'is_squeeze'])

    return df


# ====================================================================================
# Main Function: Add All Shared Features
# ====================================================================================

def add_all_shared_features(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all shared features to DataFrame (NO look-forward).

    Features added:
    - 12 macro features (GOLD, BNB, BTC_PREMIUM, ETH_PREMIUM: close + vel + ROC)
    - 20 core indicators (from indicators.py)
    - 24 momentum features
    - 32 advanced features (excluding 3 training-only statistical)
    - 4 trend strength features
    - 1 statistical feature (squeeze_duration)

    Total: 93 shared features (12 macro + 20 core + 24 momentum + 32 advanced + 4 trend + 1 statistical)

    Args:
        df: DataFrame with crypto OHLCV data (columns: open, high, low, close, volume, pair, timestamp)
        macro_df: DataFrame with macro OHLCV data (columns: open, high, low, close, volume, ticker, timestamp)

    Returns:
        DataFrame with all 93 shared features added
    """
    # Ensure required columns exist
    required_crypto = ['open', 'high', 'low', 'close', 'volume', 'pair', 'timestamp']
    missing = [col for col in required_crypto if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in crypto data: {missing}")

    required_macro = ['timestamp', 'close', 'ticker']
    missing = [col for col in required_macro if col not in macro_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in macro data: {missing}")

    # 1. Merge macro features (12 features)
    df = merge_macro_features(df, macro_df)

    # 2. Add core indicators (20 features)
    df = add_core_indicators(df)

    # 3. Add momentum features (24 features)
    df = add_momentum_features(df)

    # 4. Add advanced shared features (32 features)
    df = add_advanced_features_shared(df)

    # 5. Add trend strength features (4 features)
    df = add_trend_strength_features(df)

    # 6. Add squeeze duration (1 feature)
    df = add_squeeze_duration(df)

    # Fill NaNs with 0 (safe for derived features)
    df = df.fillna(0)

    return df


# ====================================================================================
# Feature List for Reference
# ====================================================================================

# All 93 shared features
SHARED_FEATURE_LIST = [
    # Macro features (12)
    'macro_GOLD_close', 'macro_BNB_close', 'macro_BTC_PREMIUM_close', 'macro_ETH_PREMIUM_close',
    'macro_GOLD_vel', 'macro_BNB_vel', 'macro_BTC_PREMIUM_vel', 'macro_ETH_PREMIUM_vel',
    'macro_GOLD_roc_5', 'macro_BNB_roc_5', 'macro_BTC_PREMIUM_roc_5', 'macro_ETH_PREMIUM_roc_5',

    # Core indicators (20)
    'rsi', 'rsi_vel', 'rsi_7', 'rsi_7_vel',
    'bb_position', 'bb_position_vel',
    'macd_hist', 'macd_hist_vel',
    'stoch', 'stoch_vel',
    'di_diff', 'di_diff_vel', 'adx',
    'adr', 'adr_up_bars', 'adr_down_bars', 'is_up_bar',
    'vol_ratio', 'vol_ratio_vel',
    'vwap_20',

    # Momentum features (24)
    'price_roc_3', 'price_roc_5', 'price_roc_10', 'price_roc_20',
    'price_accel_5', 'price_accel_10',
    'rsi_accel', 'rsi_7_accel', 'bb_position_accel', 'macd_hist_accel', 'stoch_accel', 'di_diff_accel',
    'volatility_regime', 'vol_regime_vel', 'vol_ratio_accel',
    'atr_14', 'atr_vel',
    'rsi_2x', 'bb_pos_2x', 'macd_hist_2x', 'price_change_2x',
    'is_up_streak', 'dist_from_high_20', 'dist_from_low_20', 'dist_from_vwap',

    # Advanced features (32)
    'rsi_4x', 'bb_pos_4x', 'macd_hist_4x', 'price_change_4x', 'vol_ratio_4x',
    'rsi_bb_interaction', 'macd_vol_interaction', 'rsi_stoch_interaction', 'bb_vol_interaction', 'adx_di_interaction',
    'price_rsi_momentum_align',
    'vol_percentile', 'vol_regime_low', 'vol_regime_med', 'vol_regime_high', 'vol_zscore',
    'is_new_high_20', 'is_new_low_20', 'price_range_position',
    'consecutive_higher_highs', 'consecutive_lower_lows',
    'vwap_distance_pct', 'price_20_high',
    'price_rsi_divergence', 'price_macd_divergence', 'price_stoch_divergence',
    'rsi_divergence_strength', 'macd_divergence_strength',
    'vol_momentum_5', 'is_high_volume',
    'price_20_low',

    # Trend strength (4)
    'adx_vel', 'adx_accel', 'is_strong_trend', 'is_weak_trend',

    # Statistical (1)
    'squeeze_duration'
]

# Verify count
assert len(SHARED_FEATURE_LIST) == 93, f"Feature count mismatch: expected 93, got {len(SHARED_FEATURE_LIST)}"
