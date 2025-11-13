#!/usr/bin/env python3
"""Training-only feature engineering.

CRITICAL: USES FUTURE DATA - CANNOT BE USED IN LIVE PREDICTION

These features are ONLY for training the model. They use future information
that is not available during live prediction.

Part of Issue #7 (Pipeline Restructuring Epic #1)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ====================================================================================
# Feature List
# ====================================================================================

TRAINING_ONLY_FEATURE_LIST = [
    'target',  # Primary target (future price change, σ normalized)
    'hurst_exponent',  # Trend persistence
    'permutation_entropy',  # Predictability
    'cusum_signal',  # Change detection
]

# Verify count
assert len(TRAINING_ONLY_FEATURE_LIST) == 4, \
    f"Expected 4 training-only features, got {len(TRAINING_ONLY_FEATURE_LIST)}"


# ====================================================================================
# Target Calculation (PRIMARY TRAINING-ONLY FEATURE)
# ====================================================================================

def detect_indicator_flips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect when indicators flip direction (turning points).

    This is the CORE of ghost signal detection - indicators change direction
    BEFORE price reversals, creating an "echo" or "ghost" of the coming move.

    Detects:
    - RSI crossing 50 (momentum shift)
    - BB position crossing 0 (price position shift)
    - MACD histogram changing sign (trend shift)
    - Stochastic crossing 50 (momentum shift)
    - DI diff crossing 0 (directional shift)

    Args:
        df: DataFrame with indicator columns (must have shared features)

    Returns:
        DataFrame with flip detection columns added
    """
    # Ensure we have the required indicators
    required = ['rsi', 'bb_position', 'macd_hist', 'stoch', 'di_diff']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required indicators for ghost detection: {missing}")

    # Detect directional flips (sign changes or threshold crosses)
    # RSI crosses 50 (neutral point)
    df['rsi_flip'] = ((df['rsi'].shift(1) < 50) & (df['rsi'] > 50)) | \
                     ((df['rsi'].shift(1) > 50) & (df['rsi'] < 50))

    # BB position crosses 0 (middle band)
    df['bb_flip'] = ((df['bb_position'].shift(1) < 0) & (df['bb_position'] > 0)) | \
                    ((df['bb_position'].shift(1) > 0) & (df['bb_position'] < 0))

    # MACD histogram changes sign
    df['macd_flip'] = ((df['macd_hist'].shift(1) < 0) & (df['macd_hist'] > 0)) | \
                      ((df['macd_hist'].shift(1) > 0) & (df['macd_hist'] < 0))

    # Stochastic crosses 50
    df['stoch_flip'] = ((df['stoch'].shift(1) < 50) & (df['stoch'] > 50)) | \
                       ((df['stoch'].shift(1) > 50) & (df['stoch'] < 50))

    # DI diff crosses 0
    df['di_flip'] = ((df['di_diff'].shift(1) < 0) & (df['di_diff'] > 0)) | \
                    ((df['di_diff'].shift(1) > 0) & (df['di_diff'] < 0))

    # Convert to integers
    df['rsi_flip'] = df['rsi_flip'].astype(int)
    df['bb_flip'] = df['bb_flip'].astype(int)
    df['macd_flip'] = df['macd_flip'].astype(int)
    df['stoch_flip'] = df['stoch_flip'].astype(int)
    df['di_flip'] = df['di_flip'].astype(int)

    # Count total flips at this candle
    df['total_flips'] = df['rsi_flip'] + df['bb_flip'] + df['macd_flip'] + \
                        df['stoch_flip'] + df['di_flip']

    return df


def detect_ghost_signals(df: pd.DataFrame, min_flips: int = 3) -> pd.DataFrame:
    """
    Detect ghost signals: when multiple indicators flip simultaneously.

    Ghost Signal = 3+ indicators flip direction at the same time
    This creates an "echo" that precedes price reversals.

    Args:
        df: DataFrame with flip detection columns
        min_flips: Minimum number of simultaneous flips to mark as ghost signal (default: 3)

    Returns:
        DataFrame with 'is_ghost_signal' column (1 = ghost signal, 0 = normal)
    """
    # Ghost signal = min_flips or more indicators flip at once
    df['is_ghost_signal'] = (df['total_flips'] >= min_flips).astype(int)

    return df


def calculate_target(df: pd.DataFrame, lookahead_periods: int = 4, min_flips: int = 3) -> pd.DataFrame:
    """
    Calculate volatility-normalized future price change (PRIMARY TARGET).

    This is the MAIN training-only feature - what we're teaching the model to predict.
    Uses FUTURE data (lookahead), so CANNOT be used in live prediction.

    CRITICAL: Only calculates target at GHOST SIGNAL points (turning points).
    Normal candles get target = 0.

    Algorithm:
    1. Detect indicator flips (RSI, BB, MACD, Stoch, DI crossing thresholds)
    2. Mark ghost signals (3+ indicators flip simultaneously)
    3. For ghost signals: measure future reversal magnitude (lookahead periods)
    4. Normalize by volatility: target = reversal / volatility_std
    5. Normal candles: target = 0

    Args:
        df: DataFrame with OHLCV data, 'pair' column, and shared features
        lookahead_periods: Hours into future to look (default: 4H)
        min_flips: Minimum simultaneous flips to mark as ghost signal (default: 3)

    Returns:
        DataFrame with 'target' column added (in σ units)

    Target Interpretation:
        - target = 0: Normal candle (no ghost signal)
        - target > +4σ: Strong upward reversal at ghost signal (BUY)
        - target < -4σ: Strong downward reversal at ghost signal (SELL)
        - ~5-10% of candles will have non-zero targets (ghost signals)
    """
    # 1. Detect indicator flips
    df = detect_indicator_flips(df)

    # 2. Mark ghost signals
    df = detect_ghost_signals(df, min_flips=min_flips)

    # 3. Calculate future price changes and volatility
    df['target'] = 0.0

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values
        is_ghost = df.loc[mask, 'is_ghost_signal'].values

        # Calculate future price change (LOOK-AHEAD!)
        future_change = np.zeros(len(closes))
        for i in range(len(closes) - lookahead_periods):
            future_close = closes[i + lookahead_periods]
            current_close = closes[i]
            pct_change = (future_close - current_close) / current_close * 100
            future_change[i] = pct_change

        # Calculate volatility (20-period rolling std of returns)
        returns = pd.Series(closes).pct_change() * 100
        volatility = returns.rolling(20).std().fillna(1.0)  # Avoid division by zero

        # Normalize: target = future_change / volatility
        normalized = future_change / volatility.values
        normalized = np.nan_to_num(normalized, 0)

        # 4. Set target = 0 for normal candles, reversal magnitude for ghost signals
        target = np.where(is_ghost == 1, normalized, 0.0)

        df.loc[mask, 'target'] = target

    # Clean up temporary columns (keep only target)
    temp_cols = ['rsi_flip', 'bb_flip', 'macd_flip', 'stoch_flip', 'di_flip',
                 'total_flips', 'is_ghost_signal']
    df = df.drop(columns=temp_cols, errors='ignore')

    return df


# ====================================================================================
# Statistical Features (3 features)
# ====================================================================================

def calculate_hurst_exponent(series, max_lag=40):
    """
    Calculate Hurst exponent using rescaled range analysis.

    H > 0.5: Trending (momentum, persistent)
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk

    Training-only because:
    - Uses 40-period rolling window
    - May be unstable on live data with limited history
    - Complex calculation that may overfit

    Args:
        series: Price series

    Returns:
        Hurst exponent value (0 to 1)
    """
    if len(series) < max_lag + 1:
        return 0.5  # Neutral (random walk)

    try:
        # Calculate log returns
        log_returns = np.log(series[1:] / series[:-1])

        # Calculate rescaled range for different lags
        lags = range(2, max_lag)
        tau = []
        rs = []

        for lag in lags:
            # Split into lag-sized chunks
            chunks = [log_returns[i:i+lag] for i in range(0, len(log_returns), lag)]
            chunks = [chunk for chunk in chunks if len(chunk) == lag]

            if not chunks:
                continue

            # Calculate R/S for each chunk
            rs_values = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue

                mean = np.mean(chunk)
                std = np.std(chunk)

                if std == 0:
                    continue

                # Cumulative deviation from mean
                cumdev = np.cumsum(chunk - mean)

                # Range
                r = np.max(cumdev) - np.min(cumdev)

                # Rescaled range
                rs_val = r / std if std > 0 else 0
                rs_values.append(rs_val)

            if rs_values:
                tau.append(lag)
                rs.append(np.mean(rs_values))

        # Fit power law: R/S = c * tau^H
        # log(R/S) = log(c) + H * log(tau)
        if len(tau) > 1:
            coeffs = np.polyfit(np.log(tau), np.log(rs), 1)
            hurst = coeffs[0]
            # Clip to valid range
            return np.clip(hurst, 0.0, 1.0)
        else:
            return 0.5

    except Exception:
        return 0.5  # Return neutral on error


def add_hurst_exponent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Hurst exponent feature (trend persistence indicator).

    Training-only because it uses 40-period rolling window and
    may be unstable on live data.

    Args:
        df: DataFrame with 'pair' and 'close' columns

    Returns:
        DataFrame with 'hurst_exponent' column added
    """
    df['hurst_exponent'] = 0.5  # Default to neutral

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        hurst_values = []
        for i in range(len(closes)):
            if i < 40:
                hurst_values.append(0.5)  # Not enough data
            else:
                window = closes[max(0, i-40):i+1]
                hurst = calculate_hurst_exponent(window)
                hurst_values.append(hurst)

        df.loc[mask, 'hurst_exponent'] = hurst_values

    return df


def calculate_permutation_entropy(series):
    """
    Calculate permutation entropy (predictability measure).

    Lower entropy = more predictable patterns
    Higher entropy = more random

    Training-only because:
    - Uses 10-period lookback for pattern analysis
    - May overfit to training data patterns
    - Complex ordinal pattern analysis

    Args:
        series: Price series (10+ values)

    Returns:
        Permutation entropy (0 to 1, normalized)
    """
    if len(series) < 3:
        return 0.5

    try:
        # Ordinal patterns of length 3
        patterns = {}
        for i in range(len(series) - 2):
            window = series[i:i+3]

            # Rank values to get ordinal pattern
            ranks = np.argsort(np.argsort(window))
            pattern = tuple(ranks)

            patterns[pattern] = patterns.get(pattern, 0) + 1

        # Calculate entropy
        total = sum(patterns.values())
        if total == 0:
            return 0.5

        probs = [count / total for count in patterns.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)

        # Normalize by max entropy (log(6) for patterns of length 3)
        max_entropy = np.log(6)  # 3! = 6 possible patterns
        normalized = entropy / max_entropy

        return np.clip(normalized, 0.0, 1.0)

    except Exception:
        return 0.5


def add_permutation_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add permutation entropy feature (predictability measure).

    Training-only because it analyzes ordinal patterns that may overfit.

    Args:
        df: DataFrame with 'pair' and 'close' columns

    Returns:
        DataFrame with 'permutation_entropy' column added
    """
    df['permutation_entropy'] = 0.5  # Default to neutral

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        perm_ent_values = []
        for i in range(len(closes)):
            if i < 10:
                perm_ent_values.append(0.5)  # Not enough data
            else:
                window = closes[max(0, i-10):i]
                perm_ent = calculate_permutation_entropy(window)
                perm_ent_values.append(perm_ent)

        df.loc[mask, 'permutation_entropy'] = perm_ent_values

    return df


def add_cusum_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CUSUM signal (cumulative sum for change detection).

    Training-only because:
    - Grows indefinitely over time
    - Not suitable for live prediction without fixed baseline
    - Requires full history to be meaningful

    Args:
        df: DataFrame with 'pair' and 'close' columns

    Returns:
        DataFrame with 'cusum_signal' column added
    """
    returns = df.groupby('pair')['close'].pct_change()
    mean_return = returns.rolling(50).mean()

    # Cumulative sum of (return - mean_return)
    # Detects shifts in mean return (regime changes)
    df['cusum_signal'] = (returns - mean_return).groupby(df['pair']).cumsum()

    # Fill NaN with 0
    df['cusum_signal'] = df['cusum_signal'].fillna(0)

    return df


# ====================================================================================
# Main Function: Add All Training-Only Features
# ====================================================================================

def add_all_training_features(df: pd.DataFrame,
                             lookahead_periods: int = 4,
                             min_flips: int = 3) -> pd.DataFrame:
    """
    Add all training-only features to DataFrame.

    ⚠️  WARNING: USES FUTURE DATA - TRAINING ONLY!

    REQUIRES:
        - df must already have OHLCV data
        - df must have 'pair' column
        - df must have shared features from Issue #6 (for ghost signal detection)

    Args:
        df: DataFrame with OHLCV + shared features
        lookahead_periods: Hours into future for target calculation (default: 4)
        min_flips: Minimum simultaneous indicator flips for ghost signal (default: 3)

    Returns:
        DataFrame with 4 additional training-only features:
        - target (PRIMARY - ghost signal reversal magnitude, σ normalized)
        - hurst_exponent (trend persistence)
        - permutation_entropy (predictability)
        - cusum_signal (change detection)

    Features added:
    - 1 target variable (PRIMARY - with ghost signal detection)
    - 3 statistical features
    Total: 4 training-only features

    Ghost Signal Detection:
        Target is calculated ONLY at ghost signal points (turning points).
        Ghost signal = 3+ indicators flip direction simultaneously.
        Normal candles get target = 0.
        This creates the sparse signal structure for V3 sample weighting.
    """
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume', 'pair']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure shared features exist (needed for ghost detection)
    ghost_required = ['rsi', 'bb_position', 'macd_hist', 'stoch', 'di_diff']
    missing_indicators = [col for col in ghost_required if col not in df.columns]
    if missing_indicators:
        raise ValueError(
            f"Missing required indicators for ghost signal detection: {missing_indicators}\n"
            f"Run scripts/05_add_shared_features.py first!"
        )

    # 1. Calculate target (PRIMARY - uses FUTURE data + ghost signal detection)
    df = calculate_target(df, lookahead_periods=lookahead_periods, min_flips=min_flips)

    # 2. Add statistical features (optional supplementary features)
    df = add_hurst_exponent(df)
    df = add_permutation_entropy(df)
    df = add_cusum_signal(df)

    # Fill any remaining NaNs
    for col in TRAINING_ONLY_FEATURE_LIST:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
