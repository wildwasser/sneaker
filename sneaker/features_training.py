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

def calculate_target(df: pd.DataFrame, lookahead_periods: int = 4) -> pd.DataFrame:
    """
    Calculate volatility-normalized future price change (PRIMARY TARGET).

    This is the MAIN training-only feature - what we're teaching the model to predict.
    Uses FUTURE data (lookahead), so CANNOT be used in live prediction.

    Algorithm:
    1. For each candle, look ahead N periods
    2. Calculate percent price change: (future_close - current_close) / current_close
    3. Normalize by volatility: target = price_change / volatility_std
    4. Result: Target in σ (sigma) units

    Args:
        df: DataFrame with OHLCV data and 'pair' column
        lookahead_periods: Hours into future to look (default: 4H)

    Returns:
        DataFrame with 'target' column added (in σ units)

    Target Interpretation:
        - target = 0: Normal candle (no significant reversal)
        - target > 0: Upward reversal (buy signal)
        - target < 0: Downward reversal (sell signal)
        - |target| > 4σ: Strong signal (typical threshold)
    """
    df['target'] = 0.0

    for pair in df['pair'].unique():
        mask = df['pair'] == pair
        closes = df.loc[mask, 'close'].values

        # Calculate future price change (LOOK-AHEAD!)
        future_change = np.zeros(len(closes))
        for i in range(len(closes) - lookahead_periods):
            future_close = closes[i + lookahead_periods]
            current_close = closes[i]
            pct_change = (future_close - current_close) / current_close * 100
            future_change[i] = pct_change

        # Calculate volatility (20-period rolling std of returns)
        returns = pd.Series(closes).pct_change() * 100
        volatility = returns.rolling(20).std()

        # Normalize: target = future_change / volatility
        # This converts raw price changes to σ units
        normalized = future_change / volatility.values
        normalized = np.nan_to_num(normalized, 0)

        df.loc[mask, 'target'] = normalized

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

def add_all_training_features(df: pd.DataFrame, lookahead_periods: int = 4) -> pd.DataFrame:
    """
    Add all training-only features to DataFrame.

    ⚠️  WARNING: USES FUTURE DATA - TRAINING ONLY!

    REQUIRES:
        - df must already have OHLCV data
        - df must have 'pair' column
        - Ideally should have shared features from Issue #6 (but not required)

    Args:
        df: DataFrame with OHLCV + shared features
        lookahead_periods: Hours into future for target calculation (default: 4)

    Returns:
        DataFrame with 4 additional training-only features:
        - target (PRIMARY - future price change, σ normalized)
        - hurst_exponent (trend persistence)
        - permutation_entropy (predictability)
        - cusum_signal (change detection)

    Features added:
    - 1 target variable (PRIMARY)
    - 3 statistical features
    Total: 4 training-only features
    """
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close', 'volume', 'pair']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 1. Calculate target (PRIMARY - uses FUTURE data)
    df = calculate_target(df, lookahead_periods=lookahead_periods)

    # 2. Add statistical features
    df = add_hurst_exponent(df)
    df = add_permutation_entropy(df)
    df = add_cusum_signal(df)

    # Fill any remaining NaNs
    for col in TRAINING_ONLY_FEATURE_LIST:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
