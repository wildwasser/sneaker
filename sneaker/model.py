"""Model utilities for Sneaker.

Functions for loading models and making predictions.
"""

import lightgbm as lgb
import numpy as np
from typing import Tuple, Dict


def load_model(model_path: str = 'models/production.txt') -> lgb.Booster:
    """
    Load trained LightGBM model.

    Args:
        model_path: Path to model file (default: models/production.txt)

    Returns:
        Loaded LightGBM booster

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        model = lgb.Booster(model_file=model_path)
        return model
    except Exception as e:
        raise FileNotFoundError(f"Could not load model from {model_path}: {e}")


def predict(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """
    Make predictions with loaded model.

    Args:
        model: Loaded LightGBM model
        X: Feature matrix (n_samples, 83)

    Returns:
        Predictions array (n_samples,) in sigma units
    """
    return model.predict(X)


def apply_threshold(
    predictions: np.ndarray,
    threshold: float = 4.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply threshold to generate trading signals.

    Args:
        predictions: Raw predictions (sigma units)
        threshold: Signal threshold (default: 4.0σ)

    Returns:
        Tuple of (buy_signals, sell_signals, hold_signals)
        Each is a boolean array indicating signal type
    """
    buy_signals = predictions > threshold
    sell_signals = predictions < -threshold
    hold_signals = np.abs(predictions) <= threshold

    return buy_signals, sell_signals, hold_signals


def get_signal_summary(
    predictions: np.ndarray,
    threshold: float = 4.0
) -> Dict[str, any]:
    """
    Get summary statistics of signals.

    Args:
        predictions: Raw predictions (sigma units)
        threshold: Signal threshold (default: 4.0σ)

    Returns:
        Dictionary with signal statistics
    """
    buy, sell, hold = apply_threshold(predictions, threshold)

    total = len(predictions)
    buy_count = buy.sum()
    sell_count = sell.sum()
    hold_count = hold.sum()

    return {
        'total': total,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': hold_count,
        'buy_pct': buy_count / total * 100,
        'sell_pct': sell_count / total * 100,
        'hold_pct': hold_count / total * 100,
        'signal_pct': (buy_count + sell_count) / total * 100,
        'pred_mean': predictions.mean(),
        'pred_std': predictions.std(),
        'pred_min': predictions.min(),
        'pred_max': predictions.max()
    }


def generate_signals(
    model: lgb.Booster,
    X: np.ndarray,
    threshold: float = 4.0
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Generate trading signals from features.

    Convenience function that combines prediction and threshold application.

    Args:
        model: Loaded LightGBM model
        X: Feature matrix (n_samples, 83)
        threshold: Signal threshold (default: 4.0σ)

    Returns:
        Tuple of (signals, summary):
        - signals: Array of signal codes (1=BUY, -1=SELL, 0=HOLD)
        - summary: Dictionary with signal statistics
    """
    # Make predictions
    predictions = predict(model, X)

    # Apply threshold
    buy, sell, hold = apply_threshold(predictions, threshold)

    # Create signal array
    signals = np.zeros(len(predictions))
    signals[buy] = 1
    signals[sell] = -1

    # Get summary
    summary = get_signal_summary(predictions, threshold)

    return signals, summary
