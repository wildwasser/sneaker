"""Sneaker: Clean cryptocurrency reversal prediction system.

Extracted from Ghost project with only the working V3 approach.
"""

__version__ = "1.0.0"
__author__ = "Extracted from Ghost Trader"

from .logging import setup_logger
from .data import (
    download_live_data,
    download_multiple_pairs,
    download_historical_data,
    BASELINE_PAIRS
)
from .indicators import add_core_indicators
from .features import add_all_features
from .features_training import add_all_training_features, TRAINING_ONLY_FEATURE_LIST
from .model import load_model, predict, generate_signals

__all__ = [
    'setup_logger',
    'download_live_data',
    'download_multiple_pairs',
    'download_historical_data',
    'BASELINE_PAIRS',
    'add_core_indicators',
    'add_all_features',
    'add_all_training_features',
    'TRAINING_ONLY_FEATURE_LIST',
    'load_model',
    'predict',
    'generate_signals'
]
