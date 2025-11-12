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
from .macro import (
    download_macro_data,
    download_ticker,
    resample_to_1h,
    MACRO_TICKERS
)
from .indicators import add_core_indicators
from .features import add_all_features
from .model import load_model, predict, generate_signals

__all__ = [
    'setup_logger',
    'download_live_data',
    'download_multiple_pairs',
    'download_historical_data',
    'BASELINE_PAIRS',
    'download_macro_data',
    'download_ticker',
    'resample_to_1h',
    'MACRO_TICKERS',
    'add_core_indicators',
    'add_all_features',
    'load_model',
    'predict',
    'generate_signals'
]
