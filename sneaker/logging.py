"""Logging utilities for Sneaker.

Provides consistent logging across all scripts with timestamped log files.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import os


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with both file and console output.

    Args:
        name: Logger name (typically script name)
        log_dir: Directory for log files (default: "logs/")
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Example:
        >>> from sneaker.logging import setup_logger
        >>> logger = setup_logger('train_model')
        >>> logger.info("Starting training...")
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create log filename: YYYY-MM-DD_HH-MM-SS_name_PID.log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid = os.getpid()
    log_file = log_path / f"{timestamp}_{name}_{pid}.log"

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear existing handlers

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initialization
    logger.info("="*60)
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Process ID: {pid}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*60)

    return logger
