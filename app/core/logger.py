import logging
import sys
from typing import Any, Dict, List

# Configure root logger
logger = logging.getLogger("ecobrandai")
logger.setLevel(logging.INFO)

# Add console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name that inherits from the root logger.

    Args:
        name: The name for the logger, typically __name__

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
