"""
Centralized structured logger for the Fintech Analytics platform.
Provides formatted console logging and persistent file logging.
"""

import logging
import sys
from pathlib import Path
from src.config import LOGS_DIR

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = LOGS_DIR / "fintech_platform.log"


def setup_logger(name: str = "fintech_analytics", log_file: Path = DEFAULT_LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a thread-safe logger with console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # Persistent File Handler
        try:
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except (OSError, PermissionError):
            pass

    return logger
