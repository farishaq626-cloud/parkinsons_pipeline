"""Shared console and file logging configuration for the PPMI pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from config import DEFAULT_LOG_LEVEL, PIPELINE_LOG_PATH


def configure_logging(
    log_path: str | Path = PIPELINE_LOG_PATH,
    level: str = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """Configure project logging for the console and a persistent log file.

    Args:
        log_path: Destination for the persistent pipeline log.
        level: Logging level name, such as ``"INFO"`` or ``"DEBUG"``.

    Returns:
        The configured root ``ppmi_pipeline`` logger.

    Raises:
        ValueError: If ``level`` is not a valid logging-level name.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level!r}.")

    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppmi_pipeline")
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
