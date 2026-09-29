"""Application logging."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from utils.paths import ensure_dirs, project_path


def setup_logging() -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger("vision_studio")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        project_path("logs", "app.log"),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream)
    logger.propagate = False
    for name in ("mmengine", "mmcv", "mmdet"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logger


def get_logger(name: str = "vision_studio") -> logging.Logger:
    return logging.getLogger(name)
