"""
Logging configuration for the RAG pipeline.
"""
import logging
import sys
from typing import Optional

from src.core.config import get_settings


def setup_logging(level: Optional[str] = None) -> None:
    """Setup logging configuration."""
    settings = get_settings()

    if level is None:
        level = "DEBUG" if settings.DEBUG else "INFO"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    if not logging.getLogger().handlers:
        setup_logging()

    return logging.getLogger(name)