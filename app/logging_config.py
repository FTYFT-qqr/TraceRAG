"""Logging setup shared by application entry points."""

from __future__ import annotations

import logging


def configure_logging(level_name: str) -> None:
    """Configure concise, timestamped process logging."""

    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
