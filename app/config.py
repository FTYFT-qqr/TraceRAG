"""Environment-backed settings for the TraceRAG service."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

from dotenv import load_dotenv


_ENV_PREFIX = "TRACERAG_"


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings sourced from environment variables."""

    app_name: str = "TraceRAG"
    environment: str = "development"
    version: str = "0.1.0"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    api_key: str | None = None
    base_url: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    index_dir: str = "data/indexes/default"
    reject_threshold: float = 0.25

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from an environment mapping without loading secret files."""

        source = os.environ if environment is None else environment
        defaults = cls()
        port_value = source.get(f"{_ENV_PREFIX}PORT", str(defaults.port))

        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError(f"{_ENV_PREFIX}PORT must be an integer.") from exc

        if not 1 <= port <= 65535:
            raise ValueError(f"{_ENV_PREFIX}PORT must be between 1 and 65535.")
        threshold_value = source.get(
            f"{_ENV_PREFIX}REJECT_THRESHOLD", str(defaults.reject_threshold)
        )
        try:
            reject_threshold = float(threshold_value)
        except ValueError as exc:
            raise ValueError(
                f"{_ENV_PREFIX}REJECT_THRESHOLD must be a number between -1 and 1."
            ) from exc
        if not -1.0 <= reject_threshold <= 1.0:
            raise ValueError(
                f"{_ENV_PREFIX}REJECT_THRESHOLD must be between -1 and 1."
            )


        return cls(
            app_name=source.get(f"{_ENV_PREFIX}APP_NAME", defaults.app_name),
            environment=source.get(f"{_ENV_PREFIX}ENVIRONMENT", defaults.environment),
            version=source.get(f"{_ENV_PREFIX}VERSION", defaults.version),
            host=source.get(f"{_ENV_PREFIX}HOST", defaults.host),
            port=port,
            api_key=(source.get(f"{_ENV_PREFIX}API_KEY") or source.get("OPENAI_API_KEY") or None),
            base_url=(source.get(f"{_ENV_PREFIX}BASE_URL") or None),
            embedding_model=source.get(
                f"{_ENV_PREFIX}EMBEDDING_MODEL", defaults.embedding_model
            ),
            chat_model=source.get(f"{_ENV_PREFIX}CHAT_MODEL", defaults.chat_model),
            index_dir=source.get(f"{_ENV_PREFIX}INDEX_DIR", defaults.index_dir),
            reject_threshold=reject_threshold,
            log_level=source.get(f"{_ENV_PREFIX}LOG_LEVEL", defaults.log_level).upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings for application startup."""
    load_dotenv(Path.cwd() / ".env", override=False)

    return Settings.from_environment()
