"""Environment-backed settings for the TraceRAG service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping


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

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from an environment mapping without loading secret files."""

        source = os.environ if environment is None else environment
        port_value = source.get(f"{_ENV_PREFIX}PORT", str(cls.port))

        try:
            port = int(port_value)
        except ValueError as exc:
            raise ValueError(f"{_ENV_PREFIX}PORT must be an integer.") from exc

        if not 1 <= port <= 65535:
            raise ValueError(f"{_ENV_PREFIX}PORT must be between 1 and 65535.")

        return cls(
            app_name=source.get(f"{_ENV_PREFIX}APP_NAME", cls.app_name),
            environment=source.get(f"{_ENV_PREFIX}ENVIRONMENT", cls.environment),
            version=source.get(f"{_ENV_PREFIX}VERSION", cls.version),
            host=source.get(f"{_ENV_PREFIX}HOST", cls.host),
            port=port,
            log_level=source.get(f"{_ENV_PREFIX}LOG_LEVEL", cls.log_level).upper(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings for application startup."""

    return Settings.from_environment()
