"""FastAPI entry point for TraceRAG."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Response returned by the service health endpoint."""

    status: str
    service: str
    environment: str
    version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application with its stage-one health endpoint."""

    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging(resolved_settings.log_level)
        logger.info(
            "TraceRAG service started (environment=%s, version=%s)",
            resolved_settings.environment,
            resolved_settings.version,
        )
        yield
        logger.info("TraceRAG service stopped")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.version,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Return the minimal process health information required for stage one."""

        logger.debug("Health check requested")
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            environment=resolved_settings.environment,
            version=resolved_settings.version,
        )

    return application


app = create_app()
