"""创建 FastAPI 应用、健康检查和服务生命周期钩子。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.api import RuntimeFactory, create_api_router

from app.config import Settings, get_settings
from app.logging_config import configure_logging

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """健康检查返回的服务状态和版本信息。"""

    status: str
    service: str
    environment: str
    version: str


def create_app(
    settings: Settings | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> FastAPI:
    """使用指定配置创建服务，并注册系统及 RAG 路由。"""

    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """在服务启停时统一初始化日志并留下生命周期记录。"""
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
    application.include_router(
        create_api_router(resolved_settings, runtime_factory=runtime_factory)
    )
    application.state.settings = resolved_settings

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """返回最小健康信息，不在此处加载模型或 API 凭据。"""

        logger.debug("Health check requested")
        return HealthResponse(
            status="ok",
            service=resolved_settings.app_name,
            environment=resolved_settings.environment,
            version=resolved_settings.version,
        )

    return application


app = create_app()
