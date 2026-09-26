"""FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.analysis_controller import router as analysis_router
from app.controllers.health_controller import router as health_router
from app.controllers.jobs_controller import router as jobs_router
from app.core.config import Settings, get_settings
from app.core.dependencies import build_services
from app.core.error_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware
from app.services.analysis_job_manager import AnalysisJobManager


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""
    resolved = settings or get_settings()
    configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.services = build_services(resolved)
        manager = AnalysisJobManager(resolved)
        application.state.job_manager = manager
        try:
            yield
        finally:
            await manager.close()

    application = FastAPI(
        lifespan=lifespan,
        title=resolved.APP_NAME,
        version="0.1.0",
        description=(
            "Prior authorization readiness analysis for patient charts "
            "and medical-necessity policies."
        ),
    )
    application.state.settings = resolved

    origins = [
        origin.strip()
        for origin in resolved.CORS_ORIGINS.split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    application.add_middleware(RequestIdMiddleware)

    register_exception_handlers(application)
    application.include_router(health_router)
    application.include_router(analysis_router)
    application.include_router(jobs_router)
    return application


app = create_app()
