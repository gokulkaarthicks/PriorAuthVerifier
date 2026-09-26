"""Health and readiness HTTP endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


def _settings(request: Request) -> Settings:
    stored = getattr(request.app.state, "settings", None)
    if isinstance(stored, Settings):
        return stored
    return get_settings()


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    """Return service health status without loading models."""
    settings = _settings(request)
    return {
        "status": "ok",
        "service": settings.APP_NAME,
    }


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    """Validate required runtime configuration without inference."""
    settings = _settings(request)
    missing: list[str] = []
    if not (settings.OPENAI_API_KEY or "").strip():
        missing.append("OPENAI_API_KEY")

    if missing:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": settings.APP_NAME,
                "missing": missing,
            },
        )
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "service": settings.APP_NAME,
        },
    )
