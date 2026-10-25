"""Application factory: wiring of settings, pipeline, storage, metrics, auth and routes.

A factory rather than a module-level app lets tests build isolated instances with an
in-memory database and a shared OCR engine, and lets the CLI pick the port.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from ledgerlens import __version__
from ledgerlens.api.logging import configure_logging, request_id_var
from ledgerlens.api.metrics import Metrics
from ledgerlens.api.routes import router
from ledgerlens.api.storage import Storage
from ledgerlens.config import Settings, get_settings
from ledgerlens.pipeline.ocr.base import OCREngine
from ledgerlens.pipeline.ocr.factory import get_engine
from ledgerlens.pipeline.run import Pipeline, Reviewer

STATIC_DIR = Path(__file__).parent / "static"
OPEN_PATHS = frozenset({"/health", "/metrics", "/", "/docs", "/openapi.json", "/redoc"})


def create_app(
    settings: Settings | None = None, engine: OCREngine | None = None, storage: Storage | None = None
) -> FastAPI:
    """Build the service; every dependency can be injected for tests."""
    settings = settings or get_settings()
    logger = configure_logging(settings.log_level)
    app = FastAPI(
        title="ledgerlens",
        version=__version__,
        description="Document intelligence for invoices, receipts and bank statements.",
    )
    app.state.settings = settings
    app.state.pipeline = Pipeline(
        engine or get_engine(settings.ocr_engine, str(settings.atlas_dir)), settings.threshold, build_reviewer(settings)
    )
    app.state.storage = storage or Storage(settings.db_path)
    app.state.metrics = Metrics()
    app.state.logger = logger
    app.include_router(router)
    _install_middleware(app)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def build_reviewer(settings: Settings) -> Reviewer | None:
    """The optional Claude reviewer, only when LEDGERLENS_LLM=1; the SDK resolves its own credentials."""
    if not settings.llm_enabled:
        return None
    from ledgerlens.llm.anthropic_adapter import ClaudeReviewer

    return ClaudeReviewer(model=settings.llm_model)


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        try:
            denial = _check_api_key(request)
            response = denial if denial is not None else await call_next(request)
        finally:
            request_id_var.reset(token)
        elapsed = time.perf_counter() - started
        path = request.url.path if request.url.path.startswith(("/v1", "/health", "/metrics")) else "other"
        metrics: Metrics = app.state.metrics
        metrics.requests.labels(request.method, path, str(response.status_code)).inc()
        metrics.request_latency.labels(path).observe(elapsed)
        app.state.logger.info(
            "request",
            extra={
                "data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                    "request_id": request_id,
                }
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response


def _check_api_key(request: Request) -> Response | None:
    """Deny protected paths without the configured key; open paths and unset keys pass."""
    expected = request.app.state.settings.api_key
    if not expected or request.url.path in OPEN_PATHS:
        return None
    provided = request.headers.get("x-api-key", "")
    if secrets.compare_digest(provided.encode(), expected.encode()):
        return None
    return JSONResponse({"detail": "invalid or missing API key"}, status_code=401)


def http_error(status: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status, detail=detail)
