"""The single error envelope (S1-B2, communication_protocol.md §5).

Every non-2xx leaves this server as:

    {"error": {"code": "<stable_machine_string>", "message": "<layman-readable>"}}

`code` is what the UI branches on; `message` may be shown to the user verbatim
(§26 — layman's terms, always). Validation errors additionally carry `fields`
so the client can land errors at the field (additive, allowed by protocol §7).

Feature modules raise `ApiError` — never hand-build a JSONResponse.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_setup import log_event

logger = logging.getLogger("app.errors")


class ApiError(Exception):
    """An application error with a stable code and a layman-readable message."""

    def __init__(self, status_code: int, code: str, message: str, fields: list[dict] | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.fields = fields
        super().__init__(message)


def _envelope(code: str, message: str, fields: list[dict] | None = None) -> dict:
    error: dict = {"code": code, "message": message}
    if fields:
        error["fields"] = fields
    return {"error": error}


# Layman messages for the framework-raised statuses we can't phrase at the site.
_INTERNAL = ("internal_error", "Something went wrong on our side. Please try again.")


class InternalErrorEnvelope:
    """Pure-ASGI catch-all that sits INSIDE CORSMiddleware.

    `@app.exception_handler(Exception)` runs in Starlette's outermost
    ServerErrorMiddleware, which wraps CORS — so a 500 left the server with no
    `access-control-allow-origin` header, and the browser app read every
    crash as "Couldn't reach the server" (audit 2026-09-02; verified with a
    TestClient carrying an Origin header). This layer returns the same
    envelope from inside the CORS wrapper, so a browser sees the message.
    `main.py` adds it BEFORE CORSMiddleware, because add_middleware makes the
    last-added layer the outermost one. The handler below stays as the
    backstop for anything raised outside this layer.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def _send(message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception as exc:
            logger.error(
                "unhandled_error", exc_info=exc,
                extra={"event_fields": {"path": scope.get("path"), "status": 500}},
            )
            if started:
                raise  # headers already sent; nothing coherent can follow
            response = JSONResponse(status_code=500, content=_envelope(*_INTERNAL))
            await response(scope, receive, send)


_GENERIC: dict[int, tuple[str, str]] = {
    404: ("not_found", "That page or item doesn't exist."),
    405: ("method_not_allowed", "That action isn't available here."),
    401: ("unauthenticated", "You need to be signed in for this."),
    403: ("forbidden", "You don't have access to this."),
}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        log_event(
            logger, "request_failed", level=logging.WARNING,
            path=request.url.path, status=exc.status_code, code=exc.code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.fields),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            code, message = exc.detail["code"], exc.detail.get("message", "")
        else:
            code, message = _GENERIC.get(
                exc.status_code, ("error", str(exc.detail) if exc.detail else "Something went wrong.")
            )
        log_event(
            logger, "request_failed", level=logging.WARNING,
            path=request.url.path, status=exc.status_code, code=code,
        )
        return JSONResponse(status_code=exc.status_code, content=_envelope(code, message))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(p) for p in err["loc"] if p != "body"),
                # pydantic prefixes a validator's own sentence with "Value error, "
                # — a phrase for a developer, in a message the UI shows verbatim.
                "message": str(err["msg"]).removeprefix("Value error, "),
            }
            for err in exc.errors()
        ]
        log_event(
            logger, "request_failed", level=logging.WARNING,
            path=request.url.path, status=422, code="validation_error",
            fields=[f["field"] for f in fields],
        )
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "Some of the information doesn't look right.",
                fields,
            ),
        )

    @app.exception_handler(OperationalError)
    @app.exception_handler(InterfaceError)
    async def db_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
        # The database went away mid-request. Its own code, like /health
        # already speaks, so a client can tell "down" from "a bug".
        logger.error(
            "database_unavailable", exc_info=exc,
            extra={"event_fields": {"path": request.url.path, "status": 503}},
        )
        return JSONResponse(
            status_code=503,
            content=_envelope(
                "database_unavailable", "The server can't reach its database right now."
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # The failure path is the one that must log (§7).
        logger.error(
            "unhandled_error",
            exc_info=exc,
            extra={"event_fields": {"path": request.url.path, "status": 500}},
        )
        return JSONResponse(status_code=500, content=_envelope(*_INTERNAL))
