"""The 500 envelope must leave the server WITH its CORS headers.

`@app.exception_handler(Exception)` runs in Starlette's outermost
ServerErrorMiddleware, outside CORSMiddleware, so a 500 carried no
`access-control-allow-origin` and the browser app read every crash as
"Couldn't reach the server" (audit 2026-09-02, verified live). The fix is a
pure-ASGI catch-all added INSIDE CORS. This test builds a throwaway app the
same way `main.py` does and sends the request a browser would.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app.errors import ApiError, InternalErrorEnvelope, register_error_handlers

ORIGIN = "http://127.0.0.1:5000"


def _app() -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("kaboom")

    @app.get("/conflict")
    async def conflict() -> dict:
        raise ApiError(409, "conflict", "That already exists.")

    @app.get("/ok")
    async def ok() -> dict:
        return {"status": "ok"}

    # Same order as main.py: envelope first, CORS last (outermost).
    app.add_middleware(InternalErrorEnvelope)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def test_500_carries_cors_headers_and_the_envelope():
    client = TestClient(_app(), raise_server_exceptions=False)
    r = client.get("/boom", headers={"Origin": ORIGIN})
    assert r.status_code == 500
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    assert r.json() == {
        "error": {
            "code": "internal_error",
            "message": "Something went wrong on our side. Please try again.",
        }
    }


def test_api_error_path_unchanged():
    client = TestClient(_app(), raise_server_exceptions=False)
    r = client.get("/conflict", headers={"Origin": ORIGIN})
    assert r.status_code == 409
    assert r.headers.get("access-control-allow-origin") == ORIGIN
    assert r.json()["error"]["code"] == "conflict"


def test_success_path_untouched():
    client = TestClient(_app())
    r = client.get("/ok", headers={"Origin": ORIGIN})
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
