"""Loopback-only job service for Folio; no paid API or database access."""
from __future__ import annotations
import os
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from research.engine import DEFAULT_SYMBOLS, VERSION, load_json, run_research

ROOT = Path(os.environ.get("FOLIO_RESEARCH_DIR", str(Path.home() / ".local/share/folio-research")))
ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
TOKEN = secrets.token_urlsafe(32)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Folio-Token"])
lock = threading.Lock()
state = {"status": "idle", "progress": 0, "message": "Ready", "error": None}


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbols: list[str] = Field(default_factory=lambda: DEFAULT_SYMBOLS.copy(), min_length=4, max_length=24)
    names: dict[str, str] = Field(default_factory=dict, max_length=24)
    years: int = Field(default=3, ge=3, le=5)
    topK: int = Field(default=3, ge=1, le=10)
    costBps: float = Field(default=20, ge=0, le=100, allow_inf_nan=False)
    includeNews: bool = True

    @field_validator("symbols")
    @classmethod
    def check_symbols(cls, values):
        import re
        if len(set(values)) != len(values) or any(not re.fullmatch(r"[A-Z0-9&_-]{1,25}\.(NS|BO)", s) for s in values):
            raise ValueError("Use distinct NSE/BSE symbols such as RELIANCE.NS.")
        return values

    @field_validator("names")
    @classmethod
    def check_names(cls, values):
        if any(len(v) > 160 or not v.strip() for v in values.values()):
            raise ValueError("Company names must be between 1 and 160 characters.")
        return values


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path == "/health":
        return JSONResponse({"ok": True, "version": VERSION})
    if request.headers.get("origin") not in ORIGINS:
        return JSONResponse({"error": "Only the local Folio dashboard can access this service."}, status_code=403)
    if request.url.path != "/session" and not secrets.compare_digest(request.headers.get("x-folio-token", ""), TOKEN):
        return JSONResponse({"error": "Reconnect to the local research service."}, status_code=401)
    length = request.headers.get("content-length", "0")
    if not length.isdecimal() or int(length) > 8192:
        return JSONResponse({"error": "Request too large."}, status_code=413)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    # Rejections also need explicit CORS so the dashboard can explain them.
    response.headers["Access-Control-Allow-Origin"] = request.headers["origin"]
    return response


@app.get("/session")
def session():
    return {"token": TOKEN, "version": VERSION}


@app.get("/status")
def status():
    with lock:
        return dict(state)


@app.get("/report")
def report():
    return {"report": load_json(ROOT / "latest.json", None)}


@app.post("/run", status_code=202)
def run(config: RunConfig):
    with lock:
        if state["status"] == "running":
            return JSONResponse({"error": "A research run is already in progress."}, status_code=409)
        state.update(status="running", progress=0, message="Starting research", error=None)

    def progress(message, percent):
        with lock:
            state.update(message=message, progress=percent)

    def job():
        try:
            run_research(config.model_dump(), ROOT, progress)
            with lock:
                state.update(status="complete", progress=100, message="Research complete")
        except Exception as e:
            with lock:
                state.update(status="error", message="Research could not finish", error=str(e)[:400])

    threading.Thread(target=job, daemon=True).start()
    return {"status": "running"}
