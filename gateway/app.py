"""Async gateway in front of one or more vLLM backends.

Responsibilities:
  - Accept chat requests and stream responses back to the client
  - Limit concurrent requests per backend (admission control)
  - Cancel the upstream request when the client disconnects, so the
    server frees its KV-cache slot immediately
  - Expose a health endpoint that reports backend reachability

Run:
    VLLM_BACKENDS=http://127.0.0.1:8000 uvicorn gateway.app:app --port 9000
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BACKENDS = [b.strip() for b in os.environ.get("VLLM_BACKENDS", "http://127.0.0.1:8000").split(",") if b.strip()]
MAX_INFLIGHT_PER_BACKEND = int(os.environ.get("MAX_INFLIGHT_PER_BACKEND", "64"))
UPSTREAM_TIMEOUT_S = float(os.environ.get("UPSTREAM_TIMEOUT_S", "600"))

app = FastAPI(title="LLM inference gateway")
_semaphores = {b: asyncio.Semaphore(MAX_INFLIGHT_PER_BACKEND) for b in BACKENDS}
_next_backend = 0
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_TIMEOUT_S))


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _client is not None:
        await _client.aclose()


def _pick_backend() -> str:
    # Round-robin across backends. Swap for least-loaded routing when you have metrics.
    global _next_backend
    backend = BACKENDS[_next_backend % len(BACKENDS)]
    _next_backend += 1
    return backend


async def _stream_upstream(request: Request, backend: str, body: dict) -> AsyncIterator[bytes]:
    sem = _semaphores[backend]
    async with sem:
        assert _client is not None
        async with _client.stream("POST", f"{backend}/v1/chat/completions", json=body) as upstream:
            async for chunk in upstream.aiter_raw():
                if await request.is_disconnected():
                    # Closing the upstream stream lets vLLM abort the request and free its blocks.
                    break
                yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    body["stream"] = True  # the gateway always streams; clients can buffer if they need a single payload
    backend = _pick_backend()
    return StreamingResponse(
        _stream_upstream(request, backend, body),
        media_type="text/event-stream",
        headers={"X-Backend": backend},
    )


@app.get("/health")
async def health() -> JSONResponse:
    assert _client is not None
    status: dict[str, str] = {}
    for backend in BACKENDS:
        try:
            resp = await _client.get(f"{backend}/health", timeout=2.0)
            status[backend] = "ok" if resp.status_code == 200 else f"http {resp.status_code}"
        except httpx.HTTPError as exc:
            status[backend] = f"down: {type(exc).__name__}"
    healthy = all(v == "ok" for v in status.values())
    return JSONResponse({"healthy": healthy, "backends": status}, status_code=200 if healthy else 503)
