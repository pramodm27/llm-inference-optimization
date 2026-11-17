"""Async streaming client for OpenAI-compatible endpoints (vLLM).

Measures per-request:
  - TTFT: time from request send to first streamed token
  - ITL: mean gap between subsequent tokens
  - end-to-end latency
  - output token count
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    ok: bool
    ttft_s: float | None = None
    itl_s: float | None = None
    e2e_s: float | None = None
    output_tokens: int = 0
    error: str | None = None
    token_times: list[float] = field(default_factory=list, repr=False)


async def stream_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,  # force full output length so runs are comparable
    }

    start = time.perf_counter()
    token_times: list[float] = []
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return RequestResult(ok=False, error=f"HTTP {resp.status_code}: {body[:200]!r}")

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    token_times.append(time.perf_counter())
    except httpx.HTTPError as exc:
        return RequestResult(ok=False, error=str(exc))

    end = time.perf_counter()
    if not token_times:
        return RequestResult(ok=False, error="no tokens streamed")

    ttft = token_times[0] - start
    gaps = [b - a for a, b in zip(token_times, token_times[1:])]
    itl = sum(gaps) / len(gaps) if gaps else 0.0
    return RequestResult(
        ok=True,
        ttft_s=ttft,
        itl_s=itl,
        e2e_s=end - start,
        output_tokens=len(token_times),
        token_times=token_times,
    )
