"""Sweep concurrency and input lengths against a running vLLM server.

Usage:
    python -m benchmarks.run_benchmark --profile fp16_tuned --port 8000
    python -m benchmarks.run_benchmark --profile fp8 --port 8001 --workload shared_prefix

Writes one JSON file per run to results/<profile>_<workload>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx
import yaml

from benchmarks.client import RequestResult, stream_completion

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "serving.yaml"
RESULTS_DIR = ROOT / "results"

# Filler text used to build prompts of a target token count (approx. 1 word ~ 1.3 tokens).
FILLER = "The quick brown fox jumps over the lazy dog while the engineer reviews the logs. "


def make_prompt(target_tokens: int, unique_tag: str = "") -> str:
    words_needed = int(target_tokens / 1.3)
    body = (FILLER * (words_needed // len(FILLER.split()) + 1)).split()[:words_needed]
    return f"[{unique_tag}] " + " ".join(body) + "\nSummarize the text above in one sentence."


def make_shared_prefix_prompt(prefix: str, index: int) -> str:
    # Identical prefix across requests lets vLLM reuse cached KV blocks.
    return f"{prefix}\n\nQuestion {index}: What is item {index} in the list?"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[k]


async def run_level(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        async def one(prompt: str) -> None:
            async with sem:
                results.append(await stream_completion(client, base_url, model, prompt, max_tokens))

        wall_start = time.perf_counter()
        await asyncio.gather(*(one(p) for p in prompts))
        wall = time.perf_counter() - wall_start

    ok = [r for r in results if r.ok]
    errors = len(results) - len(ok)
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    itls = [r.itl_s for r in ok if r.itl_s is not None]
    e2es = [r.e2e_s for r in ok if r.e2e_s is not None]
    total_out = sum(r.output_tokens for r in ok)

    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors,
        "ttft_p50_ms": round(statistics.median(ttfts) * 1000, 1) if ttfts else None,
        "ttft_p95_ms": round(percentile(ttfts, 95) * 1000, 1) if ttfts else None,
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 2) if itls else None,
        "e2e_p95_ms": round(percentile(e2es, 95) * 1000, 1) if e2es else None,
        "throughput_tok_s": round(total_out / wall, 1) if wall > 0 else None,
        "request_throughput_rps": round(len(ok) / wall, 2) if wall > 0 else None,
    }


def build_prompts(workload: str, count: int, input_len: int, shared_prefix_tokens: int) -> list[str]:
    if workload == "random":
        return [make_prompt(input_len, unique_tag=f"req{i}") for i in range(count)]
    if workload == "shared_prefix":
        prefix = make_prompt(shared_prefix_tokens, unique_tag="system")
        return [make_shared_prefix_prompt(prefix, i) for i in range(count)]
    raise ValueError(f"Unknown workload: {workload}")


async def main_async(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    bench = cfg["benchmark"]
    base_url = f"http://{args.host}:{args.port}"
    model = cfg["model"]["awq_name"] if args.profile == "awq_int4" else cfg["model"]["name"]

    # Warmup so CUDA graphs, caches, and the scheduler are steady before measuring.
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        warm_prompts = build_prompts(args.workload, bench["warmup_requests"], 512, bench["shared_prefix_tokens"])
        await asyncio.gather(*(
            stream_completion(client, base_url, model, p, bench["output_length"]) for p in warm_prompts
        ))

    report = {
        "profile": args.profile,
        "workload": args.workload,
        "model": model,
        "output_length": bench["output_length"],
        "runs": [],
    }

    input_lengths = bench["input_lengths"] if args.workload == "random" else [bench["shared_prefix_tokens"]]
    for input_len in input_lengths:
        for concurrency in bench["concurrency_levels"]:
            prompts = build_prompts(args.workload, bench["requests_per_level"], input_len, bench["shared_prefix_tokens"])
            print(f"[{args.profile}] input={input_len} concurrency={concurrency} ...", flush=True)
            row = await run_level(base_url, model, prompts, bench["output_length"], concurrency)
            row["input_tokens"] = input_len
            report["runs"].append(row)
            print(json.dumps(row), flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{args.profile}_{args.workload}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workload", choices=["random", "shared_prefix"], default="random")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
