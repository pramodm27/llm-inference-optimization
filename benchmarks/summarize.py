"""Combine results/*.json into a Markdown comparison table.

Usage:
    python -m benchmarks.summarize
    python -m benchmarks.summarize --workload random --input-tokens 512 --out docs/results.md

Compares each profile at a chosen input length and concurrency, and computes
speedups relative to the fp16_baseline profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_runs(workload: str) -> dict[str, dict]:
    runs = {}
    for path in sorted(RESULTS_DIR.glob(f"*_{workload}.json")):
        data = json.loads(path.read_text())
        runs[data["profile"]] = data
    return runs


def pick_row(report: dict, input_tokens: int, concurrency: int) -> dict | None:
    for row in report["runs"]:
        if row["input_tokens"] == input_tokens and row["concurrency"] == concurrency:
            return row
    return None


def fmt(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def build_table(runs: dict[str, dict], input_tokens: int, concurrency: int) -> str:
    baseline = runs.get("fp16_baseline")
    base_row = pick_row(baseline, input_tokens, concurrency) if baseline else None

    header = (
        "| Profile | TTFT P50 (ms) | TTFT P95 (ms) | ITL mean (ms) | E2E P95 (ms) | "
        "Throughput (tok/s) | Throughput vs baseline |"
    )
    lines = [header, "|" + "---|" * 7]
    for name, report in runs.items():
        row = pick_row(report, input_tokens, concurrency)
        if row is None:
            continue
        speedup = "n/a"
        if base_row and base_row["throughput_tok_s"]:
            speedup = f"{row['throughput_tok_s'] / base_row['throughput_tok_s']:.2f}x"
        lines.append(
            f"| {name} | {fmt(row['ttft_p50_ms'])} | {fmt(row['ttft_p95_ms'])} | "
            f"{fmt(row['itl_mean_ms'])} | {fmt(row['e2e_p95_ms'])} | "
            f"{fmt(row['throughput_tok_s'])} | {speedup} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workload", default="random", choices=["random", "shared_prefix"])
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "results.md")
    args = parser.parse_args()

    runs = load_runs(args.workload)
    if not runs:
        raise SystemExit(f"No results found in {RESULTS_DIR} for workload '{args.workload}'")

    table = build_table(runs, args.input_tokens, args.concurrency)
    title = (
        f"# Benchmark results: {args.workload}, input {args.input_tokens} tokens, "
        f"concurrency {args.concurrency}\n\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(title + table + "\n")
    print(title + table)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
