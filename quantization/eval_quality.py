"""Measure output-quality drift across quantization profiles.

Sends a fixed set of multiple-choice questions to each running server with
temperature 0 and scores exact-match answers. Compare each quantized profile
against the FP16 baseline; the delta is the quality cost of the optimization.

Usage:
    python -m quantization.eval_quality --port 8000 --name fp16_tuned
    python -m quantization.eval_quality --port 8001 --name awq_int4

Writes results/quality_<name>.json. Replace data/mcq_eval.jsonl with a
domain eval set (or an MMLU subset) for a stronger claim.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "quantization" / "data" / "mcq_eval.jsonl"
RESULTS_DIR = ROOT / "results"


def load_eval(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_letter(text: str) -> str | None:
    match = re.search(r"\b([A-D])\b", text.strip().upper())
    return match.group(1) if match else None


async def ask(client: httpx.AsyncClient, base_url: str, model: str, item: dict) -> str:
    options = "\n".join(f"{letter}. {text}" for letter, text in item["choices"].items())
    prompt = (
        f"{item['question']}\n{options}\n\n"
        "Answer with a single letter (A, B, C, or D) and nothing else."
    )
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0.0,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def main_async(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "serving.yaml").read_text())
    model = cfg["model"]["awq_name"] if args.name == "awq_int4" else cfg["model"]["name"]
    base_url = f"http://{args.host}:{args.port}"
    items = load_eval(EVAL_PATH)

    async with httpx.AsyncClient(timeout=120.0) as client:
        answers = await asyncio.gather(*(ask(client, base_url, model, it) for it in items))

    correct = 0
    details = []
    for item, raw in zip(items, answers):
        predicted = extract_letter(raw)
        is_correct = predicted == item["answer"]
        correct += int(is_correct)
        details.append({"id": item["id"], "predicted": predicted, "gold": item["answer"], "correct": is_correct})

    accuracy = correct / len(items)
    report = {"name": args.name, "n": len(items), "accuracy": round(accuracy, 4), "details": details}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"quality_{args.name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"{args.name}: accuracy={accuracy:.4f} over {len(items)} items -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Profile name, e.g. fp16_tuned or awq_int4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
