"""Launch a vLLM OpenAI-compatible server for one serving profile.

Usage:
    python -m serving.launch_vllm --profile fp16_tuned
    python -m serving.launch_vllm --profile awq_int4 --port 8001

The profile values come from configs/serving.yaml so every benchmark run
is reproducible from a single file.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "serving.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def build_command(profile_name: str, cfg: dict, port: int) -> list[str]:
    profile = cfg["profiles"][profile_name]
    model = cfg["model"]["awq_name"] if profile.get("quantization") == "awq" else cfg["model"]["name"]

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", str(profile["dtype"]),
        "--gpu-memory-utilization", str(cfg["hardware"]["gpu_memory_utilization"]),
        "--max-num-seqs", str(profile["max_num_seqs"]),
    ]

    if profile.get("quantization"):
        cmd += ["--quantization", profile["quantization"]]
    if profile.get("enable_prefix_caching"):
        cmd += ["--enable-prefix-caching"]
    if profile.get("enable_chunked_prefill"):
        cmd += ["--enable-chunked-prefill"]
    if profile.get("max_num_batched_tokens"):
        cmd += ["--max-num-batched-tokens", str(profile["max_num_batched_tokens"])]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", required=True, help="Profile name from configs/serving.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without starting the server")
    args = parser.parse_args()

    cfg = load_config()
    if args.profile not in cfg["profiles"]:
        raise SystemExit(f"Unknown profile '{args.profile}'. Choices: {', '.join(cfg['profiles'])}")

    cmd = build_command(args.profile, cfg, args.port)
    print("Launching:", shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
