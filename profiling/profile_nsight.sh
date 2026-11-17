#!/usr/bin/env bash
# Profile a vLLM serving run with Nsight Systems (timeline) and Nsight Compute (kernel metrics).
#
# Usage:
#   bash profiling/profile_nsight.sh fp16_tuned 8000
#
# Requires: nsys and ncu on PATH (installed with the CUDA toolkit), and a running server
# launched by serving/launch_vllm.py. The script drives a short benchmark load while
# nsys records, so the timeline shows prefill and decode phases under real traffic.
#
# Outputs go to results/profiles/<profile>/ and can be opened in the Nsight GUI.

set -euo pipefail

PROFILE="${1:?profile name required, e.g. fp16_tuned}"
PORT="${2:-8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/results/profiles/$PROFILE"
mkdir -p "$OUT_DIR"

command -v nsys >/dev/null || { echo "nsys not found; install the Nsight Systems CLI"; exit 1; }

echo "[1/2] Recording timeline with Nsight Systems ..."
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=none \
  --output "$OUT_DIR/timeline" \
  --force-overwrite true \
  --duration 60 \
  bash -c "cd '$ROOT' && python -m benchmarks.run_benchmark --profile '$PROFILE' --port '$PORT' --workload random" \
  || echo "nsys exited non-zero; check $OUT_DIR for partial output"

if command -v ncu >/dev/null; then
  echo "[2/2] Collecting kernel metrics with Nsight Compute on a short window ..."
  ncu \
    --set basic \
    --target-processes all \
    --launch-count 200 \
    --export "$OUT_DIR/kernels" \
    --force-overwrite \
    python -m benchmarks.run_benchmark --profile "$PROFILE" --port "$PORT" --workload random \
    || echo "ncu exited non-zero; kernel metrics may be partial"
else
  echo "[2/2] ncu not found; skipping kernel-level metrics"
fi

echo "Done. Open $OUT_DIR/timeline.nsys-rep in Nsight Systems."
