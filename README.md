# LLM Inference Optimization

A benchmark and serving project for optimizing LLM inference with vLLM on NVIDIA GPUs. It measures the effect of continuous batching, PagedAttention-based KV-cache management, prefix caching, chunked prefill, and FP8 and INT4 (AWQ) quantization on TTFT, inter-token latency, P95 latency, throughput, and output quality.

## What's included

| Area | Files | What it does |
|---|---|---|
| Serving profiles | `configs/serving.yaml`, `serving/launch_vllm.py` | Defines FP16, tuned FP16, FP8, and AWQ INT4 profiles and launches vLLM for each |
| Benchmark harness | `benchmarks/client.py`, `benchmarks/run_benchmark.py` | Streams requests at several concurrency levels; records TTFT, ITL, P95, throughput |
| Shared-prefix workload | `benchmarks/run_benchmark.py --workload shared_prefix` | Tests prefix caching on identical system context |
| Quality check | `quantization/eval_quality.py` | Scores each profile on a fixed MCQ set to measure quantization cost |
| GPU profiling | `profiling/profile_nsight.sh` | Records Nsight Systems timelines and Nsight Compute kernel metrics |
| Gateway | `gateway/app.py` | Async FastAPI proxy with admission control and cancel-on-disconnect |
| Reporting | `benchmarks/summarize.py` | Builds a comparison table with speedups vs. FP16 baseline |
| Tests | `tests/test_core.py` | CPU-only tests for prompt construction, percentiles, and launch flags |

## Setup

Benchmarks need an NVIDIA GPU. The reference setup is one H100 80GB. Tests run on any machine.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # CPU-only, runs anywhere
```

Log in to Hugging Face and accept the Llama 3.1 license before downloading weights:

```bash
huggingface-cli login
```

## Running the benchmark

Start one profile at a time on the GPU, benchmark it, then stop it before starting the next.

```bash
# Terminal 1: start a profile
python -m serving.launch_vllm --profile fp16_baseline --port 8000

# Terminal 2: benchmark it
python -m benchmarks.run_benchmark --profile fp16_baseline --port 8000 --workload random
python -m benchmarks.run_benchmark --profile fp16_baseline --port 8000 --workload shared_prefix
python -m quantization.eval_quality --name fp16_baseline --port 8000
```

Repeat for `fp16_tuned`, `fp8`, and `awq_int4`. Then summarize:

```bash
python -m benchmarks.summarize --workload random --input-tokens 512 --concurrency 32
python -m benchmarks.summarize --workload random --input-tokens 2048 --concurrency 32
```

Results land in `results/` as JSON and in `docs/results.md` as a Markdown table.

## Profiling

With a server running on a profile:

```bash
bash profiling/profile_nsight.sh fp16_tuned 8000
```

Open `results/profiles/fp16_tuned/timeline.nsys-rep` in Nsight Systems to see prefill and decode phases under load. Use the Nsight Compute output to check DRAM throughput and tensor-core utilization for the decode kernels.

## Gateway

```bash
VLLM_BACKENDS=http://127.0.0.1:8000 uvicorn gateway.app:app --port 9000
curl -N http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'
```

## Methodology

Every run uses the same model, output length (128 tokens), concurrency sweep (1, 8, 32, 64), and warmup. Requests use `temperature=0` and `ignore_eos=True` so output length is fixed and runs are comparable. Quality is measured on the same MCQ set for every profile. Keep the vLLM version pinned in `requirements.txt`, because scheduler defaults change between releases.

## Caveats

The included MCQ set is a small sample for demonstration. For a claim you'd make on a resume, replace it with a larger domain set or an MMLU subset, and report the sample size and confidence interval. Numbers depend on hardware, driver, and vLLM version, so record those alongside every result.
