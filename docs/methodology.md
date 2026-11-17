# Methodology notes

- Hardware: NVIDIA H100 80GB (single GPU). Record driver and CUDA versions for every run.
- Software: vLLM pinned in requirements.txt. Record the exact version with each result.
- Load generation: async streaming client with a semaphore for concurrency control.
- Metrics: TTFT (P50, P95), mean inter-token latency, E2E P95, output tokens per second.
- Warmup: 20 requests per profile before measurement to stabilize CUDA graphs and caches.
- Quality: exact-match accuracy on a fixed MCQ set at temperature 0.
- Known limits: single-node, single-GPU; quality set is a small sample.
