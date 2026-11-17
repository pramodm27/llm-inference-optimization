# Mapping resume claims to code

| Resume claim | Where it is implemented or measured |
|---|---|
| vLLM with PagedAttention, continuous batching, KV-cache management | serving/launch_vllm.py flags; benchmarks/run_benchmark.py concurrency sweep |
| Latency and throughput improvement | benchmarks/summarize.py speedup column vs fp16_baseline |
| INT8/INT4 and FP8 quantization evaluation | configs/serving.yaml profiles; quantization/eval_quality.py |
| CUDA profiling and bottleneck analysis | profiling/profile_nsight.sh |
| Async backend integration with request handling | gateway/app.py |
| Prefix caching | configs/serving.yaml enable_prefix_caching; --workload shared_prefix |
