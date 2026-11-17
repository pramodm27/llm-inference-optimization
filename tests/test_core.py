"""CPU-only tests. Run with: pytest -q"""

from benchmarks.run_benchmark import build_prompts, make_prompt, percentile
from serving.launch_vllm import build_command


def test_percentile_basic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 0) == 1.0


def test_percentile_empty_is_nan():
    import math
    assert math.isnan(percentile([], 95))


def test_make_prompt_is_roughly_target_length():
    prompt = make_prompt(512, unique_tag="t")
    words = len(prompt.split())
    # Target is 512 tokens at ~1.3 tokens per word, so about 394 words plus overhead.
    assert 300 < words < 500


def test_shared_prefix_prompts_share_prefix():
    prompts = build_prompts("shared_prefix", count=5, input_len=0, shared_prefix_tokens=200)
    prefixes = {p.split("\n\nQuestion")[0] for p in prompts}
    assert len(prefixes) == 1, "all requests must share an identical prefix for prefix caching"


def test_build_command_fp8_flags():
    cfg = {
        "model": {"name": "m", "awq_name": "a"},
        "hardware": {"gpu_memory_utilization": 0.9},
        "profiles": {
            "fp8": {
                "quantization": "fp8",
                "dtype": "auto",
                "max_num_seqs": 64,
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 8192,
            }
        },
    }
    cmd = build_command("fp8", cfg, port=8000)
    assert "--quantization" in cmd and "fp8" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--enable-chunked-prefill" in cmd
    assert cmd[cmd.index("--max-num-batched-tokens") + 1] == "8192"


def test_build_command_awq_uses_awq_checkpoint():
    cfg = {
        "model": {"name": "base-model", "awq_name": "awq-model"},
        "hardware": {"gpu_memory_utilization": 0.9},
        "profiles": {
            "awq_int4": {
                "quantization": "awq",
                "dtype": "float16",
                "max_num_seqs": 64,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            }
        },
    }
    cmd = build_command("awq_int4", cfg, port=8001)
    assert cmd[cmd.index("--model") + 1] == "awq-model"
    assert cmd[cmd.index("--port") + 1] == "8001"
