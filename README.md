# Efficient Attention Backends for LLM Inference in MiniTorch

Final project codebase for CMU 11-868:

- restored HW4 MiniTorch attention stack as the system baseline
- MiniTorch `naive` attention baseline
- MiniTorch-integrated `flash_tiled` prefill backend
- `contiguous` and `paged` decode-side KV cache backends
- prefill and decode benchmark harnesses
- H100 result summaries and plotting utilities

## Repo Layout

```text
minitorch/                  Restored MiniTorch codebase with backend selector in attention
attention_backend/          Project tiled backend, benchmark code, and bridge utilities
benchmark_attention_prefill.py
benchmark_decode_kv.py
plot_decode_tradeoff.py
tests/test_attention_backends.py
tests/test_minitorch_alignment.py
tests/test_decode_kv.py
results/                    Saved benchmark JSON outputs
FINAL_RESULTS.md
```

## Project Story

The project has two separate inference-side optimization stories:

1. `Prefill`
   - compare restored MiniTorch `naive` attention against `flash_tiled`
   - focus on long-context prefill latency and intermediate-memory reduction
2. `Decode`
   - compare `contiguous` KV cache against `paged`
   - focus on decode-time cache allocation efficiency and page-size tradeoffs

The recommended report summary, result tables, and interpretation are in [FINAL_RESULTS.md](FINAL_RESULTS.md).

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

If you plan to run the MiniTorch CUDA baseline on GPU, also install:

```bash
pip install pycuda
```

Run the full correctness suite:

```bash
python -m pytest tests/test_attention_backends.py tests/test_minitorch_alignment.py tests/test_decode_kv.py -q
```

## Main Benchmarks

Prefill benchmark:

```bash
python benchmark_attention_prefill.py \
  --device cuda \
  --batch-size 1 \
  --num-heads 8 \
  --head-dim 64 \
  --seq-lens 128 512 1024 2048 4096 8192 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --block-size 128 \
  --output-json results/prefill_h100_bs1_h8_d64.json | tee results/prefill_h100_bs1_h8_d64.txt
```

This compares:

- `naive`
- `flash_tiled`

Decode benchmark:

```bash
python benchmark_decode_kv.py \
  --device cuda \
  --initial-lens 512 1024 2048 4096 \
  --decode-steps 64 \
  --num-heads 8 \
  --head-dim 64 \
  --page-size 256 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json results/decode/decode_h100_bs4_p256.json | tee results/decode/decode_h100_bs4_p256.txt
```

This compares:

- `contiguous`
- `paged`

Decode page-size tradeoff plot:

```bash
python plot_decode_tradeoff.py
```

This writes:

- `results/decode/decode_tradeoff_dual_axis.png`

## Documentation

- [FINAL_RESULTS.md](FINAL_RESULTS.md)
  - final result tables, metric definitions, and report-ready interpretation
