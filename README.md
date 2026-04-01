# Efficient Attention Backends for LLM Inference in MiniTorch

Minimal midterm codebase for the CMU 11-868 project:

- restored HW4 MiniTorch base
- MiniTorch naive attention baseline
- MiniTorch-integrated `flash_tiled` attention backend
- prefill benchmark harness
- PSC GPU run instructions

## Repo Layout

```text
minitorch/                  Restored MiniTorch codebase with backend selector in attention
attention_backend/          Project tiled backend, benchmark code, and bridge utilities
benchmark_attention_prefill.py
tests/test_attention_backends.py
tests/test_minitorch_alignment.py
results/                    Saved benchmark JSON outputs
MIDTERM_PLAN.md
BENCHMARK_README.md
PRELIM_RESULTS.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

If you plan to run the MiniTorch CUDA baseline on GPU, also install:

```bash
pip install pycuda
```

Run local correctness test:

```bash
python -m pytest tests/test_attention_backends.py -q
```

MiniTorch alignment test after `numba` is installed:

```bash
python -m pytest tests/test_minitorch_alignment.py -q
```

Run a CPU benchmark:

```bash
python benchmark_attention_prefill.py \
  --device cpu \
  --batch-size 1 \
  --num-heads 4 \
  --head-dim 64 \
  --seq-lens 128 512 1024 2048 \
  --warmup-iters 3 \
  --measure-iters 7 \
  --output-json results/prefill_cpu_results.json
```

This benchmark compares the MiniTorch `self_attention` backend selected as:

- `naive`
- `flash_tiled`

For PSC GPU commands and result collection, see `BENCHMARK_README.md`.
