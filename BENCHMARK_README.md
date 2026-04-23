# Benchmark README

## Purpose

`benchmark_attention_prefill.py` benchmarks two prefill attention backends:

- `naive`: the restored MiniTorch attention path from HW4
- `flash_tiled`: exact tiled attention with online softmax accumulation, selected from inside MiniTorch

The current midterm focus is prefill only. The benchmark uses shared synthetic weights and inputs, then:

- builds one MiniTorch attention problem
- projects `Q`, `K`, and `V` once with the MiniTorch baseline weights
- runs the MiniTorch `self_attention` path with backend `naive`
- runs the MiniTorch `self_attention` path with backend `flash_tiled`

`benchmark_decode_kv.py` benchmarks two decode-time KV cache backends:

- `contiguous`: a fixed contiguous KV cache baseline
- `paged`: a page-table-backed KV cache with page-by-page decode attention

The decode benchmark uses shared synthetic weights and inputs, prefills a cache state for each sequence, then measures decode rollout only.

## Run

From the `project_attention_backend/` root:

```bash
pip install -r requirements.txt
python benchmark_attention_prefill.py --device cpu
```

For GPU runs that include the MiniTorch CUDA baseline:

```bash
pip install pycuda
```

Example CPU smoke test:

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

Example PSC GPU run:

```bash
python benchmark_attention_prefill.py \
  --device cuda \
  --batch-size 1 \
  --num-heads 8 \
  --head-dim 64 \
  --seq-lens 128 512 1024 2048 4096 \
  --warmup-iters 10 \
  --measure-iters 50 \
  --block-size 128 \
  --output-json results/prefill_a100_results.json
```

Example decode benchmark:

```bash
python benchmark_decode_kv.py \
  --device cuda \
  --initial-lens 512 1024 2048 4096 \
  --decode-steps 64 \
  --num-heads 8 \
  --head-dim 64 \
  --page-size 128 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json results/decode_h100_bs4.json
```

Recommended command with terminal log capture:

```bash
mkdir -p results
python benchmark_attention_prefill.py \
  --device cuda \
  --batch-size 1 \
  --num-heads 8 \
  --head-dim 64 \
  --seq-lens 128 512 1024 2048 4096 \
  --warmup-iters 10 \
  --measure-iters 50 \
  --block-size 128 \
  --output-json results/prefill_a100_bs1_h8_d64.json | tee results/prefill_a100_bs1_h8_d64.txt
```

Optional second sweep for batch-size scaling:

```bash
python benchmark_attention_prefill.py \
  --device cuda \
  --batch-size 4 \
  --num-heads 8 \
  --head-dim 64 \
  --seq-lens 128 512 1024 2048 \
  --warmup-iters 10 \
  --measure-iters 50 \
  --block-size 128 \
  --output-json results/prefill_a100_bs4_h8_d64.json | tee results/prefill_a100_bs4_h8_d64.txt
```

Optional decode sweep over page size:

```bash
python benchmark_decode_kv.py \
  --device cuda \
  --initial-lens 512 1024 2048 4096 \
  --decode-steps 64 \
  --num-heads 8 \
  --head-dim 64 \
  --page-size 64 \
  --warmup-iters 5 \
  --measure-iters 20 \
  --output-json results/decode_h100_bs4_p64.json | tee results/decode_h100_bs4_p64.txt
```

Batch submission on PSC:

```bash
sbatch jobs/run_h100_decode.sbatch
```

This script:

- activates the repo-local conda environment
- runs the main correctness tests
- runs the H100 decode benchmark
- writes all outputs under `results/decode/<timestamp>/`

## Metrics

- `latency_ms`
  - Median latency over measured iterations.
  - Timing is for the attention core on fixed projected `Q/K/V`, not the full decoder block.
- `tok/s`
  - `batch_size * seq_len / latency`.
- `peak_mb`
  - Actual peak memory on CUDA from `torch.cuda.max_memory_allocated`.
  - `n/a` on CPU.
  - Also `n/a` for the MiniTorch `naive` CUDA path, because its `pycuda` allocations are outside the Torch allocator.
  - For `flash_tiled`, the reported CUDA peak only covers the Torch-managed tiled backend section, not all MiniTorch allocations.
- `est_peak_mb`
  - Analytical peak intermediate footprint for the backend.
  - Useful on CPU and as a sanity check on GPU.
- `max_abs_err`
  - Maximum absolute difference against the naive baseline.
- `max_rel_err`
  - Maximum relative difference on non-negligible reference entries only.
- `lat_ms/step`
  - Median decode latency per generated token step across the measured rollout.
- `alloc_mb`
  - Allocated KV cache storage after the measured decode rollout.
- `active_mb`
  - Bytes corresponding to logically active KV tokens only.
- `util`
  - `active_mb / alloc_mb`, a simple cache-efficiency metric.
- `pages_used`
  - Number of pages allocated by the paged cache backend.

## Notes

- `flash_tiled` is not a call to PyTorch SDPA or FlashAttention. It is a project-local exact implementation using tiled score blocks plus online softmax recombination.
- The `naive` benchmark path is the restored MiniTorch implementation, not a separate handwritten baseline.
- `flash_tiled` is now selected from inside `minitorch/modules_transfomer.py`, so the comparison boundary is the MiniTorch attention module.
- The decode benchmark also runs through MiniTorch `MultiHeadAttention`, using the new `decode_step` extension rather than a standalone attention script.
- The current optimized path is forward-only and Python-level. On CPU it may be slower than naive attention because the memory savings do not offset Python loop overhead.
- The intended evaluation environment is PSC GPU, where the reduced intermediate footprint should matter much more.
- `peak_memory_bytes` is a best-effort Torch allocator measurement. Because the project mixes MiniTorch, `pycuda`, and Torch in one process, this field may be `null` when the runtime signal is not trustworthy.
- `estimated_peak_intermediate_bytes` is the more stable memory-comparison field for the midterm report because it tracks the algorithm's attention intermediates directly.
- For the decode benchmark, the more meaningful memory fields are `allocated_cache_bytes`, `active_cache_bytes`, and `cache_utilization`.

## Result Gathering

The benchmark already prints a table and writes a JSON file.

- Keep the JSON files in `results/`.
- Keep the matching terminal logs via `tee` so you have a human-readable copy.
- For the midterm report, extract these fields from the JSON:
  - `backend`
  - `seq_len`
  - `latency_ms`
  - `tokens_per_second`
  - `peak_memory_bytes`
  - `estimated_peak_intermediate_bytes`
  - `max_abs_error`
  - `max_rel_error`
  - `output_nonfinite_count`
  - `reference_nonfinite_count`

You can inspect a result file with:

```bash
cat results/prefill_a100_bs1_h8_d64.json
```

If you want a quick CSV-like view from the JSON on PSC:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/prefill_a100_bs1_h8_d64.json")
rows = json.loads(path.read_text())
print("backend,seq_len,latency_ms,tokens_per_second,peak_memory_bytes,max_abs_error,max_rel_error,output_nonfinite_count,reference_nonfinite_count")
for row in rows:
    print(
        f"{row['backend']},{row['seq_len']},{row['latency_ms']:.6f},"
        f"{row['tokens_per_second']:.2f},{row['peak_memory_bytes']},"
        f"{row['max_abs_error']},{row['max_rel_error']},"
        f"{row['output_nonfinite_count']},{row['reference_nonfinite_count']}"
    )
PY
```
