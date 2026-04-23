# Preliminary Results

## Baseline And Scope

- System baseline: restored `llmsys_hw4` MiniTorch attention stack, because HW4 already includes the transformer path and CUDA-oriented kernel plumbing relevant to this project.
- We do not use `llmsys_hw5` as the project baseline because HW5 is primarily about distributed training and pipeline/data parallel execution rather than attention-backend design.
- Midterm scope:
  - `naive`: restored MiniTorch baseline attention
  - `flash_tiled`: MiniTorch-integrated tiled attention backend
  - prefill-only attention benchmarking with synthetic inputs

## Current Status

- Implemented:
  - restored HW4 MiniTorch base inside `project_attention_backend/minitorch/`
  - backend selector in `minitorch/modules_transfomer.py`
  - tiled attention backend in `attention_backend/flash.py`
  - benchmark harness in `benchmark_attention_prefill.py`
  - contiguous and paged decode cache backends
  - decode benchmark harness in `benchmark_decode_kv.py`
  - correctness tests in `tests/`
  - formal H100 sweep in `results/prefill_h100_bs1_h8_d64.{json,txt}`
- Partial:
  - runtime CUDA peak memory is not a reliable report metric in this mixed MiniTorch/PyCUDA/Torch setup
  - decode results have not been regenerated on PSC yet
- Not done yet:
  - custom fused CUDA attention kernel
  - H100 decode-side experiment sweep for contiguous vs paged KV cache

## Experiment Setup

- Platform: PSC Bridges-2 `H100 80GB HBM3`
- Command: `python benchmark_attention_prefill.py --device cuda --batch-size 1 --num-heads 8 --head-dim 64 --seq-lens 128 512 1024 2048 4096 8192 --warmup-iters 5 --measure-iters 20 --block-size 128`
- Workload: causal prefill attention with synthetic inputs
- Model shape: `batch_size=1`, `num_heads=8`, `head_dim=64`, so `n_embd=512`
- Compared backends:
  - `naive`: restored MiniTorch baseline attention path
  - `flash_tiled`: MiniTorch-integrated tiled attention backend

Raw result files:

- `results/prefill_h100_bs1_h8_d64.json`
- `results/prefill_h100_bs1_h8_d64.txt`

## Main Table

| Seq Len | Naive Latency (ms) | Flash Latency (ms) | Speedup | Naive Tok/s | Flash Tok/s | Naive Est. Intermediates (MB) | Flash Est. Intermediates (MB) | Memory Reduction | Max Abs Err | Max Rel Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10.266 | 3.965 | 2.59x | 12,468.78 | 32,286.23 | 1.00 | 1.26 | 0.79x | 5.96e-07 | 2.67e-03 |
| 512 | 70.625 | 9.824 | 7.19x | 7,249.59 | 52,118.25 | 16.00 | 5.03 | 3.18x | 5.66e-07 | 4.53e-03 |
| 1024 | 316.684 | 21.467 | 14.75x | 3,233.51 | 47,702.03 | 64.00 | 10.06 | 6.36x | 7.75e-07 | 4.46e-03 |
| 2048 | 1222.752 | 57.797 | 21.16x | 1,674.91 | 35,434.19 | 256.00 | 20.12 | 12.72x | 7.75e-07 | 4.05e-03 |
| 4096 | 4764.853 | 215.940 | 22.07x | 859.63 | 18,968.26 | 1024.00 | 40.25 | 25.44x | 1.13e-06 | 5.08e-03 |
| 8192 | 18985.652 | 894.282 | 21.23x | 431.48 | 9,160.42 | 4096.00 | 80.50 | 50.88x | 1.18e-06 | 5.48e-03 |

## Key Observations

- The tiled backend is consistently faster than the MiniTorch naive baseline on H100 across the full sweep.
- The speedup grows with sequence length, from `2.59x` at `128` tokens to about `21-22x` at `2048-8192`.
- The estimated intermediate-memory gap also widens with sequence length:
  - `2048`: `256.00 MB -> 20.12 MB`
  - `4096`: `1024.00 MB -> 40.25 MB`
  - `8192`: `4096.00 MB -> 80.50 MB`
- This trend matches the expected motivation for FlashAttention-style tiling:
  - the naive path materializes the full attention score matrix
  - the tiled path keeps only block-sized score chunks plus online softmax statistics
- Numerical agreement is clean in the H100 run:
  - `output_nonfinite_count = 0`
  - `reference_nonfinite_count = 0`
  - `max_abs_err` stays around `1e-6`
  - `max_rel_err` stays around `4e-3` to `5e-3`

## Interpretation

- The current implementation is already strong enough for a midterm claim that tiled attention substantially improves long-context prefill performance.
- The biggest wins appear in the long-sequence regime, which is the regime the project proposal cares about most.
- At very small sequence length (`128`), the tiled path is still faster, but its estimated intermediate memory is slightly larger than naive because the score matrix is still small and the tiled path pays extra fixed overhead for block statistics and output buffers.
- For larger sequence lengths, the tiled backend's memory footprint scales much more favorably than naive attention.

## Metric Notes

- `peak_memory_bytes` is `null` in these runs and should not be used as the main memory figure in the report.
- Reason: this benchmark mixes MiniTorch, `pycuda`, and Torch in one process, so Torch allocator peak statistics are not a reliable end-to-end memory signal here.
- The report should instead use `estimated_peak_intermediate_bytes` as the primary memory comparison metric.
- `max_abs_err` and `max_rel_err` are backend-output differences measured against the MiniTorch naive reference.

## Report-Ready Summary

Suggested short writeup for the midterm:

`On PSC Bridges-2 H100, our MiniTorch-integrated tiled attention backend outperformed the restored naive MiniTorch baseline across all tested prefill sequence lengths from 128 to 8192. The speedup grew with sequence length, reaching about 22x at 4096 tokens and 21x at 8192 tokens. The estimated peak attention intermediates were also substantially reduced, from 1.0 GB to 40.25 MB at 4096 tokens and from 4.0 GB to 80.50 MB at 8192 tokens. Numerical agreement remained strong, with max absolute error around 1e-6 and no non-finite outputs in either backend.`

## Remaining Caveats

- `flash_tiled` is integrated at the MiniTorch module boundary, but internally it still uses a project-local Torch implementation rather than a custom fused CUDA kernel.
- The current benchmark focuses on the attention layer under prefill, not yet the full decoder stack.
- The decode-side contiguous and paged KV cache paths are now implemented, but their formal H100 result sweep still needs to be run.
