# Final Results

## Scope

This project now has two distinct inference-side stories:

1. `Prefill`: restored MiniTorch `naive` attention vs MiniTorch-integrated `flash_tiled`
2. `Decode`: `contiguous` KV cache vs `paged` KV cache with page-by-page decode attention

The prefill and decode optimizations target different bottlenecks and should be reported separately:

- `flash_tiled` addresses long-context prefill latency and intermediate-memory growth.
- `paged` addresses decode-time KV cache allocation efficiency and page-size tradeoffs.

We use the restored `llmsys_hw4` MiniTorch attention stack as the system baseline because HW4 already contains the transformer path and CUDA-oriented kernel plumbing relevant to attention-backend work. We do not use `llmsys_hw5` as the baseline because HW5 is mainly about distributed training, not attention-backend design.

## Result Files

Prefill H100 results:

- [prefill_h100_bs1_h8_d64.json](results/prefill_h100_bs1_h8_d64.json)
- [prefill_h100_bs1_h8_d64.txt](results/prefill_h100_bs1_h8_d64.txt)

Decode H100 results to keep in the repo:

- `results/decode/decode_h100_bs4_p32.{json,txt}`
- `results/decode/decode_h100_bs4_p64.{json,txt}`
- `results/decode/decode_h100_bs4_p128.{json,txt}`
- `results/decode/decode_h100_bs4_p256.{json,txt}`
- `results/decode/decode_h100_bs4_p512.{json,txt}`
- `results/decode/decode_h100_long_p128.{json,txt}`
- `results/decode/decode_h100_long_p256.{json,txt}`

## Experiment Setup

Platform:

- PSC Bridges-2 `H100 80GB HBM3`

Prefill workload:

- command: `python benchmark_attention_prefill.py --device cuda --batch-size 1 --num-heads 8 --head-dim 64 --seq-lens 128 512 1024 2048 4096 8192 --warmup-iters 5 --measure-iters 20 --block-size 128`
- causal prefill attention with synthetic inputs
- `batch_size=1`, `num_heads=8`, `head_dim=64`, so `n_embd=512`

Decode workload:

- command family: `python benchmark_decode_kv.py --device cuda --initial-lens ... --decode-steps 64 --num-heads 8 --head-dim 64 --page-size ... --warmup-iters 5 --measure-iters 20`
- causal decode with synthetic inputs
- decode benchmarks use variable-length prefix lengths to prefill the KV cache before rolling out 64 decode steps
- `initial_lens` means the number of past tokens already stored in the cache for each sequence before decode starts
- example: `initial_lens = [512, 1024, 2048, 4096]` means batch size 4, with those four prefix lengths already resident in the KV cache

## Metric Definitions

Prefill metrics:

- `latency_ms`: median end-to-end latency for the attention layer
- `tokens_per_second`: throughput derived from median latency
- `estimated_peak_intermediate_bytes`: estimated attention-intermediate footprint; this is the reportable memory metric
- `max_abs_error`, `max_rel_error`: backend-output differences against the MiniTorch naive reference

Decode metrics:

- `latency_ms_per_step`: median end-to-end latency for one decode step
- `tokens_per_second`: `batch_size * decode_steps / total_time`
- `allocated_cache_bytes`: total bytes reserved by the cache implementation
- `active_cache_bytes`: bytes actually occupied by live KV entries
- `cache_utilization`: `active_cache_bytes / allocated_cache_bytes`
- `pages_used`: number of allocated pages for the paged backend
- `max_abs_error`, `max_rel_error`: paged-output differences against the contiguous reference

Important caveat:

- `peak_memory_bytes` is not a reliable report metric in this mixed MiniTorch/PyCUDA/Torch setup.
- Use `estimated_peak_intermediate_bytes` for prefill and `allocated_cache_bytes` plus `cache_utilization` for decode.

## Prefill Summary

### Main Table

| Seq Len | Naive Latency (ms) | Flash Latency (ms) | Speedup | Naive Tok/s | Flash Tok/s | Naive Est. Intermediates (MB) | Flash Est. Intermediates (MB) | Memory Reduction | Max Abs Err | Max Rel Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10.266 | 3.965 | 2.59x | 12,468.78 | 32,286.23 | 1.00 | 1.26 | 0.79x | 5.96e-07 | 2.67e-03 |
| 512 | 70.625 | 9.824 | 7.19x | 7,249.59 | 52,118.25 | 16.00 | 5.03 | 3.18x | 5.66e-07 | 4.53e-03 |
| 1024 | 316.684 | 21.467 | 14.75x | 3,233.51 | 47,702.03 | 64.00 | 10.06 | 6.36x | 7.75e-07 | 4.46e-03 |
| 2048 | 1222.752 | 57.797 | 21.16x | 1,674.91 | 35,434.19 | 256.00 | 20.12 | 12.72x | 7.75e-07 | 4.05e-03 |
| 4096 | 4764.853 | 215.940 | 22.07x | 859.63 | 18,968.26 | 1024.00 | 40.25 | 25.44x | 1.13e-06 | 5.08e-03 |
| 8192 | 18985.652 | 894.282 | 21.23x | 431.48 | 9,160.42 | 4096.00 | 80.50 | 50.88x | 1.18e-06 | 5.48e-03 |

### Prefill Takeaways

- `flash_tiled` is consistently faster than the restored MiniTorch `naive` baseline.
- The speedup grows with sequence length and stabilizes around `21x-22x` for long contexts.
- Estimated intermediate-memory reduction also grows with sequence length and reaches about `51x` at `8192`.
- Numerical agreement remains strong:
  - `max_abs_err` stays around `1e-6`
  - `max_rel_err` stays around `4e-3` to `5e-3`
  - no non-finite outputs were observed

Short report-ready statement:

`On PSC Bridges-2 H100, our MiniTorch-integrated tiled attention backend outperformed the restored naive MiniTorch baseline across all tested prefill sequence lengths from 128 to 8192. The speedup grew with sequence length, reaching about 22x at 4096 tokens and 21x at 8192 tokens. The estimated peak attention intermediates were also substantially reduced, from 1.0 GB to 40.25 MB at 4096 tokens and from 4.0 GB to 80.50 MB at 8192 tokens, with max absolute error remaining around 1e-6.`

## Decode Summary

### Why There Is A Page-Size Sweep

The project-level decode comparison is still:

- `contiguous` KV cache
- vs `paged` KV cache

The page-size sweep is not the main comparison by itself. It exists to choose a reasonable default paged configuration. After that choice, the main decode comparison becomes:

- `contiguous`
- vs `paged (page_size = 256)`

### Standard Workload: Page-Size Sweep

Workload:

- `initial_lens = [512, 1024, 2048, 4096]`
- `decode_steps = 64`
- `batch_size = 4`

| Backend | Page Size | Latency per Step (ms) | Tok/s | Allocated Cache (MB) | Active Cache (MB) | Utilization | Pages Used | Max Abs Err | Max Rel Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous | n/a | 9.005 | 444.20 | 65.00 | 31.00 | 0.477 | n/a | 0.00e+00 | 0.00e+00 |
| paged | 32 | 38.600 | 103.63 | 31.00 | 31.00 | 1.000 | 248 | 1.94e-07 | 3.96e-03 |
| paged | 64 | 23.916 | 167.25 | 31.00 | 31.00 | 1.000 | 124 | 1.79e-07 | 2.51e-03 |
| paged | 128 | 16.671 | 239.93 | 32.00 | 31.00 | 0.969 | 64 | 1.53e-07 | 2.12e-03 |
| paged | 256 | 13.161 | 303.92 | 34.00 | 31.00 | 0.912 | 34 | 1.64e-07 | 3.58e-03 |
| paged | 512 | 12.080 | 331.13 | 38.00 | 31.00 | 0.816 | 19 | 2.09e-07 | 2.66e-03 |

### Standard Decode Takeaways

- All paged runs are numerically clean and closely match the contiguous reference.
- Smaller pages improve utilization but hurt latency because page traversal overhead grows quickly.
- Larger pages reduce traversal overhead, but cache utilization degrades because internal fragmentation rises.
- `page_size = 256` is the best operating point in the current implementation:
  - it is much faster than `32/64/128`
  - it preserves most of the cache-efficiency gain over contiguous
  - it avoids the sharper utilization drop seen at `512`

Reason for choosing `256` over `512`:

- `512` is only modestly faster than `256`: `12.080 ms` vs `13.161 ms`
- but utilization drops from `0.912` to `0.816`
- and allocated cache rises from `34 MB` to `38 MB`

Short report-ready statement:

`For decode, our paged KV cache prototype showed a clear page-size tradeoff. Smaller pages minimized internal fragmentation but incurred substantial per-page traversal overhead, while very large pages reduced latency slightly but gave back too much cache efficiency. In our implementation, page_size=256 provided the best balance between decode latency and allocation efficiency.`

### Long-Context Decode Comparison

Workload:

- `initial_lens = [1024, 2048, 4096, 8192]`
- `decode_steps = 64`
- `batch_size = 4`

| Backend | Page Size | Latency per Step (ms) | Tok/s | Allocated Cache (MB) | Active Cache (MB) | Utilization | Pages Used | Max Abs Err | Max Rel Err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| contiguous | n/a | 9.488 | 421.57 | 129.00 | 61.00 | 0.473 | n/a | 0.00e+00 | 0.00e+00 |
| paged | 128 | 24.269 | 164.82 | 62.00 | 61.00 | 0.984 | 124 | 1.79e-07 | 1.87e-03 |
| paged | 256 | 16.675 | 239.88 | 64.00 | 61.00 | 0.953 | 64 | 1.64e-07 | 2.20e-03 |

### Long-Context Decode Takeaways

- The same tradeoff continues at longer contexts.
- `paged (256)` remains much more allocation-efficient than `contiguous`:
  - `129 MB -> 64 MB`
  - utilization improves from `0.473 -> 0.953`
- The paged prototype is still slower than contiguous in latency:
  - `9.488 ms -> 16.675 ms`
- This is consistent with the current implementation strategy:
  - Python-level page traversal
  - page-by-page attention loop
  - no fused custom CUDA kernel yet

Short report-ready statement:

`At longer decode contexts up to 8192 cached tokens, the paged KV cache continued to reduce allocated cache capacity by about 2x relative to the contiguous baseline while maintaining numerical agreement. The current Python-level paged prototype remained slower in latency, but substantially improved cache utilization and allocation efficiency.`

## Recommended Report Story

The cleanest final report structure is:

1. Prefill optimization:
   - compare `naive` vs `flash_tiled`
   - emphasize long-context prefill speedup and intermediate-memory reduction
2. Decode optimization:
   - compare `contiguous` vs `paged`
   - use the page-size sweep only to justify `page_size = 256`
   - then report the final `contiguous vs paged(256)` comparison on both standard and long-context workloads

This keeps the story coherent:

- `flash_tiled` improves prefill efficiency
- `paged` improves decode-time cache allocation efficiency

## Recommended Figures

### Figure A: Decode Page-Size Tradeoff

Use one dual-axis figure:

- x-axis: `page size`
- left y-axis: paged `latency per step`
- right y-axis: paged `cache utilization`
- add two horizontal reference lines from the chosen contiguous baseline:
  - `latency = 9.005 ms/step`
  - `utilization = 0.477`

This figure explains why `256` was chosen.

### Figure B: Final Decode Comparison Table

Use one small table rather than another complicated plot:

- workload `[512, 1024, 2048, 4096]`
- workload `[1024, 2048, 4096, 8192]`
- compare `contiguous` vs `paged(256)`

This table is where the main project-level decode comparison should live.

## What To Keep And What To Ignore

Keep for the final repo:

- prefill formal results
- decode sweep results for `32/64/128/256/512`
- long-context decode results for `128` and `256`
- plotting script outputs once generated

Do not over-emphasize:

- raw `pages_used` in the main report body
- individual smoke runs
- unreliable runtime allocator peak-memory fields

## Current Conclusion

The project now supports a complete inference-backend story inside MiniTorch:

- `Prefill`: `naive` vs `flash_tiled`
- `Decode`: `contiguous` vs `paged`

The current final-position claims are:

- `flash_tiled` substantially improves long-context prefill latency and estimated intermediate memory.
- `paged` substantially improves decode-time cache allocation efficiency and utilization.
- `page_size = 256` is the best paged operating point in the current implementation.
- The current paged implementation is correct and memory-efficient, but still pays Python-level traversal overhead and is therefore slower than the contiguous baseline in latency.
