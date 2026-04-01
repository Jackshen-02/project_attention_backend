# Preliminary Results

## Local Result Scope

The code is now integrated at the MiniTorch module boundary: `naive` and `flash_tiled` are both selected from inside `minitorch/modules_transfomer.py`.

However, the numeric table below is still from an earlier pre-integration CPU-only smoke test. It should be treated as archived sanity-check data only, not as the current report-quality result. The archived raw JSON from that earlier run is stored as `results/prefill_cpu_results_pre_minitorch_integration.json`.

The next valid result set should be regenerated with the current MiniTorch-integrated benchmark on PSC.

## Archived Sanity-Check Table

| Backend | Seq Len | Latency (ms) | Tokens/s | Peak Memory (CUDA) | Estimated Peak Intermediates (MB) | Max Abs Err | Max Rel Err |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| naive | 128 | 0.158 | 809,695.65 | n/a | 0.50 | 0.0 | 0.0 |
| flash_tiled | 128 | 0.211 | 607,957.40 | n/a | 0.63 | 4.47e-07 | 1.26e-03 |
| naive | 512 | 1.476 | 346,863.95 | n/a | 8.00 | 0.0 | 0.0 |
| flash_tiled | 512 | 2.574 | 198,902.54 | n/a | 2.52 | 5.96e-07 | 3.54e-03 |
| naive | 1024 | 5.450 | 187,874.12 | n/a | 32.00 | 0.0 | 0.0 |
| flash_tiled | 1024 | 12.271 | 83,447.08 | n/a | 5.03 | 4.77e-07 | 3.87e-03 |
| naive | 2048 | 29.433 | 69,580.88 | n/a | 128.00 | 0.0 | 0.0 |
| flash_tiled | 2048 | 62.090 | 32,984.29 | n/a | 10.06 | 4.77e-07 | 6.45e-03 |

## Observations

- The archived numbers suggest the tiled algorithm is numerically stable and lowers intermediate-memory pressure.
- The tiled path reduces estimated intermediate memory substantially once sequence length grows:
  - `512`: `8.00 MB -> 2.52 MB`
  - `1024`: `32.00 MB -> 5.03 MB`
  - `2048`: `128.00 MB -> 10.06 MB`
- On CPU, `flash_tiled` is slower than `naive` for larger sequences. This is expected here because the implementation is Python-level and the CPU run does not expose the GPU memory-bandwidth bottleneck that motivates FlashAttention-style designs.

## Current Integration Status

- `naive` is the MiniTorch attention backend.
- `flash_tiled` is now selected from inside MiniTorch as an alternate backend.
- The benchmark compares those two MiniTorch-selected attention paths on the same projected `Q/K/V`.
- The numbers for this integrated version still need to be regenerated on PSC.

## What Needs GPU Results

- Regenerate the sweep with the current MiniTorch-integrated benchmark
- Real or at least more defensible peak-memory reporting for the integrated backends
- Latency crossover point where `flash_tiled` overtakes `naive`
- Larger-sequence sweeps such as `4096` and `8192`
- Batch-size scaling under prefill workloads
