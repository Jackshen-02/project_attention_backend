# Midterm Plan

## Base Assignment Decision

The correct project base is `llmsys_hw4/`, not `llmsys_hw5/`.

- `llmsys_hw3/` is the first assignment that introduces the decoder-only transformer and multi-head attention path.
- `llmsys_hw4/` keeps that transformer stack and adds the CUDA-oriented softmax/layernorm/kernel plumbing that is directly relevant to attention-backend work.
- `llmsys_hw5/` moves into distributed training, data parallelism, pipeline parallelism, and HuggingFace GPT2 model partitioning. That is not the right base for a prefill/decode attention-backend project.

Reference files inspected:

- `llmsys_hw3/minitorch/transformer.py`
- `llmsys_hw4/minitorch/modules_transfomer.py`
- `llmsys_hw4/minitorch/tensor.py`
- `llmsys_hw4/minitorch/tensor_functions.py`
- `llmsys_hw4/minitorch/cuda_kernel_ops.py`
- `llmsys_hw4/src/softmax_kernel.cu`
- `llmsys_hw5/project/run_data_parallel.py`
- `llmsys_hw5/pipeline/model.py`
- `llmsys_hw5/pipeline/pipe.py`

For the midterm, the implementation lives entirely inside `project_attention_backend/` and does not modify any assignment directory.

## Implemented

- Restored the HW4 MiniTorch base inside `project_attention_backend/minitorch/`
  - This is the correct system baseline for the project.
- `attention_backend/naive.py`
  - Dense attention reference math used for correctness checks.
  - Explicitly computes `QK^T`, softmax, and `AV`.
- `attention_backend/flash.py`
  - Exact tiled forward pass using online softmax accumulation.
  - Avoids materializing the full attention matrix.
  - This is FlashAttention-style, but not a custom fused CUDA kernel yet.
- `attention_backend/minitorch_bridge.py`
  - Connects the benchmark harness to the restored MiniTorch baseline.
- `minitorch/modules_transfomer.py`
  - Now exposes an attention backend selector.
  - `naive` stays as the original MiniTorch path.
  - `flash_tiled` is integrated as an alternate backend at the MiniTorch module boundary.
- `attention_backend/benchmark.py`
  - Prefill benchmark driver with CPU/GPU-aware timing and memory reporting.
  - Compares MiniTorch `self_attention` backends on the same projected `Q/K/V`.
- `benchmark_attention_prefill.py`
  - CLI entrypoint for benchmark sweeps.
- `tests/test_attention_backends.py`
  - CPU correctness checks for naive vs tiled outputs.
- `attention_backend/kv_cache.py`
  - Minimal contiguous KV cache helper.
  - Paged KV cache interface sketch for final-project expansion.

## Partial

- Peak memory reporting is only partial on CUDA.
  - The MiniTorch `naive` path uses `pycuda`, so Torch allocator stats do not capture it.
  - In practice, the report should use `estimated_peak_intermediate_bytes` as the main memory metric.
- On CPU, the script reports analytical peak intermediate bytes instead of allocator-level tensor memory.
- KV cache work is intentionally only a scaffold for the midterm.
- The tiled backend is benchmarked against the MiniTorch baseline through a focused attention-layer harness, not yet through the full decoder stack.

## Not Done Yet

- Custom CUDA kernel for tiled/fused attention.
- Decode benchmark with KV cache growth.
- Paged KV cache allocator and page-table lookup path.
- Dynamic batching / allocator fragmentation experiments.
- Full decoder-stack benchmark and deeper system profiling.

## Immediate Next Steps

1. Use the H100 prefill sweep in `results/prefill_h100_bs1_h8_d64.{json,txt}` for the midterm report.
2. Add one or two follow-up sweeps if more evidence is needed:
   batch-size scaling or a different tile size.
3. Implement a real paged KV cache path for decode experiments.
4. Decide whether the final project needs deeper integration into the full decoder stack or a custom CUDA kernel.
