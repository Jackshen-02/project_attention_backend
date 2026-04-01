from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .common import AttentionShape, max_error, median_ms
from .flash import tiled_peak_intermediate_bytes
from .minitorch_bridge import (
    build_minitorch_attention_layer,
    build_shared_attention_problem,
    make_minitorch_backend,
    make_minitorch_input,
)
from .naive import naive_peak_intermediate_bytes


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int = 1
    num_heads: int = 4
    head_dim: int = 64
    seq_lens: tuple[int, ...] = (128, 512, 1024, 2048)
    warmup_iters: int = 5
    measure_iters: int = 20
    causal: bool = True
    block_size: int = 128
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    seq_len: int
    latency_ms: float
    tokens_per_second: float
    peak_memory_bytes: int | None
    estimated_peak_intermediate_bytes: int
    max_abs_error: float | None
    max_rel_error: float | None
    output_nonfinite_count: int
    reference_nonfinite_count: int
    device: str
    dtype: str
    block_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_name)


def _benchmark_cpu(fn, iters: int) -> tuple[list[float], Any]:
    times_ms: list[float] = []
    out = None
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    assert out is not None
    return times_ms, out


def _synchronize_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    try:
        import pycuda.driver as cuda  # type: ignore

        cuda.Context.synchronize()
    except Exception:
        pass


def _benchmark_cuda_wallclock(fn, iters: int, device: torch.device) -> tuple[list[float], Any]:
    times_ms: list[float] = []
    out = None
    for _ in range(iters):
        _synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        _synchronize_device(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    assert out is not None
    return times_ms, out


def benchmark_backend(
    backend: str,
    seq_len: int,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    device = select_device(config.device)
    dtype = parse_dtype(config.dtype)
    _, mt_backend = make_minitorch_backend(device)
    shape = AttentionShape(
        batch_size=config.batch_size,
        num_heads=config.num_heads,
        query_len=seq_len,
        key_len=seq_len,
        head_dim=config.head_dim,
    )
    shared = build_shared_attention_problem(
        config.batch_size,
        seq_len,
        config.num_heads,
        config.head_dim,
        seed=config.seed,
    )
    mt_x = make_minitorch_input(shared, mt_backend)
    naive_layer = build_minitorch_attention_layer(
        shared,
        num_heads=config.num_heads,
        backend=mt_backend,
        causal=config.causal,
        attention_backend="naive",
    )
    flash_layer = build_minitorch_attention_layer(
        shared,
        num_heads=config.num_heads,
        backend=mt_backend,
        causal=config.causal,
        attention_backend="flash_tiled",
    )
    mt_q, mt_kT, mt_v = naive_layer.project_to_query_key_value(mt_x)
    reference = naive_layer.self_attention(mt_q, mt_kT, mt_v)

    if backend == "naive":
        fn = lambda: naive_layer.self_attention(mt_q, mt_kT, mt_v)
        estimated_peak = naive_peak_intermediate_bytes(shape, torch.float32)
    elif backend == "flash_tiled":
        fn = lambda: flash_layer.self_attention(mt_q, mt_kT, mt_v)
        estimated_peak = tiled_peak_intermediate_bytes(shape, torch.float32, config.block_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    for _ in range(config.warmup_iters):
        _ = fn()
    if device.type == "cuda":
        _synchronize_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        times_ms, output = _benchmark_cuda_wallclock(fn, config.measure_iters, device)
        measured_peak = int(torch.cuda.max_memory_allocated(device))
        peak_memory = measured_peak if backend == "flash_tiled" and measured_peak > 0 else None
    else:
        times_ms, output = _benchmark_cpu(fn, config.measure_iters)
        peak_memory = None

    if backend == "naive":
        max_abs_error = 0.0
        max_rel_error = 0.0
        output_nonfinite_count = 0
        reference_nonfinite_count = 0
    else:
        output_t = torch.tensor(output.to_numpy(), dtype=torch.float32)
        ref_t = torch.tensor(reference.to_numpy(), dtype=torch.float32)
        error_stats = max_error(output_t, ref_t)
        max_abs_error = error_stats.max_abs
        max_rel_error = error_stats.max_rel
        output_nonfinite_count = error_stats.candidate_nonfinite_count
        reference_nonfinite_count = error_stats.reference_nonfinite_count
    latency_ms = median_ms(times_ms)
    tokens_per_second = (
        config.batch_size * seq_len / (latency_ms / 1000.0)
        if latency_ms > 0.0
        else float("inf")
    )

    return BenchmarkResult(
        backend=backend,
        seq_len=seq_len,
        latency_ms=latency_ms,
        tokens_per_second=tokens_per_second,
        peak_memory_bytes=peak_memory,
        estimated_peak_intermediate_bytes=estimated_peak,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        output_nonfinite_count=output_nonfinite_count,
        reference_nonfinite_count=reference_nonfinite_count,
        device=str(device),
        dtype=config.dtype,
        block_size=config.block_size,
    )


def benchmark_suite(config: BenchmarkConfig) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for seq_len in config.seq_lens:
        for backend in ("naive", "flash_tiled"):
            results.append(benchmark_backend(backend, seq_len, config))
    return results


def results_to_json(results: list[BenchmarkResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)
