from __future__ import annotations

import argparse
from pathlib import Path

from attention_backend.decode_benchmark import (
    DecodeBenchmarkConfig,
    benchmark_decode_suite,
    decode_results_to_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark contiguous vs paged KV cache decode backends.")
    parser.add_argument("--initial-lens", type=int, nargs="+", required=True)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def format_megabytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 ** 2):.2f}"


def format_error(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3e}"


def main() -> None:
    args = build_parser().parse_args()
    config = DecodeBenchmarkConfig(
        initial_lens=tuple(args.initial_lens),
        decode_steps=args.decode_steps,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        warmup_iters=args.warmup_iters,
        measure_iters=args.measure_iters,
        page_size=args.page_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    results = benchmark_decode_suite(config)

    header = (
        f"{'backend':<12} {'batch':>6} {'decode_steps':>12} {'lat_ms/step':>12} "
        f"{'tok/s':>12} {'alloc_mb':>10} {'active_mb':>10} {'util':>8} "
        f"{'pages_used':>10} {'max_abs_err':>12} {'max_rel_err':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        pages_used = "n/a" if result.pages_used is None else str(result.pages_used)
        print(
            f"{result.backend:<12} {result.batch_size:>6d} {result.decode_steps:>12d} "
            f"{result.latency_ms_per_step:>12.3f} {result.tokens_per_second:>12.2f} "
            f"{format_megabytes(result.allocated_cache_bytes):>10} "
            f"{format_megabytes(result.active_cache_bytes):>10} "
            f"{result.cache_utilization:>8.3f} {pages_used:>10} "
            f"{format_error(result.max_abs_error):>12} {format_error(result.max_rel_error):>12}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(decode_results_to_json(results) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
