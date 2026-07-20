#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bruise_repro.benchmark_640 import run_benchmark_640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SegFormer and YOLO using a controlled 640x640 input/output pipeline."
        )
    )
    parser.add_argument("--project-config", default="configs/project.yaml")
    parser.add_argument("--benchmark-config", default="configs/benchmark_640.yaml")
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/tbommawa/bruise_repro_v3_runs/"
            "protocol_v3_distillation_suite/benchmarks_640"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("fp32", "fp16"),
        default=("fp32",),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=185)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark_640(
        project_config=args.project_config,
        benchmark_config=args.benchmark_config,
        output_dir=args.output_dir,
        device_name=args.device,
        precisions=args.precision,
        seed=args.seed,
        max_images=args.max_images,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
