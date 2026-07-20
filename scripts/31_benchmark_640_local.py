#!/usr/bin/env python3
"""
scripts/31_benchmark_640_local.py

Controlled 640x640 deployment-latency benchmark that runs on this laptop
(CPU, no CUDA) as well as on ORC's GPU once the cluster is back -- see
pipeline/benchmark_640.py's module docstring for the full design rationale
and the YOLO-normalization correction relative to scripts/29.

--device defaults to auto-detect (cuda:0 if available, else cpu) instead of
scripts/29's hard CUDA requirement, so this actually runs here. FPS numbers
produced on this machine are a CPU correctness/latency reference only --
not comparable to ORC's GPU numbers -- until run again with --device cuda:0.

Usage (once ORC is back, for GPU-comparable numbers):
    python scripts/31_benchmark_640_local.py --device cuda:0 --precision fp32 fp16 \
        --max-images 185 --warmup 30 --repeats 3 \
        --out-dir results/benchmark_640_gpu

Fast CPU sanity check (this laptop, right now):
    python scripts/31_benchmark_640_local.py --max-images 2 --warmup 1 --repeats 1 \
        --out-dir results/benchmark_640_smoke

fp16 note: fp16 now casts real half-precision weights (model.half()), not
torch.autocast -- see Fixed640Adapter in pipeline/benchmark_640.py. At the
default --batch-size 1, fp16 may still not beat fp32: Tensor Cores need
enough work per kernel launch to pay off, which a single 640x640 image
often can't provide. To see a real fp16 speedup, add --batch-size 8 (or
similar) to both the fp32 and fp16 runs so they're compared like-for-like:
    python scripts/31_benchmark_640_local.py --device cuda:0 --precision fp32 fp16 \
        --max-images 185 --warmup 30 --repeats 3 --batch-size 8 \
        --out-dir results/benchmark_640_gpu_batch8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.benchmark_640 import run_benchmark_640


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark SegFormer/YOLO 640x640 inference latency (CPU or CUDA).")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--models", default="configs/benchmark_640_models.yaml")
    ap.add_argument("--out-dir", default="results/benchmark_640_local")

    ap.add_argument("--device", default=default_device(),
                    help="e.g. cuda:0 or cpu (default: cuda:0 if available, else cpu)")
    ap.add_argument("--precision", nargs="+", choices=("fp32", "fp16"), default=("fp32",),
                    help="fp16 is skipped automatically off a CUDA device")

    ap.add_argument("--max-images", type=int, default=185,
                    help="Number of fixed-test images to preload (185 = full fixed_consensus_test set)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Images per forward pass (default 1 = single-frame deployment "
                         "latency). >1 measures batched-throughput amortized per-image cost -- "
                         "needed to see a real fp16 speedup, since fp16/Tensor Cores need "
                         "enough work per kernel launch to beat fp32, which batch=1 often can't "
                         "provide.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark_640(
        paths_config=args.paths,
        common_config=args.common,
        models_config=args.models,
        output_dir=args.out_dir,
        device_name=args.device,
        precisions=args.precision,
        max_images=args.max_images,
        warmup=args.warmup,
        repeats=args.repeats,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
