#!/usr/bin/env python3
"""
scripts/test_timing_mode.py

Diagnostic (not a pipeline stage) that proves, with real numbers, whether
resize/normalize is currently inside or outside benchmark_adapter()'s timed
section -- see pipeline/benchmark_640.py's "TIMING MODE TOGGLE" comment.

Design note: every measurement below re-reads the image from disk and
re-preprocesses it on EVERY iteration -- exactly matching what the real
benchmark_adapter() loop does per-iteration. An earlier version of this
script reused one cached preprocessed tensor for the "forward-only"
measurement while the real benchmark_adapter() re-read+re-resized from disk
every iteration; that mismatch in memory-allocation pattern (not just
"is resize timed") was itself enough to skew the comparison. Keeping the
per-iteration work identical between the manual measurement and the real
function isolates the one thing actually being tested: where the stopwatch
starts, nothing else.

Usage:
    python scripts/test_timing_mode.py --model yolo_sem_direct --n-iters 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.benchmark_640 import (
    SourceImage,
    _synchronize,
    build_adapter,
    benchmark_adapter,
    load_fixed_test_images,
    read_source_image,
)
from pipeline.io_utils import load_yaml


def time_mode_raw_forward(adapter, source: SourceImage, n_warmup: int, n_iters: int) -> float:
    """Per iteration: read (untimed) -> preprocess (untimed) -> [TIMED] forward+threshold.
    Matches the block currently ACTIVE in pipeline/benchmark_640.py."""
    device = adapter.device

    def one_call() -> None:
        image_rgb = read_source_image(source)
        tensor = adapter._preprocess(image_rgb)
        _synchronize(device)
        start = time.perf_counter()
        adapter._forward_and_threshold(tensor)
        _synchronize(device)
        one_call.last_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(n_warmup):
        one_call()
    total = 0.0
    for _ in range(n_iters):
        one_call()
        total += one_call.last_ms
    return total / n_iters


def time_mode_full_pipeline(adapter, source: SourceImage, n_warmup: int, n_iters: int) -> float:
    """Per iteration: read (untimed) -> [TIMED] preprocess -> forward+threshold.
    Matches the block currently COMMENTED OUT in pipeline/benchmark_640.py."""
    device = adapter.device

    def one_call() -> None:
        image_rgb = read_source_image(source)
        _synchronize(device)
        start = time.perf_counter()
        adapter.infer_full(image_rgb)
        _synchronize(device)
        one_call.last_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(n_warmup):
        one_call()
    total = 0.0
    for _ in range(n_iters):
        one_call()
        total += one_call.last_ms
    return total / n_iters


def main() -> None:
    ap = argparse.ArgumentParser(description="Prove whether resize is inside or outside the timed benchmark section.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model", default="yolo_sem_direct",
                    help="run_name to test (default yolo_sem_direct -- fast, no transformers-version issues)")
    ap.add_argument("--family", default="yolo_semantic", choices=("yolo_semantic", "segformer"))
    ap.add_argument("--pretrained-key", default=None, help="required if --family segformer")
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-iters", type=int, default=200)
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    spec = {"name": args.model, "family": args.family, "run_name": args.model}
    if args.family == "segformer":
        if not args.pretrained_key:
            raise SystemExit("--pretrained-key is required for --family segformer (e.g. segformer_b0_pretrained)")
        spec["pretrained_key"] = args.pretrained_key

    adapter = build_adapter(spec, paths, device, precision="fp32")
    images = load_fixed_test_images(paths, max_images=1)
    source = images[0]
    print(f"Test image: {source.stem} (native resolution, resized to 640x640 by _preprocess)\n")

    # (A) manual measurement, resize EXCLUDED from the timed window
    raw_ms = time_mode_raw_forward(adapter, source, args.n_warmup, args.n_iters)

    # (B) manual measurement, resize INCLUDED in the timed window
    full_ms = time_mode_full_pipeline(adapter, source, args.n_warmup, args.n_iters)

    # (C) the REAL function scripts/31 actually calls -- same iteration count
    real_summary, _ = benchmark_adapter(adapter, images, warmup=args.n_warmup, repeats=args.n_iters)

    print("=" * 70)
    print(f"(A) manual, resize EXCLUDED from timer: {raw_ms:8.3f} ms")
    print(f"(B) manual, resize INCLUDED in timer:    {full_ms:8.3f} ms   (difference = {full_ms - raw_ms:.3f} ms -- this is resize's real cost)")
    print("-" * 70)
    print(f"(C) REAL benchmark_adapter() reports:    {real_summary.mean_latency_ms:8.3f} ms")
    print("=" * 70)

    dist_to_raw = abs(real_summary.mean_latency_ms - raw_ms)
    dist_to_full = abs(real_summary.mean_latency_ms - full_ms)
    if dist_to_raw < dist_to_full:
        print(f"\n=> (C) is close to (A) [diff {dist_to_raw:.3f} ms] not (B) [diff {dist_to_full:.3f} ms]:")
        print("   resize is currently EXCLUDED from the timed benchmark (raw-forward mode is active).")
    else:
        print(f"\n=> (C) is close to (B) [diff {dist_to_full:.3f} ms] not (A) [diff {dist_to_raw:.3f} ms]:")
        print("   resize is currently INCLUDED in the timed benchmark (full-pipeline mode is active).")


if __name__ == "__main__":
    main()
