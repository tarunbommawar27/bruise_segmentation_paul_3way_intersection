#!/usr/bin/env python3
"""
scripts/29_benchmark_inference_all_models.py

Benchmark inference speed for ALL WL-deployable models in the bruise pipeline.

This combines the model list from:
  - scripts/27_evaluate_test_640_models_01_to_11.py
  - scripts/28_evaluate_test_640_models_12_plus.py

and uses the deployment-style timing logic from benchmark_inference_fair_v3.py:

Two FPS numbers are reported per model:
  1. raw_forward_fps:
       nn.Module forward only.
       Input tensor is already preprocessed and already on GPU.
       This isolates model architecture speed.

  2. full_pipeline_fps:
       camera RGB image already in RAM
       -> resize / normalize
       -> GPU tensor
       -> model forward
       -> sigmoid or temperature scaling
       -> threshold
       -> resize binary mask back to the original image resolution
       This is the real camera-to-overlay-mask deployment speed.

Important:
  - Thresholds are loaded from each model's validation threshold_search.csv.
  - Thresholds are NOT recomputed on the test images.
  - Disk imread is NOT timed because a real camera frame is already in RAM.
  - Model loading is NOT timed.
  - Warmup iterations are NOT timed.
  - segformer_b2_als_teacher is NOT included by default because it is an ALS-input
    model, not a WL-deployable model. Benchmark it separately on ALS images if needed.

Usage:
    conda activate bruise_orc
    cd /home/tbommawa/bruise_pipeline_root

    python scripts/29_benchmark_inference_all_models.py \
        --paths configs/paths.yaml \
        --common configs/common_train.yaml \
        --n-images 20 \
        --n-iters 200 \
        --warmup 20 \
        --out results/benchmark_all_models_v1.csv

Fast sanity check:
    python scripts/29_benchmark_inference_all_models.py \
        --paths configs/paths.yaml \
        --common configs/common_train.yaml \
        --n-images 3 \
        --n-iters 30 \
        --warmup 10 \
        --out results/benchmark_all_models_quick.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.models import load_segformer_model
from pipeline.yolo_threshold_temp import bruise_prob_from_logits, load_yolo_model, yolo_raw_class_logits

logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# WL-deployable models
# ─────────────────────────────────────────────────────────────────────────────

SEGFORMER_MODELS = [
    # scripts 01-11
    ("segformer_b2_teacher", "segformer_b2_pretrained", "b2"),
    ("segformer_b0_direct", "segformer_b0_pretrained", "b0"),
    ("segformer_b0_distilled", "segformer_b0_pretrained", "b0"),

    # scripts 12+
    ("segformer_b0_fairness_distill_approach_a", "segformer_b0_pretrained", "b0"),
    ("segformer_b0_fairness_distill_approach_b", "segformer_b0_pretrained", "b0"),
    ("segformer_b0_als_to_wl_distilled", "segformer_b0_pretrained", "b0"),
]

YOLO_MODELS = [
    "yolo_sem_direct",
    "yolo_sem_distilled",
]


# ─────────────────────────────────────────────────────────────────────────────
# Image loading and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_image_raw(image_path: str) -> tuple[np.ndarray, tuple[int, int], str]:
    """Load image into RAM. Disk read is setup, not part of timed camera pipeline."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    stem = Path(str(image_path)).stem
    return img_rgb, (orig_h, orig_w), stem


def make_segformer_input(img_rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """Same inference preprocessing structure as BruiseDataset for SegFormer."""
    tfm = A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    x = tfm(image=img_rgb)["image"].float().unsqueeze(0)
    return x.to(device)


def make_yolo_input(img_rgb: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    """Same inference preprocessing structure as BruiseDataset for YOLO."""
    tfm = A.Compose([
        A.Resize(height=size, width=size),
        ToTensorV2(),
    ])
    x = (tfm(image=img_rgb)["image"].float() / 255.0).unsqueeze(0)
    return x.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Generic GPU benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_gpu_fn(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int,
    n_iters: int,
) -> tuple[float, float, float]:
    """Return mean_ms, std_ms, fps for a GPU callable."""
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize(device)

    times = np.empty(n_iters, dtype=np.float64)

    for i in range(n_iters):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times[i] = (time.perf_counter() - t0) * 1000.0

    mean_ms = float(times.mean())
    std_ms = float(times.std())
    fps = 1000.0 / mean_ms
    return mean_ms, std_ms, fps


def dtype_configs(include_fp32: bool, include_fp16: bool):
    configs = []
    if include_fp32:
        configs.append(("fp32", torch.float32, False))
    if include_fp16:
        configs.append(("fp16", torch.float16, True))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Per-model benchmark functions
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_segformer_one_dtype(
    model: torch.nn.Module,
    model_name: str,
    architecture: str,
    threshold: float,
    checkpoint: Path,
    image_records: list[dict],
    size: int,
    dtype_name: str,
    torch_dtype: torch.dtype,
    use_amp: bool,
    device: torch.device,
    warmup: int,
    n_iters: int,
) -> tuple[dict, list[dict]]:

    first = image_records[0]
    x_raw = make_segformer_input(first["img_rgb"], size, device).to(torch_dtype)

    with torch.no_grad():
        def raw_forward():
            with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                model(x_raw)

        raw_ms, raw_std, raw_fps = benchmark_gpu_fn(raw_forward, device, warmup, n_iters)

        per_image_rows = []

        for rec in image_records:
            img_rgb = rec["img_rgb"]
            orig_h, orig_w = rec["orig_h"], rec["orig_w"]

            def full_pipeline():
                x = make_segformer_input(img_rgb, size, device).to(torch_dtype)

                with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                    logits = model(x)

                prob = torch.sigmoid(logits.float())
                mask_640 = (prob[0, 0] >= threshold).cpu().numpy().astype("uint8")
                cv2.resize(mask_640, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            full_ms, full_std, full_fps = benchmark_gpu_fn(full_pipeline, device, warmup, n_iters)

            per_image_rows.append({
                "model": model_name,
                "model_family": "segformer",
                "architecture": architecture,
                "dtype": dtype_name,
                "resolution": size,
                "image_stem": rec["stem"],
                "orig_w": orig_w,
                "orig_h": orig_h,
                "full_pipeline_mean_ms": full_ms,
                "full_pipeline_std_ms": full_std,
                "full_pipeline_fps": full_fps,
            })

    full_mean_ms = float(np.mean([r["full_pipeline_mean_ms"] for r in per_image_rows]))
    full_std_across_images_ms = float(np.std([r["full_pipeline_mean_ms"] for r in per_image_rows]))
    full_fps = 1000.0 / full_mean_ms

    summary = {
        "model": model_name,
        "model_family": "segformer",
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "dtype": dtype_name,
        "resolution": size,
        "n_images": len(image_records),
        "n_iters_per_image": n_iters,
        "warmup": warmup,
        "threshold": threshold,
        "temperature": "",
        "raw_forward_mean_ms": raw_ms,
        "raw_forward_std_ms": raw_std,
        "raw_forward_fps": raw_fps,
        "full_pipeline_mean_ms": full_mean_ms,
        "full_pipeline_std_across_images_ms": full_std_across_images_ms,
        "full_pipeline_fps": full_fps,
        "timed_full_pipeline": "preprocess + forward + sigmoid + threshold + resize_to_original",
    }

    return summary, per_image_rows


def benchmark_yolo_one_dtype(
    model: torch.nn.Module,
    model_name: str,
    threshold: float,
    temperature: float,
    checkpoint: Path,
    image_records: list[dict],
    size: int,
    dtype_name: str,
    torch_dtype: torch.dtype,
    use_amp: bool,
    device: torch.device,
    warmup: int,
    n_iters: int,
) -> tuple[dict, list[dict]]:

    first = image_records[0]
    x_raw = make_yolo_input(first["img_rgb"], size, device).to(torch_dtype)

    with torch.no_grad():
        def raw_forward():
            with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                yolo_raw_class_logits(model, x_raw, out_hw=(size, size))

        raw_ms, raw_std, raw_fps = benchmark_gpu_fn(raw_forward, device, warmup, n_iters)

        per_image_rows = []

        for rec in image_records:
            img_rgb = rec["img_rgb"]
            orig_h, orig_w = rec["orig_h"], rec["orig_w"]

            def full_pipeline():
                x = make_yolo_input(img_rgb, size, device).to(torch_dtype)

                with torch.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                    class_logits = yolo_raw_class_logits(model, x, out_hw=(size, size))

                prob = bruise_prob_from_logits(class_logits.float(), temperature)
                mask_640 = (prob[0] >= threshold).cpu().numpy().astype("uint8")
                cv2.resize(mask_640, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            full_ms, full_std, full_fps = benchmark_gpu_fn(full_pipeline, device, warmup, n_iters)

            per_image_rows.append({
                "model": model_name,
                "model_family": "yolo",
                "architecture": "yolo_sem",
                "dtype": dtype_name,
                "resolution": size,
                "image_stem": rec["stem"],
                "orig_w": orig_w,
                "orig_h": orig_h,
                "full_pipeline_mean_ms": full_ms,
                "full_pipeline_std_ms": full_std,
                "full_pipeline_fps": full_fps,
            })

    full_mean_ms = float(np.mean([r["full_pipeline_mean_ms"] for r in per_image_rows]))
    full_std_across_images_ms = float(np.std([r["full_pipeline_mean_ms"] for r in per_image_rows]))
    full_fps = 1000.0 / full_mean_ms

    summary = {
        "model": model_name,
        "model_family": "yolo",
        "architecture": "yolo_sem",
        "checkpoint": str(checkpoint),
        "dtype": dtype_name,
        "resolution": size,
        "n_images": len(image_records),
        "n_iters_per_image": n_iters,
        "warmup": warmup,
        "threshold": threshold,
        "temperature": temperature,
        "raw_forward_mean_ms": raw_ms,
        "raw_forward_std_ms": raw_std,
        "raw_forward_fps": raw_fps,
        "full_pipeline_mean_ms": full_mean_ms,
        "full_pipeline_std_across_images_ms": full_std_across_images_ms,
        "full_pipeline_fps": full_fps,
        "timed_full_pipeline": "preprocess + forward + temperature + threshold + resize_to_original",
    }

    return summary, per_image_rows


# ─────────────────────────────────────────────────────────────────────────────
# Image selection
# ─────────────────────────────────────────────────────────────────────────────

def select_benchmark_images(paths: dict, args: argparse.Namespace) -> list[dict]:
    if args.image:
        img_paths = [args.image]
    else:
        test_df = load_fixed_test(paths["fixed_test_manifest"])
        if "image_path" not in test_df.columns:
            raise KeyError("fixed_test_manifest must provide image_path via load_fixed_test().")

        if args.sample_seed is not None:
            test_df = test_df.sample(frac=1.0, random_state=args.sample_seed).reset_index(drop=True)

        img_paths = list(test_df["image_path"].head(args.n_images))

    records = []
    for p in img_paths:
        img_rgb, (orig_h, orig_w), stem = load_image_raw(str(p))
        records.append({
            "image_path": str(p),
            "stem": stem,
            "img_rgb": img_rgb,
            "orig_h": orig_h,
            "orig_w": orig_w,
        })

    if not records:
        raise RuntimeError("No benchmark images selected.")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark all WL-deployable bruise segmentation models.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--out", default="results/benchmark_all_models_v1.csv")
    ap.add_argument("--per-image-out", default=None)

    ap.add_argument("--image", default=None, help="Optional single image path. If omitted, fixed test manifest is used.")
    ap.add_argument("--n-images", type=int, default=20, help="Number of fixed-test images to benchmark.")
    ap.add_argument("--sample-seed", type=int, default=2026, help="Shuffle test images before selecting; use -1 to disable.")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[640])

    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)

    ap.add_argument("--fp32", action="store_true", help="Benchmark fp32 only unless --fp16 is also given.")
    ap.add_argument("--fp16", action="store_true", help="Benchmark fp16 only unless --fp32 is also given.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional subset of model names to benchmark.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_seed == -1:
        args.sample_seed = None

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for benchmark. CPU FPS is not meaningful here.")

    # Default: run both fp32 and fp16 unless user explicitly selects one.
    include_fp32 = args.fp32 or not (args.fp32 or args.fp16)
    include_fp16 = args.fp16 or not (args.fp32 or args.fp16)

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device = torch.device("cuda:0")
    size_list = args.resolutions

    image_records = select_benchmark_images(paths, args)
    logger.info("Selected %d benchmark image(s). First=%s %dx%d",
                len(image_records), image_records[0]["stem"],
                image_records[0]["orig_w"], image_records[0]["orig_h"])

    allowed_models = set(args.models) if args.models else None

    summary_rows = []
    per_image_rows = []

    # ── SegFormer models ────────────────────────────────────────────────────
    for model_name, pretrained_key, architecture in SEGFORMER_MODELS:
        if allowed_models is not None and model_name not in allowed_models:
            continue

        logger.info("========== Benchmarking %s ==========" , model_name)

        try:
            model, threshold, checkpoint = load_segformer_model(model_name, pretrained_key, paths, device)
        except Exception as e:
            logger.warning("SKIP %s: %s", model_name, e)
            continue

        try:
            for size in size_list:
                for dtype_name, torch_dtype, use_amp in dtype_configs(include_fp32, include_fp16):
                    logger.info("[%s] resolution=%d dtype=%s", model_name, size, dtype_name)
                    summary, per_img = benchmark_segformer_one_dtype(
                        model=model,
                        model_name=model_name,
                        architecture=architecture,
                        threshold=threshold,
                        checkpoint=checkpoint,
                        image_records=image_records,
                        size=size,
                        dtype_name=dtype_name,
                        torch_dtype=torch_dtype,
                        use_amp=use_amp,
                        device=device,
                        warmup=args.warmup,
                        n_iters=args.n_iters,
                    )
                    summary_rows.append(summary)
                    per_image_rows.extend(per_img)
                    logger.info("[%s %s] raw=%.2f FPS | full=%.2f FPS",
                                model_name, dtype_name,
                                summary["raw_forward_fps"], summary["full_pipeline_fps"])
        finally:
            del model
            torch.cuda.empty_cache()

    # ── YOLO models ─────────────────────────────────────────────────────────
    for model_name in YOLO_MODELS:
        if allowed_models is not None and model_name not in allowed_models:
            continue

        logger.info("========== Benchmarking %s ==========" , model_name)

        try:
            model, threshold, temperature, checkpoint = load_yolo_model(model_name, paths, device)
        except Exception as e:
            logger.warning("SKIP %s: %s", model_name, e)
            continue

        try:
            for size in size_list:
                for dtype_name, torch_dtype, use_amp in dtype_configs(include_fp32, include_fp16):
                    logger.info("[%s] resolution=%d dtype=%s", model_name, size, dtype_name)
                    summary, per_img = benchmark_yolo_one_dtype(
                        model=model,
                        model_name=model_name,
                        threshold=threshold,
                        temperature=temperature,
                        checkpoint=checkpoint,
                        image_records=image_records,
                        size=size,
                        dtype_name=dtype_name,
                        torch_dtype=torch_dtype,
                        use_amp=use_amp,
                        device=device,
                        warmup=args.warmup,
                        n_iters=args.n_iters,
                    )
                    summary_rows.append(summary)
                    per_image_rows.extend(per_img)
                    logger.info("[%s %s] raw=%.2f FPS | full=%.2f FPS",
                                model_name, dtype_name,
                                summary["raw_forward_fps"], summary["full_pipeline_fps"])
        finally:
            del model
            torch.cuda.empty_cache()

    if not summary_rows:
        raise RuntimeError("No benchmark rows produced. Check model checkpoints and threshold_search.csv files.")

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_path, index=False)

    per_image_out = Path(args.per_image_out) if args.per_image_out else out_path.with_name(out_path.stem + "_per_image.csv")
    ensure_dir(per_image_out.parent)
    per_image_df = pd.DataFrame(per_image_rows)
    per_image_df.to_csv(per_image_out, index=False)

    logger.info("Saved summary benchmark -> %s", out_path)
    logger.info("Saved per-image benchmark -> %s", per_image_out)

    display_cols = [
        "model", "dtype", "resolution",
        "raw_forward_fps", "raw_forward_mean_ms",
        "full_pipeline_fps", "full_pipeline_mean_ms",
        "threshold", "temperature",
    ]
    logger.info("\n%s", summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
