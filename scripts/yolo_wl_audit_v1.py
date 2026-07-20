#!/usr/bin/env python3
"""
scripts/yolo_wl_audit_v1.py

White-light YOLO audit — single script covering all 4 items raised in the
2026-07-07 ML model meeting with Artin. NOT numbered (not a pipeline stage,
just a diagnostic tool) and deliberately NOT modular yet — everything lives
in one file for now; will be split up once the approach is validated.

What this does, per model (yolo_sem_direct, yolo_sem_distilled):

  1. FPS BENCHMARK AT 640 ONLY — no upsample to 6024x4022. Earlier guidance
     said the deployment mask must be resized back to camera resolution
     inside the timed block (see benchmark_inference_fair_v3.py); today's
     meeting reversed that for this comparison — report raw_forward_fps and
     full_pipeline_fps purely at the model's native 640x640, since SegFormer's
     own numbers are already reported that way and Artin wants an apples to
     apples comparison at 640.

  2. TEST DICE AT 640, GT RESIZED BY THE DATA LOADER — uses
     pipeline.data.BruiseDataset, which resizes image AND mask together via
     one Albumentations A.Resize(img_h, img_w) call (pipeline/data.py:93).
     This is the "let the data loader handle it, don't manually resize"
     fix — prediction and GT are compared at the same 640x640 resolution,
     never at the image's native resolution.

  3. TEMPERATURE SCALING SANITY CHECK — threshold+temperature are searched
     ONCE on val (pipeline.yolo_threshold_temp.run_threshold_search, which
     already implements "search once on val, apply to test" -- see
     scripts/07b_threshold_yolo_direct.py) and applied to test. We report
     test Dice at the swept best (T, threshold) AND at T=1.0 (no temperature
     scaling, just its own best val threshold) side by side, so it's
     immediately visible whether temperature scaling is actually buying
     anything. If T=1.0 wins or ties, the guidance from the meeting is to
     drop temperature scaling entirely.

  4. BARE PYTORCH + OUR THRESHOLD ANALYSIS  vs.  ULTRALYTICS AS-IS — compares
     our own pipeline (raw nn.Module -> temperature -> our swept threshold,
     via pipeline.yolo_threshold_temp) against Ultralytics' native
     .predict() output (its own internal sigmoid+argmax, no temperature, no
     threshold control -- see pipeline/mask_utils.py::yolo_sem_pred_mask)
     on the SAME fixed test set. This is the check Artin asked for before
     deciding whether temperature scaling is worth keeping at all. Note the
     two paths are compared at DIFFERENT resolutions by nature of what each
     API returns (ours: 640x640 via the data loader; Ultralytics: native
     image resolution via yolo_sem_pred_mask's internal resize) -- that
     difference is exactly what's being audited, so it is reported
     explicitly rather than silently normalised away.

Usage:
    python scripts/yolo_wl_audit_v1.py --paths configs/paths.yaml --common configs/common_train.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_fixed_test, load_train_val_split
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import read_gt_mask, yolo_sem_pred_mask
from pipeline.metrics import compute_image_row, summarize
from pipeline.yolo_threshold_temp import (bruise_prob_from_logits, evaluate_yolo_raw,
                                           run_threshold_search, yolo_raw_class_logits)

logger = setup_logging()

YOLO_MODELS = ["yolo_sem_direct", "yolo_sem_distilled"]
N_WARMUP = 20
N_ITERS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — FPS benchmark at 640 only (no resize back to camera resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _make_yolo_input_640(img_rgb: np.ndarray, img_h: int, img_w: int, device: torch.device) -> torch.Tensor:
    """Same structural preprocessing as BruiseDataset(training=False) for YOLO
    (resize -> /255, no ImageNet norm) -- kept local rather than imported from
    another script, since numbered/utility scripts in this repo aren't meant
    to import each other (see scripts/16's docstring)."""
    tfm = A.Compose([A.Resize(height=img_h, width=img_w), ToTensorV2()])
    x = tfm(image=img_rgb)["image"].float() / 255.0
    return x.unsqueeze(0).to(device)


def _benchmark_fps_640(yolo_nn, img_rgb: np.ndarray, img_h: int, img_w: int,
                        best_thr: float, best_temp: float, device: torch.device) -> dict:
    """raw_forward_fps: nn.Module(x) only, input already on GPU at 640x640.
    full_pipeline_fps: preprocess + forward + temp-scale + threshold, ALL AT
    640x640 -- deliberately no cv2.resize back to original camera resolution,
    per today's meeting (item 1)."""

    def raw_forward():
        with torch.no_grad():
            yolo_raw_class_logits(yolo_nn, x_gpu, out_hw=(img_h, img_w))

    def full_pipeline():
        with torch.no_grad():
            x = _make_yolo_input_640(img_rgb, img_h, img_w, device)
            class_logits = yolo_raw_class_logits(yolo_nn, x, out_hw=(img_h, img_w))
            prob = bruise_prob_from_logits(class_logits, best_temp)
            (prob[0] >= best_thr).cpu().numpy().astype("uint8")

    x_gpu = _make_yolo_input_640(img_rgb, img_h, img_w, device)

    def _time(fn) -> tuple[float, float]:
        for _ in range(N_WARMUP):
            fn()
        torch.cuda.synchronize(device)
        times = np.empty(N_ITERS, dtype="float64")
        for i in range(N_ITERS):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            times[i] = (time.perf_counter() - t0) * 1000.0
        return float(times.mean()), float(times.std())

    raw_ms, raw_std = _time(raw_forward)
    full_ms, full_std = _time(full_pipeline)
    return {
        "raw_forward_fps": 1000.0 / raw_ms, "raw_forward_mean_ms": raw_ms, "raw_forward_std_ms": raw_std,
        "full_pipeline_fps": 1000.0 / full_ms, "full_pipeline_mean_ms": full_ms, "full_pipeline_std_ms": full_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Item 4 — Ultralytics as-is (native .predict(), no temperature, no custom thr)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_ultralytics_native(wrapper, test_df: pd.DataFrame, img_h: int, device_str: str) -> pd.DataFrame:
    """Ultralytics' own postprocessing: internal sigmoid + argmax, at the
    image's NATIVE resolution (yolo_sem_pred_mask resizes its class map back
    up with cv2.INTER_NEAREST) -- no temperature scaling, no threshold we
    control. This is the "as is" side of the item-4 comparison."""
    rows = []
    for _, r in test_df.iterrows():
        gt = read_gt_mask(r.mask_path).astype("uint8")
        res = wrapper.predict(str(r.image_path), imgsz=img_h, device=device_str, verbose=False)[0]
        pred = yolo_sem_pred_mask(res, gt.shape)
        rows.append(compute_image_row(pred, gt, str(r.stem)))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model audit
# ─────────────────────────────────────────────────────────────────────────────

def _audit_one_model(model_name: str, paths: dict, cfg: dict, device: torch.device) -> None:
    from ultralytics import YOLO

    run_dir = Path(paths["project_root"]) / model_name
    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        logger.warning("SKIP %s: no trained weights at %s.", model_name, best_pt)
        return

    out_dir = ensure_dir(run_dir / "wl_audit_v1")
    img_h, img_w = cfg["img_h"], cfg["img_w"]

    # ── Item 3 (part 1) — threshold+temperature search ONCE on val ──────────
    _, val_df = load_train_val_split(Path(paths["project_root"]) / "splits" / "train_val_split.csv")
    test_df = load_fixed_test(paths["fixed_test_manifest"])
    logger.info("[%s] val=%d test=%d device=%s", model_name, len(val_df), len(test_df), device)

    grid_df, best_thr, best_temp = run_threshold_search(str(best_pt), val_df, cfg, device, run_dir)
    t1_rows = grid_df[grid_df["temperature"] == 1.0].sort_values("mean_dice", ascending=False)
    if len(t1_rows) == 0:
        raise RuntimeError(
            f"{run_dir/'threshold_search.csv'} has no temperature=1.0 rows — "
            "cfg['yolo_temperatures'] must include 1.0 as the no-scaling baseline.")
    t1_thr = float(t1_rows.iloc[0]["threshold"])
    logger.info("[%s] val best: T=%.2f thr=%.2f dice=%.4f | T=1.0 baseline: thr=%.2f dice=%.4f",
                model_name, best_temp, best_thr, float(grid_df.iloc[0]["mean_dice"]),
                t1_thr, float(t1_rows.iloc[0]["mean_dice"]))

    # ── Item 2 + 3 (part 2) — test Dice at 640, GT resized by the data loader,
    # swept (T, thr) vs T=1.0 baseline, applied-not-recomputed-per-image ─────
    test_loader = torch.utils.data.DataLoader(
        BruiseDataset(test_df, img_h, img_w, training=False),
        batch_size=cfg.get("yolo_threshold_search_batch", 4),
        shuffle=False, num_workers=cfg.get("workers", 4), pin_memory=True,
    )
    yolo_wrapper = YOLO(str(best_pt))
    yolo_nn = copy.deepcopy(yolo_wrapper.model).to(device).eval()

    df_temp_scaled, summary_temp_scaled = evaluate_yolo_raw(yolo_nn, test_loader, device, best_thr, best_temp)
    df_t1, summary_t1 = evaluate_yolo_raw(yolo_nn, test_loader, device, t1_thr, 1.0)
    df_temp_scaled.to_csv(out_dir / "test_per_image_640_temp_scaled.csv", index=False)
    df_t1.to_csv(out_dir / "test_per_image_640_no_temp.csv", index=False)
    logger.info("[%s] TEST @640  temp-scaled(T=%.2f,thr=%.2f): mean_dice=%.4f miss_rate=%.4f",
                model_name, best_temp, best_thr,
                summary_temp_scaled["mean_dice"], summary_temp_scaled["complete_miss_rate"])
    logger.info("[%s] TEST @640  no-temp    (T=1.0,thr=%.2f): mean_dice=%.4f miss_rate=%.4f",
                model_name, t1_thr,
                summary_t1["mean_dice"], summary_t1["complete_miss_rate"])
    if summary_t1["mean_dice"] >= summary_temp_scaled["mean_dice"]:
        logger.warning("[%s] Temperature scaling does NOT improve test Dice — "
                        "per today's meeting, consider dropping it.", model_name)

    # ── Item 4 — Ultralytics as-is vs our PyTorch+threshold approach ────────
    device_str = "0" if torch.cuda.is_available() else "cpu"
    df_native = _evaluate_ultralytics_native(yolo_wrapper, test_df, img_h, device_str)
    df_native.to_csv(out_dir / "test_per_image_ultralytics_native.csv", index=False)
    summary_native = summarize(df_native.to_dict("records"))
    logger.info("[%s] TEST @native-res  Ultralytics as-is (no temp, no custom thr): "
                "mean_dice=%.4f miss_rate=%.4f",
                model_name, summary_native["mean_dice"], summary_native["complete_miss_rate"])

    # ── Item 1 — FPS at 640 only, no resize back to camera resolution ──────
    sample_img_bgr = cv2.imread(str(test_df.iloc[0]["image_path"]))
    sample_img_rgb = cv2.cvtColor(sample_img_bgr, cv2.COLOR_BGR2RGB)
    fps = _benchmark_fps_640(yolo_nn, sample_img_rgb, img_h, img_w, best_thr, best_temp, device)
    logger.info("[%s] FPS @640  raw_forward=%.1f  full_pipeline=%.1f",
                model_name, fps["raw_forward_fps"], fps["full_pipeline_fps"])

    # ── Save one consolidated summary row ───────────────────────────────────
    summary_row = {
        "model": model_name,
        "val_best_temperature": best_temp, "val_best_threshold": best_thr,
        "val_t1_threshold": t1_thr,
        "test640_temp_scaled_mean_dice": summary_temp_scaled["mean_dice"],
        "test640_temp_scaled_miss_rate": summary_temp_scaled["complete_miss_rate"],
        "test640_no_temp_mean_dice": summary_t1["mean_dice"],
        "test640_no_temp_miss_rate": summary_t1["complete_miss_rate"],
        "test_native_ultralytics_mean_dice": summary_native["mean_dice"],
        "test_native_ultralytics_miss_rate": summary_native["complete_miss_rate"],
        **fps,
    }
    pd.DataFrame([summary_row]).to_csv(out_dir / "audit_summary.csv", index=False)
    logger.info("[%s] Audit complete -> %s", model_name, out_dir / "audit_summary.csv")

    del yolo_nn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description="White-light YOLO audit: FPS@640, Dice@640, temp-scaling check, Ultralytics-vs-PyTorch")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this audit (FPS numbers are meaningless on CPU).")
    device = torch.device("cuda:0")

    for model_name in YOLO_MODELS:
        _audit_one_model(model_name, paths, cfg, device)


if __name__ == "__main__":
    main()
