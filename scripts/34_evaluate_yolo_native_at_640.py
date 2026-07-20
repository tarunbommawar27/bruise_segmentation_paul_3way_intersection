#!/usr/bin/env python3
"""
scripts/34_evaluate_yolo_native_at_640.py

Evaluates YOLO26n-sem (direct + distilled) using Ultralytics' NATIVE
inference path, but scored on the same 640x640 grid every SegFormer model is
scored on -- making YOLO and SegFormer numbers directly comparable for the
first time.

WHY THIS SCRIPT EXISTS
-----------------------
Three things differ between this project's custom raw-logit YOLO evaluation
(pipeline/yolo_threshold_temp.py + pipeline.data.BruiseDataset) and
Ultralytics' native .predict() path:

  1. NORMALIZATION. The custom path applies ImageNet mean/std normalization
     (BruiseDataset). YOLO was trained through Ultralytics' own loop
     (pipeline/yolo_stage.py::model.train(imgsz=640)), which scales to [0,1]
     with no ImageNet normalization -- a different input distribution than
     the backbone's filters ever learned. Native .predict() matches training.
  2. GEOMETRY. The custom path stretch-resizes to a square (A.Resize).
     Ultralytics letterboxes (preserves aspect ratio, pads to square) in both
     training and .predict(). Native .predict() again matches training.
  3. COMPARISON GRID. scripts/yolo_wl_audit_v1.py's native path compares at
     the image's NATIVE camera resolution, because .predict() internally runs
     the model at imgsz=640 and then upsamples its class map back to the
     original image size before returning it (verified: for a 4022x6024 input,
     result.semantic_mask.data comes back as (4022, 6024), so
     pipeline/mask_utils.py::yolo_sem_pred_mask's cv2.resize to gt.shape is a
     no-op). Every SegFormer number in this project is computed at 640x640.

(1) and (2) are NOT bugs to fix here -- they are how YOLO is correctly run,
and each architecture should be evaluated in the inference mode it was
trained for. (3) IS a confound: Dice computed at 4022x6024 is not comparable
to Dice computed at 640x640 (boundary error is a much smaller fraction of
total pixels at native resolution, which systematically inflates Dice). This
script removes confound (3) only, by bringing YOLO's native prediction back
to the 640 grid and scoring it against GT resized to 640 the same way
BruiseDataset resizes GT for every SegFormer evaluation.

WHAT "BACK TO THE 640 GRID" MEANS EXACTLY
--------------------------------------------
Ultralytics hands back a native-resolution class map (already undone from its
own letterbox). That map is resized to 640x640 with INTER_NEAREST, and the
ground-truth mask is resized from native to 640x640 with INTER_NEAREST too.
Nearest-neighbour on both sides keeps masks strictly binary (see
pipeline/data.py's own eval transform and the knowledge document's
"why nearest-neighbor is used for masks"). Both sides therefore land on the
identical grid via the identical operation, which is exactly the property the
SegFormer evaluations already have.

Deliberately NOT reusing pipeline.data.BruiseDataset here: that class applies
ImageNet normalization to the image, which is the very thing native inference
must avoid. Only its GT-resize convention is mirrored.

Usage (from project root):
    python scripts/34_evaluate_yolo_native_at_640.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# cv2 BEFORE ultralytics: on this Windows machine, importing ultralytics first
# makes cv2.imread(..., IMREAD_GRAYSCALE) return (H,W,1) instead of (H,W)
# (documented package conflict -- see SESSION_HANDOFF.md "Known environment
# bugs"). Masks are squeezed defensively below regardless, so this ordering is
# belt-and-braces rather than the sole guard.
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import read_gt_mask
from pipeline.metrics import compute_image_row, summarize

logger = setup_logging()

YOLO_MODELS = ["yolo_sem_direct", "yolo_sem_distilled"]


def _to_640_nearest(mask: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Resize a binary mask to the 640 grid with nearest-neighbour.

    Squeeze first: read_gt_mask/cv2 can hand back (H,W,1) on this machine (see
    the cv2/ultralytics import-order note in this file's header), and
    cv2.resize on a trailing-singleton array silently keeps that extra axis,
    which would then break the (H,W)-shaped metric functions downstream.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze(-1)
    resized = cv2.resize(mask.astype("uint8"), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype("uint8")


def evaluate_one(model_name: str, paths: dict, cfg: dict, device_str: str,
                  out_root: Path) -> dict:
    from ultralytics import YOLO

    run_dir = Path(paths["project_root"]) / model_name
    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {best_pt}")

    img_h, img_w = cfg["img_h"], cfg["img_w"]
    test_df = load_fixed_test(paths["fixed_test_manifest"])
    wrapper = YOLO(str(best_pt))

    rows = []
    for i, r in enumerate(test_df.itertuples(index=False), start=1):
        # Native Ultralytics inference: letterbox + /255 internally at
        # imgsz=640, then Ultralytics maps the class map back to native size.
        res = wrapper.predict(str(r.image_path), imgsz=img_h, device=device_str, verbose=False)[0]
        class_map = res.semantic_mask.data
        if hasattr(class_map, "cpu"):
            class_map = class_map.cpu().numpy()
        pred_native = (np.asarray(class_map) == 1).astype("uint8")

        gt_native = read_gt_mask(str(r.mask_path))

        # Both sides -> the identical 640 grid via the identical operation.
        pred_640 = _to_640_nearest(pred_native, img_h, img_w)
        gt_640 = _to_640_nearest(gt_native, img_h, img_w)

        rows.append(compute_image_row(pred_640, gt_640, str(r.stem)))
        if i % 25 == 0:
            logger.info("  [%s] %d/%d images", model_name, i, len(test_df))

    per_image = pd.DataFrame(rows)
    summary = summarize(rows)
    summary.update({
        "run_name": model_name,
        "eval_variant": "ultralytics_native_scored_at_640",
        "n_images": len(per_image),
        "note": ("Ultralytics native .predict() inference (letterbox + /255, "
                 "matching training); prediction and GT both resized native->640 "
                 "with INTER_NEAREST so metrics are directly comparable to the "
                 "SegFormer 640 evaluations."),
    })

    out_dir = ensure_dir(out_root / model_name)
    per_image.to_csv(out_dir / "test_per_image.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)
    logger.info("[%s] mean_dice=%.4f median_dice=%.4f miss_rate=%.4f -> %s",
                model_name, summary["mean_dice"], summary["median_dice"],
                summary["complete_miss_rate"], out_dir)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--out-dir", default="yolo_native_640_evaluation")
    ap.add_argument("--device", default=None,
                    help="Ultralytics device string ('0' for cuda:0, 'cpu'). Default: auto.")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    if args.device is not None:
        device_str = args.device
    else:
        import torch
        device_str = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device_str)

    out_root = ensure_dir(Path(paths["project_root"]) / args.out_dir)

    summaries = []
    for model_name in YOLO_MODELS:
        summaries.append(evaluate_one(model_name, paths, cfg, device_str, out_root))

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(out_root / "yolo_native_640_comparison.csv", index=False)
    logger.info("Saved comparison: %s", out_root / "yolo_native_640_comparison.csv")


if __name__ == "__main__":
    main()
