#!/usr/bin/env python3
"""
scripts/27_evaluate_test_640_models_01_to_11.py

Test Dice for the 5 models produced by scripts 01-11 (segformer_b2_teacher,
segformer_b0_direct, segformer_b0_distilled, yolo_sem_direct,
yolo_sem_distilled), ALWAYS at 640x640 with ground truth resized DOWN by the
data loader (pipeline.data.BruiseDataset's single A.Resize call) -- NEVER
upscaling a prediction back up to the image's native camera resolution
(6024x4022) to compare against a native-resolution GT.

WHY THIS SCRIPT EXISTS (separate from 09/10/11):
  scripts/09_evaluate_test.py's eval_yolo() calls Ultralytics' native
  .predict(), which internally resizes YOLO's 640x640 prediction UP to
  gt.shape (native resolution) via pipeline.mask_utils.yolo_sem_pred_mask's
  cv2.INTER_NEAREST resize, then compares against the native-resolution GT
  mask -- that upscaling step is exactly what this script avoids. Scripts
  10/11 already avoid it correctly (both SegFormer and YOLO go through
  BruiseDataset at 640x640 for GT, and YOLO's raw class logits are resized
  to out_hw=640x640, not native), but they're bundled with Track A/B
  controlled-comparison framing, bootstrap CI, Wilcoxon tests, and 3-way
  speed benchmarking -- more than is needed for a plain "what's the Dice at
  640, no upscaling, anywhere" number. This script isolates just that.

  This is the same "Item 2" methodology already established in
  scripts/yolo_wl_audit_v1.py ("TEST DICE AT 640, GT RESIZED BY THE DATA
  LOADER") generalised to all 5 pre-Phase-3 models, not just the 2 YOLO ones.

METHOD (identical for all 5 models):
  1. Load each model's own val-selected threshold from its existing
     threshold_search.csv (never re-derive on test).
  2. Run inference through pipeline.data.BruiseDataset(test_df, img_h, img_w,
     training=False) -- ONE resize call handles both the image (for the
     model) and the mask (for the GT comparison), so pred and GT are always
     compared at the same 640x640 resolution.
  3. For YOLO: bypass Ultralytics' own .predict()/argmax entirely -- use
     pipeline.yolo_threshold_temp.yolo_raw_class_logits(..., out_hw=(640,640))
     so its raw class logits are resized to 640x640 (matching BruiseDataset's
     GT), never to native resolution, then apply its own val-selected
     (threshold, temperature).
  4. Score with pipeline.metrics_extended (dice_safe / summarize_extended),
     never pipeline.metrics's dice_np() (see that module's docstring for why).

Per this repo's convention that numbered/utility scripts don't import each
other (see scripts/16's docstring, scripts/yolo_wl_audit_v1.py), the small
inference helpers here are NOT imported from 10/11/17/25 -- they're
duplicated locally, kept intentionally minimal.

Usage:
    python scripts/27_evaluate_test_640_models_01_to_11.py \\
        --paths configs/paths.yaml --common configs/common_train.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.metrics_extended import compute_image_row_extended, summarize_extended
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.yolo_threshold_temp import bruise_prob_from_logits, yolo_raw_class_logits

logger = setup_logging()

SEGFORMER_MODELS = [
    ("segformer_b2_teacher", "segformer_b2_pretrained"),
    ("segformer_b0_direct", "segformer_b0_pretrained"),
    ("segformer_b0_distilled", "segformer_b0_pretrained"),
]
YOLO_MODELS = ["yolo_sem_direct", "yolo_sem_distilled"]

SURF_DICE_DELTA = 2.0
N_BOOTSTRAP = 2000
EVAL_BATCH = 4


def _load_threshold(run_dir: Path) -> float:
    thr_csv = run_dir / "threshold_search.csv"
    if not thr_csv.exists():
        raise FileNotFoundError(f"{thr_csv} not found -- run the corresponding training script first.")
    return float(pd.read_csv(thr_csv).sort_values("mean_dice", ascending=False).iloc[0]["threshold"])


def _load_threshold_temp(run_dir: Path) -> tuple[float, float]:
    thr_csv = run_dir / "threshold_search.csv"
    if not thr_csv.exists():
        raise FileNotFoundError(f"{thr_csv} not found -- run 07b/07c first.")
    row = pd.read_csv(thr_csv).sort_values("mean_dice", ascending=False).iloc[0]
    return float(row["threshold"]), float(row["temperature"])


def _score_rows(rows: list[dict], out_dir: Path, run_name: str, best_thr: float) -> None:
    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(out_dir / "test_per_image.csv", index=False)
    summary = summarize_extended(rows, n_bootstrap=N_BOOTSTRAP)
    summary.update({"run_name": run_name, "best_threshold": best_thr, "n_images": len(rows),
                     "eval_resolution": "640x640 (GT resized down by data loader, no upscaling)"})
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)
    logger.info("[%s] TEST @640 (no upscale): median_dice=%.4f miss_rate=%.4f -> %s",
                run_name, summary["median_dice"], summary["complete_miss_rate"],
                out_dir / "test_summary.csv")


def _evaluate_segformer(model_name: str, pretrained_key: str, paths: dict, cfg: dict,
                         device: torch.device, test_df: pd.DataFrame, out_root: Path) -> None:
    run_dir = Path(paths["project_root"]) / model_name
    best_pt = run_dir / "best_model.pt"
    if not best_pt.exists():
        logger.warning("SKIP %s: no best_model.pt at %s", model_name, best_pt)
        return
    best_thr = _load_threshold(run_dir)
    amp = cfg.get("amp", True)

    model = SegformerWrapper(build_segformer(paths[pretrained_key], num_labels=1)).to(device)
    model.load_state_dict(torch.load(str(best_pt), map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=EVAL_BATCH, shuffle=False, num_workers=cfg.get("workers", 4), pin_memory=True,
    )

    rows = []
    with torch.no_grad():
        for x, y, stems, *_ in tqdm(loader, desc=f"{model_name} test@640", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            prob = torch.sigmoid(logits).float().cpu().numpy()[:, 0]
            gt = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob[i] >= best_thr).astype("uint8")
                g = (gt[i, 0] > 0.5).astype("uint8")
                rows.append(compute_image_row_extended(pred, g, str(stem), surf_dice_delta=SURF_DICE_DELTA))

    out_dir = ensure_dir(out_root / model_name)
    _score_rows(rows, out_dir, model_name, best_thr)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_yolo(model_name: str, paths: dict, cfg: dict,
                    device: torch.device, test_df: pd.DataFrame, out_root: Path) -> None:
    from ultralytics import YOLO

    run_dir = Path(paths["project_root"]) / model_name
    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists() or not (run_dir / "threshold_search.csv").exists():
        logger.warning("SKIP %s: missing best.pt or threshold_search.csv under %s "
                        "(run 06/08 then 07b/07c first)", model_name, run_dir)
        return
    best_thr, best_temp = _load_threshold_temp(run_dir)

    yolo_wrapper = YOLO(str(best_pt))
    yolo_nn = copy.deepcopy(yolo_wrapper.model).to(device).eval()

    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=EVAL_BATCH, shuffle=False, num_workers=cfg.get("workers", 4), pin_memory=True,
    )

    rows = []
    with torch.no_grad():
        for x, y, stems, *_ in tqdm(loader, desc=f"{model_name} test@640", leave=False):
            x = x.to(device, non_blocking=True)
            # out_hw=x.shape[-2:] -- resizes YOLO's raw logits to 640x640 (this batch's
            # own tensor size), matching BruiseDataset's GT -- NEVER to native resolution.
            class_logits = yolo_raw_class_logits(yolo_nn, x, out_hw=x.shape[-2:])
            prob = bruise_prob_from_logits(class_logits, best_temp).cpu().numpy()
            gt = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob[i] >= best_thr).astype("uint8")
                g = (gt[i, 0] > 0.5).astype("uint8")
                rows.append(compute_image_row_extended(pred, g, str(stem), surf_dice_delta=SURF_DICE_DELTA))

    out_dir = ensure_dir(out_root / model_name)
    _score_rows(rows, out_dir, model_name, best_thr)

    del yolo_nn, yolo_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test Dice @640 (no native-resolution upscaling) for models from scripts 01-11")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_df = load_fixed_test(paths["fixed_test_manifest"])
    out_root = ensure_dir(Path(paths["project_root"]) / "test_eval_640_no_upscale")
    logger.info("device=%s | test images=%d -> %s", device, len(test_df), out_root)

    for model_name, pretrained_key in SEGFORMER_MODELS:
        _evaluate_segformer(model_name, pretrained_key, paths, cfg, device, test_df, out_root)

    for model_name in YOLO_MODELS:
        _evaluate_yolo(model_name, paths, cfg, device, test_df, out_root)


if __name__ == "__main__":
    main()
