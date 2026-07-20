#!/usr/bin/env python3
"""
Step 6 — YOLO26n-sem direct training (no teacher, Ultralytics recipe).

Trains the YOLO semantic segmentation model using Ultralytics' own training
pipeline, NOT the SegFormer pipeline. Key differences vs SegFormer (Steps 1,3):

  - Optimizer: Ultralytics auto-selects SGD or AdamW (yolo_optimizer config)
  - LR schedule: cosine decay (NOT poly decay)
  - Loss: BCE per-pixel inside Ultralytics' framework (NOT Dice+BCE)
  - Data format: images + semantic class-index PNGs (NOT binary float masks)
  - Batch size: fixed (not VRAM-probed) — Ultralytics manages its own OOM logic

Why YOLO uses Ultralytics' own training pipeline:
  Ultralytics integrates the YOLO backbone, neck, and semantic head in a
  tightly coupled training loop. Extracting the nn.Module and plugging it
  into a custom training loop risks losing Ultralytics' automatic mixed
  precision, EMA, and multi-scale augmentation. The cost is less control over
  hyperparameters; the benefit is stability.

Why close_mosaic matters (YOLO specific):
  Mosaic augmentation (combining 4 images) helps YOLO learn diverse context.
  Disabling it for the last N epochs (close_mosaic) lets the model fine-tune
  on single-image crops that better match test-time input distribution.

MODULARITY NOTE: this script used to carry a full copy of the data.yaml
writer, the train/skip/resume block, and the native-prediction val eval loop
-- all byte-for-byte identical to 08_train_yolo_sem_distilled.py. Those three
pieces now live in pipeline.yolo_stage (write_yolo_data_yaml,
train_yolo_standard_recipe, evaluate_yolo_native_on_val). build_split() stays
here since it's genuinely different from script 08's version (no teacher
fusion for direct training).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import load_train_val_split
from pipeline.io_utils import ensure_dir, load_yaml, save_json, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import link_or_copy, read_gt_mask, save_semantic_class_mask
from pipeline.yolo_stage import evaluate_yolo_native_on_val, train_yolo_standard_recipe, write_yolo_data_yaml

logger = setup_logging()


def build_split(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    """Build the Ultralytics-format dataset for one split.

    Why symlink (link_or_copy) instead of copy:
      Forensic images are large (~10MB each). Copying the full dataset for
      every training run would rapidly fill disk. Symlinks point to the
      original files and take bytes, not megabytes.

    Why save_semantic_class_mask (not raw binary mask):
      Ultralytics' semantic segmentation expects a uint8 class-index PNG where
      0 = background and 1 = bruise. The pipeline's binary float masks store
      confidence in [0,1]. save_semantic_class_mask converts to the class-index
      format Ultralytics expects.

    Args:
        df:       DataFrame with image_path and mask_path columns.
        out_dir:  root of the Ultralytics dataset (images/ + masks/ sub-dirs built here).
        split:    "train" or "val".
    """
    img_dir  = ensure_dir(out_dir / "images" / split)
    mask_dir = ensure_dir(out_dir / "masks"  / split)
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"build {split}"):
        src = Path(r.image_path)
        link_or_copy(src, img_dir / src.name)               # symlink image
        gt = read_gt_mask(r.mask_path).astype("uint8")
        save_semantic_class_mask(gt, mask_dir / (src.stem + ".png"))    # 0/1 class-index PNG


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO26n-sem direct training (no teacher)")
    ap.add_argument("--paths",                default="configs/paths.yaml")
    ap.add_argument("--common",               default="configs/common_train.yaml")
    ap.add_argument("--force-rebuild-dataset", action="store_true",
                    help="Re-build the Ultralytics image/mask dataset even if already done")
    ap.add_argument("--force-retrain",        action="store_true",
                    help="Re-train even if best.pt already exists")
    args = ap.parse_args()

    paths      = load_yaml(args.paths)
    cfg        = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)
    device_str = "0" if torch.cuda.is_available() else "cpu"

    run_dir          = ensure_dir(Path(paths["project_root"]) / "yolo_sem_direct")
    data_dir         = run_dir / "yolo_data"
    dataset_complete = run_dir / "DATASET_COMPLETE.json"
    best_pt          = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"

    train_df, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # ── Dataset build ─────────────────────────────────────────────────────────
    if args.force_rebuild_dataset or not dataset_complete.exists():
        logger.info("Building Ultralytics dataset in %s ...", data_dir)
        build_split(train_df, data_dir, "train")
        build_split(val_df,   data_dir, "val")
        write_yolo_data_yaml(run_dir, data_dir, len(train_df), len(val_df))
        logger.info("Dataset build complete.")
    else:
        logger.info("Dataset already built (DATASET_COMPLETE.json exists).")

    # ── Training ──────────────────────────────────────────────────────────────
    train_yolo_standard_recipe(
        weights_path=paths["yolo_sem_weights"],
        data_yaml_path=run_dir / "data.yaml",
        run_dir=run_dir, cfg=cfg, device_str=device_str,
        force_retrain=args.force_retrain, log_label="YOLO direct",
    )

    # ── Val evaluation using Ultralytics .predict() (native argmax, no temperature) ──
    val_summary = evaluate_yolo_native_on_val(
        best_pt=best_pt, val_df=val_df, cfg=cfg, device_str=device_str,
        run_dir=run_dir, log_label="YOLO direct",
    )

    run_config = {
        "run_name":      "yolo_sem_direct",
        "model_family":  "yolo_semantic_native",
        "yolo_model":    "yolo26n-sem.pt",
        "training_type": "direct",
        "alpha":         None,          # no distillation
        "lrf":           cfg["yolo_lrf"],
        "cos_lr":        cfg["yolo_cos_lr"],
        "warmup_epochs": cfg["yolo_warmup_epochs"],
        "optimizer":     cfg["yolo_optimizer"],
    }
    save_json(run_dir / "run_config.json", run_config)
    pd.DataFrame([{**run_config, **val_summary}]).to_csv(run_dir / "val_summary.csv", index=False)
    logger.info("Val summary: %s", val_summary)


if __name__ == "__main__":
    main()
