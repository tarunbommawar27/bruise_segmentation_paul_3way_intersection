#!/usr/bin/env python3
"""
Step 8 — Full-length YOLO26n-sem distillation with the Optuna-found alpha.

Trains YOLO with offline pseudo-mask KD at the alpha found by Step 7.
The pseudo-masks are built once before training starts (offline KD) by fusing:
  fused = α·GT + (1−α)·σ(z_teacher/T)
  class_mask = (fused >= pseudo_threshold).astype(uint8)

The Ultralytics trainer then treats these as if they were GT labels, training
YOLO to predict a mask that agrees with both the GT annotation and the teacher's
probability map in the softened decision-boundary region.

Why offline KD (not online like SegFormer):
  Ultralytics' training loop doesn't expose hooks to inject per-batch teacher
  signals. See 07_optuna_alpha_yolo_sem.py docstring for the full explanation.

Why re-raise on missing Optuna CSV (not default to 0.75):
  Silent fallback would produce results that claim Optuna-optimised alpha but
  actually don't use it. Run 07_optuna_alpha_yolo_sem.py first, or pass --alpha.

MODULARITY NOTE: see 06_train_yolo_sem_direct.py's docstring -- the data.yaml
writer, the train/skip/resume block, and the native-prediction val eval loop
this script used to duplicate now live in pipeline.yolo_stage. This script
also uses the shared pipeline.segformer_stage.load_optuna_best_alpha() helper
(the same one script 05 uses) instead of its own private copy of that logic.
build_split() stays here since it genuinely differs from script 06's version
(this one fuses in the teacher's pseudo-mask).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import load_train_val_split
from pipeline.io_utils import ensure_dir, load_yaml, save_json, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import link_or_copy, read_gt_mask, save_semantic_class_mask, teacher_prob_for_image
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.segformer_stage import load_optuna_best_alpha
from pipeline.yolo_stage import evaluate_yolo_native_on_val, train_yolo_standard_recipe, write_yolo_data_yaml

logger = setup_logging()


def build_split(
    df: pd.DataFrame,
    out_dir: Path,
    split: str,
    alpha: float,
    teacher,
    temperature: float,
    cfg: dict,
    device: torch.device,
) -> None:
    """Build Ultralytics dataset with pseudo-mask KD for one split.

    Why use tqdm here: building pseudo-masks requires running teacher forward
    passes per image (disk read + resize + GPU inference + save). The progress
    bar shows whether the precomputation is making progress on a large dataset.
    """
    img_dir  = ensure_dir(out_dir / "images" / split)
    mask_dir = ensure_dir(out_dir / "masks"  / split)

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"build {split}"):
        src = Path(r.image_path)
        link_or_copy(src, img_dir / src.name)
        gt = read_gt_mask(r.mask_path)

        if split == "train" and teacher is not None:
            img_bgr = cv2.imread(str(src))
            prob    = teacher_prob_for_image(
                teacher, temperature, img_bgr,
                cfg["img_h"], cfg["img_w"], device)
            fused      = alpha * gt + (1.0 - alpha) * prob
            class_mask = (fused >= cfg["pseudo_threshold"]).astype("uint8")
        else:
            # Val split: always use clean GT — never fuse teacher for evaluation targets
            class_mask = gt.astype("uint8")

        save_semantic_class_mask(class_mask, mask_dir / (src.stem + ".png"))


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO26n-sem distillation training")
    ap.add_argument("--paths",                 default="configs/paths.yaml")
    ap.add_argument("--common",                default="configs/common_train.yaml")
    ap.add_argument("--alpha",                 type=float, default=None,
                    help="Override Optuna alpha (ablation use). Reads from CSV if not set.")
    ap.add_argument("--force-rebuild-dataset", action="store_true")
    ap.add_argument("--force-retrain",         action="store_true")
    args = ap.parse_args()

    paths      = load_yaml(args.paths)
    cfg        = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_str = "0" if torch.cuda.is_available() else "cpu"

    run_dir          = ensure_dir(Path(paths["project_root"]) / "yolo_sem_distilled")
    data_dir         = run_dir / "yolo_data"
    dataset_complete = run_dir / "DATASET_COMPLETE.json"
    best_pt          = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"

    # Resolve alpha: CLI override or Optuna CSV (raises if neither present)
    alpha = (float(args.alpha) if args.alpha is not None
             else load_optuna_best_alpha(
                 Path(paths["project_root"]) / "optuna_alpha_search" / "yolo_sem_best_alpha.csv",
                 prerequisite_script="07_optuna_alpha_yolo_sem.py"))

    train_df, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")
    logger.info("Train: %d | Val: %d | alpha=%.3f", len(train_df), len(val_df), alpha)

    # ── Pseudo-mask dataset build ─────────────────────────────────────────────
    if args.force_rebuild_dataset or not dataset_complete.exists():
        teacher_dir = Path(paths["project_root"]) / "segformer_b2_teacher"
        if not (teacher_dir / "best_model.pt").exists():
            raise FileNotFoundError(
                f"Teacher not found: {teacher_dir / 'best_model.pt'}\n"
                "Run 01_train_segformer_b2_teacher.py first.")

        teacher = SegformerWrapper(
            build_segformer(paths["segformer_b2_pretrained"], num_labels=1)).to(device)
        teacher.load_state_dict(
            torch.load(str(teacher_dir / "best_model.pt"),
                       map_location=device, weights_only=True))
        teacher.eval()

        temp_file   = teacher_dir / "temperature.json"
        temperature = (float(json.loads(temp_file.read_text()).get("temperature", 1.0))
                       if temp_file.exists() else 1.0)
        logger.info("Building pseudo-mask dataset (T=%.3f, pseudo_thr=%.2f) ...",
                    temperature, cfg["pseudo_threshold"])

        build_split(train_df, data_dir, "train", alpha, teacher, temperature, cfg, device)
        build_split(val_df,   data_dir, "val",   alpha, None,    1.0,         cfg, device)

        del teacher    # release GPU memory before training starts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        write_yolo_data_yaml(run_dir, data_dir, len(train_df), len(val_df),
                             extra_marker_fields={"alpha": alpha})
        logger.info("Pseudo-mask dataset built.")

    # ── YOLO training ─────────────────────────────────────────────────────────
    train_yolo_standard_recipe(
        weights_path=paths["yolo_sem_weights"],
        data_yaml_path=run_dir / "data.yaml",
        run_dir=run_dir, cfg=cfg, device_str=device_str,
        force_retrain=args.force_retrain, log_label="YOLO distilled",
    )

    # ── Val evaluation with native Ultralytics prediction ─────────────────────
    val_summary = evaluate_yolo_native_on_val(
        best_pt=best_pt, val_df=val_df, cfg=cfg, device_str=device_str,
        run_dir=run_dir, log_label="YOLO distilled",
    )

    run_config = {
        "run_name": "yolo_sem_distilled", "model_family": "yolo_semantic_native",
        "yolo_model": "yolo26n-sem.pt", "training_type": "distill", "alpha": alpha,
        "lrf": cfg["yolo_lrf"], "cos_lr": cfg["yolo_cos_lr"],
        "warmup_epochs": cfg["yolo_warmup_epochs"], "optimizer": cfg["yolo_optimizer"],
    }
    save_json(run_dir / "run_config.json", run_config)
    pd.DataFrame([{**run_config, **val_summary}]).to_csv(run_dir / "val_summary.csv", index=False)
    logger.info("Val summary: %s", val_summary)


if __name__ == "__main__":
    main()
