#!/usr/bin/env python3
"""
Step 7 — Optuna TPE search for YOLO26n-sem distillation alpha.

Finds the optimal α for YOLO's offline pseudo-mask knowledge distillation.
Unlike SegFormer's online KD (soft labels computed at training time from the
live teacher), YOLO uses OFFLINE KD: the pseudo-masks are baked into disk
before training starts, and Ultralytics reads them as if they were GT masks.

════════════════════════════════════════════════════════════════════════════
WHY YOLO USES OFFLINE KD (not online like SegFormer)
════════════════════════════════════════════════════════════════════════════
Ultralytics' training loop calls its own data pipeline internally and doesn't
expose a hook to inject custom per-batch targets at training time. To use
teacher signals we must generate fused masks (α·GT + (1−α)·teacher_prob)
before training and save them to disk where Ultralytics' dataset reader will
find them.

Why temperature-scaling the teacher logits before fusion:
  YOLO is trained on these fused masks; if the teacher probabilities are
  near-binary (0.0001 / 0.9999), the fused mask collapses to ≈ GT regardless
  of α. Temperature scaling spreads teacher probs to reveal the teacher's
  calibrated uncertainty near bruise boundaries.

Why pseudo_threshold (not 0.5):
  After fusion, values near 0.5 are uncertain. A threshold > 0.5 (e.g. 0.6)
  ensures the fused mask only labels pixels as bruise when both GT AND teacher
  agree with at least some confidence, reducing noisy training signal.

MODULARITY NOTE: see 04_optuna_alpha_segformer_b0.py's docstring -- the
Optuna study/sampler/storage orchestration and trials/best-alpha CSV writing
this script used to duplicate now live in
pipeline.optuna_stage.run_optuna_alpha_search(). This script keeps its own
build_split()/run_trial() (YOLO-specific: bakes pseudo-masks to disk, offline
KD) and its own skip-guard/teacher-loading setup in main(), both unchanged.
"""
from __future__ import annotations

import argparse
import json as _json
import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import load_train_val_split
from pipeline.io_utils import ensure_dir, load_yaml as _load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import (
    link_or_copy,
    read_gt_mask,
    save_semantic_class_mask,
    teacher_prob_for_image,
    yolo_sem_pred_mask,
)
from pipeline.metrics import compute_image_row, summarize
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.optuna_stage import run_optuna_alpha_search

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
    """Build Ultralytics dataset with teacher-fused masks for one split.

    Why only fuse on 'train' (not val):
      Val masks must be identical to GT — they're used to score the model.
      Using fused masks for val would measure student accuracy at replicating
      a soft target rather than matching the true binary GT, which is wrong.

    Args:
        alpha:       GT weight in fusion: fused = α·GT + (1−α)·teacher_prob
        teacher:     raw SegformerWrapper nn.Module (not the teacher_fn callable,
                     because teacher_prob_for_image needs the raw nn.Module)
        temperature: calibrated T from temperature.json
    """
    img_dir  = ensure_dir(out_dir / "images" / split)
    mask_dir = ensure_dir(out_dir / "masks"  / split)

    for _, r in df.iterrows():
        src = Path(r.image_path)
        link_or_copy(src, img_dir / src.name)    # symlink to avoid disk duplication
        gt = read_gt_mask(r.mask_path)

        if split == "train" and teacher is not None:
            # Offline KD: generate teacher probability map from raw image BGR
            img_bgr = cv2.imread(str(src))
            prob    = teacher_prob_for_image(
                teacher, temperature, img_bgr,
                cfg["img_h"], cfg["img_w"], device)
            # Fuse GT and teacher soft labels with alpha weighting
            fused      = alpha * gt + (1.0 - alpha) * prob
            # pseudo_threshold: only label a pixel bruise if both sources agree
            class_mask = (fused >= cfg["pseudo_threshold"]).astype("uint8")
        else:
            # Val split: pure GT — never use teacher for evaluation masks
            class_mask = gt.astype("uint8")

        save_semantic_class_mask(class_mask, mask_dir / (src.stem + ".png"))


def run_trial(
    alpha: float,
    trial_number: int,
    weights_path: str,
    out_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    teacher,
    temperature: float,
    device: torch.device,
) -> float:
    """Run one YOLO trial: build pseudo-mask dataset, train short run, return val Dice.

    Why delete data_dir after the trial:
      Each trial generates a full copy of the dataset (symlinks + new mask PNGs).
      Keeping all trial datasets would quickly exhaust disk space (15 trials × N images).
      The trial-specific masks are no longer needed once val Dice is recorded.

    Args:
        weights_path: path to yolo26n-sem.pt pretrained weights.

    Returns:
        Mean val Dice for this trial.
    """
    from ultralytics import YOLO

    run_dir  = out_dir / f"yolo_sem_trial_{trial_number}"
    data_dir = run_dir / "yolo_data"

    build_split(train_df, data_dir, "train", alpha, teacher, temperature, cfg, device)
    build_split(val_df,   data_dir, "val",   alpha, None,    1.0,         cfg, device)

    data_yaml = {
        "path":      str(data_dir.resolve()),
        "train":     "images/train",
        "val":       "images/val",
        "masks_dir": "masks",
        "names":     {0: "background", 1: "bruise"},
    }
    data_yaml_path = run_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    device_str = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)
    model.train(
        data     = str(data_yaml_path),
        task     = "semantic",
        imgsz    = cfg["img_h"],
        epochs   = cfg["optuna_search_epochs"],
        patience = cfg["optuna_search_epochs"],    # patience = epochs so no early stopping in trials
        batch    = cfg["yolo_batch"],
        device   = device_str,
        workers  = cfg.get("workers", 8),
        project  = str(run_dir / "ultralytics_runs"),
        name     = "train",
        exist_ok = True,
        optimizer       = cfg["yolo_optimizer"],
        lrf             = cfg["yolo_lrf"],
        cos_lr          = cfg["yolo_cos_lr"],
        warmup_epochs   = min(3, cfg["optuna_search_epochs"]),    # cap warmup at 3 for short trials
        weight_decay    = cfg["yolo_weight_decay"],
        seed            = cfg.get("seed", 42),
        verbose         = False,    # suppress Ultralytics' extensive output during trials
    )

    best_pt    = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    eval_model = YOLO(str(best_pt))
    rows = []
    for _, r in val_df.iterrows():
        gt  = read_gt_mask(r.mask_path).astype("uint8")
        res = eval_model.predict(
            str(r.image_path), imgsz=cfg["img_h"], device=device_str, verbose=False)[0]
        pred = yolo_sem_pred_mask(res, gt.shape)
        rows.append(compute_image_row(pred, gt, str(getattr(r, "stem", r.image_path))))
    agg = summarize(rows)

    # Delete only the data_dir (masks + symlinks), keep ultralytics_runs for debugging
    shutil.rmtree(data_dir, ignore_errors=True)
    return float(agg.get("mean_dice", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna alpha search for YOLO KD")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    args = ap.parse_args()

    paths  = _load_yaml(args.paths)
    cfg    = _load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(paths["project_root"]) / "optuna_alpha_search")
    best_csv = out_dir / "yolo_sem_best_alpha.csv"

    if best_csv.exists():
        logger.info("Already searched: %s", best_csv)
        return

    train_df, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")
    logger.info("Train: %d | Val: %d | device: %s", len(train_df), len(val_df), device)

    teacher_dir = Path(paths["project_root"]) / "segformer_b2_teacher"
    if not (teacher_dir / "best_model.pt").exists():
        raise FileNotFoundError(
            f"Teacher not found: {teacher_dir / 'best_model.pt'}\n"
            "Run 01_train_segformer_b2_teacher.py first.")

    # Load raw nn.Module (not the teacher_fn callable) because teacher_prob_for_image
    # needs direct access to the model to run on raw BGR images with its own preprocessing
    raw_model = SegformerWrapper(
        build_segformer(paths["segformer_b2_pretrained"], num_labels=1)).to(device)
    raw_model.load_state_dict(
        torch.load(str(teacher_dir / "best_model.pt"),
                   map_location=device, weights_only=True))
    raw_model.eval()

    # Load temperature — used to spread teacher logits before baking pseudo-masks
    temp_file   = teacher_dir / "temperature.json"
    temperature = (float(_json.loads(temp_file.read_text()).get("temperature", 1.0))
                   if temp_file.exists() else 1.0)
    logger.info("Teacher loaded (T=%.3f)", temperature)

    def objective(trial) -> float:
        alpha = trial.suggest_float(
            "alpha",
            cfg["optuna_alpha_min"],
            cfg["optuna_alpha_max"],
            step=cfg["optuna_alpha_step"],
        )
        try:
            return run_trial(alpha, trial.number, paths["yolo_sem_weights"], out_dir,
                             train_df, val_df, cfg, raw_model, temperature, device)
        except Exception as exc:
            logger.error("Trial %d (alpha=%.2f) failed: %s",
                         trial.number, alpha, exc, exc_info=True)
            return 0.0

    best_alpha, best_value = run_optuna_alpha_search(
        study_label="yolo_sem", out_dir=out_dir, cfg=cfg, objective_fn=objective,
    )

    logger.info("Optuna YOLO search complete: best alpha=%.2f | val Dice=%.4f",
                best_alpha, best_value)


if __name__ == "__main__":
    main()
