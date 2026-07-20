"""
pipeline/yolo_stage.py

Shared pieces of scripts 06 (YOLO direct training) and 08 (YOLO distilled
training) that were byte-for-byte identical between the two files before
this refactor: writing the Ultralytics data.yaml + completion marker,
running Ultralytics' own .train() with this project's standard recipe
(including the skip-if-already-trained / resume-if-interrupted logic), and
evaluating a trained checkpoint on validation with Ultralytics' native
prediction path.

WHAT IS DELIBERATELY NOT HERE: each script's build_split() function (how the
training masks get built -- plain GT copy for script 06, teacher-fused
pseudo-masks for script 08) stays in its own script. That function is where
the two scripts genuinely differ, not where they duplicate each other, so
moving it here would not reduce any duplication -- it would just relocate
genuinely distinct code for no benefit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from pipeline.io_utils import save_json
from pipeline.mask_utils import read_gt_mask, yolo_sem_pred_mask
from pipeline.metrics import compute_image_row, summarize

logger = logging.getLogger("pipeline")


def write_yolo_data_yaml(
    run_dir: Path,
    data_dir: Path,
    train_n: int,
    val_n: int,
    extra_marker_fields: dict | None = None,
) -> Path:
    """Write the Ultralytics data.yaml descriptor plus a DATASET_COMPLETE.json
    marker, and return the data.yaml path.

    Why write data.yaml (not pass paths as CLI args): Ultralytics' .train()
    API expects a data YAML path specifying where train/val images and masks
    live. This file also becomes the permanent record of exactly what data
    the model was trained on.

    Why a separate DATASET_COMPLETE.json marker (not just checking data.yaml
    exists): a crash partway through building the dataset (e.g. mid-way
    through writing thousands of mask files) could leave data.yaml written
    but the actual image/mask files incomplete. The marker is only written
    AFTER both splits are fully built, so its presence is a reliable "the
    dataset is really done" signal for the caller's own skip-guard, distinct
    from data.yaml merely existing.

    Args:
        run_dir: the model's run directory (data.yaml and the marker go here).
        data_dir: root of the built Ultralytics dataset (images/, masks/).
        train_n, val_n: image counts recorded in the marker for reference.
        extra_marker_fields: additional fields to record in
            DATASET_COMPLETE.json (e.g. {"alpha": alpha} for a distilled run)
            -- written before train_images/val_images, matching script 08's
            original field order. None (the default) matches script 06,
            which has no extra fields to record.

    Returns:
        Path to the written data.yaml.
    """
    data_yaml = {
        "path":      str(data_dir.resolve()),
        "train":     "images/train",
        "val":       "images/val",
        "masks_dir": "masks",
        "names":     {0: "background", 1: "bruise"},
    }
    yaml_path = run_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    marker = dict(extra_marker_fields) if extra_marker_fields else {}
    marker.update({"train_images": train_n, "val_images": val_n})
    save_json(run_dir / "DATASET_COMPLETE.json", marker)

    return yaml_path


def train_yolo_standard_recipe(
    *,
    weights_path: str,
    data_yaml_path: Path,
    run_dir: Path,
    cfg: dict,
    device_str: str,
    force_retrain: bool,
    log_label: str,
) -> None:
    """Run (or skip, or resume) Ultralytics training with this project's
    standard YOLO26n-sem recipe -- identical hyperparameter set used by both
    scripts 06 and 08 before this refactor: cosine LR decay, configurable
    optimizer, warmup epochs, and close_mosaic for the final N epochs (see
    script 06's module docstring for why close_mosaic matters).

    Handles the same three-way branch both scripts had: skip entirely if
    best.pt already exists (unless force_retrain), resume from last.pt if
    training was interrupted (Ultralytics saves last.pt every epoch), or
    start fresh from the pretrained yolo26n-sem weights.

    Args:
        weights_path: path to the pretrained yolo26n-sem.pt to start from
            (only used for a fresh training run, not a resume).
        data_yaml_path: path to the data.yaml written by write_yolo_data_yaml().
        run_dir: the model's run directory (Ultralytics writes under
            run_dir/ultralytics_runs/train/).
        cfg: parsed configs/common_train.yaml (all yolo_* hyperparameters).
        device_str: Ultralytics device string ("0" for first GPU, or "cpu").
        force_retrain: if True, retrain even if best.pt already exists.
        log_label: human-readable label for log messages (e.g. "YOLO direct"
            or "YOLO distilled") so the two callers' log output stays
            distinguishable even though the underlying code path is shared.
    """
    from ultralytics import YOLO

    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    last_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "last.pt"

    if best_pt.exists() and not force_retrain:
        logger.info("Skipping: already trained at %s.", best_pt)
        return

    if last_pt.exists() and not force_retrain:
        # Ultralytics saves last.pt every epoch -- resume if training was interrupted
        logger.info("Resuming %s training from: %s", log_label, last_pt)
        YOLO(str(last_pt)).train(resume=True)
        return

    logger.info("Starting %s training ...", log_label)
    model = YOLO(weights_path)    # load pretrained YOLO26n-sem backbone
    model.train(
        data=str(data_yaml_path),
        task="semantic",                   # Ultralytics semantic segmentation head
        imgsz=cfg["img_h"],                # square crop (640)
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        batch=cfg["yolo_batch"],           # fixed batch (not VRAM-probed -- Ultralytics manages OOM)
        workers=cfg["workers"],
        device=device_str,
        project=str(run_dir / "ultralytics_runs"),
        name="train",
        exist_ok=True,                      # allow resuming into same dir
        # LR schedule parameters (YOLO paper recipe)
        optimizer=cfg["yolo_optimizer"],           # "auto" -> SGD or AdamW
        lrf=cfg["yolo_lrf"],                       # final LR as fraction of initial
        cos_lr=cfg["yolo_cos_lr"],                 # cosine decay (not poly)
        warmup_epochs=cfg["yolo_warmup_epochs"],   # linear warmup for first N epochs
        weight_decay=cfg["yolo_weight_decay"],
        close_mosaic=cfg["yolo_close_mosaic"],     # disable mosaic for final N epochs
        seed=cfg["seed"],
    )
    logger.info("%s training finished.", log_label)


def evaluate_yolo_native_on_val(
    best_pt: Path,
    val_df: pd.DataFrame,
    cfg: dict,
    device_str: str,
    run_dir: Path,
    log_label: str,
) -> dict:
    """Evaluate a trained YOLO checkpoint on validation using Ultralytics'
    own native .predict() path (argmax postprocessing, no temperature or
    threshold control -- that tuning happens separately in scripts 07b/07c).

    Writes val_per_image.csv and returns the aggregate summary dict (from
    pipeline.metrics.summarize) -- the caller is responsible for merging
    this with its own run_config fields (which differ between direct and
    distilled: training_type, alpha, etc.) and writing val_summary.csv,
    since that merged content is where the two scripts genuinely differ.

    Args:
        best_pt: path to the trained best.pt checkpoint.
        val_df: validation manifest (needs image_path, mask_path, stem columns).
        cfg: parsed configs/common_train.yaml (needs img_h for imgsz).
        device_str: Ultralytics device string ("0" or "cpu").
        run_dir: where to write val_per_image.csv.
        log_label: human-readable label for the log message.

    Returns:
        Aggregate dict from pipeline.metrics.summarize().
    """
    from ultralytics import YOLO

    logger.info("Evaluating %s on val set with Ultralytics native prediction ...", log_label)
    eval_model = YOLO(str(best_pt))
    rows = []
    for _, r in val_df.iterrows():
        gt = read_gt_mask(r.mask_path).astype("uint8")
        res = eval_model.predict(
            str(r.image_path), imgsz=cfg["img_h"], device=device_str, verbose=False)[0]
        pred = yolo_sem_pred_mask(res, gt.shape)
        rows.append(compute_image_row(pred, gt, str(r.stem)))
    pd.DataFrame(rows).to_csv(run_dir / "val_per_image.csv", index=False)
    return summarize(rows)
