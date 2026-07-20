#!/usr/bin/env python3
"""
Step 1 — Train the SegFormer-B2 teacher from scratch.

Trains the large teacher model (27.3M params) on the full training split
using the SegFormer paper's recipe:
  - Layer-wise LR: backbone_lr=6e-5, head_lr=6e-4 (10× backbone)
  - Warmup (1% of total steps) + polynomial LR decay
  - AdamW with weight decay, no decay on bias/norm parameters
  - Gradient accumulation to reach the effective batch size on limited VRAM
  - VRAM-adaptive batch-size probe before training starts

Why train B2 first:
  B2 is the teacher for all KD experiments (Steps 4, 5, 7, 8). It must be
  trained and calibrated (Step 2) before any student can use its soft labels.
  The order matters: a weaker teacher produces weaker soft labels, which cap
  the ceiling of student performance.

Why NOT load an off-the-shelf SegFormer-B2 pretrained on ADE20k:
  ADE20k has 150 semantic classes with no forensic skin or bruise data. The
  head weights are incompatible (different number of classes). We use the
  ImageNet backbone weights and randomly initialise a new 1-class head.

MODULARITY NOTE: this script used to be near-identical to
03_train_segformer_b0_direct.py, differing only in which pretrained key/run
folder to use -- both just built a model and called
pipeline.trainer.train_pytorch(..., teacher_fn=None). That shared sequence
now lives in pipeline.segformer_stage.run_segformer_training_stage(), used
by both scripts (and by 05, which adds a teacher). Nothing about what gets
trained or checkpointed has changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.segformer_stage import run_segformer_training_stage

# setup_logging(run_dir) would route logs to a file; we skip that here because
# run_dir is created inside run_segformer_training_stage() after config validation
logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SegFormer-B2 teacher")
    ap.add_argument("--paths",         default="configs/paths.yaml")
    ap.add_argument("--common",        default="configs/common_train.yaml")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Re-train even if best_model.pt already exists")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    # Pre-flight checks: fail before any GPU memory is allocated
    validate_paths(paths)
    validate_cfg(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary = run_segformer_training_stage(
        model_name="segformer_b2_teacher",
        pretrained_key="segformer_b2_pretrained",
        paths=paths, cfg=cfg, device=device,
        force_retrain=args.force_retrain,
        teacher_dir_name=None,    # direct training, no distillation for the teacher
    )
    if summary is not None:
        logger.info("Teacher training complete: %s", summary)


if __name__ == "__main__":
    main()
