#!/usr/bin/env python3
"""
Step 3 — Train SegFormer-B0 directly (no teacher, no knowledge distillation).

Trains the small student model (3.7M params) on the same training split as
the B2 teacher (Step 1), using identical hyperparameters:
  - Layer-wise LR (backbone_lr=6e-5, head_lr=6e-4)
  - Warmup + poly LR decay
  - VRAM-adaptive batch-size probe

Why train B0 direct (not just use B0 distilled):
  B0 direct serves as the baseline for the Track A apple-to-apple comparison.
  If distillation (Step 5) outperforms B0 direct at identical conditions, the
  improvement is attributable to the teacher signal, not to any other
  hyperparameter change.

Why use the SAME trainer as B2 (not a simplified one):
  Identical optimizer, LR schedule, and loss function are required for Track A.
  Any difference in training recipe would make the comparison confounded.

MODULARITY NOTE: see 01_train_segformer_b2_teacher.py's docstring -- the
shared training-stage sequence both scripts used to duplicate now lives in
pipeline.segformer_stage.run_segformer_training_stage(). This file only
differs from 01 in the model_name/pretrained_key values it passes in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.segformer_stage import run_segformer_training_stage

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct SegFormer-B0 training (no teacher)")
    ap.add_argument("--paths",         default="configs/paths.yaml")
    ap.add_argument("--common",        default="configs/common_train.yaml")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Re-train even if best_model.pt already exists")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    summary = run_segformer_training_stage(
        model_name="segformer_b0_direct",
        pretrained_key="segformer_b0_pretrained",
        paths=paths, cfg=cfg, device=device,
        force_retrain=args.force_retrain,
        teacher_dir_name=None,    # direct training -- no soft labels
    )
    if summary is not None:
        logger.info("B0 direct training complete: %s", summary)


if __name__ == "__main__":
    main()
