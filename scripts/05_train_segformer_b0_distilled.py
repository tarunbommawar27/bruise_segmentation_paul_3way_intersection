#!/usr/bin/env python3
"""
Step 5 — Full-length SegFormer-B0 distillation with the Optuna-found alpha.

Trains B0 with knowledge distillation from the B2 teacher at the alpha value
found by the Optuna search (Step 4). The KD target is:
  ỹ = α·y_GT + (1−α)·σ(z_teacher/T)

This produces the "segformer_b0_distilled" checkpoint used in:
  - Track A evaluation (10_track_a_evaluate.py) — apple-to-apple vs B0 direct
  - Track B evaluation (11_track_b_evaluate.py) — five-model deployment comparison

Why use the Optuna alpha (not a fixed α=0.75):
  α controls the trade-off between GT supervision and teacher guidance. The
  optimal value depends on the teacher's calibrated confidence and the dataset's
  class imbalance. Optuna finds it empirically on this specific dataset rather
  than relying on a general-purpose default from a different domain.

MODULARITY NOTE: see 01_train_segformer_b2_teacher.py's docstring -- the
shared training-stage sequence (skip-guard, split loading, model building,
train_pytorch call) now lives in
pipeline.segformer_stage.run_segformer_training_stage(). This script adds
its own alpha resolution (CLI override or Optuna CSV, via the shared
pipeline.segformer_stage.load_optuna_best_alpha() -- also used by script 08's
YOLO equivalent) and passes teacher_dir_name/alpha through to turn on
distillation mode.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.segformer_stage import load_optuna_best_alpha, run_segformer_training_stage

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser(description="Distilled SegFormer-B0 training with Optuna alpha")
    ap.add_argument("--paths",         default="configs/paths.yaml")
    ap.add_argument("--common",        default="configs/common_train.yaml")
    ap.add_argument("--alpha",         type=float, default=None,
                    help="Override Optuna alpha (useful for ablations). "
                         "If not set, reads from optuna_alpha_search/segformer_b0_best_alpha.csv")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve alpha lazily: CLI flag overrides the Optuna-found best alpha.
    # This is only actually CALLED after the skip-guard inside
    # run_segformer_training_stage() passes -- see that function's docstring
    # for why (matches the original script's exact order of operations).
    def resolve_alpha() -> float:
        if args.alpha is not None:
            return float(args.alpha)
        return load_optuna_best_alpha(
            Path(paths["project_root"]) / "optuna_alpha_search" / "segformer_b0_best_alpha.csv",
            prerequisite_script="04_optuna_alpha_segformer_b0.py")

    summary = run_segformer_training_stage(
        model_name="segformer_b0_distilled",
        pretrained_key="segformer_b0_pretrained",
        paths=paths, cfg=cfg, device=device,
        force_retrain=args.force_retrain,
        teacher_dir_name="segformer_b2_teacher",
        teacher_pretrained_key="segformer_b2_pretrained",
        alpha_resolver=resolve_alpha,
    )
    if summary is not None:
        logger.info("B0 distillation complete: %s", summary)


if __name__ == "__main__":
    main()
