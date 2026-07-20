#!/usr/bin/env python3
"""
Step 4 — Optuna TPE search for SegFormer-B0 distillation alpha.

Finds the optimal α (GT weight in ỹ = α·y_GT + (1−α)·σ(z_teacher/T))
via Tree-structured Parzen Estimator (TPE), a Bayesian optimisation method.
Each trial trains B0 for a few short epochs and reports val Dice.

════════════════════════════════════════════════════════════════════════════
WHY OPTUNA / WHY NOT GRID SEARCH
════════════════════════════════════════════════════════════════════════════
α is a single continuous hyperparameter in [0.4, 0.9]. Grid search with
step 0.05 = 11 runs × full training = expensive. TPE learns from previous
trial results and concentrates samples in the high-Dice region, typically
finding the optimum in 15 trials even though grid search would need 11 trials
to cover the same range uniformly.

WHY SHORT EPOCHS PER TRIAL (not full training):
  We only need to rank α values (which is higher?), not report absolute Dice.
  Relative ranking is stable after 15–20% of full training (model has learned
  basic structure). Short trials are reproducibly faster and don't bias the
  final comparison.

WHY TPE SEED = cfg["seed"]:
  Reproducible search: re-running step 4 with the same seed produces the same
  sequence of α values, which is important for paper submission reproducibility.

MODULARITY NOTE: this script used to build the Optuna study/sampler/storage,
run study.optimize(), and save the trials + best-alpha CSVs itself -- code
that was identical to 07_optuna_alpha_yolo_sem.py's equivalent block except
for filenames. That shared orchestration now lives in
pipeline.optuna_stage.run_optuna_alpha_search(). This script keeps its own
_run_trial() (SegFormer-specific: trains via train_pytorch directly, online
KD) and its own skip-guard/teacher-loading setup in main(), both unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import load_train_val_split
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.optuna_stage import run_optuna_alpha_search
from pipeline.trainer import load_teacher, train_pytorch

logger = setup_logging()


def _run_trial(
    trial,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    search_cfg: dict,
    paths: dict,
    device: torch.device,
    teacher_fn,
    out_dir: Path,
) -> float:
    """Run one Optuna trial: suggest alpha, train short run, return val Dice.

    Why build a fresh model per trial: parameter state from a previous α
    would contaminate the trial (the model was already partially adapted to
    a different α). Starting fresh guarantees each trial measures the true
    effect of α, not a path-dependent artefact.
    """
    alpha = trial.suggest_float(
        "alpha",
        search_cfg["optuna_alpha_min"],
        search_cfg["optuna_alpha_max"],
        step=search_cfg["optuna_alpha_step"],
    )
    trial_dir = out_dir / f"segformer_b0_trial_{trial.number}"

    # Fresh model per trial — see docstring above
    model = SegformerWrapper(build_segformer(paths["segformer_b0_pretrained"], num_labels=1))
    try:
        summary = train_pytorch(
            model=model,
            model_name=f"segformer_b0_trial{trial.number}",
            run_dir=trial_dir,
            train_df=train_df,
            val_df=val_df,
            cfg=search_cfg,
            device=device,
            teacher_fn=teacher_fn,
            alpha=alpha,
        )
        # "mean_dice" is the best val Dice achieved during short training
        score = float(summary.get("mean_dice", 0.0))
    except Exception as exc:
        logger.error("Trial %d (alpha=%.2f) failed: %s", trial.number, alpha, exc, exc_info=True)
        score = 0.0    # failed trial reports 0 so Optuna continues with other alpha values

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    # release VRAM between trials to avoid OOM
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description="Optuna alpha search for SegFormer-B0 distillation")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = ensure_dir(Path(paths["project_root"]) / "optuna_alpha_search")
    best_csv = out_dir / "segformer_b0_best_alpha.csv"

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
    teacher_fn = load_teacher(
        teacher_dir, paths["segformer_b2_pretrained"], device, cfg.get("amp", True))
    logger.info("Teacher loaded (calibrated T from temperature.json if present).")

    # Short training config for trials: override epochs and patience
    # patience = search_epochs so early stopping never fires mid-trial
    n_search = cfg.get("optuna_search_epochs", 15)
    search_cfg = {**cfg, "epochs": n_search, "patience": n_search}

    best_alpha, best_value = run_optuna_alpha_search(
        study_label="segformer_b0",
        out_dir=out_dir,
        cfg=cfg,
        objective_fn=lambda trial: _run_trial(
            trial, train_df, val_df, search_cfg, paths, device, teacher_fn, out_dir),
    )

    logger.info(
        "Optuna search complete: best alpha=%.2f | best val Dice=%.4f",
        best_alpha, best_value,
    )


if __name__ == "__main__":
    main()
