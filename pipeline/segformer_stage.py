"""
pipeline/segformer_stage.py

Shared training-stage orchestration for scripts 01, 03, and 05 (SegFormer-B2
teacher, SegFormer-B0 direct, SegFormer-B0 distilled).

WHY THIS FILE EXISTS
---------------------
Before this refactor, scripts 01 and 03 were near-identical copies of each
other -- both built a SegformerWrapper with a different pretrained key and
called the same pipeline.trainer.train_pytorch(..., teacher_fn=None). Script
05 did the same thing plus two extra setup steps (loading the teacher,
resolving the Optuna-found alpha). None of that surrounding structure is
architecture-specific -- it's the same "load split, build model, optionally
attach a teacher, call train_pytorch" sequence every time, only the
model_name/pretrained_key/teacher_dir/alpha values differ.

This module extracts that shared sequence into run_segformer_training_stage(),
used by all three scripts. The actual training algorithm was already fully
centralized in pipeline.trainer.train_pytorch() before this refactor -- this
change only removes the duplicated setup code that surrounded that call in
three separate files, it does not touch train_pytorch() itself or change what
gets trained, checkpointed, or written to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from pipeline.data import load_train_val_split
from pipeline.io_utils import ensure_dir
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.trainer import load_teacher, train_pytorch

logger = logging.getLogger("pipeline")


def load_optuna_best_alpha(best_csv: Path, prerequisite_script: str) -> float:
    """Read the Optuna-found best alpha from a `*_best_alpha.csv` file.

    Shared by scripts 05 (SegFormer distillation) and, in the yolo_stage
    module, script 08 (YOLO distillation) -- both scripts had their own,
    near-identical private `_load_optuna_alpha` helper before this refactor.

    Why raise instead of defaulting to some fixed alpha (e.g. 0.75) when the
    CSV is missing: a silent fallback would let a run finish claiming to use
    "the Optuna-optimised alpha" when it actually used an arbitrary default --
    a reproducibility and reporting error that is easy to miss until much
    later. Raising forces the user to either run the search first, or
    explicitly pass an override (e.g. --alpha) so the choice is visible in
    the command that was run.

    Args:
        best_csv: path to the `*_best_alpha.csv` file (has a `best_alpha` column).
        prerequisite_script: filename of the search script to name in the
            error message (e.g. "04_optuna_alpha_segformer_b0.py").

    Returns:
        The best_alpha value from the CSV's first row.

    Raises:
        RuntimeError: if best_csv does not exist.
    """
    if not best_csv.exists():
        raise RuntimeError(
            f"Optuna alpha CSV not found: {best_csv}\n"
            f"Run {prerequisite_script} first, or pass --alpha <value> to override.")
    alpha = float(pd.read_csv(best_csv).iloc[0]["best_alpha"])
    logger.info("Loaded Optuna best alpha=%.3f from %s", alpha, best_csv)
    return alpha


def run_segformer_training_stage(
    *,
    model_name: str,
    pretrained_key: str,
    paths: dict,
    cfg: dict,
    device,
    force_retrain: bool,
    teacher_dir_name: str | None = None,
    teacher_pretrained_key: str | None = None,
    alpha_resolver: Callable[[], float] | None = None,
) -> dict | None:
    """Shared body of scripts 01, 03, and 05's main() functions.

    Handles, in order, exactly what each of the three scripts did before
    this refactor: the skip-guard (don't overwrite a finished run unless
    asked to), loading the train/val split, optionally loading a calibrated
    teacher (only for distillation runs), building a fresh model from the
    pretrained backbone, and calling train_pytorch(). Behavior for each of
    the three call sites is unchanged -- this function is a straight
    extraction of what was already common to all three, not a new design.

    Why alpha is a lazy `alpha_resolver` callable, not a plain float: in the
    original script 05, alpha was only resolved (reading the Optuna CSV,
    which raises if that CSV is missing) AFTER the skip-guard check --
    if best_model.pt already existed, the script returned early and never
    touched the Optuna CSV at all. Accepting a plain already-resolved alpha
    here would force the caller to resolve it before knowing whether this
    run even needs it, changing that order and risking an unnecessary raise
    (e.g. if best_model.pt exists but the Optuna CSV was later moved/archived).
    A callable defers that work until we've confirmed training will actually
    run, preserving the original order exactly.

    Args:
        model_name: the run's identity string passed to train_pytorch (also
            used in log messages), e.g. "segformer_b2_teacher".
        pretrained_key: key into `paths` for this model's HuggingFace
            pretrained-weights directory, e.g. "segformer_b2_pretrained".
        paths: parsed configs/paths.yaml.
        cfg: parsed configs/common_train.yaml.
        device: torch.device to train on.
        force_retrain: if True, retrain even if best_model.pt already exists.
        teacher_dir_name: run folder name of the teacher to distill from
            (e.g. "segformer_b2_teacher"), or None for direct (non-distilled)
            training -- this is the single flag that turns "direct" into
            "distilled" mode, matching scripts 01/03 (None) vs 05 (set).
        teacher_pretrained_key: pretrained-weights key for the teacher's own
            architecture (needed to rebuild the teacher's skeleton before
            loading its weights) -- required whenever teacher_dir_name is set.
        alpha_resolver: zero-argument callable returning the distillation
            GT-vs-teacher weight, invoked only when teacher_dir_name is set
            AND only after the skip-guard has already passed. Ignored for
            direct (non-distilled) runs.

    Returns:
        The summary dict returned by train_pytorch(), or None if training
        was skipped because best_model.pt already exists and force_retrain
        is False -- mirroring the original scripts' skip-and-return behavior.

    Raises:
        FileNotFoundError: if teacher_dir_name is set but that teacher's
            best_model.pt does not exist yet.
    """
    run_dir = ensure_dir(Path(paths["project_root"]) / model_name)

    if (run_dir / "best_model.pt").exists() and not force_retrain:
        logger.info("Skipping: already trained at %s (use --force-retrain to overwrite).",
                    run_dir / "best_model.pt")
        return None

    train_df, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")

    teacher_fn = None
    alpha = None
    if teacher_dir_name is not None:
        alpha = alpha_resolver()    # resolved now, i.e. AFTER the skip-guard above
        teacher_dir = Path(paths["project_root"]) / teacher_dir_name
        if not (teacher_dir / "best_model.pt").exists():
            raise FileNotFoundError(
                f"Teacher not found: {teacher_dir / 'best_model.pt'}\n"
                "Run 01_train_segformer_b2_teacher.py first.")
        teacher_fn = load_teacher(
            teacher_dir, paths[teacher_pretrained_key], device, cfg.get("amp", True))
        logger.info("Train: %d | Val: %d | alpha=%.3f | device: %s",
                    len(train_df), len(val_df), alpha, device)
    else:
        logger.info("Train: %d images | Val: %d images | device: %s",
                    len(train_df), len(val_df), device)

    model = SegformerWrapper(build_segformer(paths[pretrained_key], num_labels=1))

    train_kwargs = dict(
        model=model, model_name=model_name, run_dir=run_dir,
        train_df=train_df, val_df=val_df, cfg=cfg, device=device,
        teacher_fn=teacher_fn,
    )
    if alpha is not None:
        train_kwargs["alpha"] = alpha

    return train_pytorch(**train_kwargs)
