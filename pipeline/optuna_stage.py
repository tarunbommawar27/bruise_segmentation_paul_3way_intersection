"""
pipeline/optuna_stage.py

Shared Optuna study orchestration for scripts 04 (SegFormer-B0 alpha search)
and 07 (YOLO26n-sem alpha search).

WHY THIS FILE EXISTS
---------------------
Both scripts, before this refactor, independently built a seeded TPE
sampler, created an Optuna study with SQLite storage (resumable if a search
crashes partway through), ran study.optimize() for a configured number of
trials, saved the full trial history to a CSV, extracted the best alpha
(falling back to a configured default if Optuna's own result is missing the
key), and wrote a one-row best-alpha CSV. That orchestration is identical
between the two scripts -- what genuinely differs is what happens *inside*
one trial (SegFormer's trial calls train_pytorch directly for online KD;
YOLO's trial has to bake pseudo-masks to disk first for offline KD, since
Ultralytics has no online-KD hook -- see script 07's module docstring).

This module extracts only the shared orchestration. Each script keeps its
own trial function and its own `objective` closure passed in as
`objective_fn` -- the actual per-trial logic is untouched.

WHY THE "ALREADY SEARCHED" SKIP-GUARD STAYS IN EACH SCRIPT'S main(), NOT HERE:
in both original scripts, that check happens BEFORE the (expensive, and
potentially error-raising) work of loading the calibrated teacher and the
train/val split -- if the search already completed, none of that setup
should run at all. Moving the skip-guard into this module would mean
callers have already done that expensive setup before finding out it was
unnecessary. Keeping the check in main() preserves the original order
exactly: check first, only do expensive setup if actually needed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger("pipeline")


def run_optuna_alpha_search(
    *,
    study_label: str,
    out_dir: Path,
    cfg: dict,
    objective_fn: Callable,
) -> tuple[float, float]:
    """Create (or resume) a seeded TPE Optuna study, run the configured
    number of trials, and save the trial history + best-alpha CSVs.

    Assumes the caller has already checked whether
    `<study_label>_best_alpha.csv` exists and skipped calling this function
    if so -- see this module's docstring for why that check intentionally
    lives in each script's main(), not here.

    Args:
        study_label: short identifier used for every derived filename and
            the study name itself -- "segformer_b0" (script 04) or
            "yolo_sem" (script 07). Produces `<out_dir>/<study_label>_alpha.db`,
            `<out_dir>/<study_label>_trials.csv`, and
            `<out_dir>/<study_label>_best_alpha.csv`.
        out_dir: directory to write the SQLite DB and CSVs into (typically
            `project_root/optuna_alpha_search`).
        cfg: parsed configs/common_train.yaml. Reads `seed` (sampler seed),
            `optuna_n_trials` (default 15), and `optuna_default_alpha`
            (default 0.75, used only if Optuna's best_params is somehow
            missing the "alpha" key).
        objective_fn: the Optuna objective callable (`Callable[[optuna.Trial], float]`),
            already wired up by the caller to whatever per-trial logic that
            model family needs (suggesting alpha from the search space, then
            running one trial and returning its validation Dice).

    Returns:
        (best_alpha, best_mean_dice) -- the same two values written into
        the best-alpha CSV's row, so the caller can log its own completion
        message in its own preferred wording.

    Why SQLite storage with load_if_exists=True: persists every trial as it
    completes, so a search that crashes partway through (e.g. an
    out-of-memory trial) can be re-run and will resume from where it left
    off rather than losing all completed trials.

    Why TPESampler(seed=cfg["seed"]): reproducible search -- re-running this
    function with the same seed produces the same sequence of suggested
    alpha values, which matters for reporting reproducibility.
    """
    import optuna

    db_path = out_dir / f"{study_label}_alpha.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{study_label}_alpha",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,    # resume from a crashed search
        sampler=optuna.samplers.TPESampler(seed=cfg.get("seed", 42)),
    )

    study.optimize(objective_fn, n_trials=cfg.get("optuna_n_trials", 15))

    # Save full trial history for reproducibility reporting
    study.trials_dataframe().to_csv(out_dir / f"{study_label}_trials.csv", index=False)

    best_alpha = study.best_params.get("alpha", cfg.get("optuna_default_alpha", 0.75))
    pd.DataFrame([{
        "model_name":     study_label,
        "best_alpha":     best_alpha,
        "best_mean_dice": study.best_value,
    }]).to_csv(out_dir / f"{study_label}_best_alpha.csv", index=False)

    return best_alpha, study.best_value
