#!/usr/bin/env python3
"""
Step 0 — subject-level train/val split builder.

Reads the raw subject_5fold_split.csv (the dataset's own manifest) and builds
a brand-new single train/val split at subject level, removing any subject
overlap with the fixed test set. Same 82/18 subject split used throughout the
workspace, with a leakage check to guard against accidental contamination.

Why build the split here (not reuse an existing one):
  Each pipeline run must be self-contained. Importing a pre-built split from
  another experiment folder risks importing the other experiment's random seed,
  val set selection, or test-set contamination silently. Building it fresh here
  with an explicit seed guarantees reproducibility within this pipeline.

Why subject-level (not image-level) split:
  One forensic subject may contribute multiple images (different lighting,
  angles). An image-level split could put images of the same person in both
  train and val, leaking the subject's skin-tone and bruise appearance into
  the val set and inflating reported Dice scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import normalize_manifest
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_paths

logger = setup_logging()    # stdout-only; run_dir not yet known at split time


def main() -> None:
    ap = argparse.ArgumentParser(description="Build subject-level train/val split")
    ap.add_argument("--paths",        default="configs/paths.yaml")
    ap.add_argument("--val-fraction", type=float, default=0.18,
                    help="Fraction of subjects to hold out for validation (default 0.18 ≈ 18%)")
    ap.add_argument("--seed",         type=int,   default=42,
                    help="RNG seed for reproducible subject shuffle (default 42)")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    # Validate paths early — catches typos and missing files before any disk work
    validate_paths(paths)

    out_dir = ensure_dir(Path(paths["project_root"]) / "splits")

    # normalize_manifest: standardises column names (image_path, mask_path, subject, stem)
    # so downstream code has a consistent schema regardless of raw CSV column names
    train_raw  = normalize_manifest(pd.read_csv(paths["train_manifest"]))
    test_raw   = normalize_manifest(pd.read_csv(paths["fixed_test_manifest"]))

    # Exclude test subjects from the pool before splitting
    # (test manifest may contain subjects NOT in train_manifest if test was collected separately)
    test_subjects = set(test_raw["subject"].unique())
    all_subjects  = sorted(s for s in train_raw["subject"].unique()
                           if s not in test_subjects)

    logger.info(
        "Total subjects after test exclusion: %d | test subjects excluded: %d",
        len(all_subjects), len(test_subjects),
    )

    # Shuffle at subject level with a fixed seed for reproducibility
    rng           = np.random.default_rng(args.seed)
    subjects_arr  = np.array(all_subjects)
    rng.shuffle(subjects_arr)    # in-place shuffle of the subject array

    # Val fraction → number of subjects: max(1,...) ensures val is never empty
    n_val         = max(1, int(len(subjects_arr) * args.val_fraction))
    val_subjects  = set(subjects_arr[:n_val].tolist())
    train_subjects = set(subjects_arr[n_val:].tolist())

    # Leakage check: any overlap between the three sets is a fatal error
    tv = train_subjects & val_subjects     # train-val overlap (should be empty)
    tt = train_subjects & test_subjects    # train-test overlap (should be empty)
    vt = val_subjects   & test_subjects    # val-test overlap (should be empty)
    if tv or tt or vt:
        raise RuntimeError(
            f"Subject leakage detected — train∩val={tv} train∩test={tt} val∩test={vt}\n"
            "This is a critical error. Check the manifests for duplicate subjects.")
    logger.info("PASS — zero subject leakage across train / val / fixed-test.")

    # Annotate each row with its split assignment
    train_raw = train_raw[train_raw["subject"].isin(train_subjects | val_subjects)].copy()
    train_raw["split"] = train_raw["subject"].apply(
        lambda s: "val" if s in val_subjects else "train")
    train_raw.to_csv(out_dir / "train_val_split.csv", index=False)

    summary = train_raw.groupby("split").agg(
        n_images   = ("stem", "count"),
        n_subjects = ("subject", "nunique"),
    ).reset_index()
    summary.to_csv(out_dir / "split_summary.csv", index=False)

    logger.info("\n%s", summary.to_string(index=False))
    logger.info(
        "Fixed test: %d images, %d subjects",
        len(test_raw), test_raw["subject"].nunique(),
    )
    logger.info("Split written to: %s", out_dir)


if __name__ == "__main__":
    main()
