#!/usr/bin/env python3
"""
One-off local-laptop fixup — remap ORC absolute paths baked into manifest CSVs.

WHY THIS SCRIPT EXISTS
-----------------------
configs/paths.yaml was already repointed at this machine's local paths, but
that only controls where each *manifest CSV itself* is found on disk. The
image_path / mask_path VALUES inside those CSVs are a separate thing -- they
were written by scripts run on the ORC cluster and are Linux absolute paths
like "/home/tbommawa/labelbox_paul_orc_dataset_full_consensus/...". Those
strings do not resolve on this Windows laptop no matter what paths.yaml says,
because paths.yaml is never consulted again once a manifest has been loaded --
pipeline.data.BruiseDataset reads whatever string is sitting in the
image_path/mask_path column and hands it straight to cv2.imread().

This script rewrites exactly those baked-in strings, for exactly the 4
manifests that configs/paths.yaml's REQUIRED keys point training/eval/
benchmark code at. It does not touch pipeline/ or any numbered pipeline
script -- it only changes where a file is found on disk, never a pixel value,
a threshold, or a model decision, so it cannot affect any already-recorded
eval number.

WHY A PLAIN TEXT PREFIX REPLACE (NOT A PANDAS COLUMN EDIT)
-------------------------------------------------------------
Every one of the 3 known ORC prefixes below is a long, distinctive substring
that only ever appears in path columns in these files (the other columns are
Labelbox HTTPS URLs on a completely different domain, or plain metadata). A
whole-file text replace is simpler than parsing the CSV, deciding which
columns are "path-like", and rewriting only those -- and it's trivially
auditable with a plain diff against the .orc_backup.csv this script creates.

WHY RE-DERIVE FROM THE BACKUP ON EVERY RUN (NOT REWRITE THE LIVE FILE AGAIN)
--------------------------------------------------------------------------------
The first run copies the untouched ORC-path original to <name>.orc_backup.csv.
Every run (including the first) then reads FROM the backup, applies the
prefix replacements, and writes the result to the live path. This makes the
script idempotent and safe to re-run any number of times: the live file is
always a fresh, correct rebuild from the one true original, never a second
replace pass over an already-replaced file (which would be harmless here
since the old prefixes wouldn't match twice, but "always rebuild from the
untouched source" is a simpler invariant to reason about than "don't double
some rows").
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import normalize_manifest
from pipeline.io_utils import load_yaml, setup_logging

logger = setup_logging()

# (ORC prefix, local replacement). Order does not matter -- the 3 prefixes
# are disjoint substrings, none is a prefix of another.
PREFIX_MAP: list[tuple[str, str]] = [
    (
        "/home/tbommawa/labelbox_paul_orc_dataset_full_consensus",
        "C:/BRUISE_SEGMENTATION_PROJECT/labelbox_paul_orc_dataset_full_consensus",
    ),
    (
        "/home/tbommawa/NEW FINAL PIPELINE/labelbox_als_train",
        "C:/BRUISE_SEGMENTATION_PROJECT/labelbox_als_train",
    ),
    (
        "/home/tbommawa/bruise_pipeline_root/labelbox_als_intersection",
        "C:/BRUISE_SEGMENTATION_PROJECT/labelbox_als_intersection",
    ),
]

# The 4 manifests actually wired into configs/paths.yaml's required keys.
# (Other CSVs containing ORC paths elsewhere in the repo -- e.g.
# fairness_evaluation/*.csv, splits/train_val_split_before_ita.csv -- are
# historical outputs, not input manifests any script reads images from, and
# are deliberately left alone.)
MANIFEST_PATH_KEYS: list[str] = [
    "train_manifest",
    "als_train_manifest",
    "als_test_manifest",
    "fixed_test_manifest",
]


def remap_one_manifest(csv_path: Path) -> None:
    """Rebuild csv_path from its .orc_backup.csv copy with ORC prefixes replaced."""
    backup_path = csv_path.with_name(csv_path.stem + ".orc_backup" + csv_path.suffix)

    if not backup_path.exists():
        # First time seeing this file: the live copy IS the untouched ORC
        # original -- preserve it before we ever rewrite anything.
        shutil.copy2(csv_path, backup_path)
        logger.info("Backed up untouched original -> %s", backup_path)

    original_text = backup_path.read_text(encoding="utf-8")

    rewritten_text = original_text
    replacements_made = 0
    for old_prefix, new_prefix in PREFIX_MAP:
        count = rewritten_text.count(old_prefix)
        replacements_made += count
        rewritten_text = rewritten_text.replace(old_prefix, new_prefix)

    csv_path.write_text(rewritten_text, encoding="utf-8")
    logger.info("Rewrote %s (%d path occurrences remapped)", csv_path, replacements_made)


def verify_paths_resolve(csv_path: Path) -> list[str]:
    """Load csv_path through the same normalize_manifest() every pipeline
    script uses, and return every image_path/mask_path value that does not
    exist on disk. Reusing normalize_manifest (rather than re-guessing column
    names here) guarantees this check looks at exactly the columns the real
    pipeline will actually read."""
    df = normalize_manifest(pd.read_csv(csv_path))
    missing: list[str] = []
    for col in ("image_path", "mask_path"):
        for value in df[col]:
            if not Path(value).exists():
                missing.append(f"[{csv_path.name}:{col}] {value}")
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Remap ORC absolute paths baked into manifest CSVs to local paths.")
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)

    all_missing: list[str] = []
    for key in MANIFEST_PATH_KEYS:
        csv_path = Path(paths[key])
        if not csv_path.exists():
            logger.warning("SKIP %s: %s does not exist yet.", key, csv_path)
            continue

        remap_one_manifest(csv_path)
        all_missing.extend(verify_paths_resolve(csv_path))

    if all_missing:
        raise RuntimeError(
            f"{len(all_missing)} path(s) still do not resolve after remapping:\n"
            + "\n".join(all_missing[:20])
            + ("\n... (truncated)" if len(all_missing) > 20 else "")
        )

    logger.info("PASS -- every image_path/mask_path in all processed manifests resolves on disk.")


if __name__ == "__main__":
    main()
