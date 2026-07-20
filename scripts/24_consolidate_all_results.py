#!/usr/bin/env python3
"""
scripts/24_consolidate_all_results.py

Final step — gathers every results artifact produced by scripts 00-23 (plus
the ad-hoc benchmark/audit scripts) into ONE self-contained folder, so the
whole project's results can be reviewed, zipped, or downloaded without
hunting across a dozen scattered directories under project_root.

WHY A SEPARATE FOLDER, NOT ANOTHER MERGED CSV
------------------------------------------------
scripts/10_consolidate_results.py already exists and does row-per-model CSV
merging (val+test metrics side by side) for the original 5 models. This
script is different in kind: it does not transform or merge any numbers, it
just COPIES every small results file (CSV/JSON/TXT/XLSX/LOG) from its
original location into one mirrored tree, preserving each file's original
name and relative grouping. That existing consolidated CSV is itself one of
the files copied in (see MANIFEST below).

WHY MODEL WEIGHTS (*.pt) ARE NEVER COPIED
------------------------------------------------
Only human-readable results are consolidated (extensions in
_ALLOWED_EXTENSIONS). Checkpoints are multi-hundred-MB binaries that live
under pretrained_weights_root / each run_dir's best_model.pt or
ultralytics_runs/.../weights/ -- copying those would make this folder
multi-GB and defeats the point (easy to review/zip/download). If you need a
specific checkpoint, go to its run_dir directly.

WHY MISSING SOURCES ARE WARNINGS, NOT ERRORS
------------------------------------------------
This script is meant to be runnable at any point in the pipeline's
lifetime (not only once everything from 00-23 is finished), so an
unfinished stage (e.g. Phase 3 not yet trained) is reported in
MANIFEST.txt as "MISSING", not a crash -- consistent with how
17_fairness_eval_all_models.py skips-with-a-warning rather than failing the
whole run over one incomplete model.

Usage:
    python scripts/24_consolidate_all_results.py --paths configs/paths.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging

logger = setup_logging()

# Only human-readable result files are consolidated -- never weights/images.
_ALLOWED_EXTENSIONS = {".csv", ".json", ".txt", ".xlsx", ".log"}
_SKIP_DIRNAMES = {"__pycache__"}


def _copy_filtered_tree(src_dir: Path, dst_dir: Path, manifest: list[str]) -> None:
    """Recursively copy only _ALLOWED_EXTENSIONS files from src_dir into
    dst_dir, preserving relative subdirectory structure. Silently skips
    anything else (weights, images, ultralytics caches) by construction --
    they simply don't match the extension filter, no special-casing needed
    for e.g. ultralytics_runs/train/weights/best.pt."""
    if not src_dir.exists():
        manifest.append(f"MISSING  {src_dir}")
        return
    n_copied = 0
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if any(part in _SKIP_DIRNAMES for part in path.parts):
            continue
        if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
            continue
        rel = path.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        n_copied += 1
    manifest.append(f"OK       {src_dir}  ({n_copied} files -> {dst_dir})")


def _copy_file(src: Path, dst: Path, manifest: list[str]) -> None:
    if not src.exists():
        manifest.append(f"MISSING  {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(f"OK       {src} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate every pipeline results artifact into one folder")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--out-name", default="CONSOLIDATED_RESULTS",
                    help="Name of the output folder, created directly under project_root")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    project_root = Path(paths["project_root"])
    out_root = ensure_dir(project_root / args.out_name)
    manifest: list[str] = []

    # ── Splits (script 00, +22 adds ita_group_index_5 to the same file) ─────
    _copy_filtered_tree(project_root / "splits", out_root / "splits", manifest)

    # ── Per-model run directories (scripts 01,03,05,06,08,12,13,16) ─────────
    # training_history.csv, threshold_search.csv, val_summary.csv,
    # run_config.json, temperature.json/calibration.json, and (for YOLO)
    # ultralytics_runs/train/results.csv + this project's own wl_audit_v1/
    # subfolder if scripts/yolo_wl_audit_v1.py has been run.
    model_dirs = [
        "segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled",
        "yolo_sem_direct", "yolo_sem_distilled",
        "segformer_b0_fairness_distill_approach_a", "segformer_b0_fairness_distill_approach_b",
        "segformer_b2_als_teacher", "segformer_b0_als_to_wl_distilled",
    ]
    for name in model_dirs:
        _copy_filtered_tree(project_root / name, out_root / "models" / name, manifest)

    # ── Optuna alpha searches (scripts 04, 07, 15) ──────────────────────────
    _copy_filtered_tree(project_root / "optuna_alpha_search", out_root / "optuna_alpha_search", manifest)

    # ── Evaluation results (scripts 09, 09b, 10_track_a, 11_track_b, 17) ────
    _copy_filtered_tree(project_root / "fixed_test_evaluation", out_root / "evaluation" / "fixed_test_evaluation", manifest)
    _copy_filtered_tree(project_root / "track_a_evaluation", out_root / "evaluation" / "track_a_evaluation", manifest)
    _copy_filtered_tree(project_root / "track_b_evaluation", out_root / "evaluation" / "track_b_evaluation", manifest)
    _copy_filtered_tree(project_root / "fairness_evaluation", out_root / "evaluation" / "fairness_evaluation", manifest)
    _copy_file(project_root / "ALL_MODELS_VAL_TEST_LR_CONSOLIDATED.csv",
               out_root / "evaluation" / "ALL_MODELS_VAL_TEST_LR_CONSOLIDATED.csv", manifest)

    # ── Final publication tables (script 18) ────────────────────────────────
    _copy_filtered_tree(project_root / "paper_tables", out_root / "publication_tables", manifest)

    # ── ITA / skin-tone fairness labels (scripts 19-23) ─────────────────────
    if "ita_labels_dir" in paths:
        _copy_filtered_tree(Path(paths["ita_labels_dir"]), out_root / "ita_labels", manifest)
    else:
        manifest.append("MISSING  paths['ita_labels_dir'] not set in paths.yaml")

    # ── Ad-hoc benchmark / audit scripts (benchmark_inference_fair_v3.py,
    # yolo_wl_audit_v1.py) -- whatever landed in project_root/results ───────
    _copy_filtered_tree(project_root / "results", out_root / "benchmarks", manifest)

    manifest_text = "\n".join(manifest)
    (out_root / "MANIFEST.txt").write_text(manifest_text, encoding="utf-8")
    logger.info("Consolidation complete -> %s\n%s", out_root, manifest_text)


if __name__ == "__main__":
    main()
