#!/usr/bin/env python3
"""
Step 7b — Temperature + threshold sweep for YOLO26n-sem DIRECT on val.

Ultralytics' .predict() pipeline applies argmax to the semantic logits,
which ignores the probability scale and has no threshold to tune. This
script bypasses that pipeline, pulls raw class logits from the underlying
nn.Module, and sweeps (temperature, threshold) pairs on val to find the
best operating point for the direct (non-distilled) YOLO model.

════════════════════════════════════════════════════════════════════════════
WHY TEMPERATURE SWEEP FOR YOLO (not for SegFormer)
════════════════════════════════════════════════════════════════════════════
YOLO is trained with BCE loss which pushes logits toward ±∞ more aggressively
than SegFormer (YOLO updates every pixel with the same loss weight, while
SegFormer's Dice term modulates per-image). The resulting probability histogram
is nearly bimodal: most pixels sit at p≈0 or p≈1, leaving a near-empty
region around p=0.5 where threshold selection would matter. Dividing logits
by T > 1 spreads the distribution to a region where threshold tuning is
meaningful.

SegFormer does NOT need this: its Dice+BCE combined loss does not drive logits
as hard toward ±∞, so its probability histogram has a meaningful spread and
threshold tuning at p∈[0.3, 0.7] is well-behaved without temperature scaling.

Output: yolo_sem_direct/threshold_search.csv with columns (temperature, threshold, mean_dice)
  — same file name as SegFormer's threshold_search.csv so evaluation scripts
  (09_evaluate_test.py, 11_track_b_evaluate.py) can read both identically.

MODULARITY NOTE: this script and 07c_threshold_yolo_distilled.py used to each
carry a full copy of the sweep-orchestration logic below (skip-guard, val
split loading, calling run_threshold_search()), differing only in which
run_name/prerequisite-script string to use. That shared logic is now one
function, pipeline.yolo_threshold_temp.run_yolo_threshold_stage() -- this
script is just CLI parsing plus one call into it and its own result logging.
Nothing about what gets computed or written to disk has changed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.yolo_threshold_temp import run_yolo_threshold_stage

logger = setup_logging()


def main() -> None:
    ap = argparse.ArgumentParser(description="Temperature+threshold sweep for YOLO direct")
    ap.add_argument("--paths",       default="configs/paths.yaml")
    ap.add_argument("--common",      default="configs/common_train.yaml")
    ap.add_argument("--force-rerun", action="store_true",
                    help="Re-run sweep even if threshold_search.csv already exists")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    result = run_yolo_threshold_stage(
        model_name="yolo_sem_direct",
        prerequisite_script="06_train_yolo_sem_direct.py",
        paths=paths, cfg=cfg, force_rerun=args.force_rerun,
    )
    if result is None:
        return    # already searched -- run_yolo_threshold_stage already logged why
    grid_df, best_thr, best_temp = result

    logger.info(
        "YOLO direct: best_threshold=%.2f | best_temp=%.2f | val_dice=%.4f",
        best_thr, best_temp, float(grid_df.iloc[0]["mean_dice"]),
    )
    logger.info("Top 10 (threshold, temperature, dice):\n%s",
                grid_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
