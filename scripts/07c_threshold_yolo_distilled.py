#!/usr/bin/env python3
"""
Step 7c — Temperature + threshold sweep for YOLO26n-sem DISTILLED on val.

Identical method to 07b_threshold_yolo_direct.py, applied to the distilled
YOLO checkpoint instead. See 07b and pipeline/yolo_threshold_temp.py for
the full rationale (BCE-saturation → near-binary logits → temperature sweep).

Why run a separate sweep for distilled (not reuse 07b's result):
  Distillation changes the model's learned decision boundary and the logit
  scale. Even if the architecture is the same, the optimal (temperature,
  threshold) pair typically differs because:
    - The teacher's soft labels trained the model on different targets
    - Temperature scaling needed depends on the logit magnitude, which differs
      between direct and distilled training
  Reusing 07b's (T, threshold) for the distilled model would be incorrect.

MODULARITY NOTE: see 07b_threshold_yolo_direct.py's docstring -- the sweep
orchestration both this script and 07b used to duplicate now lives in
pipeline.yolo_threshold_temp.run_yolo_threshold_stage(). This file only
differs from 07b in the model_name/prerequisite_script values it passes in
and its own result-logging labels.
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
    ap = argparse.ArgumentParser(description="Temperature+threshold sweep for YOLO distilled")
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
        model_name="yolo_sem_distilled",
        prerequisite_script="08_train_yolo_sem_distilled.py",
        paths=paths, cfg=cfg, force_rerun=args.force_rerun,
    )
    if result is None:
        return
    grid_df, best_thr, best_temp = result

    logger.info(
        "YOLO distilled: best_threshold=%.2f | best_temp=%.2f | val_dice=%.4f",
        best_thr, best_temp, float(grid_df.iloc[0]["mean_dice"]),
    )
    logger.info("Top 10 (threshold, temperature, dice):\n%s",
                grid_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
