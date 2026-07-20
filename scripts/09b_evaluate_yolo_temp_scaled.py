#!/usr/bin/env python3
"""
Step 9b — Evaluate YOLO direct + distilled with val-selected (temperature, threshold).

This script applies the same train/val/test discipline as the SegFormer evaluation:
  1. Temperature and threshold were searched on val only (07b / 07c)
  2. The best (T, threshold) pair is loaded here and applied once on test
  3. Test results are reported, never used to re-tune anything

Why 09b is separate from 09_evaluate_test.py:
  09's eval_yolo() uses Ultralytics' native .predict() which applies argmax
  (no temperature, no threshold control). This works for a quick baseline but
  doesn't use the temperature-scaled probability path optimised in 07b/07c.
  09b applies the optimised (T, threshold) and shows whether temperature scaling
  actually helps on the test set. Keeping the two approaches separate makes the
  comparison explicit.

Why bypass Ultralytics' predict() here:
  Ultralytics' .predict() → argmax → binary mask path is implemented inside
  the Ultralytics engine and cannot be intercepted to insert temperature scaling.
  Instead we deepcopy the underlying nn.Module, call it directly to get raw
  logits, apply temperature scaling, and threshold. This is exactly what the
  benchmark and Track B scripts also do.

Output: fixed_test_evaluation/<run_name>_temp_scaled/
  — same format as 09_evaluate_test.py so consolidation scripts can combine both.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.metrics import compute_image_row, summarize
from pipeline.yolo_threshold_temp import bruise_prob_from_logits, yolo_raw_class_logits

logger = setup_logging()


def _load_temp_threshold(run_dir: Path) -> tuple[float, float]:
    """Load the val-selected (temperature, threshold) for a YOLO run.

    Why raise on missing CSV (no silent fallback):
      The val sweep CSV records the jointly optimal (T, threshold) pair.
      Without it we cannot know which T and threshold to apply on test,
      and falling back to (T=1, threshold=0.5) would give misleading numbers
      because YOLO's near-binary logits require T >> 1 to be thresholdable.

    Raises:
        FileNotFoundError: if threshold_search.csv is missing.
    """
    csv = run_dir / "threshold_search.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found.\n"
            "Run scripts/07b_threshold_yolo_direct.py or "
            "07c_threshold_yolo_distilled.py first.")
    row      = pd.read_csv(csv).sort_values("mean_dice", ascending=False).iloc[0]
    best_thr = float(row["threshold"])
    best_temp = float(row["temperature"])
    return best_thr, best_temp


def eval_yolo_with_temp_threshold(
    run_name: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Evaluate a YOLO model with temperature-scaled probabilities and val-selected threshold.

    Why deepcopy the nn.Module from YOLO wrapper:
      Ultralytics YOLO wraps the nn.Module with its own prediction pipeline.
      We need the raw nn.Module to call forward() and get class logits before
      Ultralytics' argmax postprocessing. deepcopy ensures we don't mutate
      the Ultralytics wrapper's internal state.

    Args:
        run_name: label for output CSV.
        run_dir:  training run directory (contains ultralytics best.pt and threshold_search.csv).
        out_dir:  where to save test_per_image.csv and test_summary.csv.

    Returns:
        Aggregated metrics dict.
    """
    from ultralytics import YOLO as UltralyticsYOLO

    best_thr, best_temp = _load_temp_threshold(run_dir)
    logger.info("%s: T=%.2f, threshold=%.2f", run_name, best_temp, best_thr)

    best_pt      = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    yolo_wrapper = UltralyticsYOLO(str(best_pt))
    # deepcopy: extract nn.Module without holding references to Ultralytics internals
    yolo_model   = copy.deepcopy(yolo_wrapper.model).to(device).eval()

    loader = torch.utils.data.DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    rows = []
    with torch.no_grad():
        for x, y, stems, *_ in loader:
            x = x.to(device, non_blocking=True)
            t0 = time.perf_counter()
            # Extract class logits from the semantic head (bypass argmax)
            class_logits = yolo_raw_class_logits(yolo_model, x, out_hw=x.shape[-2:])
            # Temperature scale before thresholding
            prob = bruise_prob_from_logits(class_logits, best_temp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()    # GPU done before stopping clock
            inf_time = time.perf_counter() - t0

            prob_np = prob.cpu().numpy()
            gt      = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob_np[i] >= best_thr).astype("uint8")
                g    = (gt[i, 0] > 0.5).astype("uint8")
                row  = compute_image_row(pred, g, str(stem))
                row["inference_time_sec"]  = inf_time
                row["best_threshold"]      = best_thr
                row["best_temperature"]    = best_temp
                rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "test_per_image.csv", index=False)
    agg    = summarize(rows)
    mean_t = float(pd.DataFrame(rows)["inference_time_sec"].mean())
    agg.update({
        "run_name":               run_name,
        "best_threshold":         best_thr,
        "best_temperature":       best_temp,
        "mean_inference_time_sec": mean_t,
        "fps":                    1.0 / mean_t,
    })
    pd.DataFrame([agg]).to_csv(out_dir / "test_summary.csv", index=False)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate YOLO direct+distilled with temperature-scaled threshold")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--force",  action="store_true",
                    help="Re-evaluate even if outputs already exist")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project_root = Path(paths["project_root"])
    test_df      = load_fixed_test(paths["fixed_test_manifest"])
    logger.info("Fixed test set: %d images | device: %s", len(test_df), device)

    yolo_runs = ["yolo_sem_direct", "yolo_sem_distilled"]
    all_agg   = []

    for run_name in yolo_runs:
        run_dir = project_root / run_name
        # Write to a separate sub-directory so 09's non-temp-scaled results are preserved
        out_dir = ensure_dir(project_root / "fixed_test_evaluation" / (run_name + "_temp_scaled"))

        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (already evaluated): %s_temp_scaled", run_name)
            all_agg.append(pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict())
            continue

        if not (run_dir / "threshold_search.csv").exists():
            logger.warning("SKIP %s: no threshold_search.csv (run 07b/07c first)", run_name)
            continue

        logger.info("Evaluating (temp-scaled): %s", run_name)
        agg = eval_yolo_with_temp_threshold(run_name, run_dir, test_df, cfg, device, out_dir)
        all_agg.append(agg)
        logger.info("  %s | dice=%.4f | fps=%.1f | T=%.2f",
                    run_name, agg.get("mean_dice", float("nan")),
                    agg.get("fps", float("nan")), agg.get("best_temperature", float("nan")))

    if all_agg:
        out_df = pd.DataFrame(all_agg).sort_values("mean_dice", ascending=False)
        out_df.to_csv(
            project_root / "fixed_test_evaluation" / "yolo_temp_scaled_comparison.csv",
            index=False,
        )
        logger.info("\n── YOLO Temperature-Scaled Test Results ──────────────────\n%s",
                    out_df.to_string(index=False))


if __name__ == "__main__":
    main()
