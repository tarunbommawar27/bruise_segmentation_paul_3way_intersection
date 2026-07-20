#!/usr/bin/env python3
"""
scripts/10_track_a_evaluate.py

Track A — Strict apple-to-apple SegFormer benchmark (Gut et al. 2022 style).

════════════════════════════════════════════════════════════════════════════════
WHAT TRACK A IS
════════════════════════════════════════════════════════════════════════════════
Track A evaluates ONLY the three SegFormer variants (B2 teacher, B0 direct,
B0 distilled) using IDENTICAL conditions for all three:
  - Same optimizer (AdamW)
  - Same loss (Dice+BCE)
  - Same LR schedule (warmup/poly)
  - Same preprocessing (ImageNet normalisation, 640×640)
  - Same threshold-selection protocol (val sweep, never test)

WHY identical conditions matter: if we compare B2 vs B0 direct and B0 trains
with a different optimizer, we cannot tell whether the performance difference
is due to architecture or the optimizer. Track A controls for everything except
architecture, so any performance difference is attributable to the model.

WHY YOLO is excluded from Track A:
  YOLO uses Ultralytics' own optimizer, warm-up schedule, and BCE loss with a
  fundamentally different data format. Including it in Track A would violate the
  "identical conditions" requirement. See Track B for the five-model comparison.

════════════════════════════════════════════════════════════════════════════════
METRICS REPORTED
════════════════════════════════════════════════════════════════════════════════
  Primary:   Dice, IoU, Precision, Recall
  Boundary:  Surface Dice (δ=2px), ASD (pixels), HD95 (pixels)
  Fairness:  per-ITA-group Dice (Kruskal-Wallis — in summarize_extended)
  Speed:     raw_forward_fps, mask_out_fps, e2e_fps (3 categories)
  Stats:     bootstrap 95% CI for median Dice; Wilcoxon vs B0-direct baseline

Usage:
    python scripts/10_track_a_evaluate.py \\
        --paths  configs/paths.yaml \\
        --common configs/common_train.yaml

MODULARITY NOTE: this script used to define its own copies of _gpu_timed,
_build_single_image_tensor, benchmark_speed, _run_inference_loop, and
_run_wilcoxon_tests -- all either verbatim-identical or near-identical to
the same functions in 11_track_b_evaluate.py. Those now live in
pipeline.track_eval, shared by both scripts. eval_track_a() below stays a
separate function from script 11's eval_segformer_track_b() because the two
write genuinely different output schemas (Track B's per-image CSV and
summary include extra fields Track A's do not) -- see pipeline/track_eval.py's
module docstring for why that difference is kept as separate glue code
rather than one function with conditional branches.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.metrics_extended import summarize_extended
from pipeline.models import SegformerWrapper, build_segformer
from pipeline.track_eval import (
    benchmark_segformer_speed,
    load_segformer_threshold,
    run_segformer_inference_loop,
    run_wilcoxon_tests_vs_baseline,
)

logger = setup_logging()    # stdout-only for evaluation scripts


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_track_a(
    run_name: str,
    pretrained: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    benchmark_img: str,
    test_paths: list[str],
    surf_dice_delta: float = 2.0,
    n_bootstrap: int = 2000,
    n_warmup: int = 15,
    n_iters: int = 100,
) -> dict:
    """Evaluate one SegFormer run on the fixed test set (Track A protocol).

    Args:
        run_name:        label for this model in the output CSVs.
        pretrained:      HuggingFace checkpoint dir (for building model architecture).
        run_dir:         training run directory (contains best_model.pt).
        test_df:         fixed test set DataFrame.
        out_dir:         where to write test_per_image.csv and test_summary.csv.
        benchmark_img:   one real image used for the speed benchmark.
        test_paths:      all test image paths (for end-to-end speed measurement).
        surf_dice_delta: boundary tolerance for Surface Dice (pixels).
        n_bootstrap:     resamples for bootstrap CI.

    Returns:
        Summary dict (also includes "_per_image_dice" for Wilcoxon test).
    """
    best_thr = load_segformer_threshold(run_dir)

    # Load best checkpoint — NEVER use last epoch weights (may have overfit)
    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    model.load_state_dict(
        torch.load(str(run_dir / "best_model.pt"),
                   map_location=device, weights_only=True)
    )
    model.eval()

    # batch_size=1: ensures timing measurements are per-image (not batched)
    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    amp  = cfg.get("amp", True)
    rows = run_segformer_inference_loop(model, loader, best_thr, device, amp, surf_dice_delta)

    # Save per-image CSV before aggregation (in case aggregation crashes)
    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(out_dir / "test_per_image.csv", index=False)

    # Aggregate metrics with bootstrap CI
    summary = summarize_extended(rows, n_bootstrap=n_bootstrap)

    # Speed benchmark (separate controlled measurement — not from the inference loop above)
    speed = benchmark_segformer_speed(model, best_thr, benchmark_img, test_paths,
                                       cfg, device, n_warmup, n_iters)

    summary.update({"run_name": run_name, "track": "A", "best_threshold": best_thr, **speed})
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    # Free GPU memory before evaluating the next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # _per_image_dice is kept for the Wilcoxon test (not saved to CSV)
    return {**summary, "_per_image_dice": per_image_df["dice"].values}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Track A strict SegFormer evaluation")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--n-warmup",    type=int,   default=15)
    ap.add_argument("--n-iters",     type=int,   default=100)
    ap.add_argument("--n-bootstrap", type=int,   default=2000)
    ap.add_argument("--surf-delta",  type=float, default=2.0,
                    help="Surface Dice boundary tolerance in pixels (default 2.0)")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if outputs already exist")
    args = ap.parse_args()

    paths  = load_yaml(args.paths)
    cfg    = load_yaml(args.common)
    # Pre-flight validation: fail fast before any GPU work starts
    validate_paths(paths)
    validate_cfg(cfg)

    device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project_root = Path(paths["project_root"])

    test_df      = load_fixed_test(paths["fixed_test_manifest"])
    test_paths   = test_df["image_path"].tolist()
    # Use the first test image as the speed-benchmark input (representative image)
    benchmark_img = test_paths[0]
    track_a_eval_dir = ensure_dir(project_root / "track_a_evaluation")

    logger.info("Track A | test images: %d | device: %s", len(test_df), device)

    # The three SegFormer variants — all use identical training conditions
    track_a_runs = [
        ("segformer_b2_teacher",   paths["segformer_b2_pretrained"]),
        ("segformer_b0_direct",    paths["segformer_b0_pretrained"]),
        ("segformer_b0_distilled", paths["segformer_b0_pretrained"]),
    ]

    all_results: dict = {}

    for run_name, pretrained in track_a_runs:
        run_dir = project_root / run_name
        out_dir = ensure_dir(track_a_eval_dir / run_name)

        # Skip if training hasn't finished yet
        if not (run_dir / "best_model.pt").exists():
            logger.warning("SKIP %s: best_model.pt not found (training incomplete?)", run_name)
            continue

        # Skip if already evaluated (unless --force re-evaluates everything)
        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (already done): %s", run_name)
            # Load previous results so we can still run Wilcoxon tests
            prev      = pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict()
            prev_dice = pd.read_csv(out_dir / "test_per_image.csv")["dice"].values
            prev["_per_image_dice"] = prev_dice
            all_results[run_name]   = prev
            continue

        logger.info("Evaluating [Track A]: %s", run_name)
        result = eval_track_a(
            run_name=run_name, pretrained=pretrained,
            run_dir=run_dir, test_df=test_df, cfg=cfg, device=device,
            out_dir=out_dir, benchmark_img=benchmark_img, test_paths=test_paths,
            surf_dice_delta=args.surf_delta, n_bootstrap=args.n_bootstrap,
            n_warmup=args.n_warmup, n_iters=args.n_iters,
        )
        all_results[run_name] = result
        logger.info(
            "  %s | median_dice=%.4f | surf_dice=%.4f | miss=%.3f | raw_fps=%.1f",
            run_name,
            result.get("median_dice", float("nan")),
            result.get("median_surf_dice", float("nan")),
            result.get("complete_miss_rate", float("nan")),
            result.get("raw_fwd_fps", float("nan")),
        )

    if not all_results:
        logger.warning("No results collected — did any models finish training?")
        return

    # ── Wilcoxon signed-rank tests vs B0 direct baseline ─────────────────────
    wilcoxon_rows = run_wilcoxon_tests_vs_baseline(all_results, "segformer_b0_direct", track_a_eval_dir)

    # ── Combined Track A comparison table ─────────────────────────────────────
    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in all_results.values()]
    combined = pd.DataFrame(summary_rows).sort_values("median_dice", ascending=False)
    combined.to_csv(track_a_eval_dir / "track_a_comparison.csv", index=False)

    display_cols = [
        "run_name", "median_dice", "iqr_dice", "ci95_lo_dice", "ci95_hi_dice",
        "median_iou", "median_surf_dice", "median_hd95_px",
        "complete_miss_rate", "raw_fwd_fps", "mask_out_fps", "e2e_fps",
    ]
    show = [c for c in display_cols if c in combined.columns]
    logger.info("\n── Track A Results ──────────────────────────────────────\n%s",
                combined[show].to_string(index=False))

    if wilcoxon_rows:
        logger.info("\n── Wilcoxon Tests vs segformer_b0_direct ────────────────\n%s",
                    pd.DataFrame(wilcoxon_rows).to_string(index=False))


if __name__ == "__main__":
    main()
