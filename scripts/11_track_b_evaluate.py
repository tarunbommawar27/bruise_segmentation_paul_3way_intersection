#!/usr/bin/env python3
"""
scripts/11_track_b_evaluate.py

Track B — Controlled deployment-style five-model family comparison.

════════════════════════════════════════════════════════════════════════════════
WHAT TRACK B IS (vs Track A)
════════════════════════════════════════════════════════════════════════════════
Track B evaluates ALL five trained models on the fixed held-out test set,
where each model uses its OWN best realistic training recipe:
  - SegFormer: AdamW + Dice+BCE + warmup/poly + ImageNet normalisation
  - YOLO: Ultralytics auto-optimizer + BCE + cosine LR + val temperature sweep

WHY Track B is separate from Track A:
  YOLO and SegFormer use different training frameworks, optimizers, and
  preprocessing. Differences in performance may reflect these training recipe
  differences in addition to (or instead of) architectural differences. Track B
  results can only support the claim "better results in OUR setup" — not
  "architecturally superior". Track A controls for everything else so ONLY
  architecture differs; Track B shows what each model achieves at its best.

Usage:
    python scripts/11_track_b_evaluate.py \\
        --paths  configs/paths.yaml \\
        --common configs/common_train.yaml

MODULARITY NOTE: this script used to define its own copies of _gpu_timed,
_e2e_times_segformer, and an inline Wilcoxon-test loop in main() -- all
either verbatim-identical or near-identical to the same logic in
10_track_a_evaluate.py. Those now live in pipeline.track_eval, shared by
both scripts. eval_segformer_track_b() stays a separate function from
script 10's eval_track_a() because it writes a genuinely different output
schema (extra temperature_used per-image field, extra n_params/
best_temperature summary fields) -- see pipeline/track_eval.py's module
docstring for why that's kept as separate glue rather than one conditional
function. eval_yolo_track_b() and _e2e_times_yolo() stay here unchanged
since Track A has no YOLO evaluation to share this with.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, get_augmentation, load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.metrics_extended import compute_image_row_extended, summarize_extended
from pipeline.models import SegformerWrapper, build_segformer, count_params
from pipeline.track_eval import (
    benchmark_segformer_speed,
    build_single_image_tensor,
    gpu_timed,
    load_segformer_threshold,
    load_yolo_threshold_temp,
    run_segformer_inference_loop,
    run_wilcoxon_tests_vs_baseline,
)
from pipeline.yolo_threshold_temp import bruise_prob_from_logits, yolo_raw_class_logits

logger = setup_logging()    # stdout-only for evaluation scripts


# ─────────────────────────────────────────────────────────────────────────────
# YOLO-only end-to-end timing (no Track A equivalent to share this with)
# ─────────────────────────────────────────────────────────────────────────────

def _e2e_times_yolo(
    yolo_nn: torch.nn.Module, best_thr: float, best_temp: float,
    test_paths: list[str], cfg: dict, device: torch.device,
) -> np.ndarray:
    """Measure end-to-end wall-clock time for YOLO over all test images."""
    tfm = get_augmentation(training=False, img_h=cfg["img_h"], img_w=cfg["img_w"])
    times = []
    for p in test_paths:
        t0  = time.perf_counter()
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = tfm(image=img, mask=np.zeros(img.shape[:2], "f"))
        xb  = aug["image"].float().unsqueeze(0).to(device)
        with torch.no_grad():
            cl = yolo_raw_class_logits(yolo_nn, xb, out_hw=xb.shape[-2:])
            pr = bruise_prob_from_logits(cl, best_temp)
            _  = (pr >= best_thr).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer Track B evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_segformer_track_b(
    run_name: str,
    pretrained: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    benchmark_img: str,
    test_paths: list[str],
    surf_delta: float = 2.0,
    n_bootstrap: int = 2000,
    n_warmup: int = 15,
    n_iters: int = 100,
) -> dict:
    """Evaluate one SegFormer model on the fixed test set (Track B).

    Identical logic to eval_track_a() but output goes to track_b_evaluation/
    to keep the two benchmark results separate, plus records n_params and
    a constant temperature_used=1.0 per row / best_temperature=1.0 in the
    summary (SegFormer is never temperature-scaled in this project) so the
    combined comparison table has those columns for every model, YOLO included.
    """
    best_thr = load_segformer_threshold(run_dir)

    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    model.load_state_dict(
        torch.load(str(run_dir / "best_model.pt"),
                   map_location=device, weights_only=True)
    )
    model.eval()
    total_params, _ = count_params(model)    # report param count for the comparison table

    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    amp  = cfg.get("amp", True)
    rows = run_segformer_inference_loop(
        model, loader, best_thr, device, amp, surf_delta,
        extra_row_fields={"temperature_used": 1.0},
    )

    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(out_dir / "test_per_image.csv", index=False)
    summary      = summarize_extended(rows, n_bootstrap=n_bootstrap)

    speed = benchmark_segformer_speed(model, best_thr, benchmark_img, test_paths,
                                       cfg, device, n_warmup, n_iters)

    summary.update({
        "run_name":         run_name,
        "track":            "B",
        "best_threshold":   best_thr,
        "best_temperature": 1.0,
        "n_params":         total_params,
        **speed,
    })
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {**summary, "_per_image_dice": per_image_df["dice"].values}


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Track B evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_yolo_track_b(
    run_name: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    benchmark_img: str,
    test_paths: list[str],
    surf_delta: float = 2.0,
    n_bootstrap: int = 2000,
    n_warmup: int = 15,
    n_iters: int = 100,
) -> dict:
    """Evaluate one YOLO model on the fixed test set (Track B).

    Key differences from eval_segformer_track_b:
      - Uses yolo_raw_class_logits to bypass Ultralytics' own postprocessing
        (which applies argmax without temperature scaling)
      - Applies temperature scaling before thresholding
      - Loads (threshold, temperature) pair from val sweep (07b/07c)
    """
    from ultralytics import YOLO as UltralyticsYOLO

    best_thr, best_temp = load_yolo_threshold_temp(run_dir)

    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"YOLO best.pt not found: {best_pt}\n"
            f"Run 06_train_yolo_sem_direct.py or 08_train_yolo_sem_distilled.py first.")

    yolo_wrapper = UltralyticsYOLO(str(best_pt))
    # deepcopy: detach from Ultralytics wrapper so we control forward pass directly
    yolo_nn = copy.deepcopy(yolo_wrapper.model).to(device).eval()

    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    rows = []
    with torch.no_grad():
        for x, y, stems, img_paths, _ in loader:
            x = x.to(device, non_blocking=True)
            # Raw class logits: bypasses Ultralytics' argmax postprocessing
            class_logits = yolo_raw_class_logits(yolo_nn, x, out_hw=x.shape[-2:])
            # Temperature scaling: spreads the saturated YOLO probability histogram
            prob = bruise_prob_from_logits(class_logits, best_temp)
            prob_np = prob.cpu().numpy()
            gt_np   = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob_np[i] >= best_thr).astype("uint8")
                g    = (gt_np[i, 0] > 0.5).astype("uint8")
                row  = compute_image_row_extended(pred, g, str(stem), surf_dice_delta=surf_delta)
                row["threshold_used"]   = best_thr
                row["temperature_used"] = best_temp
                rows.append(row)

    per_image_df = pd.DataFrame(rows)
    per_image_df.to_csv(out_dir / "test_per_image.csv", index=False)
    summary      = summarize_extended(rows, n_bootstrap=n_bootstrap)

    # Speed benchmark
    x_gpu = build_single_image_tensor(benchmark_img, cfg, device)

    with torch.no_grad():
        a_mean, a_std, a_fps = gpu_timed(
            lambda: yolo_raw_class_logits(yolo_nn, x_gpu, x_gpu.shape[-2:]),
            n_warmup, n_iters, device,
        )
        def mask_fn():
            cl = yolo_raw_class_logits(yolo_nn, x_gpu, x_gpu.shape[-2:])
            pr = bruise_prob_from_logits(cl, best_temp)
            _  = (pr >= best_thr).to(torch.uint8)
        b_mean, b_std, b_fps = gpu_timed(mask_fn, n_warmup, n_iters, device)

    e2e = _e2e_times_yolo(yolo_nn, best_thr, best_temp, test_paths, cfg, device)

    summary.update({
        "run_name":         run_name,
        "track":            "B",
        "best_threshold":   best_thr,
        "best_temperature": best_temp,
        "n_params":         None,   # YOLO param count not tracked here (Ultralytics internal)
        "raw_fwd_fps":      a_fps,   "raw_fwd_mean_ms":  a_mean,  "raw_fwd_std_ms":  a_std,
        "mask_out_fps":     b_fps,   "mask_out_mean_ms": b_mean,  "mask_out_std_ms": b_std,
        "e2e_fps":          1000.0 / float(e2e.mean()),
        "e2e_mean_ms":      float(e2e.mean()),
        "e2e_std_ms":       float(e2e.std()),
    })
    pd.DataFrame([summary]).to_csv(out_dir / "test_summary.csv", index=False)
    return {**summary, "_per_image_dice": per_image_df["dice"].values}


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(run_name: str, res: dict) -> None:
    """Log a one-line summary for a completed model evaluation."""
    logger.info(
        "  %s | dice=%.4f CI=[%.3f,%.3f] | hd95=%.1fpx | miss=%.3f | raw=%.1f FPS | e2e=%.1f FPS",
        run_name,
        res.get("median_dice", float("nan")),
        res.get("ci95_lo_dice", float("nan")),
        res.get("ci95_hi_dice", float("nan")),
        res.get("median_hd95_px", float("nan")),
        res.get("complete_miss_rate", float("nan")),
        res.get("raw_fwd_fps", float("nan")),
        res.get("e2e_fps", float("nan")),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Track B deployment-style five-model evaluation")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    ap.add_argument("--n-warmup",    type=int,   default=15)
    ap.add_argument("--n-iters",     type=int,   default=100)
    ap.add_argument("--n-bootstrap", type=int,   default=2000)
    ap.add_argument("--surf-delta",  type=float, default=2.0)
    ap.add_argument("--force", action="store_true", help="Re-evaluate even if outputs exist")
    args = ap.parse_args()

    paths  = load_yaml(args.paths)
    cfg    = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    project_root = Path(paths["project_root"])
    test_df      = load_fixed_test(paths["fixed_test_manifest"])
    test_paths   = test_df["image_path"].tolist()
    benchmark_img = test_paths[0]
    track_b_dir  = ensure_dir(project_root / "track_b_evaluation")

    logger.info("Track B | %d test images | device: %s", len(test_df), device)

    segformer_runs = [
        ("segformer_b2_teacher",   paths["segformer_b2_pretrained"]),
        ("segformer_b0_direct",    paths["segformer_b0_pretrained"]),
        ("segformer_b0_distilled", paths["segformer_b0_pretrained"]),
    ]
    yolo_runs = ["yolo_sem_direct", "yolo_sem_distilled"]

    all_results: dict = {}

    # ── SegFormer models ──────────────────────────────────────────────────────
    for run_name, pretrained in segformer_runs:
        run_dir = project_root / run_name
        out_dir = ensure_dir(track_b_dir / run_name)

        if not (run_dir / "best_model.pt").exists():
            logger.warning("SKIP %s: no best_model.pt", run_name)
            continue
        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (done): %s", run_name)
            prev                    = pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict()
            prev["_per_image_dice"] = pd.read_csv(out_dir / "test_per_image.csv")["dice"].values
            all_results[run_name]   = prev
            continue

        logger.info("Evaluating [Track B / SegFormer]: %s", run_name)
        result = eval_segformer_track_b(
            run_name=run_name, pretrained=pretrained,
            run_dir=run_dir, test_df=test_df, cfg=cfg, device=device,
            out_dir=out_dir, benchmark_img=benchmark_img, test_paths=test_paths,
            surf_delta=args.surf_delta, n_bootstrap=args.n_bootstrap,
            n_warmup=args.n_warmup, n_iters=args.n_iters,
        )
        all_results[run_name] = result
        _print_result(run_name, result)

    # ── YOLO models ───────────────────────────────────────────────────────────
    for run_name in yolo_runs:
        run_dir = project_root / run_name
        out_dir = ensure_dir(track_b_dir / run_name)
        best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"

        if not best_pt.exists():
            logger.warning("SKIP %s: no best.pt", run_name)
            continue
        if not (run_dir / "threshold_search.csv").exists():
            logger.warning("SKIP %s: no threshold_search.csv (run 07b/07c first)", run_name)
            continue
        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (done): %s", run_name)
            prev                    = pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict()
            prev["_per_image_dice"] = pd.read_csv(out_dir / "test_per_image.csv")["dice"].values
            all_results[run_name]   = prev
            continue

        logger.info("Evaluating [Track B / YOLO]: %s", run_name)
        result = eval_yolo_track_b(
            run_name=run_name, run_dir=run_dir,
            test_df=test_df, cfg=cfg, device=device,
            out_dir=out_dir, benchmark_img=benchmark_img, test_paths=test_paths,
            surf_delta=args.surf_delta, n_bootstrap=args.n_bootstrap,
            n_warmup=args.n_warmup, n_iters=args.n_iters,
        )
        all_results[run_name] = result
        _print_result(run_name, result)

    if not all_results:
        logger.warning("No results collected.")
        return

    # ── Wilcoxon tests vs B0 direct ───────────────────────────────────────────
    wilcoxon_rows = run_wilcoxon_tests_vs_baseline(all_results, "segformer_b0_direct", track_b_dir)

    # ── Combined Track B table ─────────────────────────────────────────────────
    summary_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in all_results.values()]
    combined = pd.DataFrame(summary_rows).sort_values("median_dice", ascending=False)
    combined.to_csv(track_b_dir / "track_b_comparison.csv", index=False)

    display_cols = [
        "run_name", "median_dice", "ci95_lo_dice", "ci95_hi_dice",
        "median_iou", "median_surf_dice", "median_hd95_px",
        "complete_miss_rate", "raw_fwd_fps", "mask_out_fps", "e2e_fps",
    ]
    show = [c for c in display_cols if c in combined.columns]
    logger.info("\n── Track B Results ──────────────────────────────────────────\n%s",
                combined[show].to_string(index=False))

    if wilcoxon_rows:
        logger.info("\n── Wilcoxon Tests ────────────────────────────────────────────\n%s",
                    pd.DataFrame(wilcoxon_rows).to_string(index=False))


if __name__ == "__main__":
    main()
