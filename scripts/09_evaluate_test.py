#!/usr/bin/env python3
"""
Step 9 — Evaluate all 5 trained models on the fixed held-out test set.

Evaluates the three SegFormer variants and two YOLO variants once each on the
fixed test set. The test set is NEVER used for training, validation, threshold
tuning, or temperature selection — only for final reported numbers.

Why use 09 for basic evaluation (not the full Track A/B scripts):
  09_evaluate_test.py gives quick per-model test Dice/IoU/Precision/Recall
  with no bootstrapping, no Wilcoxon tests, and no speed benchmark. It is
  intended as a fast sanity check. For publication-quality results with
  confidence intervals, surface Dice, and statistical tests, use:
    - scripts/10_track_a_evaluate.py (SegFormer variants, apple-to-apple)
    - scripts/11_track_b_evaluate.py (all 5 models, best realistic recipe)

Why eval_yolo() uses Ultralytics' native argmax (not temperature+threshold):
  This script is a baseline evaluator. The temperature+threshold version of
  YOLO evaluation lives in 09b_evaluate_yolo_temp_scaled.py. The two scripts
  together show whether temperature scaling actually improves test Dice for YOLO.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_fixed_test
from pipeline.io_utils import ensure_dir, load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.mask_utils import read_gt_mask, yolo_sem_pred_mask
from pipeline.metrics import compute_image_row, summarize
from pipeline.models import SegformerWrapper, build_segformer

logger = setup_logging()


def _load_best_threshold(run_dir: Path) -> float:
    """Load the val-selected threshold for a SegFormer run.

    Why raise on missing CSV (not default to 0.5):
      Silent fallback to 0.5 would produce test numbers at a sub-optimal
      threshold, not comparable to numbers produced by the Track A/B scripts
      which always use the val-selected threshold. A missing CSV means Step 3/5
      training didn't complete successfully.

    Raises:
        FileNotFoundError: if threshold_search.csv is missing.
    """
    thr_csv = run_dir / "threshold_search.csv"
    if not thr_csv.exists():
        raise FileNotFoundError(
            f"threshold_search.csv not found in {run_dir}.\n"
            "Run the corresponding training script first — it writes this file "
            "automatically at the end of training.")
    return float(
        pd.read_csv(thr_csv)
        .sort_values("mean_dice", ascending=False)
        .iloc[0]["threshold"]
    )


def eval_pytorch(
    run_name: str,
    pretrained: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Evaluate a SegFormer model on the fixed test set.

    Args:
        run_name:  label written into the output CSVs.
        pretrained: HuggingFace checkpoint (for building model architecture).
        run_dir:   training run directory (must contain best_model.pt and
                   threshold_search.csv).
        out_dir:   where to write test_per_image.csv and test_summary.csv.

    Returns:
        Aggregated metrics dict.
    """
    best_thr = _load_best_threshold(run_dir)

    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    model.load_state_dict(
        torch.load(str(run_dir / "best_model.pt"),
                   map_location=device, weights_only=True))
    model.eval()

    # batch_size=1: accurate per-image timing (not amortised across a batch)
    loader = DataLoader(
        BruiseDataset(test_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )

    amp  = cfg.get("amp", True)
    rows = []
    with torch.no_grad():
        for x, y, stems, *_ in loader:
            x = x.to(device, non_blocking=True)
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()    # wait for GPU before stopping clock
            inf_time = time.perf_counter() - t0

            prob = torch.sigmoid(logits).float().cpu().numpy()
            gt   = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob[i, 0] >= best_thr).astype("uint8")
                g    = (gt[i, 0] > 0.5).astype("uint8")
                row  = compute_image_row(pred, g, str(stem))
                row["inference_time_sec"] = inf_time
                rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "test_per_image.csv", index=False)
    agg    = summarize(rows)
    mean_t = float(pd.DataFrame(rows)["inference_time_sec"].mean())
    agg.update({
        "run_name":               run_name,
        "best_threshold":         best_thr,
        "mean_inference_time_sec": mean_t,
        "fps":                    1.0 / mean_t,
    })
    pd.DataFrame([agg]).to_csv(out_dir / "test_summary.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return agg


def eval_yolo(
    run_name: str,
    run_dir: Path,
    test_df: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Evaluate a YOLO model using Ultralytics' native .predict() (argmax, no threshold).

    This is the baseline YOLO evaluation. For temperature+threshold evaluation,
    see 09b_evaluate_yolo_temp_scaled.py.

    Why no GPU timing synchronize here: Ultralytics' .predict() is a Python-level
    call that blocks until inference completes, so time.perf_counter() captures
    the true wall-clock time including any internal GPU wait.
    """
    from ultralytics import YOLO

    best_pt    = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    device_str = "0" if torch.cuda.is_available() else "cpu"
    model      = YOLO(str(best_pt))

    rows = []
    for _, r in test_df.iterrows():
        gt  = read_gt_mask(r.mask_path).astype("uint8")
        t0  = time.perf_counter()
        res = model.predict(
            str(r.image_path), imgsz=cfg["img_h"], device=device_str, verbose=False)[0]
        inf_time = time.perf_counter() - t0
        pred = yolo_sem_pred_mask(res, gt.shape)
        row  = compute_image_row(pred, gt, str(r.stem))
        row["inference_time_sec"] = inf_time
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "test_per_image.csv", index=False)
    agg    = summarize(rows)
    mean_t = float(pd.DataFrame(rows)["inference_time_sec"].mean())
    agg.update({
        "run_name":               run_name,
        "mean_inference_time_sec": mean_t,
        "fps":                    1.0 / mean_t,
    })
    pd.DataFrame([agg]).to_csv(out_dir / "test_summary.csv", index=False)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all 5 models on fixed test set")
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

    pytorch_runs = [
        ("segformer_b2_teacher",   paths["segformer_b2_pretrained"]),
        ("segformer_b0_direct",    paths["segformer_b0_pretrained"]),
        ("segformer_b0_distilled", paths["segformer_b0_pretrained"]),
    ]
    yolo_runs = ["yolo_sem_direct", "yolo_sem_distilled"]
    all_agg   = []

    for run_name, pretrained in pytorch_runs:
        run_dir = project_root / run_name
        out_dir = ensure_dir(project_root / "fixed_test_evaluation" / run_name)

        if not (run_dir / "best_model.pt").exists():
            logger.warning("SKIP %s: no best_model.pt (training incomplete?)", run_name)
            continue
        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (already evaluated): %s", run_name)
            all_agg.append(pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict())
            continue

        logger.info("Evaluating: %s", run_name)
        agg = eval_pytorch(run_name, pretrained, run_dir, test_df, cfg, device, out_dir)
        all_agg.append(agg)
        logger.info("  %s | dice=%.4f | fps=%.1f", run_name,
                    agg.get("mean_dice", float("nan")), agg.get("fps", float("nan")))

    for run_name in yolo_runs:
        run_dir = project_root / run_name
        out_dir = ensure_dir(project_root / "fixed_test_evaluation" / run_name)
        best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"

        if not best_pt.exists():
            logger.warning("SKIP %s: no best.pt (training incomplete?)", run_name)
            continue
        if (out_dir / "test_summary.csv").exists() and not args.force:
            logger.info("SKIP (already evaluated): %s", run_name)
            all_agg.append(pd.read_csv(out_dir / "test_summary.csv").iloc[0].to_dict())
            continue

        logger.info("Evaluating (YOLO native): %s", run_name)
        agg = eval_yolo(run_name, run_dir, test_df, cfg, out_dir)
        all_agg.append(agg)
        logger.info("  %s | dice=%.4f | fps=%.1f", run_name,
                    agg.get("mean_dice", float("nan")), agg.get("fps", float("nan")))

    if all_agg:
        out_df = pd.DataFrame(all_agg).sort_values("mean_dice", ascending=False)
        out_df.to_csv(project_root / "fixed_test_evaluation" / "test_comparison.csv", index=False)
        logger.info("\n── Test Comparison ──────────────────────────────────────\n%s",
                    out_df.to_string(index=False))


if __name__ == "__main__":
    main()
