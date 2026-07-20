"""
Temperature scaling + threshold sweep for YOLO26n-sem (direct + distilled).

WHY THIS FILE EXISTS
---------------------
Ultralytics' own postprocessing (ultralytics/models/yolo/semantic/predict.py)
converts the model's raw [1, nc, H, W] logits straight into a hard per-pixel
class map via argmax (or `pred.gt(0)` in the 2-class case) -- there is no
probability map, no confidence score, and no threshold control exposed
through `.predict()` for task="semantic". That conversion happens *inside*
Ultralytics' SemanticSegmentationPredictor.postprocess(), which we never call
here -- we bypass it entirely by going straight to the underlying nn.Module.

For a 2-class semantic head (background=0, bruise=1), the raw output is
[1, 2, H, W] logits. We do NOT use Ultralytics' argmax. Instead we take the
class-logit difference (bruise_logit - background_logit) and pass it through
sigmoid -- this is mathematically identical to softmax-over-2-classes and
gives us a single bruise-probability map per pixel, exactly analogous to
SegFormer's torch.sigmoid(logits) in pipeline/trainer.py.

BCE-trained segmentation heads (which is what YOLO's semantic loss is, see
ultralytics/utils/loss.py) push logits toward +-infinity for confidently
correct pixels (see threshold_temperature_scaling_explained.pdf for the full
derivation). After full training this produces a near-binary probability
histogram -- two spikes near 0 and 1, with the middle essentially empty. Any
threshold placed in that empty middle produces an identical binary mask, so
naive threshold sweeping (same method used for SegFormer in
pipeline/trainer.py::_threshold_sweep) is close to meaningless on its own.

Temperature scaling (dividing the logit by T > 1 before sigmoid) pulls
saturated logits back into sigmoid's non-flat region, de-saturating the
histogram so a threshold sweep has something real to differentiate. This
file sweeps T and threshold jointly on val, exactly mirroring the simple
grid-sweep method already used for SegFormer (cfg["thresholds"]) -- not the
multi-method (Otsu/Triangle/Li/Kapur/MCET) approach, to keep methodology
consistent across all 5 trained models.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pipeline.data import BruiseDataset, load_train_val_split
from pipeline.metrics import compute_image_row, summarize

logger = logging.getLogger("pipeline")


def yolo_raw_class_logits(yolo_model, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Run YOLO's underlying nn.Module directly (bypassing .predict() and its
    internal argmax postprocessing), and upsample the raw [B, nc, h, w]
    semantic head output to the input resolution.

    yolo_model is the raw nn.Module (e.g. YOLO(...).model), NOT the YOLO()
    wrapper -- same extraction pattern as benchmark_inference_fair.py's
    yolo_raw_fp32 = copy.deepcopy(yolo_wrapper.model).
    """
    preds = yolo_model(x)
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.shape[-2:] != out_hw:
        preds = F.interpolate(preds.float(), size=out_hw, mode="bilinear", align_corners=False)
    return preds


def bruise_prob_from_logits(class_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert [B, nc, H, W] class logits into a single [B, H, W] bruise
    probability map.

    nc == 2 (background=0, bruise=1): take the class-logit difference and
    apply sigmoid -- equivalent to 2-class softmax, analogous to SegFormer's
    single-channel sigmoid output.
    nc == 1: head already emits a single bruise logit directly -- sigmoid it.

    Dividing by `temperature` BEFORE sigmoid is the de-saturation step: T=1.0
    is the model's native (likely near-binary) output; T>1.0 spreads the
    histogram back toward 0.5, exactly as in the BCE-saturation analysis.
    """
    if class_logits.shape[1] >= 2:
        z = class_logits[:, 1] - class_logits[:, 0]
    else:
        z = class_logits[:, 0]
    return torch.sigmoid(z / temperature)


def evaluate_yolo_raw(yolo_model, loader, device, threshold: float, temperature: float):
    """Same shape/contract as pipeline/trainer.py::evaluate(), but for YOLO's
    raw nn.Module + our own sigmoid/threshold postprocessing instead of
    Ultralytics' .predict() + argmax."""
    yolo_model.eval()
    rows = []
    with torch.no_grad():
        for x, y, stems, *_ in loader:
            x = x.to(device, non_blocking=True)
            class_logits = yolo_raw_class_logits(yolo_model, x, out_hw=x.shape[-2:])
            prob = bruise_prob_from_logits(class_logits, temperature).cpu().numpy()
            gt = y.cpu().numpy()
            for i, stem in enumerate(stems):
                pred = (prob[i] >= threshold).astype("uint8")
                g = (gt[i, 0] > 0.5).astype("uint8")
                rows.append(compute_image_row(pred, g, str(stem)))
    return pd.DataFrame(rows), summarize(rows)


def threshold_temperature_sweep(yolo_model, loader, device, thresholds: list[float],
                                 temperatures: list[float]) -> pd.DataFrame:
    """Grid sweep over threshold x temperature on val, scored by mean_dice --
    same simple grid-sweep method as SegFormer's _threshold_sweep, just
    crossed with an extra temperature axis. Returns the full grid (sorted
    best-first) so the empty-gap / divergence-temperature pattern from the
    PDF is visible in the saved CSV, not just the single winning row."""
    rows = []
    for t in temperatures:
        for thr in thresholds:
            _, s = evaluate_yolo_raw(yolo_model, loader, device, thr, t)
            rows.append({"temperature": t, "threshold": thr, **s})
    df = pd.DataFrame(rows).sort_values("mean_dice", ascending=False).reset_index(drop=True)
    return df


def run_threshold_search(weights_path: str, val_df, cfg: dict, device: torch.device,
                          run_dir: Path) -> tuple[pd.DataFrame, float, float]:
    """Full val-side pipeline for one YOLO run (direct or distilled):
    load best.pt -> extract raw nn.Module -> sweep T x threshold on val ->
    save threshold_search.csv (same filename convention as SegFormer) ->
    return (full_grid_df, best_threshold, best_temperature).
    """
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    yolo_wrapper = YOLO(str(weights_path))
    yolo_model = copy.deepcopy(yolo_wrapper.model).to(device).eval()

    val_ds = BruiseDataset(val_df, cfg["img_h"], cfg["img_w"], training=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.get("yolo_threshold_search_batch", 4),
        shuffle=False, num_workers=cfg.get("workers", 8), pin_memory=True,
    )

    temperatures = cfg.get("yolo_temperatures", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    thresholds = cfg["thresholds"]

    grid_df = threshold_temperature_sweep(yolo_model, val_loader, device, thresholds, temperatures)
    grid_df.to_csv(run_dir / "threshold_search.csv", index=False)

    best_row = grid_df.iloc[0]
    best_thr = float(best_row["threshold"])
    best_temp = float(best_row["temperature"])
    return grid_df, best_thr, best_temp


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint + threshold/temperature loading (moved here from
# scripts/29_benchmark_inference_all_models.py, which originally duplicated
# this logic inline). Kept alongside run_threshold_search() since both read
# the same threshold_search.csv this file writes.
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo_threshold_temp(run_dir: Path) -> tuple[float, float]:
    """Read the best val-set (threshold, temperature) pair for one YOLO run.

    threshold_search.csv here has one row per (temperature, threshold)
    combination from the joint grid sweep in threshold_temperature_sweep();
    the best pair is whichever row has the highest mean_dice. temperature
    defaults to 1.0 for any older CSV written before temperature scaling was
    added, so this stays readable against historical runs.
    """
    csv = run_dir / "threshold_search.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing threshold_search.csv: {csv}")

    row = pd.read_csv(csv).sort_values("mean_dice", ascending=False).iloc[0]
    thr = float(row["threshold"])
    temp = float(row["temperature"]) if "temperature" in row.index else 1.0
    return thr, temp


def load_yolo_model(
    model_name: str,
    paths: dict,
    device: torch.device,
) -> tuple[torch.nn.Module, float, float, Path]:
    """Load a trained YOLO run's raw nn.Module + its calibrated (threshold,
    temperature) pair.

    Returns the raw underlying nn.Module (deep-copied out of the Ultralytics
    YOLO() wrapper), NOT the wrapper itself -- callers must go through
    yolo_raw_class_logits()/bruise_prob_from_logits() above, exactly like
    run_threshold_search() does, so a benchmark or eval script can never
    accidentally fall back to Ultralytics' own .predict() argmax
    postprocessing (see this module's docstring for why that path is
    bypassed entirely).
    """
    from ultralytics import YOLO

    run_dir = Path(paths["project_root"]) / model_name
    ckpt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing YOLO checkpoint: {ckpt}")

    thr, temp = load_yolo_threshold_temp(run_dir)

    wrapper = YOLO(str(ckpt))
    nn_model = copy.deepcopy(wrapper.model).to(device).eval()

    return nn_model, thr, temp, ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Shared stage runner for scripts/07b_threshold_yolo_direct.py and
# scripts/07c_threshold_yolo_distilled.py
# ─────────────────────────────────────────────────────────────────────────────

def run_yolo_threshold_stage(
    model_name: str,
    prerequisite_script: str,
    paths: dict,
    cfg: dict,
    force_rerun: bool,
) -> tuple[pd.DataFrame, float, float] | None:
    """Shared body of scripts 07b (direct) and 07c (distilled) -- before this
    refactor, both scripts copy-pasted an identical sequence of steps (only
    differing in which run_name/prerequisite-script-name string to use) around
    the one function that actually does the work, run_threshold_search().
    This function IS that identical sequence, extracted once so the two
    scripts are now both a few lines of argument-passing plus their own
    result-logging, instead of two near-duplicate copies of the same logic.

    Behavior is unchanged from the pre-refactor scripts: same checkpoint
    resolution, same val split, same call into run_threshold_search(), same
    threshold_search.csv written by that function -- this function only
    relocates the surrounding skip-guard/existence-check boilerplate, it does
    not alter any computation.

    Args:
        model_name: run folder name under project_root, e.g. "yolo_sem_direct"
            or "yolo_sem_distilled" -- determines both the checkpoint path
            searched and the threshold_search.csv location.
        prerequisite_script: filename of the training script to name in the
            error message if best.pt does not exist yet (e.g.
            "06_train_yolo_sem_direct.py"), so the user knows what to run first.
        paths: parsed configs/paths.yaml (must contain "project_root").
        cfg: parsed configs/common_train.yaml (thresholds/temperatures grid).
        force_rerun: if True, re-run the sweep even if threshold_search.csv
            already exists; if False (default CLI behavior), skip and return
            None when a previous sweep's output is already on disk.

    Returns:
        (grid_df, best_threshold, best_temperature) -- the same three values
        run_threshold_search() returns -- or None if the sweep was skipped
        because threshold_search.csv already exists and force_rerun is False.
        Returning None (not raising) for the skip case mirrors the original
        scripts exactly: "already searched" is an expected, normal outcome,
        not an error condition.

    Raises:
        FileNotFoundError: if this model's best.pt does not exist yet -- the
        sweep cannot run without trained weights to load.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = Path(paths["project_root"]) / model_name
    best_pt = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    out_csv = run_dir / "threshold_search.csv"

    if not best_pt.exists():
        raise FileNotFoundError(
            f"No trained weights found at {best_pt}.\n"
            f"Run {prerequisite_script} first.")

    if out_csv.exists() and not force_rerun:
        logger.info("Already searched: %s (use --force-rerun to redo).", out_csv)
        return None

    _, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")
    logger.info("Val: %d images | device: %s", len(val_df), device)

    grid_df, best_thr, best_temp = run_threshold_search(
        str(best_pt), val_df, cfg, device, run_dir)

    return grid_df, best_thr, best_temp
