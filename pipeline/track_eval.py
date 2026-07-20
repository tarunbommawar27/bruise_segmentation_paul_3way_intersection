"""
pipeline/track_eval.py

Shared pieces of scripts 10 (Track A: strict SegFormer-only comparison) and
11 (Track B: deployment-style five-model comparison) that were duplicated
--- in some cases verbatim --- between the two files before this refactor.

WHAT IS SHARED HERE: GPU timing (_gpu_timed was a literal copy in both
files), threshold/temperature loading, building a single-image GPU tensor
for the speed benchmark, the SegFormer end-to-end wall-clock timer, the
3-category SegFormer speed benchmark (raw forward / mask output / e2e), the
SegFormer test-set inference loop, and the Wilcoxon-vs-baseline runner.

WHAT IS DELIBERATELY NOT MERGED: eval_track_a() (script 10) and
eval_segformer_track_b() (script 11) look similar but write genuinely
different output schemas -- Track B's per-image CSV has an extra
temperature_used column, and Track B's summary has extra n_params and
best_temperature fields that Track A's does not. Rather than merge these
into one function with conditional branches controlling which columns
appear (higher risk of a subtle schema bug), each script keeps its own
per-model eval function, built out of the shared pieces here plus its own
few extra lines for the fields that genuinely differ. This mirrors the same
shared-mechanics / per-family-glue split already used for the YOLO training
scripts (pipeline.yolo_stage) and the Optuna search scripts
(pipeline.optuna_stage).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pipeline.data import BruiseDataset, get_augmentation
from pipeline.metrics_extended import compute_image_row_extended, wilcoxon_compare

logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Threshold / temperature loading
# ─────────────────────────────────────────────────────────────────────────────

def load_segformer_threshold(run_dir: Path) -> float:
    """Load the val-selected threshold from a SegFormer training run.

    Why raise on missing CSV: both tracks' results must be reported at the
    val-selected threshold. Silent fallback to 0.5 would produce numbers
    that are not comparable across runs because each model would use a
    different effective threshold.

    Raises:
        FileNotFoundError: if threshold_search.csv is absent.
    """
    thr_csv = run_dir / "threshold_search.csv"
    if not thr_csv.exists():
        raise FileNotFoundError(
            f"{thr_csv} not found.\n"
            "Run the training script first — it saves threshold_search.csv "
            "automatically at the end of training.")
    return float(
        pd.read_csv(thr_csv).sort_values("mean_dice", ascending=False).iloc[0]["threshold"]
    )


def load_yolo_threshold_temp(run_dir: Path) -> tuple[float, float]:
    """Load the val-selected (threshold, temperature) pair for a YOLO run.

    Why temperature needed for YOLO: YOLO's BCE loss saturates logits toward
    ±∞. Temperature T > 1 spreads probabilities to enable meaningful
    thresholding. Temperature was found by scripts 07b/07c.

    Raises:
        FileNotFoundError: if threshold_search.csv is absent.
    """
    csv = run_dir / "threshold_search.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"threshold_search.csv not found in {run_dir}.\n"
            "Run scripts/07b_threshold_yolo_direct.py or "
            "07c_threshold_yolo_distilled.py first.")
    row = pd.read_csv(csv).sort_values("mean_dice", ascending=False).iloc[0]
    return float(row["threshold"]), float(row["temperature"])


# ─────────────────────────────────────────────────────────────────────────────
# GPU timing
# ─────────────────────────────────────────────────────────────────────────────

def _synchronize(device: torch.device) -> None:
    """No-op off CUDA -- there is no async queue to drain on CPU, so timing
    a CPU-only run doesn't need (and can't use) torch.cuda.synchronize().
    Mirrors pipeline/benchmark_640.py's identical helper."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_timed(fn, n_warmup: int, n_iters: int, device: torch.device) -> tuple[float, float, float]:
    """Return (mean_ms, std_ms, fps) for a GPU-bound callable (device-agnostic
    despite the name -- also runs, as a CPU-latency reference, when device is
    CPU).

    Why synchronize() before and after each timed call: GPU work is
    submitted asynchronously. Without synchronize() we measure kernel-LAUNCH
    time (microseconds), not actual COMPUTE time (milliseconds).
    """
    for _ in range(n_warmup):
        fn()
    _synchronize(device)    # ensure all warmup GPU work is done before timing

    times = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        _synchronize(device)            # GPU idle before this iteration
        t0 = time.perf_counter()
        fn()
        _synchronize(device)            # GPU finished before stopping clock
        times[i] = (time.perf_counter() - t0) * 1000.0   # seconds → ms
    return float(times.mean()), float(times.std()), 1000.0 / float(times.mean())


def build_single_image_tensor(image_path: str, cfg: dict, device: torch.device) -> torch.Tensor:
    """Load one test image and return a GPU tensor for the speed benchmark.

    Why build the tensor outside the timed function: the controlled
    benchmark measures only nn.Module forward + postprocessing. Disk I/O is
    excluded because it is not reproducible (disk caching effects vary
    between runs).
    """
    ds = BruiseDataset(
        pd.DataFrame([{"image_path": image_path,
                       "mask_path":  image_path,
                       "stem":       Path(image_path).stem}]),
        cfg["img_h"], cfg["img_w"], training=False,
    )
    # ds[0] returns (x, y, stem, img_path, mask_path) — we only need x
    x_cpu = ds[0][0].unsqueeze(0)    # add batch dim: [C, H, W] → [1, C, H, W]
    return x_cpu.to(device)


def e2e_times_segformer(
    model: nn.Module, best_thr: float, test_paths: list[str],
    cfg: dict, device: torch.device,
) -> np.ndarray:
    """Measure end-to-end wall-clock time for SegFormer over all test images
    (disk read + preprocess + GPU transfer + forward + postprocess), one
    reading per test image (not a repeated single image) so disk-cache
    variance is averaged out rather than measured once and extrapolated."""
    tfm = get_augmentation(training=False, img_h=cfg["img_h"], img_w=cfg["img_w"])
    times = []
    for p in test_paths:
        t0  = time.perf_counter()
        img = cv2.imread(p)                                  # disk I/O
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # BGR→RGB
        aug = tfm(image=img, mask=np.zeros(img.shape[:2], dtype=np.float32))
        xb  = aug["image"].float().unsqueeze(0).to(device)   # H2D transfer
        with torch.no_grad():
            logits = model(xb)
            prob   = torch.sigmoid(logits.float())
            _      = (prob >= best_thr).cpu().numpy()        # threshold + D2H
        _synchronize(device)    # GPU done before stopping clock
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.array(times)


def benchmark_segformer_speed(
    model: nn.Module,
    best_thr: float,
    benchmark_img: str,
    test_paths: list[str],
    cfg: dict,
    device: torch.device,
    n_warmup: int,
    n_iters: int,
) -> dict:
    """Measure three timing categories for one SegFormer model:
      (a) raw_forward:   nn.Module(x) only — architecture-isolated speed
      (b) mask_output:   forward + sigmoid + threshold (no disk, no resize)
      (c) end_to_end:    disk imread + preprocess + GPU transfer + forward +
                         postprocess — one reading per test image

    Why separate (a) and (b): (a) isolates the architectural bottleneck;
    (b) shows postprocessing cost; (c) shows real-world deployment cost.
    """
    model.eval()
    x_gpu = build_single_image_tensor(benchmark_img, cfg, device)

    with torch.no_grad():
        a_mean, a_std, a_fps = gpu_timed(lambda: model(x_gpu), n_warmup, n_iters, device)

        def mask_out():
            logits = model(x_gpu)
            prob   = torch.sigmoid(logits.float())
            _      = (prob >= best_thr).to(torch.uint8)
        b_mean, b_std, b_fps = gpu_timed(mask_out, n_warmup, n_iters, device)

    e2e = e2e_times_segformer(model, best_thr, test_paths, cfg, device)

    return {
        "raw_fwd_fps":    a_fps,    "raw_fwd_mean_ms":  a_mean,  "raw_fwd_std_ms":  a_std,
        "mask_out_fps":   b_fps,    "mask_out_mean_ms": b_mean,  "mask_out_std_ms": b_std,
        "e2e_fps":        1000.0 / float(e2e.mean()),
        "e2e_mean_ms":    float(e2e.mean()),
        "e2e_std_ms":     float(e2e.std()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test-set inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_segformer_inference_loop(
    model: nn.Module,
    loader: DataLoader,
    best_thr: float,
    device: torch.device,
    amp: bool,
    surf_dice_delta: float,
    extra_row_fields: dict | None = None,
) -> list[dict]:
    """Run inference over the test set and return per-image rows.

    Why per-image rows (not just aggregate): per-image CSVs enable post-hoc
    analysis (per-ITA-group breakdown, Wilcoxon test, outlier inspection)
    without re-running inference.

    Args:
        extra_row_fields: additional constant fields merged into every row
            (e.g. {"temperature_used": 1.0} for Track B, matching that
            script's original per-image CSV schema) -- None (Track A's
            original schema) adds nothing beyond threshold_used.
    """
    rows = []
    with torch.no_grad():
        for x, y, stems, img_paths, mask_paths in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            # .float(): convert fp16 logits to fp32 before sigmoid to avoid overflow
            prob_np = torch.sigmoid(logits).float().cpu().numpy()
            gt_np   = y.cpu().numpy()

            for i, stem in enumerate(stems):
                pred = (prob_np[i, 0] >= best_thr).astype("uint8")
                g    = (gt_np[i, 0] > 0.5).astype("uint8")
                row  = compute_image_row_extended(
                    pred, g, str(stem), surf_dice_delta=surf_dice_delta)
                row["threshold_used"] = best_thr    # record which threshold produced this result
                if extra_row_fields:
                    row.update(extra_row_fields)
                rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon tests
# ─────────────────────────────────────────────────────────────────────────────

def run_wilcoxon_tests_vs_baseline(all_results: dict, baseline_name: str, out_dir: Path) -> list[dict]:
    """Run Wilcoxon signed-rank tests comparing each model vs the baseline.

    Why B0 direct as baseline: B0 direct is the simplest student model —
    no teacher, just straightforward training. Comparing against it answers:
    "Does distillation help?" and "Does the teacher help over direct training?".

    Args:
        all_results:   {run_name: summary_dict} including "_per_image_dice".
        baseline_name: name of the baseline model (typically "segformer_b0_direct").
        out_dir:       directory to save wilcoxon_tests.csv.

    Returns:
        List of Wilcoxon result dicts (one per non-baseline model). Empty
        list (and no CSV written) if baseline_name is not in all_results.
    """
    if baseline_name not in all_results:
        logger.warning("Baseline '%s' not in results — skipping Wilcoxon tests.", baseline_name)
        return []

    baseline_dice = all_results[baseline_name]["_per_image_dice"]
    wilcoxon_rows = []

    for run_name, res in all_results.items():
        if run_name == baseline_name:
            continue    # skip comparison with self
        w = wilcoxon_compare(res["_per_image_dice"], baseline_dice)
        wilcoxon_rows.append({"model": run_name, "vs_baseline": baseline_name, **w})

    if wilcoxon_rows:
        pd.DataFrame(wilcoxon_rows).to_csv(out_dir / "wilcoxon_tests.csv", index=False)

    return wilcoxon_rows
