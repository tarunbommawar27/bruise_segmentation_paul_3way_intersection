"""
pipeline/metrics_extended.py

Extended evaluation metrics for the Gut-style apple-to-apple benchmark.

Adds to the standard pipeline/metrics.py:
  - dice_safe()                   — Dice with correct empty-mask handling
  - iou_safe()                    — IoU with correct empty-union handling
  - surface_dice()                — boundary Dice within pixel tolerance δ
  - asd_px()                      — Average Surface Distance in pixels
  - hd95_px()                     — 95th-percentile Hausdorff Distance
  - compute_image_row_extended()  — drops-in for compute_image_row, adds
                                    boundary metrics and fixed empty-GT Dice
  - bootstrap_ci()                — 95% CI for any scalar metric
  - wilcoxon_compare()            — paired Wilcoxon signed-rank test
  - summarize_extended()          — full aggregate: median/IQR/mean/std/CI

WHY A SEPARATE FILE (not editing metrics.py):
  metrics.py is used by scripts 06, 07, 08, 09 which have already
  produced committed results. Changing dice_np() there would change those
  numbers and make them incomparable. This module adds the fixed versions
  alongside the originals so both sets of results remain reproducible.

EMPTY-GT DICE FIX (dice_safe):
  The original dice_np() returns 1.0 whenever denom==0, which includes
  the case where pred is non-empty but GT is empty (a false-positive on
  a clean image). During threshold sweep this inflates mean_dice for
  overly-aggressive thresholds and can cause the sweep to pick the WRONG
  threshold. dice_safe() corrects all four empty-mask cases:

      pred empty  AND  GT empty   → 1.0  (true negative, correct)
      pred non-empty  AND  GT empty → 0.0  (false positive)
      pred empty  AND  GT non-empty → 0.0  (complete miss)
      both non-empty              → standard 2|P∩G|/(|P|+|G|)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-count helpers (used by multiple metric functions)
# ─────────────────────────────────────────────────────────────────────────────

def _bool_masks(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cast pred and gt to bool in one place.

    Why: every metric function needs bool arrays. Casting in a helper
    prevents both the risk of forgetting to cast and the overhead of
    casting multiple times within the same function.
    """
    # astype(bool): non-zero → True, zero → False. Handles uint8 and float masks.
    return pred.astype(bool), gt.astype(bool)


def _pixel_counts(pred_b: np.ndarray, gt_b: np.ndarray) -> tuple[int, int, int, int, int]:
    """Compute TP, FP, FN, pred-positive, gt-positive pixel counts.

    Why: every evaluation metric (Dice, IoU, precision, recall) needs these
    five numbers. Computing them once and sharing avoids redundant bitwise
    operations across the metrics in compute_image_row_extended().

    Args:
        pred_b: bool prediction mask, shape (H, W).
        gt_b:   bool ground-truth mask, shape (H, W).

    Returns:
        (pp, gp, tp, fp, fn) — all integers.
    """
    # pp/gp: positive pixel counts used by empty-mask guards
    pp = int(pred_b.sum())
    gp = int(gt_b.sum())
    # logical_and: pixels correctly predicted as bruise
    tp = int(np.logical_and(pred_b, gt_b).sum())
    # logical_and with ~gt_b: pixels predicted bruise but are background
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    # logical_and with ~pred_b: bruise pixels the model missed entirely
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    return pp, gp, tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# Fixed Dice and IoU (empty-mask safe)
# ─────────────────────────────────────────────────────────────────────────────

def dice_safe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient with correct four-case empty-mask handling.

    Why this replaces dice_np() from metrics.py:
      dice_np() returns 1.0 whenever denom==0, even when pred is non-empty
      and GT is empty (a false-positive). That inflates mean_dice during the
      threshold sweep and can cause the sweep to select an overly-aggressive
      threshold. See module docstring for the full correction table.

    Args:
        pred: binary prediction array (any numeric dtype, non-zero = positive).
        gt:   binary ground-truth array.

    Returns:
        float in [0.0, 1.0].
    """
    pred_b, gt_b = _bool_masks(pred, gt)
    pp = int(pred_b.sum())   # predicted positive pixel count
    gp = int(gt_b.sum())     # ground-truth positive pixel count

    if pp == 0 and gp == 0:
        # Both masks empty: model correctly predicts no bruise → 1.0
        return 1.0
    if pp == 0 or gp == 0:
        # One mask is empty but not the other: either complete miss or false-positive
        return 0.0

    # Standard Dice: 2|P∩G| / (|P|+|G|)
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(2 * inter / (pp + gp))


def iou_safe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-Union with correct empty-union handling.

    Why: IoU = |P∩G|/|P∪G|. When both masks are empty, union=0 → division
    by zero. The correct answer is 1.0 (both empty means perfect agreement).

    Args:
        pred: binary prediction array.
        gt:   binary ground-truth array.

    Returns:
        float in [0.0, 1.0].
    """
    pred_b, gt_b = _bool_masks(pred, gt)
    # Union: all pixels predicted or labelled as bruise
    union = int(np.logical_or(pred_b, gt_b).sum())
    if union == 0:
        # Both masks empty → perfect IoU by convention
        return 1.0
    # Intersection: pixels both predicted and labelled as bruise
    inter = int(np.logical_and(pred_b, gt_b).sum())
    return float(inter / union)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary extraction (shared by all three boundary metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _boundary(mask: np.ndarray) -> np.ndarray:
    """Return the boundary pixels of a binary mask.

    Why erosion-based boundary: eroding the mask by one pixel and XORing with
    the original gives exactly the outer ring of pixels. This is the standard
    morphological boundary definition used by Surface Dice literature.

    Args:
        mask: bool or binary array (H, W).

    Returns:
        bool array (H, W) — True only at boundary pixels.
    """
    mask_b = mask.astype(bool)
    # binary_erosion shrinks the mask by 1 pixel — the ring lost is the boundary
    eroded = ndimage.binary_erosion(mask_b)
    # XOR: pixels in original but NOT in eroded = boundary ring
    return mask_b & ~eroded


def _distance_from_boundary(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance transform from the boundary of a binary mask.

    Why distance transform: surface metrics (Surface Dice, ASD, HD95) need
    to know how far each point is from the nearest boundary. The EDT
    gives this efficiently in O(n) for an n-pixel image.

    Args:
        mask: bool or binary array (H, W).

    Returns:
        float32 array (H, W) — distance (pixels) from nearest boundary pixel.
        Returns array of np.inf if the mask has no boundary (empty mask).
    """
    bnd = _boundary(mask)
    if bnd.sum() == 0:
        # No boundary (empty mask) → every pixel is infinitely far from boundary
        # Returning inf causes downstream checks to return NaN/0 correctly
        return np.full(mask.shape, np.inf, dtype=np.float32)
    # distance_transform_edt: ~bnd marks non-boundary pixels; EDT gives distance to nearest True
    return ndimage.distance_transform_edt(~bnd).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Surface Dice (boundary Dice with tolerance δ)
# ─────────────────────────────────────────────────────────────────────────────

def surface_dice(pred: np.ndarray, gt: np.ndarray, delta: float = 2.0) -> float:
    """Surface Dice at boundary tolerance δ pixels (Nikolov et al. 2018).

    Why Surface Dice alongside standard Dice:
      Standard Dice counts all pixels equally. Surface Dice focuses only on
      the boundary and counts a boundary pixel as "correct" if the nearest
      boundary on the other mask is within δ pixels. This is clinically
      relevant: for bruise segmentation, a 2-pixel boundary error is acceptable
      while completely missing the bruise edge is not.

    Formula:
      Surface_Dice = (|bnd_pred within δ of bnd_gt| + |bnd_gt within δ of bnd_pred|)
                     / (|bnd_pred| + |bnd_gt|)

    Args:
        pred:  binary prediction array (H, W).
        gt:    binary ground-truth array (H, W).
        delta: tolerance in pixels (default 2.0 per supervisor spec).

    Returns:
        float in [0.0, 1.0]. Returns 1.0 if both masks empty, 0.0 if one is empty.
    """
    pred_b, gt_b = _bool_masks(pred, gt)
    pp, gp = int(pred_b.sum()), int(gt_b.sum())

    if pp == 0 and gp == 0:
        # Both empty: correct prediction of no bruise → perfect score
        return 1.0
    if pp == 0 or gp == 0:
        # One boundary is missing: cannot compute a meaningful surface overlap
        return 0.0

    bnd_pred = _boundary(pred_b)
    bnd_gt   = _boundary(gt_b)

    if bnd_pred.sum() == 0 or bnd_gt.sum() == 0:
        # Pathological case: filled mask with no boundary (single-pixel masks etc.)
        return 0.0

    # Distance from each boundary to the other mask's boundary
    dist_from_gt   = _distance_from_boundary(gt_b)    # distance of each pixel from GT boundary
    dist_from_pred = _distance_from_boundary(pred_b)  # distance of each pixel from pred boundary

    # Count pred boundary pixels within δ of GT boundary
    pred_on_gt = int((dist_from_gt[bnd_pred] <= delta).sum())
    # Count GT boundary pixels within δ of pred boundary
    gt_on_pred = int((dist_from_pred[bnd_gt] <= delta).sum())

    # Denominator: total boundary pixels in both masks
    denom = int(bnd_pred.sum()) + int(bnd_gt.sum())
    return float((pred_on_gt + gt_on_pred) / denom)


# ─────────────────────────────────────────────────────────────────────────────
# Average Surface Distance (ASD)
# ─────────────────────────────────────────────────────────────────────────────

def asd_px(pred: np.ndarray, gt: np.ndarray) -> float:
    """Symmetric Average Surface Distance in pixels.

    Why ASD: Surface Dice only tells you whether boundaries overlap within δ.
    ASD tells you the average misalignment distance when there IS an error —
    a small ASD means small boundary errors, a large ASD means the predicted
    boundary is far from the true one.

    Formula (symmetric):
      ASD = (mean d(bnd_pred → bnd_gt) + mean d(bnd_gt → bnd_pred)) / 2

    Args:
        pred: binary prediction array.
        gt:   binary ground-truth array.

    Returns:
        float ≥ 0 pixels. Returns 0.0 if both empty. Returns nan if one is empty
        (ASD is undefined when one boundary is absent).
    """
    pred_b, gt_b = _bool_masks(pred, gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        # Both empty: zero distance by convention — model correctly found nothing
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        # One boundary missing: ASD is undefined (nan propagates to summarize_extended
        # where it is excluded from mean/median to avoid corrupting the aggregate)
        return float("nan")

    bnd_pred = _boundary(pred_b)
    bnd_gt   = _boundary(gt_b)

    dist_from_gt   = _distance_from_boundary(gt_b)
    dist_from_pred = _distance_from_boundary(pred_b)

    # Mean distance from pred boundary to nearest GT boundary
    p2g = float(dist_from_gt[bnd_pred].mean()) if bnd_pred.sum() > 0 else 0.0
    # Mean distance from GT boundary to nearest pred boundary
    g2p = float(dist_from_pred[bnd_gt].mean()) if bnd_gt.sum() > 0 else 0.0

    # Symmetric: average both directed distances
    return float((p2g + g2p) / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# 95th-percentile Hausdorff Distance (HD95)
# ─────────────────────────────────────────────────────────────────────────────

def hd95_px(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff Distance in pixels.

    Why HD95 instead of HD100: the classical Hausdorff distance is the maximum
    over all boundary pixels. One outlier pixel (e.g. a tiny disconnected
    prediction fragment) can dominate HD100 and make the metric uninterpretable.
    HD95 uses the 95th percentile instead, which is robust to outliers while
    still penalising systematic boundary errors.

    Formula:
      HD95 = max(percentile_95(d(bnd_pred → bnd_gt)),
                 percentile_95(d(bnd_gt → bnd_pred)))

    Args:
        pred: binary prediction array.
        gt:   binary ground-truth array.

    Returns:
        float ≥ 0 pixels. Returns 0.0 if both empty. Returns nan if one is empty.
    """
    pred_b, gt_b = _bool_masks(pred, gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        # Both empty: zero Hausdorff distance by convention
        return 0.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        # One boundary missing: HD95 is undefined
        return float("nan")

    bnd_pred = _boundary(pred_b)
    bnd_gt   = _boundary(gt_b)

    dist_from_gt   = _distance_from_boundary(gt_b)
    dist_from_pred = _distance_from_boundary(pred_b)

    # Distances from each boundary pixel to the other mask's boundary
    p_dists = dist_from_gt[bnd_pred]   if bnd_pred.sum() > 0 else np.array([0.0])
    g_dists = dist_from_pred[bnd_gt]   if bnd_gt.sum()   > 0 else np.array([0.0])

    # HD95: take the max of the 95th percentiles from each direction
    # The max makes HD95 symmetric (both boundaries must agree)
    return float(max(np.percentile(p_dists, 95), np.percentile(g_dists, 95)))


# ─────────────────────────────────────────────────────────────────────────────
# Per-image precision and recall helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP+FP). Returns 1.0 when no positive predictions made.

    Why 1.0 for empty prediction: if the model predicts nothing, it makes no
    false positives, so precision is perfectly defined as 1.0 (vacuously).
    This matches scikit-learn's zero_division=1 behaviour.
    """
    # tp + fp == 0 means model made no positive prediction (all background)
    return 1.0 if (tp + fp) == 0 else float(tp / (tp + fp))


def _safe_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP+FN). Returns 1.0 when GT has no positives.

    Why 1.0 for empty GT: if there is no bruise to find, the model cannot
    miss anything, so recall is 1.0 by convention.
    """
    # tp + fn == 0 means GT has no positive pixels (no bruise present)
    return 1.0 if (tp + fn) == 0 else float(tp / (tp + fn))


# ─────────────────────────────────────────────────────────────────────────────
# Extended per-image row (drop-in for compute_image_row)
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_row_extended(
    pred: np.ndarray,
    gt: np.ndarray,
    stem: str,
    surf_dice_delta: float = 2.0,
) -> dict:
    """Drop-in replacement for compute_image_row() with boundary metrics and
    fixed empty-GT Dice.

    Why a new function rather than patching compute_image_row:
      Scripts 06-09 have already recorded results using compute_image_row.
      Changing that function retroactively would break reproducibility of those
      committed results. This function adds the extended metrics alongside the
      original ones without touching the original code path.

    Args:
        pred:            binary prediction mask (H, W), uint8 or bool.
        gt:              binary ground-truth mask (H, W), uint8 or bool.
        stem:            image filename stem for indexing results.
        surf_dice_delta: boundary tolerance in pixels for Surface Dice (default 2).

    Returns:
        Dict with keys: stem, dice, iou, precision, recall, tp_pixels, fp_pixels,
        fn_pixels, pred_positive_pixels, gt_positive_pixels, zero_dice,
        complete_miss, pred_gt_ratio, surf_dice, asd_px, hd95_px.
    """
    pred_b, gt_b = _bool_masks(pred, gt)

    # Get all pixel counts in a single pass (avoids redundant bitwise ops)
    pp, gp, tp, fp, fn = _pixel_counts(pred_b, gt_b)

    # Dice and IoU with correct empty-mask handling (see module docstring)
    d = dice_safe(pred, gt)
    j = iou_safe(pred, gt)

    # Precision and recall with vacuous-truth convention for empty sets
    precision = _safe_precision(tp, fp)
    recall    = _safe_recall(tp, fn)

    # Boundary metrics — expensive (requires distance transforms) but essential
    # for the supervisor's boundary-accuracy requirement
    sd  = surface_dice(pred, gt, delta=surf_dice_delta)
    asd = asd_px(pred, gt)
    hd  = hd95_px(pred, gt)

    return {
        "stem":                  stem,
        "dice":                  d,
        "iou":                   j,
        "precision":             precision,
        "recall":                recall,
        "tp_pixels":             tp,
        "fp_pixels":             fp,
        "fn_pixels":             fn,
        "pred_positive_pixels":  pp,
        "gt_positive_pixels":    gp,
        # zero_dice: model produced non-zero output but Dice is still 0 (badly wrong)
        "zero_dice":             (d == 0.0 and gp > 0),
        # complete_miss: model predicted nothing on an image that has a bruise
        "complete_miss":         (pp == 0 and gp > 0),
        # pred_gt_ratio: how much bigger/smaller the prediction is vs GT area
        "pred_gt_ratio":         pp / gp if gp > 0 else float("nan"),
        "surf_dice":             sd,
        "asd_px":                asd,
        "hd95_px":               hd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    stat: str = "median",
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap confidence interval for median or mean.

    Why bootstrapping: parametric CIs (e.g. t-interval) assume normality.
    Per-image Dice scores are skewed (many zeros from complete misses, many
    near-1.0 from easy negatives). Bootstrap makes no distributional assumption
    and is the standard for non-normal medical imaging metrics.

    Args:
        values:      1D array of per-image metric values (e.g. Dice scores).
        n_bootstrap: number of resamples (2000 gives stable 95% CI).
        ci:          confidence level (default 0.95 = 95%).
        stat:        "median" (default) or "mean".
        seed:        RNG seed for reproducibility across scripts.

    Returns:
        (lower, upper) bounds of the CI at the given confidence level.
        Returns (nan, nan) if values is empty after NaN removal.
    """
    rng  = np.random.default_rng(seed)    # seeded for reproducibility
    vals = np.asarray(values, dtype=float)
    # Drop NaN — ASD/HD95 are nan when one mask is empty; we exclude those
    # images rather than letting them poison the confidence interval
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        # No valid values to bootstrap → return nan pair (propagates to table)
        return float("nan"), float("nan")

    # Choose statistic function: median for skewed Dice, mean available for completeness
    fn = np.median if stat == "median" else np.mean

    # Draw n_bootstrap resamples WITH replacement (standard bootstrap procedure)
    boots = np.array(
        [fn(rng.choice(vals, size=len(vals), replace=True))
         for _ in range(n_bootstrap)]
    )

    # Percentile method (empirical): no normal approximation required
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boots, 100 * alpha))
    hi = float(np.percentile(boots, 100 * (1.0 - alpha)))
    return lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank test
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test: model A vs model B on the same images.

    Why Wilcoxon (not t-test): Dice scores are non-normal (many zeros, many
    near-1.0). Wilcoxon is non-parametric and valid for any distribution.
    The paired version is correct here because each row in a[] and b[]
    corresponds to the SAME test image, so the measurements are dependent.

    Why effect size: p-value alone is inflated by large test sets (185 images
    can detect trivially small differences as "significant"). The rank-biserial
    correlation r is bounded in [-1, 1] and gives a scale-independent measure
    of practical importance.

    Args:
        a: per-image Dice scores for model A (same image order as b).
        b: per-image Dice scores for model B.

    Returns:
        Dict with keys: statistic, p_value, effect_size, n_pairs, and optionally
        a "warning" key if fewer than 10 valid pairs exist.
    """
    from scipy.stats import wilcoxon   # deferred import — scipy is heavy
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Keep only pairs where BOTH values are finite (drop NaN rows)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b  = a[valid], b[valid]
    n     = int(len(a))

    if n < 10:
        # Wilcoxon is unreliable with very few pairs — flag rather than crash
        return {
            "statistic":   float("nan"),
            "p_value":     float("nan"),
            "effect_size": float("nan"),
            "n_pairs":     n,
            "warning":     "fewer than 10 valid pairs — result unreliable",
        }

    # two-sided: test whether model A and B differ (not which is better)
    stat, p = wilcoxon(a, b, alternative="two-sided")

    # Rank-biserial correlation: r = 1 - 2W / (n(n+1)/2)
    # r=1 means A always wins, r=-1 means B always wins, r=0 means no effect
    r = 1.0 - (2.0 * stat) / (n * (n + 1) / 2.0)

    return {
        "statistic":   float(stat),
        "p_value":     float(p),
        "effect_size": float(r),
        "n_pairs":     n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extended summarize (replaces summarize() for Track A/B scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_column(df: pd.DataFrame, col: str, n_bootstrap: int) -> dict:
    """Compute median/IQR/mean/std and bootstrap 95% CI for one metric column.

    Why IQR alongside std: Dice is non-normal. Std is reported for
    completeness but IQR (Q3-Q1) is the correct spread measure for skewed
    distributions.

    Args:
        df:          DataFrame of per-image rows.
        col:         Column name to aggregate.
        n_bootstrap: Number of bootstrap resamples for the CI.

    Returns:
        Dict with keys: median_{col}, iqr_{col}, mean_{col}, std_{col},
        ci95_lo_{col}, ci95_hi_{col}.
    """
    vals = df[col].dropna()    # drop NaN before aggregation (e.g. empty-mask ASD rows)
    lo, hi = bootstrap_ci(vals.values, n_bootstrap=n_bootstrap)
    return {
        f"median_{col}":  float(vals.median()),
        f"iqr_{col}":     float(vals.quantile(0.75) - vals.quantile(0.25)),
        f"mean_{col}":    float(vals.mean()),
        f"std_{col}":     float(vals.std()),
        f"ci95_lo_{col}": lo,
        f"ci95_hi_{col}": hi,
    }


def summarize_extended(rows: list[dict], n_bootstrap: int = 2000) -> dict:
    """Full aggregate summary over per-image rows from compute_image_row_extended().

    Why here instead of inline in scripts: every evaluation script (09, 10, 11)
    needs the same summary table with the same columns. A single function
    guarantees consistency — if we add a new metric we add it here and all
    scripts get it automatically.

    Args:
        rows:        List of per-image row dicts from compute_image_row_extended().
        n_bootstrap: Number of bootstrap resamples (default 2000).

    Returns:
        Dict with aggregate statistics. Empty rows → empty dict.
    """
    if not rows:
        # Return empty dict rather than crashing — caller can check for it
        return {}

    df = pd.DataFrame(rows)

    # Initialise output with the image count for sanity-checking table sizes
    out: dict = {"n_images": len(df)}

    # Aggregate all continuous metrics using the same helper
    for col in ["dice", "iou", "surf_dice", "asd_px", "hd95_px", "precision", "recall"]:
        if col in df.columns:
            out.update(_aggregate_column(df, col, n_bootstrap))

    # Miss-rate statistics (binary columns → use mean for rate)
    # complete_miss: model predicted nothing on an image that has a bruise
    complete_miss = df.get("complete_miss",
                            (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0))
    # zero_dice: model output non-empty but Dice is still 0 (highly wrong)
    zero_dice = df.get("zero_dice",
                        (df["dice"] == 0) & (df["gt_positive_pixels"] > 0))

    out["zero_dice_count"]     = int(zero_dice.sum())
    out["zero_dice_rate"]      = float(zero_dice.mean())
    out["complete_miss_count"] = int(complete_miss.sum())
    out["complete_miss_rate"]  = float(complete_miss.mean())

    # Prediction size ratio: > 1 means over-segmenting, < 1 means under-segmenting
    # Replace inf (from gp=0 divisions) with nan before averaging
    out["mean_pred_gt_ratio"] = float(
        df["pred_gt_ratio"].replace([np.inf, -np.inf], np.nan).mean()
    )

    return out
