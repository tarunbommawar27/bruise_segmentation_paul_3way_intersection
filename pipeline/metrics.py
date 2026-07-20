import numpy as np
import pandas as pd


def dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(inter / union)


def precision_recall_np(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int, int]:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = 1.0 if tp + fp == 0 else float(tp / (tp + fp))
    recall = 1.0 if tp + fn == 0 else float(tp / (tp + fn))
    return precision, recall, tp, fp, fn


def compute_image_row(pred: np.ndarray, gt: np.ndarray, stem: str) -> dict:
    d = dice_np(pred, gt)
    j = iou_np(pred, gt)
    pr, rc, tp, fp, fn = precision_recall_np(pred, gt)
    pp = int(pred.astype(bool).sum())
    gp = int(gt.astype(bool).sum())
    return {
        "stem": stem, "dice": d, "iou": j, "precision": pr, "recall": rc,
        "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn,
        "pred_positive_pixels": pp, "gt_positive_pixels": gp,
        "pred_gt_ratio": pp / gp if gp > 0 else float("nan"),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    complete_miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
    return {
        "n_images": int(len(df)),
        "mean_dice": float(df["dice"].mean()),
        "median_dice": float(df["dice"].median()),
        "mean_iou": float(df["iou"].mean()),
        "median_iou": float(df["iou"].median()),
        "zero_dice_count": int((df["dice"] == 0).sum()),
        "zero_dice_rate": float((df["dice"] == 0).mean()),
        "complete_miss_count": int(complete_miss.sum()),
        "complete_miss_rate": float(complete_miss.mean()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_pred_gt_ratio": float(df["pred_gt_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
    }
