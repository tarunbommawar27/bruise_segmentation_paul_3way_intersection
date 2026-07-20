import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def read_gt_mask(path: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return (mask > 0).astype("float32")


def teacher_prob_for_image(model, temperature: float, img_bgr: np.ndarray, img_h: int, img_w: int,
                            device: torch.device) -> np.ndarray:
    """Run the teacher at its trained resolution (img_h x img_w), then resize the
    probability map back up to the image's native resolution -- running the teacher
    at native clinical-photo resolution instead would cause a self-attention OOM,
    since attention memory scales with image area."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_w, img_h), interpolation=cv2.INTER_LINEAR).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
    std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
    resized = (resized - mean) / std
    x = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x) / temperature)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(prob, 0.0, 1.0)


def save_semantic_class_mask(mask01: np.ndarray, path: Path) -> None:
    """0=background, 1=bruise -- Ultralytics' semantic ('task=semantic') label format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask01.astype("uint8")):
        raise RuntimeError(f"Could not write semantic mask: {path}")


def yolo_sem_pred_mask(result, shape: tuple[int, int]) -> np.ndarray:
    """task="semantic" results expose neither .masks nor .probs (those are
    instance-segmentation / classification attributes) -- the dense per-pixel
    class-ID map lives at result.semantic_mask.data, shape (H, W), values are
    class indices (0=background, 1=bruise for this 2-class setup)."""
    h, w = shape
    if hasattr(result, "semantic_mask") and result.semantic_mask is not None:
        class_map = result.semantic_mask.data
        if hasattr(class_map, "cpu"):
            class_map = class_map.cpu().numpy()
        class_map = np.asarray(class_map)
        m = (class_map == 1).astype("uint8")
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.zeros((h, w), dtype="uint8")
