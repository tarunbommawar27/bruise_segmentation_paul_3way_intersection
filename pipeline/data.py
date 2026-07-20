from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    img_col = _find_col(df, ["image_path", "image", "img_path", "resolved_image_path"])
    msk_col = _find_col(
        df, ["mask_path", "majority_mask_path", "mask", "gt_mask_path", "resolved_paul_mask_path"]
    )
    if img_col is None:
        raise RuntimeError(f"No image column found. Available: {list(df.columns)}")
    if msk_col is None:
        raise RuntimeError(f"No mask column found. Available: {list(df.columns)}")
    df["image_path"] = df[img_col].astype(str)
    df["mask_path"] = df[msk_col].astype(str)
    if "stem" not in df.columns:
        df["stem"] = df["image_path"].apply(lambda p: Path(p).stem)
    if "subject" not in df.columns:
        df["subject"] = df["stem"].apply(lambda s: str(s).split("_")[0].split("-")[0])
    return df


def load_train_val_split(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_manifest(pd.read_csv(csv_path))
    if "split" not in df.columns:
        raise RuntimeError(f"{csv_path} is missing a 'split' column. Run 00_build_split.py first.")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    return train_df, val_df


def load_fixed_test(csv_path: str) -> pd.DataFrame:
    return normalize_manifest(pd.read_csv(csv_path))


def get_augmentation(training: bool, img_h: int, img_w: int) -> A.Compose:
    imagenet_norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if training:
        return A.Compose([
            A.Resize(height=img_h, width=img_w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            imagenet_norm,
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(height=img_h, width=img_w), imagenet_norm, ToTensorV2()])


class BruiseDataset(Dataset):
    """Reads BGR images and binary masks from disk, resized to img_h x img_w."""

    def __init__(self, df: pd.DataFrame, img_h: int, img_w: int, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.tfm = get_augmentation(training, img_h, img_w)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img = cv2.imread(str(r.image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {r.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {r.mask_path}")
        # IMREAD_GRAYSCALE is documented to return (H, W), but importing ultralytics
        # ANYWHERE in the process monkey-patches cv2.imread so it returns (H, W, 1)
        # instead -- import ORDER does not help, only whether ultralytics is imported
        # at all. That trailing axis survives albumentations and ToTensorV2, so y comes
        # out [B,1,H,W,1] instead of [B,1,H,W]. Nothing crashes: downstream
        # dice_np(pred[H,W], gt[H,W,1]) BROADCASTS to [H,W,H] and returns nonsense (a
        # pixel-perfect prediction scores Dice 63.9 instead of 1.0). Squeezing here
        # fixes every consumer at once, and is a no-op when cv2 is unpatched.
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        return x, y, str(r.stem), str(r.image_path), str(r.mask_path)
