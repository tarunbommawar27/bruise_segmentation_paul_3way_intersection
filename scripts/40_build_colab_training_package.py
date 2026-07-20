#!/usr/bin/env python3
"""
scripts/40_build_colab_training_package.py

Builds bruise_colab_train.zip -- everything bruise_colab_train_all.ipynb needs
to TRAIN all 5 models from scratch on Colab (the existing
32_build_colab_gpu_package.py builds an INFERENCE-only package and ships no
train images; this is its training counterpart, and neither replaces the other).

Contents: 640x640 pre-resized images+masks for all three splits, three
manifests, the pretrained backbones, and the inter-labeler agreement table.
Deliberately NO pipeline code -- the notebook writes its own modules, so the
notebook is the single source of truth and cannot drift out of sync with a
stale copy baked into the zip.

WHY THE IMAGES ARE PRE-RESIZED TO 640 (and why that is not a shortcut)
----------------------------------------------------------------------
The source photos are 4022x6024 (~24MP, ~2.5MB JPEG each). Every epoch, the
training dataloader decoded all 697 of them at full resolution and immediately
threw ~99% of the pixels away resizing to 640x640 -- ~287ms of CPU per image,
measured on the build machine. On Colab (2-8 vCPU) that makes every epoch
CPU-bound no matter which GPU is attached.

Pre-resizing is *mathematically equivalent*, not an approximation, because
A.Resize is the FIRST op in the augmentation pipeline -- it runs before every
flip, brightness and noise transform, so nothing upstream of it depends on the
native resolution. Doing that one deterministic resize once at build time and
caching the result produces the identical 640x640 array the dataloader would
have produced itself, and PNG is lossless so the cached copy is bit-exact.
Interpolation is matched to albumentations' own defaults on purpose:
INTER_LINEAR for images, INTER_NEAREST for masks.

Masks are binarised BEFORE the nearest-neighbour resize, matching
BruiseDataset's read -> (mask > 0) -> A.Resize(mask_interpolation=NEAREST)
order. Binarising after would let a nearest-sampled raw grey value land on
the wrong side of the > 0 test.

WHY ZIP_STORED, NOT ZIP_DEFLATED
---------------------------------
PNG payloads are already deflate-compressed. Re-deflating them costs minutes
of CPU at build time and minutes again at unzip time on Colab, and buys back
well under 1%. The manifests and YAML are small enough that compressing them
separately is not worth the extra code path.

THE ASPECT RATIO IS DELIBERATELY NOT PRESERVED
-----------------------------------------------
6024x4022 -> 640x640 is an anisotropic stretch, not a letterbox. That is what
the original pipeline did and what all five trained models saw, so changing it
here would silently invalidate any comparison against the historical runs. It
is recorded in the manifest (native_h/native_w) so the choice stays visible.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import setup_logging

logger = setup_logging()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 640

TRAIN_VAL_SPLIT = PROJECT_ROOT / "splits" / "train_val_split.csv"
TEST_MANIFEST = (PROJECT_ROOT / "labelbox_paul_orc_dataset_full_consensus"
                 / "fixed_consensus_test" / "manifest.csv")
TEST_ITA = PROJECT_ROOT / "ita_labels" / "wl_test_per_image_ita.csv"
INTERLABELER = PROJECT_ROOT / "interlabeler_agreement_640.csv"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_weights"

# Cache of resized PNGs, kept between runs so a rebuild costs seconds not minutes.
# Manifests store ARCHIVE-relative paths ("data640/train/images/x.png"); the same
# string appended to CACHE_ROOT gives the on-disk location. Keeping the two in one
# mapping (rather than two parallel path builders) is why cache_path() exists.
CACHE_ROOT = PROJECT_ROOT / ".colab_train_cache"
DATA_ARC = f"data{IMG_SIZE}"


def cache_path(arc_path: str) -> Path:
    """On-disk location of an archive-relative manifest path."""
    return CACHE_ROOT / arc_path

EXPECTED = {"train": 697, "val": 134, "test": 185}

# Columns carried into every manifest. ita_group_index_5 / skin_tone_category
# are what the notebook's fairness section stratifies on; without them in the
# zip that section cannot run and the fairness eval would silently be skipped.
KEEP_COLS = ["stem", "subject", "ita_group_index_5", "skin_tone_category", "ITA"]


def resize_image_640(path: Path) -> np.ndarray:
    """Decode a native-resolution photo and resize to 640x640, BGR, INTER_LINEAR.

    Reads and writes BGR (never RGB): the training dataloader does
    imread -> cvtColor(BGR2RGB) -> resize, and resize is per-channel, so
    resizing in BGR and letting the loader do its own BGR2RGB afterwards is
    identical to resizing in RGB here. Keeping BGR means cv2.imwrite stores
    the channels in the order cv2.imread will read them back.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def resize_mask_640(path: Path) -> np.ndarray:
    """Decode a mask, binarise, then resize 640x640 with INTER_NEAREST.

    IMREAD_GRAYSCALE is documented to return (H, W) but returns (H, W, 1) in
    any process where ultralytics has been imported -- it monkey-patches
    cv2.imread at import time, and import ORDER does not help. That trailing
    axis silently broadcasts downstream into garbage Dice. Squeezing here is a
    no-op on an unpatched cv2 and makes the written PNG 2-D either way.
    Returns 0/255 so the PNG is human-viewable; the loader re-binarises with
    (> 0), so the exact positive value is not load-bearing.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    binary = (mask > 0).astype(np.uint8)
    resized = cv2.resize(binary, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (resized * 255).astype(np.uint8)


def _write_png(array: np.ndarray, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.png")
    if not cv2.imwrite(str(tmp), array, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"Could not write {dest}")
    tmp.replace(dest)    # atomic: a killed build never leaves a truncated PNG in the cache


def build_split_cache(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Resize one split into CACHE_DIR and return its manifest with relative paths.

    Skips any file already cached, so re-running after a crash (or after only
    the zip step changed) costs a directory listing instead of ~5 minutes of
    JPEG decoding.
    """
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"resize {split}"):
        stem = str(r["stem"])
        img_arc = f"{DATA_ARC}/{split}/images/{stem}.png"
        mask_arc = f"{DATA_ARC}/{split}/masks/{stem}.png"

        if not cache_path(img_arc).exists():
            _write_png(resize_image_640(Path(r["image_path"])), cache_path(img_arc))
        if not cache_path(mask_arc).exists():
            _write_png(resize_mask_640(Path(r["mask_path"])), cache_path(mask_arc))

        row = {c: r[c] for c in KEEP_COLS if c in df.columns}
        row.update({
            "split": split,
            "image_path": img_arc,
            "mask_path": mask_arc,
            "native_h": int(r["image_h"]) if "image_h" in df.columns and pd.notna(r["image_h"]) else -1,
            "native_w": int(r["image_w"]) if "image_w" in df.columns and pd.notna(r["image_w"]) else -1,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def load_splits() -> dict[str, pd.DataFrame]:
    """Assemble the three source manifests, with skin-tone labels attached to test.

    train/val already carry ITA columns (00_build_split.py merged them in);
    the test manifest does not, so it is joined against wl_test_per_image_ita.csv
    on `stem`. The join is asserted to be total -- a partial join would quietly
    drop images out of the fairness analysis rather than fail.
    """
    tv = pd.read_csv(TRAIN_VAL_SPLIT)
    if "split" not in tv.columns:
        raise RuntimeError(f"{TRAIN_VAL_SPLIT} has no 'split' column -- run 00_build_split.py.")

    test = pd.read_csv(TEST_MANIFEST)
    test["mask_path"] = test["majority_mask_path"] if "majority_mask_path" in test.columns else test["mask_path"]

    ita = pd.read_csv(TEST_ITA)[["stem", "ITA", "skin_tone_category", "ita_group_index_5"]]
    before = len(test)
    test = test.merge(ita, on="stem", how="left", validate="one_to_one")
    unmatched = int(test["ita_group_index_5"].isna().sum())
    if len(test) != before or unmatched:
        raise RuntimeError(
            f"Test/ITA join is not total: {len(test)} rows (expected {before}), "
            f"{unmatched} without a skin-tone label. The fairness section stratifies "
            "on this column and would silently drop those images.")

    return {
        "train": tv[tv["split"] == "train"].reset_index(drop=True),
        "val": tv[tv["split"] == "val"].reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def assert_no_leakage(manifests: dict[str, pd.DataFrame]) -> None:
    """Refuse to build a package whose splits share a subject or an image.

    This is the guard the whole design rests on: the threshold is fitted on val
    and reported on test, so any val/test contact silently inflates the reported
    test score instead of crashing. Subject-level (not just image-level) because
    one person contributes several photos -- same skin, same bruise, different
    angle -- and an image-level check would pass while leaking the subject.
    """
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        subj = set(manifests[a]["subject"]) & set(manifests[b]["subject"])
        if subj:
            raise RuntimeError(
                f"{len(subj)} subject(s) in BOTH {a} and {b} (e.g. {sorted(subj)[0]}). "
                "The split is supposed to be subject-level.")
        stems = set(manifests[a]["stem"]) & set(manifests[b]["stem"])
        if stems:
            raise RuntimeError(f"{len(stems)} image(s) in BOTH {a} and {b} (e.g. {sorted(stems)[0]}).")

    for split, expected in EXPECTED.items():
        if len(manifests[split]) != expected:
            raise RuntimeError(
                f"{split}: {len(manifests[split])} rows, expected {expected}. "
                "Refusing to build a package whose splits changed size unnoticed.")

    logger.info("PASS -- no subject or image overlap across train/val/test; counts %s", EXPECTED)


def verify_cache(manifests: dict[str, pd.DataFrame]) -> None:
    """Re-open every cached PNG and assert its shape before it goes in the zip.

    Catches a truncated or wrongly-shaped file here, on a machine with the
    source data, instead of 40 minutes into a Colab training run.
    """
    for split, df in manifests.items():
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"verify {split}"):
            img = cv2.imread(str(cache_path(r["image_path"])), cv2.IMREAD_COLOR)
            if img is None or img.shape != (IMG_SIZE, IMG_SIZE, 3):
                raise RuntimeError(f"Bad cached image {r['image_path']}: {None if img is None else img.shape}")
            msk = cv2.imread(str(cache_path(r["mask_path"])), cv2.IMREAD_GRAYSCALE)
            if msk is None:
                raise RuntimeError(f"Bad cached mask {r['mask_path']}")
            if msk.ndim == 3:
                msk = msk[..., 0]
            if msk.shape != (IMG_SIZE, IMG_SIZE):
                raise RuntimeError(f"Bad cached mask shape {r['mask_path']}: {msk.shape}")
            if not set(np.unique(msk)).issubset({0, 255}):
                raise RuntimeError(f"Mask {r['mask_path']} is not binary: {np.unique(msk)[:8]}")
    logger.info("PASS -- every cached PNG is %dx%d and every mask is binary.", IMG_SIZE, IMG_SIZE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "bruise_colab_train.zip"))
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip re-decoding every cached PNG (only safe on an unchanged cache)")
    args = ap.parse_args()
    out_path = Path(args.out)

    sources = load_splits()
    manifests = {split: build_split_cache(df, split) for split, df in sources.items()}

    assert_no_leakage(manifests)
    if not args.skip_verify:
        verify_cache(manifests)

    logger.info("Writing %s ...", out_path)
    # ZIP_STORED: the payload is PNG, already deflated -- see module docstring.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for split, df in manifests.items():
            for _, r in tqdm(df.iterrows(), total=len(df), desc=f"zip {split}"):
                zf.write(cache_path(r["image_path"]), r["image_path"])
                zf.write(cache_path(r["mask_path"]), r["mask_path"])
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(f"manifests/{split}.csv", buf.getvalue())

        for path in sorted(PRETRAINED_DIR.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                zf.write(path, f"pretrained_weights/{path.relative_to(PRETRAINED_DIR).as_posix()}")

        zf.write(INTERLABELER, "interlabeler_agreement_640.csv")

    # Post-write verification against the real zip, not against what we think we wrote.
    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())
        for split, df in manifests.items():
            missing = [p for p in list(df["image_path"]) + list(df["mask_path"]) if p not in names]
            if missing:
                raise RuntimeError(f"{len(missing)} {split} path(s) missing from zip, e.g. {missing[0]}")
        for required in ["manifests/train.csv", "manifests/val.csv", "manifests/test.csv",
                         "interlabeler_agreement_640.csv", "pretrained_weights/yolo26n-sem.pt"]:
            if required not in names:
                raise RuntimeError(f"Missing {required} in built package")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:16]
    logger.info("PASS -- wrote %s (%.0f MB, sha256:%s...)", out_path, size_mb, sha)
    logger.info("Upload to: Google Drive -> MyDrive/bruise_segmentation_gpu/%s", out_path.name)


if __name__ == "__main__":
    main()
