#!/usr/bin/env python3
"""
scripts/42_build_colab_final_package.py

Builds bruise_colab_final.zip for bruise_colab_final.ipynb -- the FINAL notebook
that trains SegFormer with the custom loop (per-model best batch) AND YOLO with
NATIVE Ultralytics training, then evaluates YOLO two ways (native argmax + custom
/255 swept). Neither 40's nor 32's package fits this; this is their successor.

WHY NATIVE RESOLUTION, NOT PRE-RESIZED 640 (the key difference from script 40)
------------------------------------------------------------------------------
Script 40 pre-resized everything to a 640 stretch because SegFormer only ever
sees 640 and the resize is deterministic, so caching it once is free and correct.
That does NOT work here, because native YOLO must reproduce its ~0.83 result, and
that number comes from Ultralytics letterboxing the FULL-resolution image to 640 --
a different geometry from the anisotropic 640 stretch. Feeding native YOLO a
pre-stretched 640 image would silently change what it was trained on and break the
reproduction. So this package ships the native-resolution images, and the notebook
builds a 640 stretch cache ONCE on Colab (at setup, not per-epoch) for SegFormer
and the custom-YOLO path. Native YOLO trains straight off the native images.

WHAT ELSE IS IN THE ZIP
-----------------------
Native images + binary masks for all three splits; three manifests carrying the
skin-tone (ITA) columns the fairness section stratifies on; the pretrained
backbones (SegFormer b0/b2 + yolo26n-sem.pt); and the inter-labeler agreement CSV
for the annotation-ceiling section. No pipeline code -- the notebook writes its own
modules, so it is the single source of truth.

Masks are binarised (0/255) at native resolution and shipped as PNG. They are NOT
resized here: the notebook resizes image and mask together (stretch for the 640
cache, or Ultralytics' letterbox for native YOLO), and pre-resizing the mask to one
geometry would make it wrong for the other consumer.
"""
from __future__ import annotations

import argparse
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
TRAIN_VAL_SPLIT = PROJECT_ROOT / "splits" / "train_val_split.csv"
TEST_MANIFEST = (PROJECT_ROOT / "labelbox_paul_orc_dataset_full_consensus"
                 / "fixed_consensus_test" / "manifest.csv")
TEST_ITA = PROJECT_ROOT / "ita_labels" / "wl_test_per_image_ita.csv"
INTERLABELER = PROJECT_ROOT / "interlabeler_agreement_640.csv"
PRETRAINED_DIR = PROJECT_ROOT / "pretrained_weights"

# Cache of binarised native-res masks, so a rebuild does not re-decode them each time.
CACHE_ROOT = PROJECT_ROOT / ".colab_final_cache"

EXPECTED = {"train": 697, "val": 134, "test": 185}
KEEP_COLS = ["stem", "subject", "ita_group_index_5", "skin_tone_category", "ITA"]


def load_splits() -> dict[str, pd.DataFrame]:
    """Assemble the three source manifests, with skin-tone labels on every split.

    train/val already carry ITA columns (00_build_split merged them); test is joined
    against wl_test_per_image_ita.csv on `stem`. The join is asserted total -- a
    partial join would silently drop images out of the fairness analysis.
    """
    tv = pd.read_csv(TRAIN_VAL_SPLIT)
    if "split" not in tv.columns:
        raise RuntimeError(f"{TRAIN_VAL_SPLIT} has no 'split' column -- run 00_build_split.py.")

    test = pd.read_csv(TEST_MANIFEST)
    test["mask_path"] = test["majority_mask_path"] if "majority_mask_path" in test.columns else test["mask_path"]
    ita = pd.read_csv(TEST_ITA)[["stem", "ITA", "skin_tone_category", "ita_group_index_5"]]
    before = len(test)
    test = test.merge(ita, on="stem", how="left", validate="one_to_one")
    if len(test) != before or int(test["ita_group_index_5"].isna().sum()):
        raise RuntimeError("Test/ITA join is not total -- fairness would drop images.")

    return {
        "train": tv[tv["split"] == "train"].reset_index(drop=True),
        "val": tv[tv["split"] == "val"].reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def binarise_mask_to_cache(src: Path, dst: Path) -> None:
    """Read a native mask, binarise to 0/255, write PNG at NATIVE resolution.

    IMREAD_GRAYSCALE returns (H,W,1) once ultralytics is imported anywhere in the
    process (its cv2 monkey-patch); squeezing here is a no-op otherwise and keeps the
    written PNG 2-D. No resize -- see module docstring.
    """
    if dst.exists():
        return
    m = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {src}")
    if m.ndim == 3:
        m = m[..., 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.png")
    if not cv2.imwrite(str(tmp), ((m > 0).astype(np.uint8) * 255)):
        raise RuntimeError(f"Cannot write mask: {dst}")
    tmp.replace(dst)


def build_manifest(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[tuple[Path, str]]]:
    """Return (manifest with archive-relative paths, list of (src_file, arcname) to zip).

    Images are shipped as-is (JPEG passthrough, lossless copy into the zip); masks are
    binarised into the cache first, then shipped from there.
    """
    rows, files = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"stage {split}"):
        stem = str(r["stem"])
        img_src = Path(r["image_path"])
        img_arc = f"images/{split}/{img_src.name}"
        mask_arc = f"masks/{split}/{stem}.png"
        mask_cache = CACHE_ROOT / mask_arc

        binarise_mask_to_cache(Path(r["mask_path"]), mask_cache)
        files.append((img_src, img_arc))
        files.append((mask_cache, mask_arc))

        row = {c: r[c] for c in KEEP_COLS if c in df.columns}
        row.update({"split": split, "image_path": img_arc, "mask_path": mask_arc})
        rows.append(row)
    return pd.DataFrame(rows), files


def assert_no_leakage(manifests: dict[str, pd.DataFrame]) -> None:
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        if set(manifests[a]["subject"]) & set(manifests[b]["subject"]):
            raise RuntimeError(f"Subject overlap between {a} and {b} -- split is supposed to be subject-level.")
        if set(manifests[a]["stem"]) & set(manifests[b]["stem"]):
            raise RuntimeError(f"Image overlap between {a} and {b}.")
    for split, n in EXPECTED.items():
        if len(manifests[split]) != n:
            raise RuntimeError(f"{split}: {len(manifests[split])} rows, expected {n}.")
    logger.info("PASS -- no subject/image overlap; counts %s", EXPECTED)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "bruise_colab_final.zip"))
    args = ap.parse_args()
    out_path = Path(args.out)

    sources = load_splits()
    manifests, all_files = {}, []
    for split, df in sources.items():
        man, files = build_manifest(df, split)
        manifests[split] = man
        all_files += files
    assert_no_leakage(manifests)

    logger.info("Writing %s (native-resolution) ...", out_path)
    # ZIP_STORED: JPEG + PNG payloads are already compressed; re-deflating wastes minutes.
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for src, arc in tqdm(all_files, desc="zip images/masks"):
            zf.write(src, arc)
        for split, man in manifests.items():
            buf = io.StringIO(); man.to_csv(buf, index=False)
            zf.writestr(f"manifests/{split}.csv", buf.getvalue())
        for p in sorted(PRETRAINED_DIR.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zf.write(p, f"pretrained_weights/{p.relative_to(PRETRAINED_DIR).as_posix()}")
        zf.write(INTERLABELER, "interlabeler_agreement_640.csv")

    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())
        for split, man in manifests.items():
            missing = [p for p in list(man["image_path"]) + list(man["mask_path"]) if p not in names]
            if missing:
                raise RuntimeError(f"{len(missing)} {split} path(s) missing from zip, e.g. {missing[0]}")
        for req in ["manifests/train.csv", "manifests/val.csv", "manifests/test.csv",
                    "interlabeler_agreement_640.csv", "pretrained_weights/yolo26n-sem.pt"]:
            if req not in names:
                raise RuntimeError(f"Missing {req} in built package")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("PASS -- wrote %s (%.0f MB, native resolution)", out_path, size_mb)
    logger.info("Upload to: Drive -> MyDrive/bruise_segmentation_gpu/%s", out_path.name)


if __name__ == "__main__":
    main()
