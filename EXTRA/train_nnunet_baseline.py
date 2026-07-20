#!/usr/bin/env python3
"""
train_nnunet_baseline.py
========================
nnU-Net v2 baseline for WL bruise segmentation, trained NATIVELY (its own
self-configuring framework), then scored with the SAME metrics as the other
baselines so the number sits in one comparable table.

WHY NATIVE (and not the custom loop)
------------------------------------
nnU-Net is not a model you drop into a training loop; it is a framework that
fingerprints the dataset and configures its own preprocessing, architecture,
patch size, batch size, LR schedule (poly, 1000-epoch default) and inference.
Forcing it onto the SegFormer/U-Net recipe throws away the exact thing that makes
it a strong baseline -- the same reason YOLO is trained natively in the reference
notebook. So this script drives nnU-Net's own CLI end to end.

WHAT "SINGLE DIRECT TRAIN, NO 5-FOLD" MEANS HERE
------------------------------------------------
nnU-Net's default is 5-fold CV. For one direct model we train a SINGLE fold (fold 0)
whose train/val membership is a custom, SUBJECT-GROUPED split we write into
splits_final.json (val_fraction / seed from the v3 config). That reproduces the
same no-subject-leakage discipline the other baselines use, in one model.

PIPELINE
--------
  1. convert  : WL manifests -> nnU-Net raw (RGB split to 3 channel files; masks -> 0/1 labels)
  2. plan     : nnUNetv2_plan_and_preprocess -d ID --verify_dataset_integrity
  3. split    : write subject-grouped splits_final.json (fold 0)
  4. train    : nnUNetv2_train ID 2d 0
  5. predict  : nnUNetv2_predict on the consensus test images
  6. score    : Dice / IoU / complete-miss at matched 640 geometry -> CSVs

Prereqs
-------
    pip install "nnunetv2>=2.5" "opencv-python-headless" "pandas" "numpy"
    export nnUNet_raw=/home/tbommawa/bruise_repro_v3_runs/nnunet/raw
    export nnUNet_preprocessed=/home/tbommawa/bruise_repro_v3_runs/nnunet/preprocessed
    export nnUNet_results=/home/tbommawa/bruise_repro_v3_runs/nnunet/results

Example
-------
    python train_nnunet_baseline.py \
        --train-manifest /home/tbommawa/labelbox_paul_orc_dataset_full_consensus/train_paul_wl_minus_test_subjects/subject_5fold_split.csv \
        --test-manifest  /home/tbommawa/labelbox_paul_orc_dataset_full_consensus/fixed_consensus_test/manifest.csv \
        --dataset-id 501 --dataset-name WLBruise --all           # do everything

    # or run stages individually: --convert  --plan  --train  --predict  --score
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---- reuse the exact manifest loader + metrics from the smp script if present -----------
try:
    from train_smp_baselines import (load_manifest, subject_val_split,
                                      dice_np, iou_np, compute_image_row, summarize)
except Exception:
    # self-contained fallbacks (kept byte-identical in spirit to the smp script) ----------
    _IMAGE_KEYS   = ["image_path", "img_path", "image", "wl_image_path", "image_file", "filepath", "path", "rgb_path"]
    _MASK_KEYS    = ["mask_path", "mask", "label_path", "gt_path", "annotation_path", "mask_file", "seg_path"]
    _SUBJECT_KEYS = ["subject", "subject_id", "patient", "patient_id", "case_id", "case", "person_id"]
    _STEM_KEYS    = ["stem", "id", "name", "image_id", "sample_id"]
    _FOLD_KEYS    = ["fold", "fold_id", "fold_index", "split_fold", "cv_fold"]

    def _pick(df, keys, required, what):
        lower = {c.lower(): c for c in df.columns}
        for k in keys:
            if k in lower:
                return lower[k]
        if required:
            raise KeyError(f"no {what} column; looked for {keys}, have {list(df.columns)}")
        return None

    def load_manifest(csv_path, data_root):
        df = pd.read_csv(csv_path)
        img_c = _pick(df, _IMAGE_KEYS, True, "image path")
        mask_c = _pick(df, _MASK_KEYS, True, "mask path")
        subj_c = _pick(df, _SUBJECT_KEYS, False, "subject")
        stem_c = _pick(df, _STEM_KEYS, False, "stem")
        root = Path(data_root) if data_root else None

        def _abs(p):
            p = Path(str(p))
            return str(p) if (p.is_absolute() or root is None) else str(root / p)

        out = pd.DataFrame()
        out["image_path"] = df[img_c].apply(_abs)
        out["mask_path"] = df[mask_c].apply(_abs)
        out["stem"] = df[stem_c].astype(str) if stem_c else out["image_path"].apply(lambda p: Path(p).stem)
        out["subject"] = df[subj_c].astype(str) if subj_c else out["stem"]
        return out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)

    def subject_val_split(train_df, val_fraction, seed):
        subjects = sorted(train_df["subject"].unique())
        rng = np.random.default_rng(seed); rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_fraction)))
        val_subjects = set(subjects[:n_val])
        val = train_df[train_df["subject"].isin(val_subjects)].reset_index(drop=True)
        trn = train_df[~train_df["subject"].isin(val_subjects)].reset_index(drop=True)
        return trn, val

    def dice_np(pred, gt):
        pred, gt = pred.astype(bool), gt.astype(bool)
        d = pred.sum() + gt.sum()
        return 1.0 if d == 0 else float(2 * np.logical_and(pred, gt).sum() / d)

    def iou_np(pred, gt):
        pred, gt = pred.astype(bool), gt.astype(bool)
        u = np.logical_or(pred, gt).sum()
        return 1.0 if u == 0 else float(np.logical_and(pred, gt).sum() / u)

    def compute_image_row(pred, gt, stem):
        pb, gb = pred.astype(bool), gt.astype(bool)
        tp = int(np.logical_and(pb, gb).sum()); fp = int(np.logical_and(pb, ~gb).sum())
        fn = int(np.logical_and(~pb, gb).sum())
        return {"stem": stem, "dice": dice_np(pred, gt), "iou": iou_np(pred, gt),
                "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
                "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
                "pred_positive_pixels": int(pb.sum()), "gt_positive_pixels": int(gb.sum())}

    def summarize(rows):
        df = pd.DataFrame(rows)
        miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
        return {"n_images": int(len(df)), "mean_dice": float(df["dice"].mean()),
                "median_dice": float(df["dice"].median()), "mean_iou": float(df["iou"].mean()),
                "median_iou": float(df["iou"].median()), "mean_recall": float(df["recall"].mean()),
                "complete_miss_count": int(miss.sum()), "complete_miss_rate": float(miss.mean())}


VAL_FRACTION = 0.18   # v3 split.val_fraction
SPLIT_SEED   = 42     # v3 split.seed


# ======================================================================================
# helpers
# ======================================================================================
def _dataset_dirname(dataset_id: int, name: str) -> str:
    return f"Dataset{int(dataset_id):03d}_{name}"


def _require_env():
    missing = [v for v in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Set these env vars before running nnU-Net stages: {missing}\n"
                 f"  export nnUNet_raw=...  nnUNet_preprocessed=...  nnUNet_results=...")


def _run(cmd: list[str]):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _write_case(image_path: str, out_img_dir: Path, case: str):
    """RGB image -> three single-channel PNGs case_0000/_0001/_0002 (canonical nnU-Net RGB layout)."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for ch in range(3):
        cv2.imwrite(str(out_img_dir / f"{case}_{ch:04d}.png"), rgb[:, :, ch])


def _write_label(mask_path: str, out_lbl_dir: Path, case: str):
    """Mask -> single-channel PNG with integer class ids {0,1} (NOT 0/255)."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read mask {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    cv2.imwrite(str(out_lbl_dir / f"{case}.png"), (m > 0).astype(np.uint8))


def _case_id(stem: str, idx: int) -> str:
    # nnU-Net is happy with arbitrary case ids; keep them unique & filesystem-safe.
    safe = "".join(c if c.isalnum() else "_" for c in str(stem))
    return f"c{idx:04d}_{safe}"[:64]


# ======================================================================================
# stage 1 — convert
# ======================================================================================
def stage_convert(a, ds_dir: Path):
    _require_env()
    raw = Path(os.environ["nnUNet_raw"]) / ds_dir.name
    imagesTr, labelsTr = raw / "imagesTr", raw / "labelsTr"
    imagesTs, labelsTs = raw / "imagesTs", raw / "labelsTs"
    for d in (imagesTr, labelsTr, imagesTs, labelsTs):
        d.mkdir(parents=True, exist_ok=True)

    full_train = load_manifest(a.train_manifest, a.data_root)
    test_df     = load_manifest(a.test_manifest,  a.data_root)
    train_df, val_df = subject_val_split(full_train, VAL_FRACTION, SPLIT_SEED)

    # training cases: all of full_train (train+val); the fold-0 split decides val internally.
    train_cases, mapping = [], []
    for i, r in full_train.reset_index(drop=True).iterrows():
        case = _case_id(r.stem, i)
        _write_case(r.image_path, imagesTr, case)
        _write_label(r.mask_path, labelsTr, case)
        train_cases.append(case)
        split = "val" if r.subject in set(val_df.subject) else "train"
        mapping.append({"case": case, "stem": r.stem, "subject": r.subject, "split": split})

    # test cases (images + GT kept separately for scoring)
    test_map = []
    for i, r in test_df.reset_index(drop=True).iterrows():
        case = _case_id(r.stem, 100000 + i)
        _write_case(r.image_path, imagesTs, case)
        _write_label(r.mask_path, labelsTs, case)   # GT for our own scoring, not used by nnU-Net
        test_map.append({"case": case, "stem": r.stem, "subject": r.subject,
                         "gt_mask": r.mask_path})

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "bruise": 1},
        "numTraining": len(train_cases),
        "file_ending": ".png",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    pd.DataFrame(mapping).to_csv(raw / "train_case_mapping.csv", index=False)
    pd.DataFrame(test_map).to_csv(raw / "test_case_mapping.csv", index=False)
    print(f"  raw dataset -> {raw}  ({len(train_cases)} train / {len(test_map)} test cases)")
    print(f"  fold-0 split: {sum(m['split']=='train' for m in mapping)} train / "
          f"{sum(m['split']=='val' for m in mapping)} val (subject-grouped)")


# ======================================================================================
# stage 2 — plan & preprocess
# ======================================================================================
def stage_plan(a, ds_dir: Path):
    _require_env()
    _run(["nnUNetv2_plan_and_preprocess", "-d", str(a.dataset_id), "--verify_dataset_integrity"])


# ======================================================================================
# stage 3 — write subject-grouped splits_final.json (fold 0)
# ======================================================================================
def stage_split(a, ds_dir: Path):
    _require_env()
    raw = Path(os.environ["nnUNet_raw"]) / ds_dir.name
    pre = Path(os.environ["nnUNet_preprocessed"]) / ds_dir.name
    if not pre.exists():
        sys.exit(f"preprocessed dir not found ({pre}); run --plan first.")
    m = pd.read_csv(raw / "train_case_mapping.csv")
    fold0 = {"train": m[m.split == "train"].case.tolist(),
             "val":   m[m.split == "val"].case.tolist()}
    # nnU-Net expects a list of folds; we only train fold 0, but provide a couple so the
    # file is well-formed if someone later trains more folds.
    splits = [fold0, fold0, fold0, fold0, fold0]
    (pre / "splits_final.json").write_text(json.dumps(splits, indent=2))
    print(f"  wrote {pre/'splits_final.json'}  (fold 0: {len(fold0['train'])} train / {len(fold0['val'])} val)")


# ======================================================================================
# stage 4 — train (single fold)
# ======================================================================================
def stage_train(a, ds_dir: Path):
    _require_env()
    cmd = ["nnUNetv2_train", str(a.dataset_id), a.configuration, str(a.fold)]
    if a.trainer:
        cmd += ["-tr", a.trainer]
    if a.epochs:
        # nnUNetv2 >= 2.5 supports -num_epochs; harmless override for a faster baseline.
        cmd += ["-num_epochs", str(a.epochs)]
    if a.continue_train:
        cmd += ["--c"]
    _run(cmd)


# ======================================================================================
# stage 5 — predict on the consensus test set
# ======================================================================================
def stage_predict(a, ds_dir: Path):
    _require_env()
    raw = Path(os.environ["nnUNet_raw"]) / ds_dir.name
    out = Path(a.out_dir) / "nnunet_test_pred"
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["nnUNetv2_predict", "-i", str(raw / "imagesTs"), "-o", str(out),
           "-d", str(a.dataset_id), "-c", a.configuration, "-f", str(a.fold)]
    if a.trainer:
        cmd += ["-tr", a.trainer]
    _run(cmd)
    print(f"  predictions -> {out}")


# ======================================================================================
# stage 6 — score with matched metrics (640 geometry, to match the smp baselines)
# ======================================================================================
def _load_bin(path: str, size: int, interp) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read {path}")
    if m.ndim == 3:
        m = m[..., 0]
    b = (m > 0).astype(np.uint8)
    return cv2.resize(b, (size, size), interpolation=interp)


def stage_score(a, ds_dir: Path):
    _require_env()
    raw = Path(os.environ["nnUNet_raw"]) / ds_dir.name
    pred_dir = Path(a.out_dir) / "nnunet_test_pred"
    tmap = pd.read_csv(raw / "test_case_mapping.csv")
    size = a.img_size

    rows = []
    for _, r in tmap.iterrows():
        pred_path = pred_dir / f"{r.case}.png"
        if not pred_path.exists():
            print(f"  !! missing prediction {pred_path}; skipping {r.stem}"); continue
        # pred & GT resized together (nearest) to matched geometry -- same as the smp path.
        pred = _load_bin(pred_path, size, cv2.INTER_NEAREST)
        gt   = _load_bin(r.gt_mask, size, cv2.INTER_NEAREST)
        rows.append(compute_image_row(pred, gt, str(r.stem)))

    if not rows:
        sys.exit("no scored images; did --predict run?")
    res_dir = Path(a.out_dir) / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res_dir / "nnunet_test_per_image.csv", index=False)
    summ = summarize(rows)
    summ.update({"model": "nnunet", "configuration": a.configuration, "fold": a.fold})
    (res_dir / "nnunet_FINAL.json").write_text(json.dumps(summ, indent=2))
    print("\n" + "=" * 60 + "\nnnU-Net TEST (matched 640 geometry)\n" + "=" * 60)
    print(f"  mean_dice   {summ['mean_dice']:.4f}")
    print(f"  median_dice {summ['median_dice']:.4f}")
    print(f"  mean_iou    {summ['mean_iou']:.4f}")
    print(f"  miss_rate   {summ['complete_miss_rate']*100:.2f}%  "
          f"({summ['complete_miss_count']}/{summ['n_images']})")
    print("  outputs ->", res_dir)


# ======================================================================================
# main
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Native nnU-Net v2 WL bruise baseline.")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest",  required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--dataset-id", type=int, default=501)
    p.add_argument("--dataset-name", default="WLBruise")
    p.add_argument("--configuration", default="2d", help="nnU-Net config (2d recommended for these PNGs).")
    p.add_argument("--fold", default=0)
    p.add_argument("--trainer", default="", help="e.g. nnUNetTrainer_250epochs for a faster baseline.")
    p.add_argument("--epochs", type=int, default=0, help="Override num_epochs (0 = nnU-Net default 1000).")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--continue-train", action="store_true")
    # stage flags
    p.add_argument("--all", action="store_true", help="convert+plan+split+train+predict+score")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--split", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--score", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    ds_dir = Path(_dataset_dirname(a.dataset_id, a.dataset_name))
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    stages = []
    if a.all:
        stages = ["convert", "plan", "split", "train", "predict", "score"]
    else:
        for s in ("convert", "plan", "split", "train", "predict", "score"):
            if getattr(a, s):
                stages.append(s)
    if not stages:
        sys.exit("Nothing to do. Pass --all or one/more of "
                 "--convert --plan --split --train --predict --score")

    fns = {"convert": stage_convert, "plan": stage_plan, "split": stage_split,
           "train": stage_train, "predict": stage_predict, "score": stage_score}
    for s in stages:
        print(f"\n########## nnU-Net stage: {s} ##########")
        fns[s](a, ds_dir)


if __name__ == "__main__":
    main()
