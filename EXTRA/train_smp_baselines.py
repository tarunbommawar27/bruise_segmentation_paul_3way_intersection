#!/usr/bin/env python3
"""
train_smp_baselines.py
======================
U-Net and DeepLabv3+ baselines for WL bruise segmentation, trained with the SAME
custom loop / recipe as the SegFormer models in `bruise_colab_final.ipynb`.

WHY THESE TWO GO THROUGH THE CUSTOM LOOP (and nnU-Net does not)
--------------------------------------------------------------
U-Net and DeepLabv3+ (segmentation_models_pytorch) are plain nn.Modules: a
pretrained ImageNet encoder + a randomly-initialised decoder/head -- structurally
identical to SegFormer (pretrained backbone + random 1-class head). So they take
the reference recipe verbatim: Dice+BCE per-image loss, AdamW with an
encoder/decoder LR split, no-decay on norms & biases, linear-warmup -> poly decay
stepped per optimizer step, AMP + grad-clip, epoch selection on threshold-free
val AP, then threshold swept on VAL and scored ONCE on TEST. Holding the recipe
fixed across architectures is the whole point of an apples-to-apples baseline.
nnU-Net is a self-configuring FRAMEWORK, so it is trained natively in a separate
script (train_nnunet_baseline.py), the same way YOLO was.

This run: single direct train (folds pooled), subject-level val split, NO distillation.

Usage
-----
    pip install "segmentation-models-pytorch>=0.3" "albumentations>=2.0,<3" \
                "torch" "opencv-python-headless" "pandas" "tqdm"

    python train_smp_baselines.py \
        --train-manifest /home/tbommawa/labelbox_paul_orc_dataset_full_consensus/train_paul_wl_minus_test_subjects/subject_5fold_split.csv \
        --test-manifest  /home/tbommawa/labelbox_paul_orc_dataset_full_consensus/fixed_consensus_test/manifest.csv \
        --out-dir /home/tbommawa/bruise_repro_v3_runs/smp_baselines \
        --models unet deeplabv3plus --encoder resnet50 --seeds 42

Manifest columns are auto-detected (image / mask / subject / fold), so this works
whether the CSV uses image_path/mask_path/subject/fold or common variants.
Paths may be absolute or relative to --data-root.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# albumentations & smp are imported lazily inside the functions that need them so
# `--help` and manifest inspection work without a full ML stack installed.


# ======================================================================================
# §0 · Config defaults (from protocol_v3_distillation_suite; overridable via CLI)
# ======================================================================================
DEFAULTS = dict(
    img_size        = 640,
    epochs          = 100,
    patience        = 15,
    workers         = 8,
    amp             = True,
    backbone_lr     = 6e-5,      # encoder LR (pretrained -> conservative)
    head_lr         = 6e-4,      # decoder + seg head LR (random -> 10x)
    betas           = (0.9, 0.999),
    weight_decay    = 0.01,
    warmup_fraction = 0.01,
    poly_power      = 1.0,
    gradient_clip   = 1.0,
    aux_weight      = 0.0,       # smp U-Net/DeepLab have no aux head
    # batch: probe VRAM by default (like the notebook). Override with --batch for
    # the v3 fixed batch (batch_size_b2/b0 = 8, accumulation_steps = 1).
    batch_mode      = "per_model",
    effective_batch = 8,
    max_probe_batch = 64,
    vram_target     = 0.75,
    # val split (v3: split.val_fraction / split.seed)
    val_fraction    = 0.18,
    split_seed      = 42,
    # threshold sweep (SegFormer logit-cut sweep from the notebook)
    cut_min         = -6.0,
    cut_max         = 6.0,
    cut_steps       = 481,
    drive_sync_every = 2,
)


# ======================================================================================
# §1 · Manifest loading  (robust column detection + subject-level val split)
# ======================================================================================
_IMAGE_KEYS   = ["image_path", "img_path", "image", "wl_image_path", "image_file", "filepath", "path", "rgb_path"]
_MASK_KEYS    = ["mask_path", "mask", "label_path", "gt_path", "annotation_path", "mask_file", "seg_path"]
_SUBJECT_KEYS = ["subject", "subject_id", "patient", "patient_id", "case_id", "case", "person_id"]
_STEM_KEYS    = ["stem", "id", "name", "image_id", "sample_id"]
_FOLD_KEYS    = ["fold", "fold_id", "fold_index", "split_fold", "cv_fold"]


def _pick(df: pd.DataFrame, keys: list[str], required: bool, what: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in lower:
            return lower[k]
    if required:
        raise KeyError(f"Could not find a {what} column. Looked for {keys}; "
                       f"manifest has {list(df.columns)}.")
    return None


def load_manifest(csv_path: str, data_root: str) -> pd.DataFrame:
    """Return a normalised manifest: columns stem, subject, image_path, mask_path (absolute)."""
    df = pd.read_csv(csv_path)
    img_c  = _pick(df, _IMAGE_KEYS, True,  "image path")
    mask_c = _pick(df, _MASK_KEYS,  True,  "mask path")
    subj_c = _pick(df, _SUBJECT_KEYS, False, "subject")
    stem_c = _pick(df, _STEM_KEYS,  False, "stem")
    fold_c = _pick(df, _FOLD_KEYS,  False, "fold")

    root = Path(data_root) if data_root else None

    def _abs(p):
        p = Path(str(p))
        if p.is_absolute() or root is None:
            return str(p)
        return str(root / p)

    out = pd.DataFrame()
    out["image_path"] = df[img_c].apply(_abs)
    out["mask_path"]  = df[mask_c].apply(_abs)
    out["stem"]       = df[stem_c].astype(str) if stem_c else out["image_path"].apply(lambda p: Path(p).stem)
    out["subject"]    = df[subj_c].astype(str) if subj_c else out["stem"]
    if fold_c:
        out["fold"] = df[fold_c]
    # de-dup identical rows (5-fold CSVs sometimes repeat)
    out = out.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return out


def subject_val_split(train_df: pd.DataFrame, val_fraction: float, seed: int):
    """Carve a VALIDATION set by SUBJECT (never split a subject across train/val)."""
    subjects = sorted(train_df["subject"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])
    val = train_df[train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    trn = train_df[~train_df["subject"].isin(val_subjects)].reset_index(drop=True)
    assert not (set(trn["subject"]) & set(val["subject"])), "subject leak train/val"
    return trn, val


# ======================================================================================
# §2 · Data  (verbatim recipe from the notebook: one read, one resize, one aug)
# ======================================================================================
def build_augmentation(training: bool, img_size: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    # A.Normalize(mean=0,std=1,max_pixel_value=255) == x/255. The MODEL applies
    # ImageNet norm (see SmpNet), so the loader emits raw [0,1] just like the notebook.
    to_unit = A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
    resize = [A.Resize(height=img_size, width=img_size)]  # bilinear img / nearest mask
    if not training:
        return A.Compose(resize + [to_unit, ToTensorV2()])
    return A.Compose(resize + [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        to_unit, ToTensorV2(),
    ])


class BruiseDataset(torch.utils.data.Dataset):
    """Reads native-res image+mask, resizes to img_size. Returns (x[3,H,W] in [0,1], y[1,H,W]{0,1}, stem)."""

    def __init__(self, df: pd.DataFrame, img_size: int, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tfm = build_augmentation(training, img_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = cv2.imread(str(r.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {r.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {r.mask_path}")
        if mask.ndim == 3:               # ultralytics monkey-patches cv2.imread -> (H,W,1)
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        assert y.shape == (1, self.img_size, self.img_size), f"bad mask shape {y.shape} for {r.stem}"
        return x, y, str(r.stem)


def make_loader(df, img_size, batch_size, training, workers, seed=0):
    ds = BruiseDataset(df, img_size, training=training)
    gen = torch.Generator(); gen.manual_seed(seed)

    def _init_worker(worker_id):
        np.random.seed(seed * 1000 + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_init_worker, generator=gen,
    )


# ======================================================================================
# §3 · Models  (U-Net / DeepLabv3+ behind the notebook's forward_train interface)
# ======================================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class SmpNet(nn.Module):
    """segmentation_models_pytorch model with a 1-class head. Input scale: ImageNet.

    forward_train(x) -> (logits[B,1,H,W], None)   -- x is RAW [0,1]; norm applied here.
    forward(x)       -> logits[B,1,H,W]
    .backbone        -> the pretrained encoder (for the LR split)
    """

    def __init__(self, arch: str, encoder: str, encoder_weights: str | None = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        builders = {
            "unet":          smp.Unet,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "deeplabv3":     smp.DeepLabV3,
            "unetplusplus":  smp.UnetPlusPlus,
            "fpn":           smp.FPN,
            "manet":         smp.MAnet,
        }
        if arch not in builders:
            raise ValueError(f"unknown smp arch: {arch}. choices: {list(builders)}")
        self.net = builders[arch](
            encoder_name=encoder, encoder_weights=encoder_weights,
            in_channels=3, classes=1, activation=None,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.net.encoder     # decoder + segmentation_head fall into the "head" group

    def forward_train(self, x):
        x = (x - self.mean) / self.std          # [0,1] -> ImageNet
        logits = self.net(x)                    # smp upsamples to input resolution -> [B,1,H,W]
        return logits, None

    def forward(self, x):
        return self.forward_train(x)[0]


def count_params(model): return sum(p.numel() for p in model.parameters())


def build_param_groups(model, backbone_lr, head_lr, weight_decay):
    """Encoder/decoder LR split + no weight decay on norms and biases (by id(), not name)."""
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = {
        "backbone_decay":    {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay":        {"params": [], "lr": head_lr,     "weight_decay": weight_decay},
        "head_no_decay":     {"params": [], "lr": head_lr,     "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        where = "backbone" if id(p) in backbone_ids else "head"
        decay = "_no_decay" if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower()) else "_decay"
        groups[where + decay]["params"].append(p)
    out = [g for g in groups.values() if g["params"]]
    n_grouped = sum(len(g["params"]) for g in out)
    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    assert n_grouped == n_total, f"param grouping lost {n_total - n_grouped} tensors"
    return out


# ======================================================================================
# §4 · Losses & metrics  (verbatim from the notebook)
# ======================================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__(); self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1.0 - dice.mean())


class SupervisedLoss(nn.Module):
    def __init__(self, aux_weight: float = 0.0):
        super().__init__(); self.main = DiceBCELoss(); self.aux_weight = aux_weight

    def forward(self, logits, aux_logits, target):
        loss = self.main(logits, target)
        if aux_logits is not None and self.aux_weight > 0:
            loss = loss + self.aux_weight * F.binary_cross_entropy_with_logits(aux_logits, target)
        return loss


def dice_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def iou_np(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def compute_image_row(pred, gt, stem):
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    return {"stem": stem, "dice": dice_np(pred, gt), "iou": iou_np(pred, gt),
            "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
            "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
            "pred_positive_pixels": int(pred_b.sum()), "gt_positive_pixels": int(gt_b.sum())}


def summarize(rows):
    df = pd.DataFrame(rows)
    miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
    return {"n_images": int(len(df)),
            "mean_dice": float(df["dice"].mean()), "median_dice": float(df["dice"].median()),
            "mean_iou": float(df["iou"].mean()), "median_iou": float(df["iou"].median()),
            "mean_precision": float(df["precision"].mean()), "mean_recall": float(df["recall"].mean()),
            "complete_miss_count": int(miss.sum()), "complete_miss_rate": float(miss.mean())}


class BinnedAP:
    """Threshold-free AP over pixels via probability histograms (model-selection metric)."""
    def __init__(self, bins=4096, device="cuda"):
        self.bins = bins
        self.pos = torch.zeros(bins, dtype=torch.float64, device=device)
        self.neg = torch.zeros(bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prob, gt):
        p = prob.reshape(-1).float().clamp(0, 1)
        g = gt.reshape(-1) > 0.5
        idx = (p * (self.bins - 1)).round().long()
        self.pos += torch.bincount(idx[g], minlength=self.bins).double()
        self.neg += torch.bincount(idx[~g], minlength=self.bins).double()

    def compute(self):
        total_pos = self.pos.sum()
        if total_pos == 0:
            return float("nan")
        tp = torch.cumsum(self.pos.flip(0), dim=0)
        fp = torch.cumsum(self.neg.flip(0), dim=0)
        precision = tp / (tp + fp).clamp_min(1e-12)
        recall = tp / total_pos
        d_recall = torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype, device=recall.device))
        return float((d_recall * precision).sum())


# ======================================================================================
# §5 · Engine  (seed, schedule, VRAM probe, AP eval, resumable train loop)
# ======================================================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_multiplier(step, total_steps, warmup_steps, power=1.0):
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return (1.0 - progress) ** power


def resolve_micro_batch(model, cfg, device):
    """Largest power-of-2 micro-batch that fits under vram_target (probe on a deepcopy)."""
    if cfg.get("fixed_batch", 0) and cfg["fixed_batch"] > 0:
        return int(cfg["fixed_batch"]), int(cfg.get("accum_steps", 1))
    if not torch.cuda.is_available():
        return 1, 1
    probe_model = deepcopy(model).to(device)
    probe_opt = torch.optim.SGD(probe_model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda") if cfg["amp"] else None
    total_vram = torch.cuda.get_device_properties(device).total_memory
    chosen, batch = 1, 1
    while batch <= cfg["max_probe_batch"]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand(batch, 3, cfg["img_size"], cfg["img_size"], device=device)
            y = torch.randint(0, 2, (batch, 1, cfg["img_size"], cfg["img_size"]), device=device).float()
            probe_model.train(); probe_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                logits, aux = probe_model.forward_train(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if aux is not None:
                    loss = loss + F.binary_cross_entropy_with_logits(aux, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(probe_opt); scaler.update()
            else:
                loss.backward(); probe_opt.step()
            frac = torch.cuda.max_memory_reserved(device) / total_vram
            del x, y, logits, loss
            if frac > cfg.get("vram_target", 0.75):
                break
            chosen = batch; batch *= 2
        except torch.cuda.OutOfMemoryError:
            break
    del probe_model, probe_opt; torch.cuda.empty_cache()
    return max(1, chosen), 1


@torch.no_grad()
def eval_ap(model, loader, device, amp):
    model.eval()
    ap = BinnedAP(device=str(device))
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        ap.update(torch.sigmoid(logits.float()), y)
    return ap.compute()


def _atomic_save(obj, dest: Path):
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, dest)


def train_run(run_id, arch, encoder, seed, cfg, manifests, runs_dir, device):
    """Train one (model, seed). Idempotent & resumable. No distillation."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k): return x

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "DONE.json"
    if done_file.exists():
        return {"run_id": run_id, "status": "skipped", **json.loads(done_file.read_text())}

    seed_everything(seed)
    amp = cfg["amp"]
    model = SmpNet(arch, encoder, encoder_weights=cfg.get("encoder_weights", "imagenet")).to(device)

    micro, accum = resolve_micro_batch(model, cfg, device)
    train_loader = make_loader(manifests["train"], cfg["img_size"], micro, True,  cfg["workers"], seed)
    val_loader   = make_loader(manifests["val"],   cfg["img_size"], micro, False, cfg["workers"], seed)

    param_groups = build_param_groups(model, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg["betas"]))
    peak_lrs = [g["lr"] for g in param_groups]
    scaler = torch.amp.GradScaler("cuda") if amp else None

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_fraction"]))
    criterion = SupervisedLoss(cfg["aux_weight"])

    start_epoch, best_ap, patience, global_step, history = 1, float("-inf"), 0, 0, []
    resume_path = run_dir / "resume.pt"
    if resume_path.exists():
        st = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best_ap = st["epoch"] + 1, st["best_ap"]
        patience, global_step, history = st["patience"], st["global_step"], st["history"]
        print(f"  [resume] {run_id} from epoch {start_epoch} (best_ap={best_ap:.4f})"); del st

    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "arch": arch, "encoder": encoder, "seed": seed,
        "micro_batch": micro, "accum_steps": accum, "effective_batch": micro * accum,
        "total_steps": total_steps, "warmup_steps": warmup_steps, "params": count_params(model),
        "n_train": int(len(manifests["train"])), "n_val": int(len(manifests["val"])),
    }, indent=2))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()
        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{run_id} e{epoch}", leave=False)):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits, aux = model.forward_train(x)
                loss = criterion(logits, aux, y) / accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                mult = lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
                for g, peak in zip(optimizer.param_groups, peak_lrs):
                    g["lr"] = peak * mult
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_ap = eval_ap(model, val_loader, device, amp)
        train_loss = running / max(1, len(train_loader))
        cur_lr = peak_lrs[0] * lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_ap": val_ap,
                        "backbone_lr": cur_lr, "sec": round(time.time() - t0, 1)})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if val_ap > best_ap:
            best_ap, patience = val_ap, 0
            _atomic_save(model.state_dict(), run_dir / "best.pt"); flag = " *"
        else:
            patience += 1; flag = ""
        print(f"  {run_id} e{epoch:3d} loss={train_loss:.4f} val_ap={val_ap:.4f} lr={cur_lr:.2e} "
              f"{time.time()-t0:.0f}s{flag}")

        last = (epoch == cfg["epochs"]) or (patience >= cfg["patience"])
        if epoch % cfg["drive_sync_every"] == 0 or last:
            _atomic_save({"epoch": epoch, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict() if scaler else None,
                          "best_ap": best_ap, "patience": patience,
                          "global_step": global_step, "history": history}, resume_path)
        if patience >= cfg["patience"]:
            print(f"  early stop at epoch {epoch} (patience={cfg['patience']})"); break

    summary = {"run_id": run_id, "arch": arch, "encoder": encoder, "seed": seed,
               "best_val_ap": best_ap, "epochs_trained": len(history), "params": count_params(model),
               "micro_batch": micro, "accum_steps": accum}
    done_file.write_text(json.dumps(summary, indent=2))
    if resume_path.exists():
        resume_path.unlink()
    del model; torch.cuda.empty_cache()
    return {"status": "trained", **summary}


# ======================================================================================
# §6 · Sweep threshold on VAL  +  score once on TEST
# ======================================================================================
@torch.no_grad()
def cache_logits(model, loader, device, amp):
    model.eval()
    logits, gts, stems = [], [], []
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        logits.append(z.float().half()[:, 0])
        gts.append((y[:, 0] > 0.5).to(device))
        stems.extend(s)
    return torch.cat(logits), torch.cat(gts), stems


def sweep_cuts(logits, gts, cuts):
    rows = []
    gts = gts.bool()
    gt_sum = gts.sum(dim=(1, 2)); gt_has = gt_sum > 0; n = len(gt_sum)
    for c in cuts:
        pred = logits >= c
        inter = (pred & gts).sum(dim=(1, 2))
        pred_sum = pred.sum(dim=(1, 2))
        denom = pred_sum + gt_sum
        dice = torch.where(denom > 0, 2.0 * inter.double() / denom.double().clamp_min(1.0),
                           torch.ones_like(denom, dtype=torch.float64))
        miss = ((pred_sum == 0) & gt_has).double()
        rows.append({"cut": float(c), "threshold": float(torch.sigmoid(torch.tensor(c))),
                     "mean_dice": float(dice.mean()),
                     "se_dice": float(dice.std(unbiased=True) / np.sqrt(n)),
                     "complete_miss_rate": float(miss.mean())})
    return pd.DataFrame(rows)


def select_cut(df):
    """Tie-band's lowest-miss cut (Dice-tied within 1 SE of the peak; break ties on miss then Dice)."""
    peak = df.loc[df["mean_dice"].idxmax()]
    band = df[df["mean_dice"] >= peak["mean_dice"] - peak["se_dice"]]
    best_miss = band["complete_miss_rate"].min()
    tied = band[band["complete_miss_rate"] <= best_miss + 1e-12]
    top_dice = tied["mean_dice"].max()
    finalists = tied[tied["mean_dice"] >= top_dice - 1e-12]
    chosen = finalists.iloc[len(finalists) // 2]
    return {"cut": float(chosen["cut"]), "threshold": float(chosen["threshold"]),
            "val_dice_at_cut": float(chosen["mean_dice"]),
            "val_miss_at_cut": float(chosen["complete_miss_rate"]),
            "val_peak_dice": float(peak["mean_dice"]), "peak_cut": float(peak["cut"]),
            "band_lo_threshold": float(band["threshold"].min()),
            "band_hi_threshold": float(band["threshold"].max()),
            "band_width_cuts": int(len(band)), "n_cuts": int(len(df))}


@torch.no_grad()
def evaluate_at_cut(model, loader, device, cut, amp):
    model.eval(); rows = []
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        pred = (z.float()[:, 0] >= cut).cpu().numpy().astype(np.uint8)
        gt = (y[:, 0] > 0.5).numpy().astype(np.uint8)
        for i, stem in enumerate(stems):
            rows.append(compute_image_row(pred[i], gt[i], stem))
    return pd.DataFrame(rows), summarize(rows)


# ======================================================================================
# §7 · Main
# ======================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net / DeepLabv3+ WL bruise baselines (custom loop).")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--test-manifest",  required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--data-root", default="", help="Prefix for relative paths in manifests (else use as-is).")
    p.add_argument("--models", nargs="+", default=["unet", "deeplabv3plus"],
                   help="Any of: unet deeplabv3plus deeplabv3 unetplusplus fpn manet")
    p.add_argument("--encoder", default="resnet50", help="smp encoder (resnet34/resnet50/efficientnet-b3/...).")
    p.add_argument("--encoder-weights", default="imagenet")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--img-size", type=int, default=DEFAULTS["img_size"])
    p.add_argument("--workers", type=int, default=DEFAULTS["workers"])
    p.add_argument("--batch", type=int, default=0, help="Fixed micro-batch (0 = probe VRAM like the notebook).")
    p.add_argument("--accum", type=int, default=1, help="Grad-accum steps when --batch is fixed.")
    p.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    p.add_argument("--split-seed", type=int, default=DEFAULTS["split_seed"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! WARNING: no CUDA -> running on CPU. This is only sane for a smoke test.")

    cfg = dict(DEFAULTS)
    cfg.update(img_size=a.img_size, epochs=a.epochs, patience=a.patience, workers=a.workers,
               amp=not a.no_amp, val_fraction=a.val_fraction, split_seed=a.split_seed,
               fixed_batch=a.batch, accum_steps=a.accum, encoder_weights=a.encoder_weights)

    out_dir = Path(a.out_dir); runs_dir = out_dir / "runs"; res_dir = out_dir / "results"
    runs_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)

    # --- manifests ---------------------------------------------------------------------
    full_train = load_manifest(a.train_manifest, a.data_root)
    test_df     = load_manifest(a.test_manifest,  a.data_root)
    train_df, val_df = subject_val_split(full_train, cfg["val_fraction"], cfg["split_seed"])

    # leakage guards vs the consensus test set
    for col in ("subject", "stem"):
        leak = set(train_df[col]) & set(test_df[col]) | set(val_df[col]) & set(test_df[col])
        if leak:
            print(f"!! WARNING: {col} overlap between train/val and TEST: {sorted(leak)[:10]}...")
    manifests = {"train": train_df, "val": val_df, "test": test_df}
    print(f"train {len(train_df)} imgs / {train_df.subject.nunique()} subj | "
          f"val {len(val_df)} imgs / {val_df.subject.nunique()} subj | test {len(test_df)} imgs")
    for s, d in manifests.items():
        d.to_csv(res_dir / f"manifest_{s}.csv", index=False)

    CUTS = np.linspace(cfg["cut_min"], cfg["cut_max"], cfg["cut_steps"])
    all_rows = []

    for arch in a.models:
        for seed in a.seeds:
            run_id = f"{arch}_{a.encoder}__seed{seed}"
            print(f"\n{'='*72}\n{run_id}\n{'='*72}")
            t0 = time.time()
            res = train_run(run_id, arch, a.encoder, seed, cfg, manifests, runs_dir, device)
            print(f"-> {res['status']} best_val_ap={res.get('best_val_ap', float('nan')):.4f} "
                  f"({(time.time()-t0)/60:.1f} min)")

            rd = runs_dir / run_id
            if not (rd / "best.pt").exists():
                print(f"  no best.pt for {run_id}; skipping eval"); continue
            model = SmpNet(arch, a.encoder, encoder_weights=None).to(device)
            model.load_state_dict(torch.load(str(rd / "best.pt"), map_location=device, weights_only=True))

            logits, gts, _ = cache_logits(
                model, make_loader(val_df, cfg["img_size"], 8, False, cfg["workers"], seed), device, cfg["amp"])
            grid = sweep_cuts(logits, gts, CUTS); sel = select_cut(grid)
            grid.to_csv(rd / "val_sweep.csv", index=False)
            (rd / "operating_point.json").write_text(json.dumps(sel, indent=2))

            pi, summ = evaluate_at_cut(
                model, make_loader(test_df, cfg["img_size"], 8, False, cfg["workers"], seed),
                device, sel["cut"], cfg["amp"])
            pi.to_csv(rd / "test_per_image.csv", index=False)
            row = {"model": arch, "encoder": a.encoder, "seed": seed,
                   "cut": sel["cut"], "threshold": sel["threshold"], **summ}
            all_rows.append(row)
            print(f"  TEST dice={summ['mean_dice']:.4f} median={summ['median_dice']:.4f} "
                  f"miss={summ['complete_miss_rate']*100:.2f}%")
            del model, logits, gts; torch.cuda.empty_cache()

    if all_rows:
        per_seed = pd.DataFrame(all_rows)
        per_seed.to_csv(res_dir / "baselines_test_per_seed.csv", index=False)
        agg = (per_seed.groupby(["model", "encoder"])
               .agg(mean_dice=("mean_dice", "mean"), std_dice=("mean_dice", "std"),
                    median_dice=("median_dice", "mean"),
                    miss=("complete_miss_rate", "mean"), miss_std=("complete_miss_rate", "std"))
               .reset_index())
        agg.to_csv(res_dir / "baselines_FINAL.csv", index=False)
        print("\n" + "=" * 72 + f"\nFINAL (mean over {len(a.seeds)} seed[s])\n" + "=" * 72)
        with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
            print(agg.to_string(index=False))
        print("\noutputs ->", res_dir)


if __name__ == "__main__":
    main()
