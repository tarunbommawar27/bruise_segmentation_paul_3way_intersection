#!/usr/bin/env python3
"""
scripts/45_generate_baselines_notebook.py  ->  bruise_colab_baselines.ipynb

The BASELINES notebook: direct (no distillation) segmentation baselines trained
each **to its own nativity**, then scored against the exact same 2-of-3 majority
target at 640 as the 5 core models — so the numbers drop straight into the paper's
comparison.

WHICH MODELS, AND WHY EACH IS TRAINED THE WAY IT IS
---------------------------------------------------
  U-Net (ResNet50)         -- plain nn.Module (pretrained encoder + random decoder),
  DeepLabV3+ (ResNet50)       structurally identical to SegFormer. So they take the
                              SegFormer custom-loop recipe VERBATIM (same Dice+BCE,
                              same encoder/head LR split, same poly schedule, same
                              threshold-free AP model-selection, same val-swept
                              threshold applied once to test). Holding the recipe
                              fixed across plain-nn.Module architectures is the whole
                              point of a fair baseline. 3 seeds each.
  nnU-Net v2 (2d)          -- a self-configuring FRAMEWORK, not a drop-in module. It
                              fingerprints the dataset and picks its own
                              preprocessing / architecture / patch & batch size / LR
                              schedule. Forcing it onto the shared recipe throws away
                              exactly what makes it strong — the same reason YOLO is
                              trained natively in bruise_colab_final.ipynb. So it runs
                              its OWN CLI end to end (convert -> plan -> train -> predict
                              -> score). One native fold-0 run (subject-grouped), the
                              "each-at-its-strongest" datapoint — not the 3-seed shared
                              recipe (that would defeat the purpose).

HOW IT REUSES THE TESTED PIPELINE
---------------------------------
`bruisekit` is reused **verbatim** from `bruise_colab_final.ipynb`. U-Net/DeepLab go
through the SAME `engine.train_run` — the loop is fully architecture-blind, so the
only thing added is one module (`smp_models.py`) that teaches `build_model` to build
SMP architectures and a second (`nnunet_native.py`) that drives nnU-Net's CLI. Data
comes from the SAME `bruise_colab_final.zip` already on Drive — no new upload.

Consistency fixes from docs/baselines_paper_todo.md are all applied: the identical
697/134 split (the package manifests), seeds 0/1/2, scoring vs the 2-of-3 majority at
640, and complete-miss rate reported alongside Dice. EDIT THIS GENERATOR, not the
.ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NB = PROJECT_ROOT / "bruise_colab_final.ipynb"      # tested modules live here
OUT = PROJECT_ROOT / "bruise_colab_baselines.ipynb"

REUSE_MODULES = ["__init__", "data", "models", "losses", "metrics",
                 "engine", "sweep", "evaluate", "postopt", "yolo_native"]

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": dedent(text).strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": dedent(text).strip("\n").splitlines(keepends=True)})


def raw_code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


def writefile(path: str, body: str) -> None:
    code(f"%%writefile {path}\n" + dedent(body).strip("\n"))


def reused_module_cells() -> dict[str, str]:
    if not SOURCE_NB.exists():
        raise SystemExit(f"{SOURCE_NB} not found -- run scripts/43_generate_final_notebook.py first.")
    nb = json.loads(SOURCE_NB.read_text(encoding="utf-8"))
    found = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        first = src.split("\n", 1)[0]
        if first.startswith("%%writefile bruisekit/"):
            name = first.split("bruisekit/")[1].removesuffix(".py").strip()
            found[name] = src
    missing = [m for m in REUSE_MODULES if m not in found]
    if missing:
        raise SystemExit(f"Reused modules missing from {SOURCE_NB.name}: {missing}")
    return found


# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
# Bruise segmentation — direct baselines: U-Net, DeepLabV3+, nnU-Net

External segmentation baselines the reviewers expect, each trained **to its own
nativity**, scored on the **same** 185-image 2-of-3 majority target at 640 as the 5
core models.

| model | trained by | seeds | why |
|---|---|---|---|
| U-Net (ResNet50) | SegFormer custom loop (shared recipe) | 0,1,2 | plain nn.Module → same recipe is fair |
| DeepLabV3+ (ResNet50) | SegFormer custom loop (shared recipe) | 0,1,2 | plain nn.Module → same recipe is fair |
| nnU-Net v2 (2d) | **native nnU-Net CLI** (self-configuring) | 1 native fold-0 | a framework, not a module — run it its way |

**All direct (no distillation).** Everything reuses the tested `bruisekit` from
`bruise_colab_final.ipynb` and the **same** `bruise_colab_final.zip` on Drive (no new
upload). U-Net/DeepLab flow through the identical, architecture-blind `train_run`;
nnU-Net drives its own CLI.

**Read alongside `docs/baselines_paper_todo.md`.** The point is not "SegFormer beats
U-Net" — at n=28 a ~0.04 mean-Dice gap may not survive the cluster bootstrap. The
point is that **5+ architectures all cluster at 0.74–0.79 → the task is
label-limited, not capacity-limited**, and the miss-rate story extends (SegFormer /
DeepLab safe, YOLO not).
""")

# ══════════════════════════════════════════════════════════════════════════════
# §1 config
# ══════════════════════════════════════════════════════════════════════════════
md("## §1 · Configuration")

code(r'''
CFG = dict(
    img_size        = 640,
    zip_name        = "bruise_colab_final.zip",
    drive_dir       = "/content/drive/MyDrive/bruise_segmentation_gpu",
    work_dir        = "/content/bruise_final",

    # ── shared custom-loop recipe (identical to SegFormer in the final notebook) ──
    epochs          = 100,
    patience        = 15,
    batch_mode      = "per_model",
    effective_batch = 8,
    max_probe_batch = 64,
    vram_target     = 0.75,
    backbone_lr     = 6e-5,
    head_lr         = 6e-4,
    betas           = (0.9, 0.999),
    weight_decay    = 0.01,
    warmup_fraction = 0.01,
    poly_power      = 1.0,
    gradient_clip   = 1.0,
    amp             = True,
    workers         = 4,
    aux_weight      = 0.0,          # smp U-Net/DeepLab have no aux head
    alpha           = 0.5,          # unused (all baselines are direct); kept for train_run's cfg key

    smp_encoder     = "resnet50",   # ImageNet-pretrained encoder for U-Net / DeepLabV3+
    smp_micro_batch = 16,           # FIXED batch for SMP models (see smp_models.py): DeepLabV3+'s
                                    # ASPP image-pool BN cannot take a batch-1 train-mode tensor, and
                                    # engine's VRAM probe starts at batch 1 -> we skip the probe for
                                    # SMP and use this fixed, batch-safe size (same for all SMP seeds,
                                    # which is cleaner for the baseline comparison anyway).

    # ── nnU-Net (native) ─────────────────────────────────────────────────────────
    nnunet_dataset_id  = 501,
    nnunet_name        = "WLBruise",
    nnunet_config      = "2d",
    nnunet_epochs      = 250,       # nnU-Net default is 1000 -> far too long on Colab; 250 is a fair baseline
    nnunet_fold        = 0,

    # ── shared ───────────────────────────────────────────────────────────────────
    seeds           = (0, 1, 2),
    cut_min = -6.0, cut_max = 6.0, cut_steps = 481,
    drive_sync_every = 2,
    run_nnunet = True,              # set False to skip the (heavier) nnU-Net run
)

# arch names must match the SMP_ARCHS set in bruisekit/smp_models.py below.
BASELINE_MODELS = {
    "unet_r50":          dict(arch="unet",          size=None, distill=False),
    "deeplabv3plus_r50": dict(arch="deeplabv3plus", size=None, distill=False),
}
print(f"{len(BASELINE_MODELS)} shared-recipe baselines x {len(CFG['seeds'])} seeds"
      f"{' + native nnU-Net' if CFG['run_nnunet'] else ''}")
''')

# ══════════════════════════════════════════════════════════════════════════════
# §2 env
# ══════════════════════════════════════════════════════════════════════════════
md("## §2 · Drive, GPU, dependencies")

code(r'''
import os, sys, time, json
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> GPU (A100). Refusing to run on CPU.")
print("GPU:", torch.cuda.get_device_name(0))
''')

code(r'''
%pip install -q "transformers>=4.40,<6" "albumentations>=2.0,<3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.7" "pyyaml" "segmentation-models-pytorch>=0.3" "opencv-python-headless"
if CFG["run_nnunet"]:
    !pip install -q "nnunetv2>=2.5"
import segmentation_models_pytorch as smp
print("segmentation-models-pytorch", smp.__version__)
''')

# ══════════════════════════════════════════════════════════════════════════════
# §3 unpack + 640 cache (train + val + test)  -- verbatim from the final notebook
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §3 · Unpack (native-res) and build the 640 cache once

Same package as the final notebook. U-Net/DeepLab train on the 640 cache (fast, no
per-epoch resize); nnU-Net reads the **native-res** images directly (it does its own
preprocessing). We build train + val + test caches here (training needs train).
""")

code(r'''
import zipfile, cv2, numpy as np, pandas as pd

ZIP_SRC = Path(CFG["drive_dir"]) / CFG["zip_name"]
WORK    = Path(CFG["work_dir"])
if not ZIP_SRC.exists():
    raise FileNotFoundError(f"{ZIP_SRC} not found. Build with scripts/42 and upload.")
if ZIP_SRC.stat().st_size < 2e9:
    raise RuntimeError(f"{ZIP_SRC} is only {ZIP_SRC.stat().st_size/1e9:.2f} GB; expected ~2.7 GB native-res package.")

if not (WORK / "manifests" / "train.csv").exists():
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP_SRC) as zf:
        zf.extractall(WORK)
    print(f"unzipped in {time.time()-t0:.0f}s")

MAN = {s: pd.read_csv(WORK / "manifests" / f"{s}.csv") for s in ("train", "val", "test")}
for s, df in MAN.items():
    print(f"{s:>5}: {len(df):>3} images, {df['subject'].nunique():>3} subjects")
for a, b in [("train","val"),("train","test"),("val","test")]:
    assert not (set(MAN[a]["subject"]) & set(MAN[b]["subject"])), f"subject leak {a}/{b}"
assert (len(MAN["train"]),len(MAN["val"]),len(MAN["test"]))==(697,134,185)
print("PASS -- 697/134/185, no leakage.")

RUNS_DIR = Path(CFG["drive_dir"]) / "runs_baselines"
OUT_DIR  = Path(CFG["drive_dir"]) / "results_baselines"
RUNS_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", RUNS_DIR)
''')

code(r'''
CACHE640 = WORK / "cache640"
def build_cache(df, split):
    idir = CACHE640 / split / "images"; mdir = CACHE640 / split / "masks"
    idir.mkdir(parents=True, exist_ok=True); mdir.mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        ip = idir / f"{r.stem}.png"; mp = mdir / f"{r.stem}.png"
        if not ip.exists():
            im = cv2.imread(str(WORK / r.image_path), cv2.IMREAD_COLOR)
            cv2.imwrite(str(ip), cv2.resize(im, (640,640), interpolation=cv2.INTER_LINEAR))
        if not mp.exists():
            m = cv2.imread(str(WORK / r.mask_path), cv2.IMREAD_GRAYSCALE)
            if m.ndim == 3: m = m[..., 0]
            b = (m > 0).astype(np.uint8)
            cv2.imwrite(str(mp), cv2.resize(b, (640,640), interpolation=cv2.INTER_NEAREST) * 255)

if not (CACHE640 / "test" / "images").exists():
    t0 = time.time()
    for s in ("train","val","test"): build_cache(MAN[s], s)
    print(f"640 cache built in {time.time()-t0:.0f}s")
else:
    print("640 cache present")

MAN640 = {}
for s, df in MAN.items():
    d = df.copy()
    d["image_path"] = d["stem"].apply(lambda x: f"{s}/images/{x}.png")
    d["mask_path"]  = d["stem"].apply(lambda x: f"{s}/masks/{x}.png")
    MAN640[s] = d
''')

# ══════════════════════════════════════════════════════════════════════════════
# §4 library
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §4 · The library

`bruisekit` reused **verbatim** from `bruise_colab_final.ipynb`, plus two new modules:
`smp_models.py` (U-Net / DeepLabV3+ behind the same `forward_train` interface, and a
`build_model` shim so the tested loop can build them) and `nnunet_native.py` (drives
nnU-Net's own CLI). mkdir first so `%%writefile` has somewhere to land.
""")

code(r'''
%cd /content
from pathlib import Path
Path("/content/bruisekit").mkdir(parents=True, exist_ok=True)
print("package dir ready:", Path("/content/bruisekit").is_dir())
''')

modules = reused_module_cells()
for name in REUSE_MODULES:
    raw_code(modules[name])

# ── new module: smp_models ───────────────────────────────────────────────────
writefile("bruisekit/smp_models.py", r'''
"""U-Net / DeepLabV3+ (segmentation_models_pytorch) behind bruisekit's interface.

WHY THESE GO THROUGH THE SHARED LOOP
------------------------------------
An SMP model is a pretrained ImageNet encoder + a randomly-initialised decoder/head
-- structurally identical to SegFormer (pretrained backbone + random 1-class head).
So they take the reference recipe VERBATIM: the loader emits raw [0,1] pixels and the
MODEL applies ImageNet normalisation (exactly like SegFormerNet), a 1-class head, the
encoder/head LR split, Dice+BCE, poly schedule, threshold-free AP selection, and the
val-swept threshold applied once to test. Holding the recipe fixed is the whole point
of a fair baseline.

THE build_model SHIM
--------------------
`engine.train_run` is architecture-blind: it calls `build_model(arch, size, paths)`
and never branches on a model's name. We only need to teach that one function to build
SMP architectures. Because `engine` did `from bruisekit.models import build_model` at
import time (binding the original by value), we reassign the name in BOTH namespaces
-- `bruisekit.models` (for eval cells that import it fresh) and `bruisekit.engine`
(the copy the training loop actually calls). SegFormer/YOLO still route to the
original builder untouched.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SMP_ARCHS = {"unet", "deeplabv3plus", "deeplabv3", "unetplusplus", "fpn", "manet"}


class SMPNet(nn.Module):
    """segmentation_models_pytorch model with a 1-class head. Input scale: ImageNet.

        forward_train(x) -> (logits[B,1,H,W], None)   -- x is RAW [0,1]; norm applied here
        forward(x)       -> logits[B,1,H,W]
        .backbone        -> the pretrained encoder (for the encoder/head LR split)
    """

    def __init__(self, arch: str, encoder: str = "resnet50", encoder_weights: str | None = "imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp
        builders = {
            "unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "deeplabv3": smp.DeepLabV3,
            "unetplusplus": smp.UnetPlusPlus, "fpn": smp.FPN, "manet": smp.MAnet,
        }
        if arch not in builders:
            raise ValueError(f"unknown smp arch: {arch}. choices: {list(builders)}")
        self.net = builders[arch](encoder_name=encoder, encoder_weights=encoder_weights,
                                  in_channels=3, classes=1, activation=None)
        # Buffers (not constants) so they move with .to(device) and save with the module.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.net.encoder      # decoder + segmentation_head fall into the "head" LR group

    @property
    def head(self):
        return self.net.decoder

    def forward_train(self, x):
        x = (x - self.mean) / self.std               # [0,1] -> ImageNet
        return self.net(x), None                     # smp upsamples to input res -> [B,1,H,W]

    def forward(self, x):
        return self.forward_train(x)[0]


def install_build_model_shim(smp_micro_batch: int = 16):
    """Route SMP archs to SMPNet, and make batch selection safe for DeepLabV3+.

    Two patches into bruisekit.engine's namespace (train_run looks both names up as
    module globals at call time, so reassigning the module attribute takes effect):

    1. build_model -> builds SMP architectures for the SMP arch names.
    2. resolve_micro_batch -> for SMPNet models, SKIP the VRAM probe and return a
       FIXED batch. The probe escalates from batch=1 in TRAIN mode; DeepLabV3+'s ASPP
       image-pooling branch produces a [B, C, 1, 1] tensor whose BatchNorm raises
       "Expected more than 1 value per channel" at B=1. A fixed batch also means every
       SMP baseline (and every seed) trains at the SAME batch, which is cleaner for the
       baseline comparison than per-model probed batches. Other archs are untouched.
    """
    import bruisekit.models as _bm
    import bruisekit.engine as _be
    original = _bm.build_model

    def build_model(arch, size, paths):
        if arch in SMP_ARCHS:
            return SMPNet(arch, encoder=paths.get("smp_encoder", "resnet50"))
        return original(arch, size, paths)

    _bm.build_model = build_model
    _be.build_model = build_model     # the name the training loop already bound

    original_probe = _be.resolve_micro_batch

    def resolve_micro_batch(model, cfg, device, teacher=None):
        if isinstance(model, SMPNet):
            b = max(2, int(cfg.get("smp_micro_batch", smp_micro_batch)))
            return b, 1               # (micro_batch, accum_steps); no probe, no batch-1 forward
        return original_probe(model, cfg, device, teacher)

    _be.resolve_micro_batch = resolve_micro_batch
    return build_model
''')

# ── new module: nnunet_native ────────────────────────────────────────────────
writefile("bruisekit/nnunet_native.py", r'''
"""Native nnU-Net v2 baseline: drive its own CLI end to end, then score with the
same metrics as everything else so the number sits in one comparable table.

Adapted from EXTRA/train_nnunet_baseline.py. The one change vs that script: the
fold-0 split is taken DIRECTLY from the package's 697/134 train/val manifests
(not re-derived from a val_fraction), so it is bit-identical to the split the
SegFormer/SMP models use. nnU-Net trains on train+val cases with fold 0 deciding
which are validation.

WHY NATIVE: nnU-Net fingerprints the dataset and self-configures preprocessing,
architecture, patch/batch size and LR schedule. Forcing it onto the shared recipe
would discard exactly what makes it a strong baseline -- so it runs its own way,
the same reason YOLO is trained natively in the final notebook.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bruisekit.metrics import compute_image_row, summarize


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def set_env(root: Path) -> dict:
    """nnU-Net reads three env vars; point them at local SSD (fast, wiped on disconnect)."""
    raw = root / "raw"; pre = root / "preprocessed"; res = root / "results"
    for d in (raw, pre, res):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(res)
    return {"raw": raw, "pre": pre, "res": res}


def dataset_dirname(dataset_id: int, name: str) -> str:
    return f"Dataset{int(dataset_id):03d}_{name}"


def _case_id(stem: str, idx: int) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(stem))
    return f"c{idx:04d}_{safe}"[:64]


def _write_case(work_root: Path, image_rel: str, out_img_dir: Path, case: str) -> None:
    """RGB image -> three single-channel PNGs case_0000/_0001/_0002 (nnU-Net RGB layout)."""
    bgr = cv2.imread(str(work_root / image_rel), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image {image_rel}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for ch in range(3):
        cv2.imwrite(str(out_img_dir / f"{case}_{ch:04d}.png"), rgb[:, :, ch])


def _write_label(work_root: Path, mask_rel: str, out_lbl_dir: Path, case: str) -> None:
    m = cv2.imread(str(work_root / mask_rel), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read mask {mask_rel}")
    if m.ndim == 3:
        m = m[..., 0]
    cv2.imwrite(str(out_lbl_dir / f"{case}.png"), (m > 0).astype(np.uint8))     # 0/1, not 0/255


def convert(env, ds_dir, work_root, train_df, val_df, test_df):
    """Stage 1: manifests -> nnU-Net raw. fold-0 split comes from the train/val manifests."""
    raw = env["raw"] / ds_dir
    imagesTr, labelsTr = raw / "imagesTr", raw / "labelsTr"
    imagesTs = raw / "imagesTs"
    for d in (imagesTr, labelsTr, imagesTs):
        d.mkdir(parents=True, exist_ok=True)

    val_subjects = set(val_df["subject"])
    combined = pd.concat([train_df, val_df], ignore_index=True)
    mapping = []
    for i, r in combined.iterrows():
        case = _case_id(r.stem, i)
        _write_case(work_root, r.image_path, imagesTr, case)
        _write_label(work_root, r.mask_path, labelsTr, case)
        mapping.append({"case": case, "stem": r.stem, "subject": r.subject,
                        "split": "val" if r.subject in val_subjects else "train"})

    test_map = []
    for i, r in test_df.iterrows():
        case = _case_id(r.stem, 100000 + i)
        _write_case(work_root, r.image_path, imagesTs, case)
        test_map.append({"case": case, "stem": r.stem, "gt_mask": str(work_root / r.mask_path)})

    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "bruise": 1},
        "numTraining": len(mapping), "file_ending": ".png",
    }, indent=2))
    pd.DataFrame(mapping).to_csv(raw / "train_case_mapping.csv", index=False)
    pd.DataFrame(test_map).to_csv(raw / "test_case_mapping.csv", index=False)
    n_tr = sum(m["split"] == "train" for m in mapping); n_va = len(mapping) - n_tr
    print(f"  raw -> {raw}  ({n_tr} train / {n_va} val / {len(test_map)} test cases)")


def plan(cfg):
    _run(["nnUNetv2_plan_and_preprocess", "-d", cfg["nnunet_dataset_id"], "--verify_dataset_integrity"])


def write_splits(env, ds_dir):
    """Stage 3: subject-grouped splits_final.json (fold 0) from our mapping."""
    raw = env["raw"] / ds_dir
    pre = env["pre"] / ds_dir
    if not pre.exists():
        raise FileNotFoundError(f"{pre} missing; run plan() first.")
    m = pd.read_csv(raw / "train_case_mapping.csv")
    fold0 = {"train": m[m.split == "train"].case.tolist(),
             "val":   m[m.split == "val"].case.tolist()}
    (pre / "splits_final.json").write_text(json.dumps([fold0] * 5, indent=2))
    print(f"  splits_final.json (fold 0: {len(fold0['train'])} train / {len(fold0['val'])} val)")


def train(cfg):
    cmd = ["nnUNetv2_train", cfg["nnunet_dataset_id"], cfg["nnunet_config"], cfg["nnunet_fold"]]
    if cfg.get("nnunet_epochs"):
        cmd += ["-num_epochs", cfg["nnunet_epochs"]]
    # --c continues from the latest checkpoint if a previous session was interrupted.
    cmd += ["--c"]
    try:
        _run(cmd)
    except subprocess.CalledProcessError:
        # first run has no checkpoint to continue from -> retry without --c
        _run([c for c in cmd if c != "--c"])


def predict(cfg, env, ds_dir, out_dir):
    raw = env["raw"] / ds_dir
    out = Path(out_dir) / "nnunet_test_pred"; out.mkdir(parents=True, exist_ok=True)
    _run(["nnUNetv2_predict", "-i", raw / "imagesTs", "-o", out,
          "-d", cfg["nnunet_dataset_id"], "-c", cfg["nnunet_config"], "-f", cfg["nnunet_fold"]])
    return out


def _load_bin(path, size, interp):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"cannot read {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return cv2.resize((m > 0).astype(np.uint8), (size, size), interpolation=interp)


def score(env, ds_dir, pred_dir, size=640):
    """Stage 6: pred & GT resized together (nearest) to 640 -- same geometry as everything else."""
    raw = env["raw"] / ds_dir
    tmap = pd.read_csv(raw / "test_case_mapping.csv")
    rows = []
    for _, r in tmap.iterrows():
        pp = Path(pred_dir) / f"{r.case}.png"
        if not pp.exists():
            print(f"  !! missing prediction {pp}; skipping {r.stem}"); continue
        pred = _load_bin(pp, size, cv2.INTER_NEAREST)
        gt = _load_bin(r.gt_mask, size, cv2.INTER_NEAREST)
        rows.append(compute_image_row(pred, gt, str(r.stem)))
    if not rows:
        raise RuntimeError("no scored images; did predict() run?")
    per_image = pd.DataFrame(rows)
    return per_image, summarize(rows)
''')

# ══════════════════════════════════════════════════════════════════════════════
# §5 imports + self-test
# ══════════════════════════════════════════════════════════════════════════════
md("## §5 · Import, install the build_model shim, and self-test")

code(r'''
sys.path.insert(0, "/content")
import importlib
import bruisekit.data, bruisekit.models, bruisekit.losses, bruisekit.metrics
import bruisekit.engine, bruisekit.sweep, bruisekit.evaluate, bruisekit.smp_models, bruisekit.nnunet_native
for m in ("data","models","losses","metrics","engine","sweep","evaluate","smp_models","nnunet_native"):
    importlib.reload(sys.modules[f"bruisekit.{m}"])

from bruisekit.data import make_loader
from bruisekit.engine import train_run
from bruisekit.evaluate import evaluate_at_cut, fairness_analysis
from bruisekit.models import count_params
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts
import bruisekit.smp_models as smpm
import bruisekit.nnunet_native as nn_native

# Route SMP archs through the tested train_run. build_model now dispatches unet/deeplab,
# and resolve_micro_batch uses a fixed batch for SMP (DeepLabV3+ ASPP BN can't take batch-1).
build_model = smpm.install_build_model_shim(CFG["smp_micro_batch"])

import numpy as np, pandas as pd, torch
PATHS = {"smp_encoder": CFG["smp_encoder"]}
DEVICE = torch.device("cuda:0")
''')

code(r'''
# SegFormer-shaped sanity for the SMP wrappers: emit [B,1,640,640] from raw [0,1].
_ld = make_loader(MAN640["val"].head(4), CACHE640, CFG["img_size"], 2, False, 0)
_x, _y, _s = next(iter(_ld))
assert _x.shape == (2,3,640,640) and _y.shape == (2,1,640,640)
assert 0.0 <= float(_x.min()) and float(_x.max()) <= 1.0
for _name, _spec in BASELINE_MODELS.items():
    _m = build_model(_spec["arch"], _spec["size"], PATHS).to(DEVICE).eval()
    with torch.no_grad():
        _z, _aux = _m.forward_train(_x.to(DEVICE))
    assert _z.shape == (2,1,640,640) and _aux is None, (_z.shape, _aux)
    print(f"{_name:<20} {_spec['arch']:<14} OK  {count_params(_m)/1e6:.2f}M")
    del _m
torch.cuda.empty_cache()
assert hasattr(nn_native, "convert") and hasattr(nn_native, "score")
print("smp_models + nnunet_native OK; all imports resolved.")
''')

# ══════════════════════════════════════════════════════════════════════════════
# §6 train U-Net + DeepLab (shared loop)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §6 · Train U-Net + DeepLabV3+ (shared custom loop, 3 seeds)

The exact same `train_run` the SegFormer models use — resumable (`resume.pt` every 2
epochs, `DONE.json` skips finished runs), threshold-free AP model-selection, per-model
best batch. Direct only (no teacher).
""")

code(r'''
BASE_QUEUE = []
for _seed in CFG["seeds"]:
    for _name, _spec in BASELINE_MODELS.items():
        BASE_QUEUE.append((f"{_name}__seed{_seed}", _name, _spec, _seed))

base_results = []
for run_id, name, spec, seed in BASE_QUEUE:
    print(f"\n{'='*70}\n{run_id}\n{'='*70}")
    t0 = time.time()
    res = train_run(run_id, spec, seed, CFG, PATHS, MAN640, CACHE640, RUNS_DIR, DEVICE)
    res["minutes"] = round((time.time()-t0)/60, 1)
    base_results.append(res)
    print(f"-> {res['status']} best_val_ap={res.get('best_val_ap', float('nan')):.4f}")
pd.DataFrame(base_results)
''')

# ══════════════════════════════════════════════════════════════════════════════
# §7 threshold sweep + test eval
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §7 · Fit threshold on val, score on test (185)

Identical to the SegFormer path: sweep the logit cut on val, fix it, evaluate once on
test. Per-image CSVs are saved for the analysis/bootstrap.
""")

code(r'''
CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
base_test_rows = []; base_per_image = {}
for run_id, name, spec, seed in BASE_QUEUE:
    rd = RUNS_DIR / run_id
    if not (rd / "best.pt").exists(): continue
    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(rd/"best.pt"), map_location=DEVICE, weights_only=True))
    logits, gts, _ = cache_logits(model, make_loader(MAN640["val"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed), DEVICE, CFG["amp"])
    grid = sweep_cuts(logits, gts, CUTS); sel = select_cut(grid)
    (rd/"operating_point.json").write_text(json.dumps(sel, indent=2))
    pi, summ = evaluate_at_cut(model, make_loader(MAN640["test"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed), DEVICE, sel["cut"], CFG["amp"])
    pi.to_csv(rd/"test_per_image.csv", index=False)
    base_per_image[run_id] = pi
    base_test_rows.append({"run_id": run_id, "model": name, "seed": seed, "cut": sel["cut"], **summ})
    print(f"  {run_id:<28} dice={summ['mean_dice']:.4f} miss={summ['complete_miss_rate']*100:.2f}%")
    del model, logits, gts; torch.cuda.empty_cache()
BASE_TEST = pd.DataFrame(base_test_rows)
BASE_TEST.to_csv(OUT_DIR/"smp_baselines_test_per_seed.csv", index=False)
BASE_TEST
''')

# ══════════════════════════════════════════════════════════════════════════════
# §8 nnU-Net native
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §8 · nnU-Net v2 — **native** self-configuring pipeline (one fold-0 run)

Drives nnU-Net's own CLI: convert → plan/preprocess → write the subject-grouped fold-0
split (from the 697/134 manifests) → train → predict on the 185 test images → score at
640. This is the heaviest cell; each stage is skip-guarded so a disconnect resumes.
Set `CFG["run_nnunet"] = False` in §1 to skip.
""")

code(r'''
if not CFG["run_nnunet"]:
    print("nnU-Net skipped (CFG['run_nnunet'] = False)")
    NNUNET = None
else:
    NN_ROOT = WORK / "nnunet"
    env = nn_native.set_env(NN_ROOT)
    ds_dir = nn_native.dataset_dirname(CFG["nnunet_dataset_id"], CFG["nnunet_name"])
    raw = env["raw"] / ds_dir

    # stage 1: convert (skip if already converted)
    if not (raw / "test_case_mapping.csv").exists():
        print("converting to nnU-Net raw...")
        nn_native.convert(env, ds_dir, WORK, MAN["train"], MAN["val"], MAN["test"])
    else:
        print("raw dataset present")

    # stage 2: plan/preprocess (skip if preprocessed present)
    if not (env["pre"] / ds_dir).exists():
        print("planning + preprocessing...")
        nn_native.plan(CFG)
    else:
        print("preprocessed present")

    # stage 3: write our fold-0 split (always -- cheap, and must exist before train)
    nn_native.write_splits(env, ds_dir)
    print("nnU-Net dataset ready.")
''')

code(r'''
if CFG["run_nnunet"]:
    # stage 4: train (resumes via --c). This is the long one -- ~nnunet_epochs epochs.
    t0 = time.time()
    nn_native.train(CFG)
    print(f"nnU-Net train done in {(time.time()-t0)/60:.1f} min")
''')

code(r'''
NNUNET = None
if CFG["run_nnunet"]:
    NN_ROOT = WORK / "nnunet"; env = nn_native.set_env(NN_ROOT)
    ds_dir = nn_native.dataset_dirname(CFG["nnunet_dataset_id"], CFG["nnunet_name"])
    pred_dir = nn_native.predict(CFG, env, ds_dir, OUT_DIR)     # stage 5
    nn_per_image, nn_summ = nn_native.score(env, ds_dir, pred_dir, size=CFG["img_size"])   # stage 6
    nn_per_image.to_csv(OUT_DIR/"nnunet_test_per_image.csv", index=False)
    (OUT_DIR/"nnunet_summary.json").write_text(json.dumps(nn_summ, indent=2))
    NNUNET = {"per_image": nn_per_image, "summary": nn_summ}
    print(f"nnU-Net TEST: mean_dice={nn_summ['mean_dice']:.4f} "
          f"median={nn_summ['median_dice']:.4f} miss={nn_summ['complete_miss_rate']*100:.2f}%")
''')

# ══════════════════════════════════════════════════════════════════════════════
# §9 combined table
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §9 · Combined baseline table

U-Net / DeepLabV3+ as mean±std over 3 seeds; nnU-Net as its single native run. These
drop into the paper's comparison against the 5 core models (`results_final/`).
""")

code(r'''
def agg(df):
    g = df.groupby("model").agg(mean_dice=("mean_dice","mean"), std_dice=("mean_dice","std"),
                                median_dice=("median_dice","mean"), miss=("complete_miss_rate","mean"),
                                miss_std=("complete_miss_rate","std")).reset_index()
    g["variant"] = g["model"]
    return g

rows = agg(BASE_TEST)[["variant","mean_dice","std_dice","median_dice","miss","miss_std"]]
if NNUNET is not None:
    s = NNUNET["summary"]
    rows = pd.concat([rows, pd.DataFrame([{"variant":"nnunet_2d","mean_dice":s["mean_dice"],"std_dice":np.nan,
        "median_dice":s["median_dice"],"miss":s["complete_miss_rate"],"miss_std":np.nan}])], ignore_index=True)
rows["dice"]   = rows.apply(lambda r: f"{r.mean_dice:.4f} ± {r.std_dice:.4f}" if pd.notna(r.std_dice) else f"{r.mean_dice:.4f} (1 run)", axis=1)
rows["miss_%"] = rows.apply(lambda r: f"{r.miss*100:.2f} ± {r.miss_std*100:.2f}" if pd.notna(r.miss_std) else f"{r.miss*100:.2f}", axis=1)
rows.to_csv(OUT_DIR/"BASELINES_RESULTS.csv", index=False)
print("185 test images\n" + "="*66)
print(rows[["variant","dice","median_dice","miss_%"]].to_string(index=False))
''')

# ══════════════════════════════════════════════════════════════════════════════
# §10 fairness
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §10 · Fairness across skin tone (ITA) — baselines

Per-image Dice averaged over seeds (single run for nnU-Net), stratified by ITA group,
via the same `fairness_analysis` used for the core models. **Exploratory at n=28** —
each ITA group has only ~9–17 subjects, so read direction, not significance.
""")

code(r'''
def per_image_over_seeds(run_ids, store):
    frames = [store[k] for k in run_ids if k in store]
    if not frames: return None
    stacked = pd.concat(frames)
    return (stacked.groupby("stem", as_index=False)
            .agg({"dice":"mean","recall":"mean","pred_positive_pixels":"mean","gt_positive_pixels":"first"}))

fair_group, fair_stats = [], []
for name in BASELINE_MODELS:
    mpi = per_image_over_seeds([f"{name}__seed{s}" for s in CFG["seeds"]], base_per_image)
    if mpi is None: continue
    out = fairness_analysis(mpi, MAN["test"], name)
    fair_group.append(out["per_group"]); fair_stats.append(out["stats"])
if NNUNET is not None:
    out = fairness_analysis(NNUNET["per_image"][["stem","dice","recall","pred_positive_pixels","gt_positive_pixels"]],
                            MAN["test"], "nnunet_2d")
    fair_group.append(out["per_group"]); fair_stats.append(out["stats"])

if fair_group:
    FAIR_GROUP = pd.concat(fair_group, ignore_index=True)
    FAIR_STATS = pd.DataFrame(fair_stats)
    FAIR_GROUP.to_csv(OUT_DIR/"baselines_fairness_per_group.csv", index=False)
    FAIR_STATS.to_csv(OUT_DIR/"baselines_fairness_stats.csv", index=False)
    print(FAIR_STATS[["model","kruskal_p","significant","fairness_gap","best_group","worst_group","max_miss_rate_gap"]].to_string(index=False))

    import matplotlib.pyplot as plt
    GROUP_ORDER = ["Light (II-III)","Intermediate (III-IV)","Tan (IV)","Brown (V)","Dark (VI)"]
    piv = FAIR_GROUP.pivot_table(index="skin_tone_category", columns="model", values="median_dice")
    piv = piv.reindex([g for g in GROUP_ORDER if g in piv.index])
    fig, ax = plt.subplots(figsize=(9,4.6))
    piv.plot.bar(ax=ax, rot=15, width=0.8)
    ax.set_ylabel("median Dice"); ax.set_xlabel(""); ax.set_ylim(0,1)
    ax.set_title("Baseline median Dice by ITA group (exploratory, n=28)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_DIR/"baselines_fairness_by_group.png", dpi=140, bbox_inches="tight"); plt.show()
''')

# ══════════════════════════════════════════════════════════════════════════════
# finish
# ══════════════════════════════════════════════════════════════════════════════
md("## §11 · Saved outputs")

code(r'''
print("All baseline outputs ->", OUT_DIR)
for f in sorted(OUT_DIR.glob("*")): print("   ", f.name)
print("\nPer-seed checkpoints + per-image CSVs ->", RUNS_DIR)
''')

md(r"""
### How to fold these into the paper

1. Drop U-Net + DeepLabV3+ (+ nnU-Net) into the **same paired subject-level cluster
   bootstrap** as the 5 core models (28 subjects, B=4000, paired) — per-image CSVs are
   in `runs_baselines/*/test_per_image.csv` and `results_baselines/nnunet_test_per_image.csv`.
2. Report **Δ mean/median Dice + 95% CI + P(Δ>0)** and the miss-rate gap for each
   baseline-vs-core comparison. **Do not claim "SegFormer beats U-Net" without the CI**
   — ~0.04 mean is near the minimum detectable effect at n=28.
3. The paper claim is the **annotation ceiling**: 5+ architectures all clustering at
   0.74–0.79 is evidence the task is *label-limited, not capacity-limited*.
""")

# ══════════════════════════════════════════════════════════════════════════════
# build
# ══════════════════════════════════════════════════════════════════════════════
notebook = {
    "cells": cells,
    "metadata": {"accelerator": "GPU",
                 "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 0,
}
OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}  ({len(cells)} cells: {n_code} code, {len(cells)-n_code} markdown)")
