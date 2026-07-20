#!/usr/bin/env python3
"""
scripts/41_generate_training_notebook.py

Generates bruise_colab_train_all.ipynb.

WHY A GENERATOR AND NOT A HAND-EDITED .ipynb
---------------------------------------------
The notebook embeds ~1000 lines of Python across seven modules. Maintaining
that as raw .ipynb JSON means escaped newlines, no syntax checking, and diffs
that are unreadable in review. Here the notebook's real source is ordinary
Python in a normal file, so it can be linted, diffed and reasoned about; the
.ipynb is a build artifact. Re-run this script after any change.

The generated notebook writes its modules to disk as real .py files at run
time (rather than defining everything in cells) so that: a failure has a real
traceback with a filename and line number instead of "<ipython-input-42>";
the modules are importable from a plain script if the work ever leaves Colab;
and the notebook stays readable as a sequence of decisions rather than a wall
of class definitions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Two notebooks come out of this one generator so they cannot drift apart -- they
# are byte-identical except the batch-size decision:
#   matched   (default) -> effective batch 8 for EVERY model. Recipe-matched, so
#                          B0-vs-B2 is a fair comparison. This is the paper notebook.
#   per_model (--per-model) -> each model uses the largest batch its own size allows
#                          (B2~8, B0/YOLO much higher). Faster, better GPU use, but
#                          NOT recipe-matched across model sizes -- see the warning
#                          the per-model notebook carries in §1 and §6.
MODE = "per_model" if "--per-model" in sys.argv else "matched"
OUT = PROJECT_ROOT / ("bruise_colab_train_per_model_batch.ipynb" if MODE == "per_model"
                      else "bruise_colab_train_all.ipynb")

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": dedent(text).strip("\n").splitlines(keepends=True)})


def writefile(path: str, body: str) -> None:
    """Emit a cell that writes one module to disk via %%writefile."""
    code(f"%%writefile {path}\n" + dedent(body).strip("\n"))


# ══════════════════════════════════════════════════════════════════════════════
_MODE_BANNER = {
    "matched": "",
    "per_model": (
        "\n> ⚠️ **This is the PER-MODEL-BATCH variant.** Each model trains at the "
        "largest batch its own size allows (B2 ≈ 8, B0/YOLO much higher) for faster "
        "training and fuller GPU use. Because the models no longer share an effective "
        "batch or a step count, **this notebook is NOT recipe-matched across model "
        "sizes — do not use it for the B0-vs-B2 comparison.** Use "
        "`bruise_colab_train_all.ipynb` (effective batch 8 for all) for that. "
        "Everything else is identical.\n"),
}[MODE]

_BULLET_1 = {
    "matched": (
        "1. **Every model now trains at the same effective batch size (8).** Previously a\n"
        "   VRAM probe silently gave B0 batch 32 / 2100 steps and B2 batch 8 / 8700 steps\n"
        "   at the same LR, so \"identical hyperparameters\" was not true."),
    "per_model": (
        "1. **Each model trains at its own best batch** (probed per model, `accum=1`). This\n"
        "   is the *faster* variant, not the *fair* one: the small models take fewer, larger\n"
        "   steps at the same LR, so B0-vs-B2 is no longer apples-to-apples. That is the one\n"
        "   deliberate difference from `bruise_colab_train_all.ipynb`."),
}[MODE]

md("""
# Bruise segmentation — train / validate / test / fairness, all five models

One notebook, one recipe, five models, three seeds each. Trains from scratch on
Colab and **survives session death**: every run checkpoints to Drive and Run All
picks up exactly where it stopped.
""" + _MODE_BANNER + """
| model | params | teacher | pixel scale |
|---|---|---|---|
| SegFormer-B2 teacher | 27.35 M | — | ImageNet |
| SegFormer-B0 direct | 3.71 M | — | ImageNet |
| SegFormer-B0 distilled | 3.71 M | B2 (same seed) | ImageNet |
| YOLO26n-sem direct | 1.63 M | — | `/255` |
| YOLO26n-sem distilled | 1.63 M | B2 (same seed) | `/255` |

**Read `docs/training_notebook_explained.md` before running this.** It says what
each design decision is, why it is that way, and which paper it comes from —
including the four correctness bugs in the previous training pipeline that this
notebook exists to fix.

The short version of what changed:

""" + _BULLET_1 + """
2. **YOLO trains in the same custom loop as SegFormer**, with an `nc=1` head that
   emits a single bruise logit. Previously YOLO's "distillation" was a hard
   pseudo-mask that is *algebraically identical to the ground truth* whenever
   α > 0.5 — six of fifteen Optuna trials returned bit-identical numbers.
3. **No Optuna. Fixed α = 0.5, three seeds per model.** The old search selected
   differences ~7× smaller than its own seed-to-seed noise.
4. **Model selection is on threshold-free val AP**, not Dice at a fixed 0.5.
5. **The threshold ties are broken on complete-miss rate**, not on the median cut.
""")

# ── §1 config ────────────────────────────────────────────────────────────────
md(r"""
## §1 · Configuration

The single place anything is decided. Every later cell reads from `CFG`; nothing
below hardcodes a hyperparameter.
""")

_BATCH_BLOCK = {
    "matched": '''    batch_mode      = "matched",   # same effective batch for EVERY model (fair B0-vs-B2)
    effective_batch = 8,      # ENFORCED for every model -- see §"why 8" in the .md
    max_probe_batch = 8,      # never probe above effective_batch: accumulation covers the rest''',
    "per_model": '''    batch_mode      = "per_model", # each model uses the largest batch its size allows
    effective_batch = 8,      # matched-mode fallback only; unused when batch_mode="per_model"
    max_probe_batch = 64,     # probe climbs to here; B2 stays ~8, B0/YOLO go much higher
    vram_target     = 0.75,   # stop the probe at 75% VRAM -- headroom for loss + checkpoint''',
}[MODE]

code(r'''
CFG = dict(
    # ── data ─────────────────────────────────────────────────────────────────
    img_size        = 640,
    zip_name        = "bruise_colab_train.zip",
    drive_dir       = "/content/drive/MyDrive/bruise_segmentation_gpu",
    work_dir        = "/content/bruise",          # local SSD: fast, wiped on disconnect

    # ── the shared recipe (SegFormer paper, Xie et al. 2021 §4.1) ────────────
    epochs          = 100,
    patience        = 15,
''' + _BATCH_BLOCK + '''
    backbone_lr     = 6e-5,
    head_lr         = 6e-4,   # 10x backbone: the head is randomly initialised
    betas           = (0.9, 0.999),
    weight_decay    = 0.01,
    warmup_fraction = 0.01,
    poly_power      = 1.0,
    gradient_clip   = 1.0,
    amp             = True,
    workers         = 4,      # Colab gives 2-8 vCPU; the 640 PNGs make this cheap

    # ── distillation ─────────────────────────────────────────────────────────
    alpha           = 0.5,    # fixed, NOT searched -- the old search selected noise
    aux_weight      = 0.4,    # Ultralytics' own semantic aux weight (YOLO only)

    # ── seeds ────────────────────────────────────────────────────────────────
    seeds           = (0, 1, 2),

    # ── threshold sweep (on val, never on test) ──────────────────────────────
    cut_min         = -6.0,   # sweeps the raw-logit cut c; threshold = sigmoid(c)
    cut_max         = 6.0,
    cut_steps       = 481,

    # ── benchmark ────────────────────────────────────────────────────────────
    bench_repeats   = 3,
    bench_warmup    = 10,

    # ── checkpointing ────────────────────────────────────────────────────────
    # Cost of durability: a B2 resume checkpoint is ~330 MB (weights + AdamW's two
    # moments), and Drive writes at ~20-40 MB/s, so syncing every epoch would add
    # ~10-15 s to a ~60 s epoch. At 2 you lose at most two epochs to a disconnect.
    # Set to 1 if your session is dying constantly; the correctness is identical.
    drive_sync_every = 2,
)

MODELS = {
    "segformer_b2_teacher":   dict(arch="segformer", size="b2", distill=False),
    "segformer_b0_direct":    dict(arch="segformer", size="b0", distill=False),
    "segformer_b0_distilled": dict(arch="segformer", size="b0", distill=True),
    "yolo_sem_direct":        dict(arch="yolo",      size="n",  distill=False),
    "yolo_sem_distilled":     dict(arch="yolo",      size="n",  distill=True),
}

# The distilled runs take the teacher trained with THEIR OWN seed, not a single
# fixed teacher. That way the reported spread over seeds includes the teacher's
# own variance, which is part of the pipeline being measured -- pinning one
# teacher would make distillation look more reproducible than it is.
TEACHER_OF = "segformer_b2_teacher"

print(f"{len(MODELS)} models x {len(CFG['seeds'])} seeds = {len(MODELS)*len(CFG['seeds'])} runs")
''')

# ── §2 environment ───────────────────────────────────────────────────────────
md(r"""
## §2 · Drive, GPU, dependencies

Fails fast rather than producing bad numbers: no GPU is a hard stop, because a
CPU run would silently take days and the benchmark section would be meaningless.
""")

code(r'''
import os, sys, subprocess, time
from pathlib import Path

from google.colab import drive
drive.mount("/content/drive")

import torch
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU. Runtime -> Change runtime type -> GPU (A100 recommended).\n"
        "Refusing to continue: a CPU run would take days and every timing in the "
        "benchmark section would be meaningless.")
print("GPU:", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print("torch:", torch.__version__)
''')

code(r'''
# Pinned so a Colab image refresh cannot silently change the numbers.
# transformers>=4.40 for the SegFormer parameter naming this notebook's
# checkpoints use; ultralytics>=8.4 for task="semantic" (SemanticSegmentationModel).
%pip install -q "transformers>=4.40,<6" "ultralytics>=8.4,<9" "albumentations>=2.0,<3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.7"

import transformers, ultralytics, albumentations
print("transformers:", transformers.__version__)
print("ultralytics :", ultralytics.__version__)
print("albumentations:", albumentations.__version__)
''')

# ── §3 unzip ─────────────────────────────────────────────────────────────────
md(r"""
## §3 · Unpack the data package

The zip holds 640×640 pre-resized PNGs. That resize is **bit-exact** to what the
old dataloader computed from the native 4022×6024 JPEGs (verified: max tensor
difference 0.0, zero mask pixels differing) — it just moves the work out of the
training loop, where it cost ~287 ms of CPU *per image per epoch*.

Unzipping to `/content` (local SSD), never to Drive: training reads every image
every epoch, and Drive's FUSE mount would make that the bottleneck all over again.
""")

code(r'''
import zipfile, shutil

ZIP_SRC = Path(CFG["drive_dir"]) / CFG["zip_name"]
WORK    = Path(CFG["work_dir"])

if not ZIP_SRC.exists():
    raise FileNotFoundError(
        f"{ZIP_SRC} not found.\nBuild it with scripts/40_build_colab_training_package.py "
        f"and upload it to {CFG['drive_dir']}/")

size_gb = ZIP_SRC.stat().st_size / 1e9
if size_gb < 0.8:
    raise RuntimeError(
        f"{ZIP_SRC} is only {size_gb:.2f} GB. The training package is ~1.0 GB; "
        "this looks like the INFERENCE package (bruise_colab_gpu_full.zip), which "
        "ships no train images.")

if not (WORK / "manifests" / "train.csv").exists():
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(ZIP_SRC) as zf:
        zf.extractall(WORK)
    print(f"unzipped {size_gb:.2f} GB in {time.time()-t0:.0f}s")
else:
    print("already unpacked")

# The batch policy gets its OWN Drive folders. The matched and per-model notebooks
# share run IDs (segformer_b2_teacher__seed0, ...), so writing to one folder would
# make the second notebook find the first's DONE.json and skip all training -- then
# silently evaluate the wrong checkpoints. The suffix keeps the two runs separate.
_tag = "" if CFG.get("batch_mode", "matched") == "matched" else "_per_model_batch"
RUNS_DIR = Path(CFG["drive_dir"]) / f"runs_v2{_tag}"      # checkpoints: Drive, survives death
OUT_DIR  = Path(CFG["drive_dir"]) / f"results_v2{_tag}"   # tables/figures: Drive
RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"batch_mode={CFG.get('batch_mode','matched')}  ->  checkpoints:", RUNS_DIR)
''')

code(r'''
import pandas as pd

# Guard the property the whole evaluation rests on: the threshold is FITTED on val
# and REPORTED on test, so any contact between them inflates the reported score
# silently instead of crashing. Re-asserted here (the build script also checks) so
# the notebook is honest even if someone hands you a zip built by something else.
MAN = {s: pd.read_csv(WORK / "manifests" / f"{s}.csv") for s in ("train", "val", "test")}
for s, df in MAN.items():
    print(f"{s:>5}: {len(df):>3} images, {df['subject'].nunique():>3} subjects")

for a, b in [("train","val"), ("train","test"), ("val","test")]:
    shared_subj  = set(MAN[a]["subject"]) & set(MAN[b]["subject"])
    shared_stems = set(MAN[a]["stem"])    & set(MAN[b]["stem"])
    assert not shared_subj,  f"LEAK: {len(shared_subj)} subject(s) in both {a} and {b}"
    assert not shared_stems, f"LEAK: {len(shared_stems)} image(s) in both {a} and {b}"
assert (len(MAN["train"]), len(MAN["val"]), len(MAN["test"])) == (697, 134, 185)
print("\nPASS -- no subject or image overlap between any two splits.")
''')

# ── §4 modules ───────────────────────────────────────────────────────────────
md(r"""
## §4 · The library

Written to disk as real modules, so tracebacks have filenames and the code can be
re-used outside Colab. Seven small files:

| module | responsibility |
|---|---|
| `data.py` | one dataloader, one resize, emits raw `[0,1]` pixels |
| `models.py` | the two architectures behind one interface; **pixel scale lives here** |
| `losses.py` | Dice+BCE, and the distillation fusion |
| `metrics.py` | Dice/IoU, and threshold-free AP for model selection |
| `engine.py` | the training loop: AMP, accumulation, poly LR, **resume** |
| `sweep.py` | threshold fitting on val |
| `evaluate.py` | test scoring, fairness, speed |
| `postopt.py` | reduce misses **without retraining** — ensemble / TTA / no-blank, fit on val |
""")

code(r'''
# %%writefile does NOT create parent directories -- it raises FileNotFoundError if
# the package dir does not already exist. Every cell below writes a RELATIVE path,
# so the cwd has to be pinned too: Colab starts at /content, but any earlier %cd
# (or a re-run from a different cell) would silently scatter the modules somewhere
# else and then import a stale copy from /content instead.
%cd /content
from pathlib import Path
Path("/content/bruisekit").mkdir(parents=True, exist_ok=True)
print("cwd:", Path.cwd(), "| package dir ready:", Path("/content/bruisekit").is_dir())
''')

writefile("bruisekit/__init__.py", '''
"""Bruise segmentation training kit. See docs/training_notebook_explained.md."""
''')

writefile("bruisekit/data.py", r'''
"""One dataloader for every model: one disk read, one resize, one augmentation.

THE DATASET EMITS RAW [0,1] PIXELS -- IT DOES NOT NORMALISE
------------------------------------------------------------
SegFormer wants ImageNet-normalised input; YOLO wants plain /255. That is not a
style preference, it is a property of the trained weights: Ultralytics trains on
/255 and its BatchNorms carry frozen running statistics for that distribution.
Feeding YOLO ImageNet-normalised pixels makes it under-fire by 4x and caps it at
Dice 0.479 with NO threshold able to recover it.

So pixel scale belongs to the MODEL, not the loader (see models.py). The loader
emits raw [0,1] and each wrapper applies its own scale. Every model therefore
shares one disk read, one resize and one augmentation -- the geometry is
identical by construction -- while each still sees the distribution it was
trained for. The alternative (normalise here, un-normalise for YOLO) works but
carries a needless roundtrip and invites exactly the bug it is working around.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_augmentation(training: bool, img_size: int) -> A.Compose:
    """The augmentation pipeline. Identical for all five models.

    A.Resize is a NO-OP on the packaged 640x640 PNGs -- it is kept as a cheap
    guard so a wrong-sized file fails as a shape error here rather than as a
    silent geometry mismatch 40 minutes into training.

    A.Normalize(mean=0, std=1, max_pixel_value=255) is exactly x/255: albumentations
    computes (img - mean*max_pixel_value) / (std*max_pixel_value).
    """
    to_unit = A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
    resize = [A.Resize(height=img_size, width=img_size)]
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


class BruiseDataset(Dataset):
    """Reads the 640x640 PNG package. Returns (x[3,H,W] in [0,1], y[1,H,W] in {0,1}, stem)."""

    def __init__(self, df: pd.DataFrame, root: Path, img_size: int, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.img_size = img_size
        self.tfm = build_augmentation(training, img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]

        img = cv2.imread(str(self.root / r.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Cannot read image: {r.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {r.mask_path}")
        # IMREAD_GRAYSCALE is documented to return (H, W) but returns (H, W, 1) in any
        # process where ultralytics has been imported -- it monkey-patches cv2.imread,
        # and import ORDER does not help. That trailing axis survives ToTensorV2 and
        # makes y [B,1,H,W,1]; dice(pred[H,W], gt[H,W,1]) then BROADCASTS to [H,W,H]
        # and returns nonsense -- a pixel-perfect prediction scores 63.9 instead of 1.0.
        # Squeezing here is a no-op on an unpatched cv2 and fixes every consumer at once.
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        aug = self.tfm(image=img, mask=mask)
        x = aug["image"].float()
        y = aug["mask"].unsqueeze(0).float()
        assert y.shape == (1, self.img_size, self.img_size), f"bad mask shape {y.shape} for {r.stem}"
        return x, y, str(r.stem)


def make_loader(df, root, img_size, batch_size, training, workers, seed=0):
    """DataLoader with seeded, reproducible shuffling and worker RNG."""
    ds = BruiseDataset(df, root, img_size, training=training)
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _init_worker(worker_id: int) -> None:
        # Each worker gets a distinct but seed-derived RNG stream, so augmentation
        # is reproducible for a given seed AND workers never draw identical noise.
        np.random.seed(seed * 1000 + worker_id)

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=training, drop_last=training,
        num_workers=workers, pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=_init_worker, generator=gen,
    )
''')

writefile("bruisekit/models.py", r'''
"""The two architectures behind one interface.

THE INTERFACE
--------------
    forward_train(x) -> (logits[B,1,H,W], aux_logits[B,1,H,W] | None)
    forward(x)       -> logits[B,1,H,W]

x is RAW [0,1] pixels (see data.py). Each wrapper applies its own pixel scale.
Both models emit ONE bruise logit at full input resolution, so every downstream
consumer -- loss, sweep, metric, benchmark -- is architecture-blind. Nothing
below this line ever branches on a model's name.

WHY YOLO IS BUILT WITH nc=1 (not nc=2)
---------------------------------------
The pretrained yolo26n-sem.pt has nc=19 (Cityscapes). Rebuilding the head with
nc=2 gives [B,2,H,W], from which the bruise logit is z1-z0 (2-class softmax and
sigmoid-of-the-difference are the same function). Rebuilding with nc=1 gives that
single logit directly, and Ultralytics' own loss supports it (nn.BCEWithLogitsLoss
for nc==1). One less transformation, one less place to get the sign wrong, and
structurally identical to SegFormer's 1-channel head. `.load()` transfers 360/364
tensors -- only the head is new, exactly mirroring SegFormer's randomly-initialised
1-class head on a pretrained backbone.

WHY THE HEAD GETS 10x THE BACKBONE'S LR (both architectures)
-------------------------------------------------------------
The backbone is pretrained and already has good features; a conservative LR
preserves them. The head is randomly initialised and has to catch up. This is the
SegFormer paper's recipe (Xie et al. 2021) and it is applied to YOLO here too --
not because YOLO's paper says so, but because holding the recipe fixed across
architectures is the entire point of an apple-to-apple comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegFormerNet(nn.Module):
    """HuggingFace SegFormer with a 1-class head. Input scale: ImageNet."""

    def __init__(self, pretrained_dir: str):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation
        self.net = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_dir, num_labels=1, ignore_mismatched_sizes=True)
        # Buffers (not constants) so they move with .to(device) and are saved with
        # the module -- the pixel scale travels with the weights it belongs to.
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @property
    def backbone(self):
        return self.net.segformer

    @property
    def head(self):
        return self.net.decode_head

    def forward_train(self, x):
        x = (x - self.mean) / self.std                      # [0,1] -> ImageNet
        logits = self.net(pixel_values=x).logits            # [B,1,H/4,W/4]
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, None                                 # SegFormer has no aux head

    def forward(self, x):
        return self.forward_train(x)[0]


class YoloSemNet(nn.Module):
    """Ultralytics YOLO26n-sem with an nc=1 head. Input scale: /255 (i.e. x as-is).

    In train() the underlying module returns (main[B,1,H/8,W/8], aux[B,1,H/16,W/16]);
    in eval() it returns only the main tensor. Both are upsampled to input
    resolution here so callers never see the stride.

    NOTE the stride: YOLO's main head is at H/8 (80x80 for a 640 input) against
    SegFormer's H/4 (160x160). YOLO predicts at a quarter of SegFormer's spatial
    resolution in area, which is an architectural ceiling on boundary precision,
    not something training can fix. Worth stating in the paper.
    """

    def __init__(self, pretrained_pt: str):
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.tasks import SemanticSegmentationModel

        pretrained = YOLO(str(pretrained_pt))
        self.net = SemanticSegmentationModel("yolo26n-sem.yaml", nc=1, ch=3, verbose=False)
        self.net.load(pretrained.model)     # transfers the backbone; head stays random
        del pretrained

    @property
    def backbone(self):
        return self.net.model[:-1]

    @property
    def head(self):
        return self.net.model[-1]

    def forward_train(self, x):
        out = self.net(x)                                   # x already /255
        if isinstance(out, (list, tuple)):
            main, aux = out[0], out[1]
        else:
            main, aux = out, None
        size = x.shape[-2:]
        main = F.interpolate(main.float(), size=size, mode="bilinear", align_corners=False)
        if aux is not None:
            aux = F.interpolate(aux.float(), size=size, mode="bilinear", align_corners=False)
        return main, aux

    def forward(self, x):
        return self.forward_train(x)[0]


def build_model(arch: str, size: str, paths: dict) -> nn.Module:
    if arch == "segformer":
        return SegFormerNet(paths[f"segformer_{size}"])
    if arch == "yolo":
        return YoloSemNet(paths["yolo"])
    raise ValueError(f"unknown arch: {arch}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_param_groups(model, backbone_lr: float, head_lr: float, weight_decay: float):
    """Backbone/head LR split + no weight decay on norms and biases.

    No decay on 1-D params (norm weights, biases): decaying a normalisation scale
    or a bias shrinks it toward zero with no regularising benefit -- standard in
    every transformer recipe including SegFormer's.

    Membership is by id(), not by name: the two architectures name their
    parameters completely differently, and a name-prefix rule would silently put
    every YOLO parameter in the wrong group.
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    groups = {
        "backbone_decay":    {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay":        {"params": [], "lr": head_lr,     "weight_decay": weight_decay},
        "head_no_decay":     {"params": [], "lr": head_lr,     "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad or name in ("mean", "std"):
            continue
        where = "backbone" if id(p) in backbone_ids else "head"
        decay = "_no_decay" if (p.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower()) else "_decay"
        groups[where + decay]["params"].append(p)

    out = [g for g in groups.values() if g["params"]]
    n_grouped = sum(len(g["params"]) for g in out)
    n_total = sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in ("mean", "std"))
    assert n_grouped == n_total, f"param grouping lost {n_total - n_grouped} tensors"
    return out
''')

writefile("bruisekit/losses.py", r'''
"""Losses. One supervised loss for every model; one distillation fusion.

WHY Dice+BCE
-------------
BCE alone is dominated by the background: bruises cover ~4.7% of pixels, so a
model predicting all-background already scores well on BCE and gets almost no
gradient toward the bruise. The Dice term is scale-invariant with respect to
object size and supplies gradient proportional to overlap, which is what we
actually report. Summing them is the standard combination for imbalanced medical
segmentation (Milletari et al. 2016 V-Net for Dice; Drozdzal et al. 2016 for the
combination), and it is also what Ultralytics' own SemanticSegmentationLoss does
(BCEWithLogits + binary Dice), so using it for both architectures does not
disadvantage YOLO relative to its native recipe.

WHAT THE DISTILLATION LOSS IS, AND WHAT IT IS NOT
--------------------------------------------------
    loss = alpha * DiceBCE(student_logits, GT)
         + (1 - alpha) * BCE(student_logits, sigmoid(teacher_logits / T_cal))

This is CALIBRATED SOFT-TARGET DISTILLATION, not Hinton et al. (2015) KD. The
difference matters for the paper and should not be papered over:

  - Hinton's KD divides BOTH the student's and the teacher's logits by a shared
    temperature T and multiplies the soft term by T^2 (to keep the soft gradient's
    magnitude comparable to the hard term's as T varies).
  - Here the student's logits are NOT temperature-scaled, and there is no T^2.
    T_cal is not a KD knob at all: it is the temperature fitted by NLL on val
    (Guo et al. 2017) that makes the teacher's probabilities CALIBRATED.

So the teacher is used as a better-calibrated estimate of P(bruise | pixel), and
the student regresses onto that probability. The justification is Menon et al.
(2021), "A statistical perspective on distillation": distillation helps to the
extent the teacher approximates the Bayes class-probability, which is exactly
what calibration improves. Do NOT cite Hinton for this formula as written.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """BCE + (1 - mean per-image soft Dice).

    Dice is computed PER IMAGE and then averaged, not pooled over the batch. A
    batch-pooled Dice lets one large bruise dominate the whole batch's gradient
    and lets an image with no predicted pixels hide inside a batch that scored
    well -- and per-image is what the reported metric does, so the loss and the
    metric agree.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(1, 2, 3))
        denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return bce + (1.0 - dice.mean())


class SupervisedLoss(nn.Module):
    """DiceBCE on the main head + aux_weight * BCE on the auxiliary head.

    The aux term applies only to YOLO (SegFormer returns aux=None). 0.4 is
    Ultralytics' own weight for its semantic aux head; keeping their value means
    the aux head is supervised as its designers intended even though the rest of
    the recipe is ours.
    """

    def __init__(self, aux_weight: float = 0.4):
        super().__init__()
        self.main = DiceBCELoss()
        self.aux_weight = aux_weight

    def forward(self, logits, aux_logits, target):
        loss = self.main(logits, target)
        if aux_logits is not None:
            loss = loss + self.aux_weight * F.binary_cross_entropy_with_logits(aux_logits, target)
        return loss


class DistillLoss(nn.Module):
    """alpha * supervised(GT) + (1-alpha) * BCE(student, calibrated teacher prob).

    See the module docstring for what this is and is not.
    """

    def __init__(self, alpha: float = 0.5, aux_weight: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.sup = SupervisedLoss(aux_weight)

    def forward(self, logits, aux_logits, target, teacher_prob):
        hard = self.sup(logits, aux_logits, target)
        soft = F.binary_cross_entropy_with_logits(logits, teacher_prob)
        return self.alpha * hard + (1.0 - self.alpha) * soft
''')

writefile("bruisekit/metrics.py", r'''
"""Metrics: Dice/IoU per image, and threshold-free AP for model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, gt).sum() / denom)


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def compute_image_row(pred: np.ndarray, gt: np.ndarray, stem: str) -> dict:
    pred_b, gt_b = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    return {
        "stem": stem,
        "dice": dice_np(pred, gt), "iou": iou_np(pred, gt),
        "precision": 1.0 if tp + fp == 0 else tp / (tp + fp),
        "recall": 1.0 if tp + fn == 0 else tp / (tp + fn),
        "pred_positive_pixels": int(pred_b.sum()),
        "gt_positive_pixels": int(gt_b.sum()),
    }


def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    # "Complete miss" = the model output ZERO pixels on an image that has a bruise.
    # This is the metric that separates the models by more than label noise, and for
    # an injury-documentation tool it is the one that actually matters: a blank mask
    # is a missed injury. Reported as a first-class number, not buried in the tail.
    miss = (df["pred_positive_pixels"] == 0) & (df["gt_positive_pixels"] > 0)
    return {
        "n_images": int(len(df)),
        "mean_dice": float(df["dice"].mean()),
        "median_dice": float(df["dice"].median()),
        "mean_iou": float(df["iou"].mean()),
        "median_iou": float(df["iou"].median()),
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "complete_miss_count": int(miss.sum()),
        "complete_miss_rate": float(miss.mean()),
    }


class BinnedAP:
    """Threshold-free average precision over pixels, via probability histograms.

    WHY AP IS THE MODEL-SELECTION METRIC
    -------------------------------------
    The old pipeline saved best_model.pt by val Dice AT A FIXED 0.5 -- but the
    threshold is re-fitted afterwards anyway, and the fitted operating points are
    nowhere near 0.5 (YOLO's lands around 0.18). So 0.5-Dice selection asks "which
    epoch is best at an operating point we will not use?" and can pick the wrong
    epoch for any model whose calibration drifts during training. AP integrates
    over ALL thresholds, so the epoch choice cannot be biased by one arbitrary cut.

    WHY HISTOGRAMS AND NOT sklearn.average_precision_score
    -------------------------------------------------------
    134 val images x 640 x 640 = 55M pixels. Sorting 55M floats per epoch costs
    seconds of wall-clock and ~450 MB. Binning into 4096 buckets on the GPU makes
    the whole thing O(bins) in memory and effectively free, at a quantisation
    error of ~1/4096 in probability -- three orders of magnitude below the
    epoch-to-epoch differences it has to rank.
    """

    def __init__(self, bins: int = 4096, device: str = "cuda"):
        self.bins = bins
        self.pos = torch.zeros(bins, dtype=torch.float64, device=device)
        self.neg = torch.zeros(bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prob: torch.Tensor, gt: torch.Tensor) -> None:
        p = prob.reshape(-1).float().clamp(0, 1)
        g = gt.reshape(-1) > 0.5
        idx = (p * (self.bins - 1)).round().long()
        self.pos += torch.bincount(idx[g], minlength=self.bins).double()
        self.neg += torch.bincount(idx[~g], minlength=self.bins).double()

    def compute(self) -> float:
        """AP = sum over thresholds of (recall_k - recall_{k-1}) * precision_k."""
        total_pos = self.pos.sum()
        if total_pos == 0:
            return float("nan")
        # Walk bins from the highest probability downward: each step admits one more
        # bucket as "predicted positive", which is exactly sweeping the threshold down.
        tp = torch.cumsum(self.pos.flip(0), dim=0)
        fp = torch.cumsum(self.neg.flip(0), dim=0)
        precision = tp / (tp + fp).clamp_min(1e-12)
        recall = tp / total_pos
        d_recall = torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype, device=recall.device))
        return float((d_recall * precision).sum())
''')

writefile("bruisekit/engine.py", r'''
"""The training loop. One loop for every model, and it survives session death."""
from __future__ import annotations

import json
import os
import random
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from bruisekit.data import make_loader
from bruisekit.losses import DistillLoss, SupervisedLoss
from bruisekit.metrics import BinnedAP
from bruisekit.models import build_model, build_param_groups, count_params


def seed_everything(seed: int) -> None:
    """Seed every RNG that touches training.

    cudnn.deterministic without use_deterministic_algorithms(True): the latter
    makes several ops here (bilinear interpolate backward, scatter_add) raise
    instead of run. Bitwise GPU determinism is therefore NOT claimed -- which is
    precisely why this notebook runs three seeds and reports spread rather than
    pretending one run is the answer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, power: float = 1.0) -> float:
    """Linear warmup 0->1, then poly decay 1->0. SegFormer's schedule (Xie 2021)."""
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return (1.0 - progress) ** power


def resolve_micro_batch(model, cfg, device, teacher=None) -> tuple[int, int]:
    """Choose (micro_batch, accum_steps) by probing what actually fits in VRAM.

    Two policies, selected by cfg["batch_mode"]:

    "matched" (the paper recipe) -- the probe is CAPPED AT effective_batch, and
    accumulation makes up any shortfall, so EVERY model trains at the same effective
    batch (8) and the same number of optimizer steps regardless of GPU. This is the
    fix for the old pipeline's central bug: the probe used to return 32 for the small
    models while accum collapsed to 1, so B2 trained at batch 8 / 8700 steps and both
    B0s at batch 32 / 2100 steps at the same LR -- making every "identical
    hyperparameters" claim between B0 and B2 false. A T4 just uses more accumulation
    than an A100 and lands in the same place.

    "per_model" -- the probe is NOT capped; each model uses the largest power-of-2
    batch its own size allows (accum = 1). B2 stays ~8, B0/YOLO climb much higher.
    Faster and better GPU utilisation, but the models NO LONGER share an effective
    batch or a step count, so this policy is not recipe-matched across model sizes
    (see the per-model notebook's §1/§6 warning). Same LR is kept deliberately (no
    linear-scaling adjustment), so the small models take fewer, larger steps.

    The probe runs on a DEEPCOPY: it does real forward/backward/step, and probing the
    live model would perturb the pretrained weights before training starts by a
    model-dependent amount.
    """
    mode = cfg.get("batch_mode", "matched")
    effective = cfg["effective_batch"]
    if not torch.cuda.is_available():
        # No GPU to probe: matched keeps its fixed effective batch via accumulation;
        # per_model has no target to hit, so fall back to a single sample.
        return (1, effective) if mode == "matched" else (1, 1)

    # matched caps the probe at effective_batch (accum fills the rest); per_model
    # lets it climb to max_probe_batch, whatever the model can hold.
    probe_ceiling = min(cfg["max_probe_batch"], effective) if mode == "matched" else cfg["max_probe_batch"]

    probe_model = deepcopy(model).to(device)
    probe_opt = torch.optim.SGD(probe_model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda") if cfg["amp"] else None
    total_vram = torch.cuda.get_device_properties(device).total_memory

    chosen, batch = 1, 1
    while batch <= probe_ceiling:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.rand(batch, 3, cfg["img_size"], cfg["img_size"], device=device)
            y = torch.randint(0, 2, (batch, 1, cfg["img_size"], cfg["img_size"]), device=device).float()
            probe_model.train()
            probe_opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["amp"]):
                # The teacher runs on every real training step too; excluding it here
                # would choose a batch that OOMs the moment distillation starts.
                if teacher is not None:
                    _ = teacher(x)
                logits, aux = probe_model.forward_train(x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
                if aux is not None:
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(aux, y)
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(probe_opt); scaler.update()
            else:
                loss.backward(); probe_opt.step()
            frac = torch.cuda.max_memory_reserved(device) / total_vram
            del x, y, logits, loss
            # 0.75 leaves headroom for the real loss (Dice+BCE+aux), augmentation
            # variance and the checkpoint write -- the probe only sees a plain BCE.
            if frac > cfg.get("vram_target", 0.75):
                break
            chosen = batch
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            break

    del probe_model, probe_opt
    torch.cuda.empty_cache()

    micro = max(1, chosen)
    if mode == "per_model":
        return micro, 1                    # use the probed batch directly, no accumulation
    while effective % micro != 0:          # matched: keep micro*accum EXACT, never approximate
        micro -= 1
    accum = effective // micro
    assert micro * accum == effective, f"{micro} x {accum} != {effective}"
    return micro, accum


def load_teacher(teacher_dir: Path, paths: dict, device, amp: bool):
    """Load a trained B2 as a frozen, calibrated soft-label generator.

    Returns a callable so the training loop never learns anything about the
    teacher's architecture -- it just calls teacher(x) and gets a probability map.
    """
    from bruisekit.models import build_model as _bm
    ckpt = teacher_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Teacher not trained yet: {ckpt}")

    model = _bm("segformer", "b2", paths).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    temperature = json.loads((teacher_dir / "calibration.json").read_text())["temperature"]

    def teacher_fn(x):
        # no_grad: the teacher is frozen. This is the single largest memory saving
        # in the distillation setup -- without it we would build a backward graph
        # through 27M parameters we never update.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        return torch.sigmoid(logits.float() / temperature)

    teacher_fn.temperature = temperature
    return teacher_fn


def calibrate_temperature(model, loader, device, amp: bool) -> dict:
    """Fit the scalar T minimising val NLL of sigmoid(z/T). Guo et al. (2017).

    WHY: BCE drives a trained model's logits toward +-inf, because sigmoid(+-inf)
    is a perfect loss. The teacher's probability histogram ends up nearly binary,
    so sigmoid(z_teacher) as a soft label is almost indistinguishable from the hard
    GT -- which defeats the point of soft-label distillation. Dividing by T > 1
    pulls saturated logits back into sigmoid's responsive region, so the student
    can see where the teacher is UNCERTAIN, which is the information distillation
    is supposed to transfer.

    Optimises log(T), not T: keeps T > 0 automatically without a constraint, and
    avoids the singularity at T = 0.
    L-BFGS, not SGD: one scalar over a fixed dataset -- second-order curvature
    converges in ~10 iterations where SGD needs hundreds.
    """
    model.eval()
    logits_all, targets_all = [], []
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="collect val logits", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                z = model(x)
            # Downsample 4x before storing: calibration fits ONE scalar, and a
            # regular 1-in-16 pixel subsample of 55M pixels estimates it to far
            # more precision than it is worth -- while keeping this in RAM.
            logits_all.append(z.float()[..., ::4, ::4].cpu())
            targets_all.append(y[..., ::4, ::4].cpu())

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    nll_before = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / torch.exp(log_t), targets)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(torch.exp(log_t).item())
    nll_after = float(torch.nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets))

    # A well-trained BCE model is OVER-confident, so T lands a little above 1
    # (this project's B2 previously calibrated to 1.84). Anything far outside that
    # is a symptom, not a temperature:
    #   T >> 10  -> the logits are near zero, i.e. the model is barely trained and
    #               calibration is degenerately flattening an already-flat output.
    #               Distilling from it would teach the student a constant 0.5.
    #   T < 0.5  -> the model is UNDER-confident, which BCE training does not
    #               normally produce -- suspect the checkpoint or the val split.
    # Loud warning rather than a raise: the number is still recorded and the run can
    # proceed, but this must never pass silently into a paper.
    if not (0.5 <= temperature <= 10.0):
        print(f"  !! WARNING: calibrated T={temperature:.3f} is outside the plausible "
              f"[0.5, 10] range. The teacher is probably under-trained (near-zero "
              f"logits) -- check its val AP before trusting any distilled student.")

    return {"temperature": temperature, "nll_before": nll_before, "nll_after": nll_after,
            "n_pixels_used": int(targets.numel()), "plausible": bool(0.5 <= temperature <= 10.0)}


@torch.no_grad()
def eval_ap(model, loader, device, amp: bool) -> float:
    """Threshold-free val AP -- the model-selection metric. See metrics.BinnedAP."""
    model.eval()
    ap = BinnedAP(device=str(device))
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(x)
        ap.update(torch.sigmoid(logits.float()), y)
    return ap.compute()


def _atomic_save(obj, dest: Path) -> None:
    """Write to a temp file, then rename.

    Drive rename is atomic; a plain torch.save straight to Drive that is killed
    halfway leaves a TRUNCATED checkpoint, and the next session then dies trying
    to resume from it -- turning one lost session into a lost run.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dest)


def train_run(run_id: str, spec: dict, seed: int, cfg: dict, paths: dict,
              manifests: dict, root: Path, runs_dir: Path, device) -> dict:
    """Train one (model, seed). Idempotent and resumable.

    RESUME CONTRACT
    ----------------
    - DONE.json exists          -> return immediately, touch nothing.
    - resume.pt exists          -> restore model+optimizer+scaler+epoch+best and continue.
    - neither                   -> fresh start.

    resume.pt is written every cfg["drive_sync_every"] epochs (and always on the
    final epoch), so a disconnect costs at most that many epochs. It is saved even
    on epochs where val AP did NOT improve: the best weights live in best.pt, but
    resuming must continue from where training actually WAS, not from the last
    good epoch, or the LR schedule and optimizer moments silently rewind.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    done_file = run_dir / "DONE.json"
    if done_file.exists():
        return {"run_id": run_id, "status": "skipped", **json.loads(done_file.read_text())}

    seed_everything(seed)
    amp = cfg["amp"]

    model = build_model(spec["arch"], spec["size"], paths).to(device)

    teacher = None
    if spec["distill"]:
        # Same-seed teacher: the distilled run's spread over seeds then includes the
        # teacher's own variance, which is part of the pipeline being measured.
        teacher = load_teacher(runs_dir / f"segformer_b2_teacher__seed{seed}", paths, device, amp)

    micro, accum = resolve_micro_batch(model, cfg, device, teacher)

    train_loader = make_loader(manifests["train"], root, cfg["img_size"], micro,
                               training=True, workers=cfg["workers"], seed=seed)
    val_loader = make_loader(manifests["val"], root, cfg["img_size"], micro,
                             training=False, workers=cfg["workers"], seed=seed)

    param_groups = build_param_groups(model, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"])
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg["betas"]))
    peak_lrs = [g["lr"] for g in param_groups]
    scaler = torch.amp.GradScaler("cuda") if amp else None

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_fraction"]))

    criterion = (DistillLoss(cfg["alpha"], cfg["aux_weight"]) if teacher is not None
                 else SupervisedLoss(cfg["aux_weight"]))

    start_epoch, best_ap, patience, global_step, history = 1, float("-inf"), 0, 0, []
    resume_path = run_dir / "resume.pt"
    if resume_path.exists():
        st = torch.load(str(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, best_ap = st["epoch"] + 1, st["best_ap"]
        patience, global_step, history = st["patience"], st["global_step"], st["history"]
        print(f"  [resume] {run_id} from epoch {start_epoch} (best_ap={best_ap:.4f})")
        del st

    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id, "seed": seed, **spec,
        "micro_batch": micro, "accum_steps": accum, "effective_batch": micro * accum,
        "total_steps": total_steps, "warmup_steps": warmup_steps,
        "params": count_params(model),
        "alpha": cfg["alpha"] if spec["distill"] else None,
        "teacher_temperature": getattr(teacher, "temperature", None),
    }, indent=2))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, t0 = 0.0, time.time()

        for step, (x, y, _) in enumerate(tqdm(train_loader, desc=f"{run_id} e{epoch}", leave=False)):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp):
                # Teacher FIRST, inside its own no_grad: its activations are freed
                # before the student's backward graph is built. Student-first would
                # hold both alive at once and roughly double peak VRAM.
                tprob = teacher(x) if teacher is not None else None
                logits, aux = model.forward_train(x)
                loss = (criterion(logits, aux, y, tprob) if teacher is not None
                        else criterion(logits, aux, y))
                loss = loss / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += loss.item() * accum

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                global_step += 1
                # LR is set per OPTIMIZER STEP, not per micro-batch: the schedule is
                # defined over gradient updates, and updating it per micro-batch would
                # advance it accum_steps times too fast.
                mult = lr_multiplier(global_step, total_steps, warmup_steps, cfg["poly_power"])
                for g, peak in zip(optimizer.param_groups, peak_lrs):
                    g["lr"] = peak * mult
                if scaler is not None:
                    scaler.unscale_(optimizer)   # real gradients before clipping
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
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
            _atomic_save(model.state_dict(), run_dir / "best.pt")
            flag = " *"
        else:
            patience += 1
            flag = ""
        print(f"  {run_id} e{epoch:3d} loss={train_loss:.4f} val_ap={val_ap:.4f}"
              f" lr={cur_lr:.2e} {time.time()-t0:.0f}s{flag}")

        last = (epoch == cfg["epochs"]) or (patience >= cfg["patience"])
        if epoch % cfg["drive_sync_every"] == 0 or last:
            _atomic_save({"epoch": epoch, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict() if scaler else None,
                          "best_ap": best_ap, "patience": patience,
                          "global_step": global_step, "history": history}, resume_path)
        if patience >= cfg["patience"]:
            print(f"  early stop at epoch {epoch} (patience={cfg['patience']})")
            break

    # The teacher must be calibrated before any student can distil from it, and
    # calibration needs the BEST weights -- so it happens here, not in a separate step
    # that could be forgotten or run against the wrong checkpoint.
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=device, weights_only=True))
    if run_id.startswith("segformer_b2_teacher"):
        cal = calibrate_temperature(model, val_loader, device, amp)
        (run_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
        print(f"  calibrated T={cal['temperature']:.4f} "
              f"(val NLL {cal['nll_before']:.4f} -> {cal['nll_after']:.4f})")

    summary = {"run_id": run_id, "seed": seed, "best_val_ap": best_ap,
               "epochs_trained": len(history), "params": count_params(model),
               "micro_batch": micro, "accum_steps": accum}
    done_file.write_text(json.dumps(summary, indent=2))
    if resume_path.exists():
        resume_path.unlink()   # training finished: the resume state is now dead weight

    del model, teacher
    torch.cuda.empty_cache()
    return {"status": "trained", **summary}
''')

writefile("bruisekit/sweep.py", r'''
"""Fit each model's operating point on val. Never on test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def cache_logits(model, loader, device, amp: bool):
    """One forward pass per image; keep the logits.

    The sweep tries 481 cuts. Re-running the model for each would be 481 x 134 =
    64,454 forward passes; caching the logits once makes all 481 cuts pure tensor
    arithmetic on data already on the GPU. fp16 keeps 134x640x640 at ~110 MB --
    the quantisation is ~3 decimal places on a logit, far below the resolution any
    of this can distinguish.
    """
    model.eval()
    logits, gts, stems = [], [], []
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        logits.append(z.float().half()[:, 0])
        # GT as BOOL, never float: the sweep counts pixels, and counting is what
        # bool+int64 do exactly. See sweep_cuts for why that matters.
        gts.append((y[:, 0] > 0.5).to(device))
        stems.extend(s)
    return torch.cat(logits), torch.cat(gts), stems


def sweep_cuts(logits, gts, cuts):
    """Per-cut mean Dice, its standard error, and complete-miss rate. Vectorised.

    THE REDUCTIONS ARE EXACT, ON PURPOSE. Dice is a ratio of PIXEL COUNTS, so the
    intersection and the two cardinalities are computed as boolean masks summed to
    int64 -- exactly what metrics.dice_np does in numpy, and exact by construction.

    Doing the same reductions in fp16 (tempting, since the cached logits are fp16)
    is wrong: fp16 carries an 11-bit mantissa, and a 640x640 image sums ~410k terms,
    so the running total stops being able to represent +1 long before the sum
    finishes. Measured against the numpy implementation that drifted by ~1.5e-4 per
    cut -- small, but the tie band is selected by comparing cuts whose Dice differ
    by ~1e-3, so the error is a tenth of the signal it is used to rank. The logits
    stay fp16 (storage only; the >= comparison is exact regardless).
    """
    rows = []
    gts = gts.bool()
    gt_sum = gts.sum(dim=(1, 2))              # int64, exact
    gt_has = gt_sum > 0
    n = len(gt_sum)
    for c in cuts:
        pred = logits >= c                    # bool; comparison is exact in fp16
        inter = (pred & gts).sum(dim=(1, 2))
        pred_sum = pred.sum(dim=(1, 2))
        denom = pred_sum + gt_sum
        # denom == 0 means both prediction and GT are empty: a correct agreement,
        # scored 1.0 -- matching metrics.dice_np so the sweep and the report agree.
        dice = torch.where(denom > 0,
                           2.0 * inter.double() / denom.double().clamp_min(1.0),
                           torch.ones_like(denom, dtype=torch.float64))
        miss = ((pred_sum == 0) & gt_has).double()
        d = dice
        rows.append({
            "cut": float(c), "threshold": float(torch.sigmoid(torch.tensor(c))),
            "mean_dice": float(d.mean()),
            "se_dice": float(d.std(unbiased=True) / np.sqrt(n)),
            "complete_miss_rate": float(miss.mean()),
        })
    return pd.DataFrame(rows)


def select_cut(df: pd.DataFrame) -> dict:
    """Pick the operating point: the tie band's LOWEST-MISS cut.

    WHY A TIE BAND AT ALL
    ----------------------
    These sweeps are extraordinarily flat -- on the previous models, B2's val Dice
    moved by 0.009 across thresholds from 0.154 to 0.959. That is not a peak, it is
    noise on a plateau, and taking argmax of it fits the val set's sampling error.
    Every cut within ONE STANDARD ERROR of the peak is statistically tied, so the
    band -- not the argmax -- is the honest answer to "which cut is best?".

    WHY MISS RATE BREAKS THE TIE
    -----------------------------
    Cuts in the band are Dice-equivalent but they are NOT miss-equivalent: a lower
    cut predicts more pixels, so fewer images come back completely blank, at
    statistically identical Dice. The old rule took the band's MEDIAN cut, which
    optimises for stability -- but a blank mask is a missed injury, and the miss
    rate is the one metric that separates these models by more than label noise.
    So: minimise miss rate within the band; break remaining ties on Dice; break
    what is left on the median cut, for reproducibility.
    """
    peak = df.loc[df["mean_dice"].idxmax()]
    band = df[df["mean_dice"] >= peak["mean_dice"] - peak["se_dice"]]
    best_miss = band["complete_miss_rate"].min()
    tied = band[band["complete_miss_rate"] <= best_miss + 1e-12]
    top_dice = tied["mean_dice"].max()
    finalists = tied[tied["mean_dice"] >= top_dice - 1e-12]
    chosen = finalists.iloc[len(finalists) // 2]
    return {
        "cut": float(chosen["cut"]), "threshold": float(chosen["threshold"]),
        "val_dice_at_cut": float(chosen["mean_dice"]),
        "val_miss_at_cut": float(chosen["complete_miss_rate"]),
        "val_peak_dice": float(peak["mean_dice"]),
        "peak_cut": float(peak["cut"]),
        "band_lo_threshold": float(band["threshold"].min()),
        "band_hi_threshold": float(band["threshold"].max()),
        "band_width_cuts": int(len(band)),
        "n_cuts": int(len(df)),
    }
''')

writefile("bruisekit/evaluate.py", r'''
"""Test scoring, fairness across skin tone, and the speed benchmark."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats

from bruisekit.metrics import compute_image_row, summarize


@torch.no_grad()
def evaluate_at_cut(model, loader, device, cut: float, amp: bool) -> tuple[pd.DataFrame, dict]:
    """Score on test at an operating point ALREADY FIXED on val. One pass, no tuning."""
    model.eval()
    rows = []
    for x, y, stems in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = model(x)
        pred = (z.float()[:, 0] >= cut).cpu().numpy().astype(np.uint8)
        gt = (y[:, 0] > 0.5).numpy().astype(np.uint8)
        for i, stem in enumerate(stems):
            rows.append(compute_image_row(pred[i], gt[i], stem))
    return pd.DataFrame(rows), summarize(rows)


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the MEDIAN.

    The median, not the mean: per-image Dice is strongly bimodal (a model either
    localises the bruise or misses it completely), so the mean mixes two different
    populations. And a bootstrap, not a t-interval, because that bimodality makes
    the normal approximation wrong.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def fairness_analysis(per_image: pd.DataFrame, manifest: pd.DataFrame, model_name: str) -> dict:
    """Is performance equitable across Fitzpatrick/ITA skin-tone groups?

    THE STAKES: this is a forensic injury-documentation tool. A model that
    segments bruises well on light skin and poorly on dark skin does not have a
    metric problem, it has an evidentiary one -- it would under-document injuries
    on exactly the population most likely to need the documentation. So skin tone
    is not a robustness ablation here, it is a primary result.

    METHOD, and why each piece:
      - Groups are ITA (Individual Typology Angle, Chardon et al. 1991), the
        standard objective skin-tone measure -- computed from image pixels, not
        from a rater's Fitzpatrick guess, so it is reproducible.
      - Kruskal-Wallis across all 5 groups: an omnibus test for "does ANY group
        differ?". Non-parametric because per-image Dice is bimodal and bounded,
        so ANOVA's normality assumption fails.
      - Pairwise Mann-Whitney U, Bonferroni-corrected over the 10 pairs: with 5
        groups, uncorrected pairwise testing finds a "significant" pair ~40% of
        the time on pure noise.
      - fairness_gap = best group's median Dice - worst group's. The effect size.
        A p-value says a gap is real; only the gap says whether it matters.
    """
    df = per_image.merge(manifest[["stem", "skin_tone_category", "ita_group_index_5"]],
                         on="stem", how="left", validate="one_to_one")
    if df["ita_group_index_5"].isna().any():
        raise RuntimeError(f"{int(df['ita_group_index_5'].isna().sum())} test images have no skin-tone label")

    per_group, samples = [], []
    for gidx, g in sorted(df.groupby("ita_group_index_5"), key=lambda kv: kv[0]):
        vals = g["dice"].to_numpy()
        lo, hi = bootstrap_ci(vals)
        per_group.append({
            "model": model_name, "ita_group_index_5": int(gidx),
            "skin_tone_category": g["skin_tone_category"].iloc[0],
            "n_images": len(g), "median_dice": float(np.median(vals)),
            "iqr_dice": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
            "ci95_lo": lo, "ci95_hi": hi,
            "mean_recall": float(g["recall"].mean()),
            "miss_rate": float(((g["pred_positive_pixels"] == 0) & (g["gt_positive_pixels"] > 0)).mean()),
        })
        samples.append(vals)

    H, p = stats.kruskal(*samples)
    pairwise = []
    raw = []
    pairs = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    for i, j in pairs:
        raw.append(stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided").pvalue)
    for (i, j), pv in zip(pairs, raw):
        adj = min(1.0, pv * len(pairs))    # Bonferroni over the 10 pairs
        pairwise.append({"model": model_name, "group_a": per_group[i]["skin_tone_category"],
                         "group_b": per_group[j]["skin_tone_category"],
                         "pvalue": pv, "bonferroni_p": adj, "significant": bool(adj < 0.05)})

    pg = pd.DataFrame(per_group)
    best, worst = pg.loc[pg["median_dice"].idxmax()], pg.loc[pg["median_dice"].idxmin()]
    stats_row = {
        "model": model_name, "kruskal_H": float(H), "kruskal_p": float(p),
        "significant": bool(p < 0.05),
        "fairness_gap": float(best["median_dice"] - worst["median_dice"]),
        "best_group": best["skin_tone_category"], "worst_group": worst["skin_tone_category"],
        "max_miss_rate_gap": float(pg["miss_rate"].max() - pg["miss_rate"].min()),
    }
    return {"per_group": pg, "pairwise": pd.DataFrame(pairwise), "stats": stats_row}


@torch.no_grad()
def benchmark_speed(model, images, device, cut: float, repeats: int, warmup: int) -> dict:
    """Time 640-tensor-on-GPU -> mask-on-GPU. Nothing else.

    WHAT IS DELIBERATELY NOT TIMED: disk read, JPEG decode, resize, host->GPU copy,
    and GPU->host copy. Those are identical for every model (one dataloader) and
    are dominated by I/O, so including them would compress the real architectural
    differences into measurement noise. The mask never leaves the GPU.

    cuda.synchronize() around each call is not optional: CUDA is asynchronous, so
    without it this would measure how long it takes to QUEUE the work, not do it --
    which reports every model as equally, impossibly fast.
    """
    model.eval()
    for _ in range(warmup):
        _ = torch.sigmoid(model(images[:1])) >= torch.sigmoid(torch.tensor(cut, device=device))
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        for i in range(len(images)):
            x = images[i:i + 1]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = model(x)
            _ = z >= cut                      # threshold on the logit == sigmoid >= sigmoid(cut)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {"median_ms": float(np.median(arr)), "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)), "fps": float(1000.0 / np.median(arr)),
            "n_timed": len(arr)}


import time  # noqa: E402  (used by benchmark_speed above)
''')

writefile("bruisekit/postopt.py", r'''
"""Reduce complete misses WITHOUT retraining. Everything here is fitted on val and
applied to test; nothing changes a trained weight.

THE PROBLEM THIS SOLVES
------------------------
A single global threshold couples two failure modes: pushing it down to stop blank
masks (complete misses) also floods the easy images with false positives, so miss-%
and Dice move together -- two sides of one knob. Lowering the threshold slides ALONG
the miss-vs-Dice curve; it cannot move the curve. The three techniques here try to
move the curve (fewer misses at the SAME Dice), and one deliberately games the miss
metric so its cost is visible for comparison:

  ensemble   -- average the 3 seeds' probability maps. A miss needs ALL three seeds
                to blank the same image, which is rare (the per-seed misses are
                different images). Free: no retraining, just averaging maps we can
                already produce. This is the honest, recommended lever.
  TTA        -- average probs over horizontal+vertical flips. Raises the probability
                on borderline images so they clear the threshold without lowering it.
  no-blank   -- if a mask is still empty, recover the most-confident region instead of
                returning blank. This GAMES the miss metric (guarantees a non-zero
                prediction whether or not anything real was found), so it is reported
                as a separate floor, never as the main method.

HOW TO READ THE RESULT (this is the whole point of the question)
----------------------------------------------------------------
Plot miss-% against Dice. A real improvement sits BELOW-AND-LEFT of the single-model
threshold-sweep curve -- lower miss at equal-or-better Dice. A point that just slid
down the same curve (Dice fell as much as miss) is the threshold in disguise, not an
improvement. Everything below fits its threshold on VAL with the same miss-tie-break
rule as the baseline, so the comparison is like-for-like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bruisekit.metrics import compute_image_row, summarize
from bruisekit.sweep import select_cut


@torch.no_grad()
def probs_plain(model, loader, device, amp: bool):
    """One forward pass per image -> sigmoid probability maps [N,H,W] (fp16), GT
    (bool), stems. fp16 storage keeps the whole split in memory; the >= comparison
    the sweep does is exact regardless of storage dtype."""
    model.eval()
    P, G, S = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            z = model(x)
        P.append(torch.sigmoid(z.float())[:, 0].half().cpu())
        G.append((y[:, 0] > 0.5).cpu())
        S.extend(s)
    return torch.cat(P), torch.cat(G), S


@torch.no_grad()
def probs_tta(model, loader, device, amp: bool, flips=("none", "h", "v")):
    """TTA: average sigmoid probs over identity + horizontal + vertical flip.

    Each flip is applied to the INPUT and UNDONE on the OUTPUT before averaging, so
    the maps stay pixel-aligned -- averaging misaligned maps would blur the boundary
    rather than sharpen the confidence. TTA is used identically on val (to fit the
    threshold) and test (to score); mismatching them would fit a threshold for a
    distribution the test pass never sees.
    """
    model.eval()
    P, G, S = [], [], []
    use_amp = amp and device.type == "cuda"
    for x, y, s in loader:
        x = x.to(device, non_blocking=True)
        acc = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        for f in flips:
            xf = torch.flip(x, [3]) if f == "h" else torch.flip(x, [2]) if f == "v" else x
            with torch.amp.autocast("cuda", enabled=use_amp):
                z = model(xf)
            p = torch.sigmoid(z.float())[:, 0]
            p = torch.flip(p, [2]) if f == "h" else torch.flip(p, [1]) if f == "v" else p
            acc = acc + p
        P.append((acc / len(flips)).half().cpu())
        G.append((y[:, 0] > 0.5).cpu())
        S.extend(s)
    return torch.cat(P), torch.cat(G), S


def mean_over_seeds(prob_list, stem_list):
    """Average probability maps across runs, aligned by stem.

    Loaders are shuffle=False so every seed already returns images in the same order,
    but this re-indexes by stem anyway rather than trusting that -- a silent order
    mismatch would average seed A's image 5 onto seed B's image 6 and quietly corrupt
    every ensemble number. Asserts the image SETS are identical first.
    """
    ref = stem_list[0]
    ref_set = set(ref)
    for sl in stem_list:
        if set(sl) != ref_set:
            raise ValueError("ensemble seeds cover different image sets")
    acc = None
    for probs, sl in zip(prob_list, stem_list):
        pos = {s: i for i, s in enumerate(sl)}
        reordered = torch.stack([probs[pos[s]] for s in ref]).float()
        acc = reordered if acc is None else acc + reordered
    return (acc / len(prob_list)).half(), list(ref)


def sweep_prob(probs, gts, thresholds):
    """Per-threshold mean Dice / SE / complete-miss on probability maps.

    Same exact-integer pixel counting as sweep.sweep_cuts (bool masks summed to
    int64), so the numbers are directly comparable to the logit-cut sweep. Emits the
    columns select_cut expects, with `cut` == `threshold` (probability space).

    The comparison is done in float32, NOT the fp16 the maps are stored in, so the
    threshold this sweep FITS on val is applied at the exact same numeric boundary
    score_prob_at() uses on test (it also compares in float32). Comparing fp16 here
    would put the val fit and the test apply on slightly different boundaries for any
    pixel sitting right at the threshold -- the same "the sweep must match the score"
    trap the logit sweep already guards against.
    """
    probs = probs.float()
    gts = gts.bool()
    gt_sum = gts.sum(dim=(1, 2))
    gt_has = gt_sum > 0
    n = len(gt_sum)
    rows = []
    for t in thresholds:
        pred = probs >= t
        inter = (pred & gts).sum(dim=(1, 2))
        ps = pred.sum(dim=(1, 2))
        den = ps + gt_sum
        dice = torch.where(den > 0, 2.0 * inter.double() / den.double().clamp_min(1.0),
                           torch.ones_like(den, dtype=torch.float64))
        miss = ((ps == 0) & gt_has).double()
        rows.append({"cut": float(t), "threshold": float(t),
                     "mean_dice": float(dice.mean()),
                     "se_dice": float(dice.std(unbiased=True) / np.sqrt(n)),
                     "complete_miss_rate": float(miss.mean())})
    return pd.DataFrame(rows)


def score_prob_at(probs, gts, stems, thr, no_blank=False, rel=0.5):
    """Score probability maps at a fixed threshold; optional no-blank floor.

    no_blank: when the thresholded mask is empty on an image that has a bruise, fall
    back to the region at >= rel * max-probability -- the most-confident blob the
    model saw. This never returns blank, which is exactly why it must be reported
    separately: it converts a genuine miss into a (possibly wrong) small prediction.
    """
    p_np = probs.float().numpy()
    g_np = gts.numpy()
    rows = []
    for i, s in enumerate(stems):
        p = p_np[i]
        pred = (p >= thr).astype("uint8")
        if no_blank and pred.sum() == 0 and p.max() > 0:
            pred = (p >= rel * float(p.max())).astype("uint8")
        rows.append(compute_image_row(pred, g_np[i].astype("uint8"), s))
    return pd.DataFrame(rows), summarize(rows)


def fit_on_val_apply_to_test(val_probs, val_gts, test_probs, test_gts, test_stems,
                             thresholds, no_blank=False):
    """The honest protocol: sweep val -> select_cut (miss-tie-break) -> score test.

    Returns (operating_point_dict, test_per_image_df, test_summary). The threshold is
    chosen ONLY from val, then applied once to test, exactly like the baseline in the
    main notebook -- so any miss/Dice difference is the technique, not a re-tuned cut.
    """
    grid = sweep_prob(val_probs, val_gts, thresholds)
    op = select_cut(grid)
    per_img, summ = score_prob_at(test_probs, test_gts, test_stems, op["threshold"], no_blank=no_blank)
    return op, per_img, summ
''')

# ── §5 wire up ───────────────────────────────────────────────────────────────
md(r"""
## §5 · Import and self-test

Cheap assertions on the pieces the whole run depends on. Each has cost real debugging
time on this project at least once.
""")

code(r'''
sys.path.insert(0, "/content")
import importlib
import bruisekit.data, bruisekit.models, bruisekit.losses, bruisekit.metrics
import bruisekit.engine, bruisekit.sweep, bruisekit.evaluate, bruisekit.postopt
for m in ("data", "models", "losses", "metrics", "engine", "sweep", "evaluate", "postopt"):
    importlib.reload(sys.modules[f"bruisekit.{m}"])

from bruisekit.data import make_loader
from bruisekit.engine import seed_everything, train_run
from bruisekit.evaluate import benchmark_speed, evaluate_at_cut, fairness_analysis
from bruisekit.metrics import BinnedAP, dice_np
from bruisekit.models import build_model, build_param_groups, count_params
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts
from bruisekit.postopt import (fit_on_val_apply_to_test, mean_over_seeds,
                               probs_plain, probs_tta, score_prob_at)

# Notebook-scope stdlib used by later sections (§7-§13 read/write JSON sidecars,
# time runs, and open images). Imported here, once, so no downstream cell trips on
# a missing name after a fresh Run All -- the library modules import their own.
import json, time
import numpy as np, torch

PATHS = {
    "segformer_b0": str(WORK / "pretrained_weights" / "segformer_mit_b0"),
    "segformer_b2": str(WORK / "pretrained_weights" / "segformer_mit_b2"),
    "yolo":         str(WORK / "pretrained_weights" / "yolo26n-sem.pt"),
}
DEVICE = torch.device("cuda:0")
''')

code(r'''
# 1. The mask must be [B,1,H,W] -- ultralytics' cv2.imread patch adds a trailing
#    axis that broadcasts a pixel-perfect prediction to Dice 63.9 instead of 1.0.
_loader = make_loader(MAN["val"].head(4), WORK, CFG["img_size"], 2, False, 0)
_x, _y, _s = next(iter(_loader))
assert _x.shape == (2, 3, 640, 640), _x.shape
assert _y.shape == (2, 1, 640, 640), f"MASK SHAPE BUG: {_y.shape}"
assert 0.0 <= float(_x.min()) and float(_x.max()) <= 1.0, "loader must emit RAW [0,1] pixels"
assert set(np.unique(_y.numpy())).issubset({0.0, 1.0}), "mask must be binary"
print(f"data      OK  x{tuple(_x.shape)} in [{_x.min():.3f},{_x.max():.3f}]  y{tuple(_y.shape)}")

# 2. Both models must emit ONE bruise logit at full resolution, or every
#    downstream consumer would need to branch on architecture.
for _arch, _size, _expect, _wants_aux in [("segformer", "b0", 3.71, False),
                                          ("segformer", "b2", 27.35, False),
                                          ("yolo", "n", 1.63, True)]:
    _m = build_model(_arch, _size, PATHS).to(DEVICE)
    # train() mode, not eval(): YOLO only emits its auxiliary head while training,
    # and the aux term is part of the loss -- an eval()-mode check would pass while
    # silently proving nothing about what the training loop actually receives.
    _m.train()
    with torch.no_grad():
        _z, _aux = _m.forward_train(_x.to(DEVICE))
    _p = count_params(_m) / 1e6
    assert _z.shape == (2, 1, 640, 640), f"{_arch}{_size} -> {_z.shape}"
    assert abs(_p - _expect) < 0.05, f"{_arch}{_size}: {_p:.2f}M params, expected ~{_expect}M"
    assert (_aux is not None) == _wants_aux, f"{_arch}{_size}: aux head presence wrong in train mode"
    if _aux is not None:
        assert _aux.shape == _z.shape, f"aux {_aux.shape} != main {_z.shape}"
    assert len(build_param_groups(_m, 6e-5, 6e-4, 0.01)) == 4, "expected 4 param groups"
    print(f"{_arch}-{_size:<3} OK  logits{tuple(_z.shape)}  aux={'yes' if _aux is not None else 'no '}  {_p:.2f}M params")
    del _m
torch.cuda.empty_cache()
''')

code(r'''
# 3. AP must be a real average precision: a perfect ranking scores 1.0, and a
#    random one scores the positive rate. If this is wrong, every model-selection
#    decision in the run is wrong, silently.
_ap = BinnedAP(device="cuda")
_gt = (torch.rand(1, 200, 200, device="cuda") > 0.9).float()
_ap.update(_gt.clone(), _gt)                       # perfect probabilities
assert abs(_ap.compute() - 1.0) < 1e-6, _ap.compute()

_ap2 = BinnedAP(device="cuda")
_ap2.update(torch.rand(1, 200, 200, device="cuda"), _gt)   # uninformative
_base = float(_gt.mean())
assert abs(_ap2.compute() - _base) < 0.03, (_ap2.compute(), _base)
print(f"AP        OK  perfect={1.0:.3f}  random={_ap2.compute():.3f} (positive rate={_base:.3f})")

# 4. The GPU sweep's Dice must equal the numpy Dice we report. These are two
#    independent implementations; if they disagree, one of the two headline
#    numbers in the paper is wrong.
_l = torch.randn(3, 64, 64, device="cuda").half()
_g = (torch.rand(3, 64, 64, device="cuda") > 0.7).half()
_sw = sweep_cuts(_l, _g, [0.0])
_np_dice = np.mean([dice_np((_l[i] >= 0).cpu().numpy(), _g[i].cpu().numpy()) for i in range(3)])
# 1e-9, not a loose tolerance: both sides count pixels with exact integer
# arithmetic, so they must agree to double precision. A loose bound here would
# have hidden the fp16-reduction drift this assertion was written to catch.
assert abs(_sw.iloc[0]["mean_dice"] - _np_dice) < 1e-9, (_sw.iloc[0]["mean_dice"], _np_dice)
print(f"sweep     OK  gpu={_sw.iloc[0]['mean_dice']:.9f} == numpy={_np_dice:.9f}")
print("\nAll self-tests passed.")
''')

# ── §6 train ─────────────────────────────────────────────────────────────────
_SIX_BATCH_NOTE = {
    "matched": "Each run's `config.json` records `micro_batch × accum_steps = 8` — the same "
               "effective batch for every model.",
    "per_model": "Each run probes its own largest batch (`accum=1`), so `config.json`'s "
                 "`effective_batch` differs by model — B2 stays small, B0/YOLO climb. Faster, "
                 "but not step-count-matched across sizes.",
}[MODE]

md("""
## §6 · Train — 15 runs

**This is the long cell.** Expect roughly 12–20 GPU-hours in total on an A100
(early stopping usually cuts it well under the 100-epoch worst case). """ + _SIX_BATCH_NOTE + """

**You do not need to finish it in one session.** Every run writes `resume.pt` to
Drive; if Colab disconnects, reconnect, Run All, and it continues from the last
synced epoch. Completed runs are skipped via `DONE.json` and cost nothing to re-run.

Order matters: the B2 teacher for a seed must finish (and be calibrated) before
that seed's distilled students can start. The loop below enforces it.
""")

code(r'''
# Teachers first, per seed: a distilled run needs its OWN seed's calibrated teacher.
RUN_QUEUE = []
for _seed in CFG["seeds"]:
    for _name, _spec in MODELS.items():
        RUN_QUEUE.append((f"{_name}__seed{_seed}", _name, _spec, _seed))
RUN_QUEUE.sort(key=lambda r: (r[3], not r[1].startswith("segformer_b2_teacher"), r[1]))

print(f"{len(RUN_QUEUE)} runs queued:\n")
for _rid, _, _, _ in RUN_QUEUE:
    _status = "DONE" if (RUNS_DIR / _rid / "DONE.json").exists() else "todo"
    print(f"  [{_status}] {_rid}")
''')

code(r'''
CFG_RUN = {**CFG, "img_size": CFG["img_size"]}
results = []
for run_id, name, spec, seed in RUN_QUEUE:
    print(f"\n{'='*72}\n{run_id}\n{'='*72}")
    t0 = time.time()
    try:
        res = train_run(run_id, spec, seed, CFG_RUN, PATHS, MAN, WORK, RUNS_DIR, DEVICE)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        results.append(res)
        print(f"-> {res['status']}: best_val_ap={res.get('best_val_ap', float('nan')):.4f} "
              f"in {res['minutes']} min")
    except torch.cuda.OutOfMemoryError:
        # Do not swallow: an OOM means the batch resolution is wrong for this GPU
        # and every later run would hit it too. Better to stop and be told.
        torch.cuda.empty_cache()
        raise

pd.DataFrame(results).to_csv(OUT_DIR / "training_summary.csv", index=False)
pd.DataFrame(results)
''')

# ── §7 sweep ─────────────────────────────────────────────────────────────────
md(r"""
## §7 · Fit each run's operating point — on **val**

A model emits a probability per pixel; a mask needs a threshold; different
thresholds give different Dice. The threshold is therefore *a parameter we choose*,
and choosing it on test would mean reporting a score tuned on the exam.

So: **val (134) chooses the threshold and is never reported. Test (185) is
reported and never chooses anything.**
""")

code(r'''
CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
sweep_rows = []

for run_id, name, spec, seed in RUN_QUEUE:
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "best.pt").exists():
        print(f"  skip {run_id} (not trained)"); continue

    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=DEVICE, weights_only=True))

    val_loader = make_loader(MAN["val"], WORK, CFG["img_size"], 8, False, CFG["workers"], seed)
    logits, gts, _ = cache_logits(model, val_loader, DEVICE, CFG["amp"])
    grid = sweep_cuts(logits, gts, CUTS)
    grid.to_csv(run_dir / "threshold_sweep.csv", index=False)

    sel = select_cut(grid)
    (run_dir / "operating_point.json").write_text(json.dumps(sel, indent=2))
    sweep_rows.append({"run_id": run_id, "model": name, "seed": seed, **sel})
    print(f"  {run_id:<38} cut={sel['cut']:+.3f} thr={sel['threshold']:.3f} "
          f"val_dice={sel['val_dice_at_cut']:.4f} miss={sel['val_miss_at_cut']:.3f} "
          f"band=[{sel['band_lo_threshold']:.3f},{sel['band_hi_threshold']:.3f}]")

    del model, logits, gts
    torch.cuda.empty_cache()

SWEEP = pd.DataFrame(sweep_rows)
SWEEP.to_csv(OUT_DIR / "operating_points.csv", index=False)
SWEEP[["run_id", "cut", "threshold", "val_peak_dice", "val_dice_at_cut",
       "val_miss_at_cut", "band_lo_threshold", "band_hi_threshold", "band_width_cuts"]]
''')

md(r"""
### How flat is the sweep, really?

If the tie band spans most of the threshold range, the threshold is close to
arbitrary — and that is a finding about the **labels**, not a defect: you cannot
resolve an operating point more finely than the annotations are self-consistent.
""")

code(r'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 3.4), sharey=True)
for ax, (name, _) in zip(axes, MODELS.items()):
    for seed in CFG["seeds"]:
        f = RUNS_DIR / f"{name}__seed{seed}" / "threshold_sweep.csv"
        if f.exists():
            g = pd.read_csv(f)
            ax.plot(g["threshold"], g["mean_dice"], lw=1.2, label=f"seed {seed}")
    row = SWEEP[SWEEP["model"] == name]
    if len(row):
        ax.axvspan(row["band_lo_threshold"].mean(), row["band_hi_threshold"].mean(),
                   color="tab:orange", alpha=0.15, label="tie band (mean)")
        ax.axvline(row["threshold"].mean(), color="k", ls="--", lw=1, label="chosen")
    ax.set_title(name.replace("_", "\n"), fontsize=9)
    ax.set_xlabel("threshold"); ax.grid(alpha=0.3)
axes[0].set_ylabel("val mean Dice"); axes[0].legend(fontsize=7)
plt.suptitle("Val threshold sweeps — a flat curve means the labels, not the cut, are the limit", y=1.02)
plt.tight_layout(); plt.savefig(OUT_DIR / "threshold_sweeps.png", dpi=140, bbox_inches="tight")
plt.show()
''')

# ── §8 test ──────────────────────────────────────────────────────────────────
md(r"""
## §8 · Score on **test** — one pass, operating point already frozen
""")

code(r'''
per_image_all, test_rows = {}, []

for run_id, name, spec, seed in RUN_QUEUE:
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "operating_point.json").exists():
        continue
    cut = json.loads((run_dir / "operating_point.json").read_text())["cut"]

    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=DEVICE, weights_only=True))

    test_loader = make_loader(MAN["test"], WORK, CFG["img_size"], 8, False, CFG["workers"], seed)
    per_img, summ = evaluate_at_cut(model, test_loader, DEVICE, cut, CFG["amp"])
    per_img.to_csv(run_dir / "test_per_image.csv", index=False)
    per_image_all[run_id] = per_img
    test_rows.append({"run_id": run_id, "model": name, "seed": seed, "cut": cut, **summ})
    print(f"  {run_id:<38} dice={summ['mean_dice']:.4f} (med {summ['median_dice']:.4f}) "
          f"miss={summ['complete_miss_rate']*100:.2f}%")

    del model
    torch.cuda.empty_cache()

TEST = pd.DataFrame(test_rows)
TEST.to_csv(OUT_DIR / "test_per_seed.csv", index=False)
TEST
''')

md(r"""
### Aggregate over seeds — this is the headline table

**Read the ± column before the mean.** Three seeds is enough to see whether a
difference between two models is larger than the noise of retraining the same
model. On the previous run of this project it usually was not — and the old
alpha search was selecting differences ~7× smaller than its own seed spread.
""")

code(r'''
agg = (TEST.groupby("model")
       .agg(n_seeds=("seed", "count"),
            mean_dice=("mean_dice", "mean"), std_dice=("mean_dice", "std"),
            median_dice=("median_dice", "mean"), std_median=("median_dice", "std"),
            mean_iou=("mean_iou", "mean"),
            miss_rate=("complete_miss_rate", "mean"), std_miss=("complete_miss_rate", "std"),
            mean_recall=("mean_recall", "mean"))
       .reindex(MODELS.keys()).reset_index())

agg["dice"] = agg.apply(lambda r: f"{r.mean_dice:.4f} ± {r.std_dice:.4f}", axis=1)
agg["miss_%"] = agg.apply(lambda r: f"{r.miss_rate*100:.2f} ± {r.std_miss*100:.2f}", axis=1)
agg.to_csv(OUT_DIR / "test_aggregate.csv", index=False)
print(agg[["model", "n_seeds", "dice", "median_dice", "mean_iou", "miss_%"]].to_string(index=False))
''')

code(r'''
# Does distillation actually beat its direct baseline, or is it inside seed noise?
# Paired over seeds: seed i's distilled run vs seed i's direct run share the split,
# the schedule and the RNG stream, so the pairing removes the seed's common effect
# and the test is about the ONE thing that differs.
from scipy import stats as sps

print("Distillation effect (paired over seeds, test mean Dice)\n" + "-" * 62)
for direct, distilled in [("segformer_b0_direct", "segformer_b0_distilled"),
                          ("yolo_sem_direct", "yolo_sem_distilled")]:
    a = TEST[TEST.model == direct].set_index("seed")["mean_dice"]
    b = TEST[TEST.model == distilled].set_index("seed")["mean_dice"]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        print(f"{distilled}: need >=2 shared seeds"); continue
    d = (b[common] - a[common]).to_numpy()
    t, p = sps.ttest_rel(b[common], a[common])
    verdict = "REAL" if p < 0.05 else "within seed noise"
    print(f"{distilled:<26} delta={d.mean():+.4f} +/- {d.std(ddof=1):.4f}  "
          f"p={p:.3f}  -> {verdict}")
print("\nWith n=3 seeds this test has low power: a 'within seed noise' result means")
print("NOT PROVEN, not 'proven absent'. Report the interval, not just the verdict.")
''')

# ── §8b reduce misses without retraining ─────────────────────────────────────
md(r"""
## §8b · Reducing complete misses **without retraining**

The models are already trained, so this section changes no weight. It asks: given the
five trained families, can we cut the complete-miss rate **without** just lowering the
threshold — which trades misses for false positives one-for-one?

**The distinction that matters.** Lowering the global threshold slides you *along* the
miss-vs-Dice curve. What we want is to move the *curve* — fewer misses at the same Dice
(a Pareto improvement). Three no-retrain techniques, each fitted on **val** and applied
to **test** with the same miss-tie-break rule as the baseline, so any difference is the
technique and not a re-tuned cut:

1. **3-seed ensemble** — average the three seeds' probability maps. A miss now needs
   *all three* seeds to blank the same image, and the per-seed misses are different
   images (direct blanked 1 / 13 / 10). Free, and the honest recommended lever.
2. **Ensemble + TTA** — also average over horizontal/vertical flips, raising the
   probability on borderline images so they clear the threshold without lowering it.
3. **Ensemble + no-blank floor** — if a mask is *still* empty, recover the
   most-confident region instead of returning blank. This **games the miss metric**
   (guarantees a non-zero prediction), so it is reported separately as a floor.

Results go to a **new** folder, `results_v2/miss_reduction/` — nothing existing is
overwritten. Read the Pareto plot: a real win sits **below-and-left** of the
single-seed baseline cloud; a point that just slid down the same curve is the
threshold in disguise.
""")

code(r'''
POST_DIR = OUT_DIR / "miss_reduction"
POST_DIR.mkdir(parents=True, exist_ok=True)
PROB_THRESHOLDS = np.linspace(0.01, 0.99, 197)   # probability-space sweep, fitted on val

post_rows, baseline_points = [], []

for name, spec in MODELS.items():
    seeds = [s for s in CFG["seeds"] if (RUNS_DIR / f"{name}__seed{s}" / "best.pt").exists()]
    if len(seeds) < 1:
        continue

    # Per-seed probability maps on val and test (one forward pass each), plus the
    # single-seed baseline point (its own val-fitted threshold) for the Pareto cloud.
    val_probs, test_probs, val_gts, test_gts, test_stems = [], [], None, None, None
    for s in seeds:
        run_dir = RUNS_DIR / f"{name}__seed{s}"
        model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=DEVICE, weights_only=True))

        vp, vg, vs = probs_plain(model, make_loader(MAN["val"],  WORK, CFG["img_size"], 8, False, CFG["workers"], s), DEVICE, CFG["amp"])
        tp, tg, ts = probs_plain(model, make_loader(MAN["test"], WORK, CFG["img_size"], 8, False, CFG["workers"], s), DEVICE, CFG["amp"])
        val_probs.append((vp, vs)); test_probs.append((tp, ts))
        val_gts, test_gts, test_stems = vg, tg, ts

        op, _, summ = fit_on_val_apply_to_test(vp, vg, tp, tg, ts, PROB_THRESHOLDS)
        baseline_points.append({"model": name, "seed": s, "technique": "single-seed",
                                "mean_dice": summ["mean_dice"], "miss_rate": summ["complete_miss_rate"]})

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if len(seeds) < 2:
        continue    # ensembling needs at least two seeds

    # 1) plain 3-seed ensemble
    ens_val,  _        = mean_over_seeds([p for p, _ in val_probs],  [st for _, st in val_probs])
    ens_test, ens_stem = mean_over_seeds([p for p, _ in test_probs], [st for _, st in test_probs])
    op_e, pi_e, summ_e = fit_on_val_apply_to_test(ens_val, val_gts, ens_test, test_gts, ens_stem, PROB_THRESHOLDS)
    pi_e.to_csv(POST_DIR / f"{name}__ensemble_test_per_image.csv", index=False)

    # 3) ensemble + no-blank floor (re-score the SAME ensemble at the SAME threshold)
    _, summ_nb = score_prob_at(ens_test, test_gts, ens_stem, op_e["threshold"], no_blank=True)

    # 2) ensemble + TTA (extra forward passes over flips; fit + apply both with TTA)
    tta_v, tta_t = [], []
    for s in seeds:
        run_dir = RUNS_DIR / f"{name}__seed{s}"
        model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
        model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=DEVICE, weights_only=True))
        vp, _, vs = probs_tta(model, make_loader(MAN["val"],  WORK, CFG["img_size"], 8, False, CFG["workers"], s), DEVICE, CFG["amp"])
        tp, _, ts = probs_tta(model, make_loader(MAN["test"], WORK, CFG["img_size"], 8, False, CFG["workers"], s), DEVICE, CFG["amp"])
        tta_v.append((vp, vs)); tta_t.append((tp, ts))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    tv, _        = mean_over_seeds([p for p, _ in tta_v], [st for _, st in tta_v])
    tt, tt_stem  = mean_over_seeds([p for p, _ in tta_t], [st for _, st in tta_t])
    op_t, _, summ_t = fit_on_val_apply_to_test(tv, val_gts, tt, test_gts, tt_stem, PROB_THRESHOLDS)

    base = [b for b in baseline_points if b["model"] == name]
    base_dice = float(np.mean([b["mean_dice"] for b in base]))
    base_miss = float(np.mean([b["miss_rate"] for b in base]))
    for tech, summ, thr in [("ensemble", summ_e, op_e["threshold"]),
                            ("ensemble+TTA", summ_t, op_t["threshold"]),
                            ("ensemble+no-blank", summ_nb, op_e["threshold"])]:
        post_rows.append({
            "model": name, "technique": tech, "threshold": thr,
            "mean_dice": summ["mean_dice"], "median_dice": summ["median_dice"],
            "miss_rate": summ["complete_miss_rate"],
            "d_dice_vs_seedmean": summ["mean_dice"] - base_dice,
            "d_miss_vs_seedmean": summ["complete_miss_rate"] - base_miss,
        })
    print(f"{name}: seed-mean miss={base_miss*100:.2f}% dice={base_dice:.4f} | "
          f"ensemble miss={summ_e['complete_miss_rate']*100:.2f}% dice={summ_e['mean_dice']:.4f} | "
          f"+TTA miss={summ_t['complete_miss_rate']*100:.2f}% | +no-blank miss={summ_nb['complete_miss_rate']*100:.2f}%")

POST = pd.DataFrame(post_rows)
pd.DataFrame(baseline_points).to_csv(POST_DIR / "baseline_single_seed_points.csv", index=False)
POST.to_csv(POST_DIR / "miss_reduction_summary.csv", index=False)
POST
''')

code(r'''
# Pareto view: does a technique move the curve, or just slide down it?
import matplotlib.pyplot as plt

base_df = pd.DataFrame(baseline_points)
fig, ax = plt.subplots(figsize=(8, 6))
markers = {"ensemble": "o", "ensemble+TTA": "s", "ensemble+no-blank": "^"}
colors = {name: c for name, c in zip(MODELS, plt.cm.tab10.colors)}

for name in MODELS:
    b = base_df[base_df.model == name]
    if len(b):
        ax.scatter(b.mean_dice, b.miss_rate * 100, marker="x", s=70, color=colors[name], alpha=0.5)
    for _, r in POST[POST.model == name].iterrows():
        ax.scatter(r.mean_dice, r.miss_rate * 100, marker=markers[r.technique], s=110,
                   color=colors[name], edgecolor="k", linewidth=0.6, zorder=3)

# legend: colour = model, marker = technique (x = single-seed baseline)
from matplotlib.lines import Line2D
model_leg = [Line2D([], [], marker="o", ls="", color=colors[n], label=n) for n in MODELS]
tech_leg = ([Line2D([], [], marker="x", ls="", color="grey", label="single-seed (baseline)")]
            + [Line2D([], [], marker=m, ls="", color="grey", mec="k", label=t) for t, m in markers.items()])
l1 = ax.legend(handles=model_leg, loc="upper right", fontsize=8, title="model")
ax.add_artist(l1); ax.legend(handles=tech_leg, loc="lower left", fontsize=8, title="technique")
ax.set_xlabel("mean Dice (higher better) →"); ax.set_ylabel("complete-miss % (lower better) ↓")
ax.set_title("Miss vs Dice — a real win is DOWN-and-RIGHT of the baseline ×'s\n"
             "(down-only with Dice unchanged = moved the curve; down-left = just slid the threshold)")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(POST_DIR / "miss_reduction_pareto.png", dpi=140, bbox_inches="tight")
plt.show()
print("written ->", POST_DIR)
for f in sorted(POST_DIR.glob("*")):
    print("   ", f.name)
''')

md(r"""
**How to judge it.** For each model, compare the technique markers to that model's
single-seed × cloud:

- **down and right** (lower miss, equal-or-higher Dice) → the curve moved; a genuine
  improvement you get for free.
- **down and left** (lower miss, lower Dice) → you slid along the same curve; that's the
  threshold in disguise, not a win.
- **no-blank floor** should always show ~0 miss by construction — treat its Dice, not
  its miss, as the honest signal, and remember it can fire on nothing real.

Expect the ensemble to help SegFormer more than YOLO: YOLO's stride-8 head misses small
faint bruises for architectural reasons no post-processing fixes (§7). If even the
ensemble leaves YOLO's miss rate high, that is the evidence it belongs in the paper as a
*recall-limited* model, and the durable fix is a recall-weighting loss (Focal-Tversky,
Salehi 2017 / Abraham 2018) at the **next** retrain — out of scope here because these
weights are frozen.
""")

# ── §9 fairness ──────────────────────────────────────────────────────────────
md(r"""
## §9 · Fairness across skin tone

This is a forensic tool. A model that finds bruises on light skin and misses them
on dark skin under-documents injuries on exactly the population most likely to need
the documentation. That makes skin tone a **primary result**, not an ablation.

Groups are ITA (Individual Typology Angle, Chardon et al. 1991) — measured from the
image, not guessed by a rater. Test ITA distribution: Dark (VI) 55, Light (II-III) 39,
Intermediate (III-IV) 38, Brown (V) 29, Tan (IV) 24.
""")

code(r'''
# Per-image Dice averaged over seeds first: one run's seed noise is not a property
# of the model, and the fairness question is about the model. n stays 185 images.
fair_per_group, fair_pairwise, fair_stats = [], [], []

for name in MODELS:
    seeds_present = [rid for rid in per_image_all if rid.startswith(name + "__")]
    if not seeds_present:
        continue
    stacked = pd.concat([per_image_all[rid] for rid in seeds_present])
    mean_per_image = (stacked.groupby("stem", as_index=False)
                      .agg({"dice": "mean", "recall": "mean",
                            "pred_positive_pixels": "mean", "gt_positive_pixels": "first"}))
    out = fairness_analysis(mean_per_image, MAN["test"], name)
    fair_per_group.append(out["per_group"])
    fair_pairwise.append(out["pairwise"])
    fair_stats.append(out["stats"])

FAIR_GROUP = pd.concat(fair_per_group, ignore_index=True)
FAIR_PAIR  = pd.concat(fair_pairwise, ignore_index=True)
FAIR_STATS = pd.DataFrame(fair_stats)
for _df, _n in [(FAIR_GROUP, "fairness_per_group"), (FAIR_PAIR, "fairness_pairwise"),
                (FAIR_STATS, "fairness_stats")]:
    _df.to_csv(OUT_DIR / f"{_n}.csv", index=False)

print(FAIR_STATS.to_string(index=False))
print("\nkruskal_p < 0.05 => at least one skin-tone group differs.")
print("fairness_gap = best group's median Dice - worst group's. The p-value says the")
print("gap is real; only the gap says whether it MATTERS.")
''')

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

pivot = FAIR_GROUP.pivot(index="skin_tone_category", columns="model", values="median_dice")
pivot = pivot.reindex(columns=[m for m in MODELS if m in pivot.columns])
pivot.plot.bar(ax=axes[0], width=0.8, rot=20)
axes[0].set_ylabel("median Dice"); axes[0].set_xlabel("")
axes[0].set_title("Median Dice by skin tone (ITA group)")
axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)

pivot_m = FAIR_GROUP.pivot(index="skin_tone_category", columns="model", values="miss_rate") * 100
pivot_m = pivot_m.reindex(columns=[m for m in MODELS if m in pivot_m.columns])
pivot_m.plot.bar(ax=axes[1], width=0.8, rot=20)
axes[1].set_ylabel("complete-miss rate (%)"); axes[1].set_xlabel("")
axes[1].set_title("Complete misses by skin tone — a blank mask is a missed injury")
axes[1].legend(fontsize=7); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout(); plt.savefig(OUT_DIR / "fairness.png", dpi=140, bbox_inches="tight")
plt.show()
''')

# ── §10 benchmark ────────────────────────────────────────────────────────────
md(r"""
## §10 · Speed

640 tensor already on the GPU → 640 mask still on the GPU. Nothing else. Disk,
decode, resize and both host copies are excluded: they are identical for all five
models and would swamp the architectural difference in I/O noise.
""")

code(r'''
# Stage the test images on the GPU once (~0.9 GB), so no loader work is ever timed.
_bench_loader = make_loader(MAN["test"], WORK, CFG["img_size"], 8, False, CFG["workers"])
GPU_IMAGES = torch.cat([x for x, _, _ in _bench_loader]).to(DEVICE)
print("staged", tuple(GPU_IMAGES.shape), f"= {GPU_IMAGES.element_size()*GPU_IMAGES.nelement()/1e9:.2f} GB")

bench_rows = []
for name, spec in MODELS.items():
    rid = f"{name}__seed{CFG['seeds'][0]}"     # speed is a property of the architecture, not the seed
    run_dir = RUNS_DIR / rid
    if not (run_dir / "best.pt").exists():
        continue
    cut = json.loads((run_dir / "operating_point.json").read_text())["cut"]
    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(run_dir / "best.pt"), map_location=DEVICE, weights_only=True))

    torch.cuda.reset_peak_memory_stats(DEVICE)
    b = benchmark_speed(model, GPU_IMAGES, DEVICE, cut, CFG["bench_repeats"], CFG["bench_warmup"])
    b.update({"model": name, "params_M": count_params(model) / 1e6,
              "peak_activation_MB": torch.cuda.max_memory_allocated(DEVICE) / 1e6})
    bench_rows.append(b)
    print(f"  {name:<26} {b['median_ms']:6.2f} ms  {b['fps']:6.1f} FPS  "
          f"p95={b['p95_ms']:.2f}ms  {b['params_M']:.2f}M")
    del model
    torch.cuda.empty_cache()

BENCH = pd.DataFrame(bench_rows)
BENCH.to_csv(OUT_DIR / "benchmark.csv", index=False)
del GPU_IMAGES
torch.cuda.empty_cache()
BENCH
''')

# ── §11 annotation ceiling ───────────────────────────────────────────────────
md(r"""
## §11 · The annotation ceiling — read this before believing any ranking above

Three human experts labelled the same 185 test images. They agree with **each
other** at Dice 0.58–0.75. If the spread between our five models is smaller than
the spread between our humans, then the differences we are ranking are label noise,
and chasing +0.01 Dice is chasing an artefact of who held the mouse.

There is also a **label-standard mismatch that cannot be fixed by retraining**:
train and val masks are *Paul's*, test masks are the *majority consensus*. Paul is
the outlier annotator (≈0.58 with each of the other two, who agree 0.755 with each
other). So we train on one standard, fit the threshold on that same standard, and
score against a different one. Say this in the paper rather than leaving it implicit
in a folder name.
""")

code(r'''
IL = pd.read_csv(WORK / "interlabeler_agreement_640.csv")
human = pd.DataFrame([
    {"comparison": c.replace("_", " ↔ "), "mean_dice": IL[c].mean(), "median_dice": IL[c].median()}
    for c in ["paul_vs_gbarimah", "paul_vs_erik", "gbarimah_vs_erik",
              "paul_vs_majority", "gbarimah_vs_majority", "erik_vs_majority"]
    if c in IL.columns])
print("HUMAN vs HUMAN (and vs the consensus they define), same 185 test images")
print(human.to_string(index=False))

model_lo, model_hi = agg["mean_dice"].min(), agg["mean_dice"].max()
h_lo = human[human.comparison.str.contains("majority")]["mean_dice"].min()
h_hi = human[human.comparison.str.contains("majority")]["mean_dice"].max()
print(f"\nour 5 models vs consensus : {model_lo:.4f} .. {model_hi:.4f}  (spread {model_hi-model_lo:.4f})")
print(f"the 3 humans vs consensus : {h_lo:.4f} .. {h_hi:.4f}  (spread {h_hi-h_lo:.4f})")
if model_hi - model_lo < h_hi - h_lo:
    print("\n=> The models are closer to each other than the annotators are to each other.")
    print("   Lead the paper with complete-miss rate, which separates them by more than")
    print("   label noise. Dice is supporting evidence, not the headline.")
''')

# ── §12 inference demo ───────────────────────────────────────────────────────
md(r"""
## §12 · Qualitative inference

Numbers hide bimodality. These are the images where the models disagree most —
which is where a reader's intuition about the failure mode actually forms.
""")

code(r'''
import cv2

ref = f"{list(MODELS)[0]}__seed{CFG['seeds'][0]}"
spread = pd.DataFrame({rid: per_image_all[rid].set_index("stem")["dice"] for rid in per_image_all})
spread["disagreement"] = spread.max(axis=1) - spread.min(axis=1)
worst = spread.sort_values("disagreement", ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(len(worst), len(MODELS) + 2, figsize=(3 * (len(MODELS) + 2), 3 * len(worst)))
for r, stem in enumerate(worst):
    row = MAN["test"][MAN["test"].stem == stem].iloc[0]
    img = cv2.cvtColor(cv2.imread(str(WORK / row.image_path)), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(WORK / row.mask_path), cv2.IMREAD_GRAYSCALE)
    if gt.ndim == 3:
        gt = gt[..., 0]
    axes[r, 0].imshow(img); axes[r, 0].set_ylabel(f"{stem}\n{row.skin_tone_category}", fontsize=7)
    axes[r, 1].imshow(img); axes[r, 1].imshow(gt > 0, alpha=0.45, cmap="Reds")
    if r == 0:
        axes[r, 0].set_title("image", fontsize=9); axes[r, 1].set_title("consensus GT", fontsize=9)

    for c, (name, spec) in enumerate(MODELS.items()):
        rid = f"{name}__seed{CFG['seeds'][0]}"
        ax = axes[r, c + 2]
        pi = per_image_all.get(rid)
        d = float(pi[pi.stem == stem]["dice"].iloc[0]) if pi is not None and (pi.stem == stem).any() else float("nan")
        m = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE).eval()
        m.load_state_dict(torch.load(str(RUNS_DIR / rid / "best.pt"), map_location=DEVICE, weights_only=True))
        cut = json.loads((RUNS_DIR / rid / "operating_point.json").read_text())["cut"]
        x, _, _ = next(iter(make_loader(MAN["test"][MAN["test"].stem == stem], WORK, CFG["img_size"], 1, False, 0)))
        with torch.no_grad():
            pred = (m(x.to(DEVICE))[0, 0] >= cut).cpu().numpy()
        ax.imshow(img); ax.imshow(pred, alpha=0.45, cmap="Blues")
        ax.set_title(f"{name.replace('segformer_','sf_').replace('yolo_sem_','yolo_')}\nDice {d:.3f}"
                     + ("  BLANK" if pred.sum() == 0 else ""), fontsize=8)
        del m
        torch.cuda.empty_cache()

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Highest-disagreement test images", y=1.001)
plt.tight_layout(); plt.savefig(OUT_DIR / "qualitative.png", dpi=130, bbox_inches="tight")
plt.show()
''')

# ── §13 final ────────────────────────────────────────────────────────────────
md(r"""
## §13 · Final table → Drive
""")

code(r'''
final = (agg.merge(BENCH[["model", "median_ms", "fps", "params_M", "peak_activation_MB"]],
                   on="model", how="left")
            .merge(FAIR_STATS[["model", "fairness_gap", "kruskal_p", "worst_group"]],
                   on="model", how="left")
            .merge(SWEEP.groupby("model")["threshold"].mean().rename("mean_threshold"),
                   on="model", how="left"))

cols = ["model", "params_M", "mean_threshold", "dice", "median_dice", "mean_iou",
        "miss_%", "fairness_gap", "kruskal_p", "worst_group", "median_ms", "fps",
        "peak_activation_MB"]
final = final[[c for c in cols if c in final.columns]]
final.to_csv(OUT_DIR / "FINAL_RESULTS.csv", index=False)

print("FINAL — 185 test images, mean ± std over", len(CFG["seeds"]), "seeds")
print("=" * 118)
print(final.to_string(index=False))
print("\nWritten to:", OUT_DIR)
for f in sorted(OUT_DIR.glob("*")):
    print("   ", f.name)
''')

md(r"""
### How to read this table

1. **`dice` carries a ± for a reason.** If two models' intervals overlap, you have
   not shown a difference between them. Say so.
2. **`miss_%` is the axis that separates the models** by more than label noise. For
   an injury-documentation tool a blank mask is a missed injury, so a model with
   9% blanks is disqualified regardless of its FPS.
3. **`fairness_gap` is an effect size; `kruskal_p` only says the gap is real.** A
   significant p with a 0.02 gap is a curiosity. A 0.3 gap is the paper's headline
   whether or not p clears 0.05.
4. **Nothing here beats the annotation ceiling** (§11). If the model spread sits
   inside the human spread, the ranking is about labels, not architectures.
""")

# ── build ────────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}  ({len(cells)} cells: {n_code} code, {len(cells)-n_code} markdown)")
