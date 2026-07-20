#!/usr/bin/env python3
"""
scripts/43_generate_final_notebook.py  ->  bruise_colab_final.ipynb

The FINAL notebook. Differs from 41 (bruise_colab_train_all / _per_model_batch) in
exactly the way the user asked for:

  - SegFormer: the SAME custom-loop pipeline as 41 (its own recipe, per-model best
    batch), distilled at alpha=0.6 -- matching the project's established value.
  - YOLO: trained with NATIVE Ultralytics `.train()` (mosaic, EMA, letterbox, its
    own LR) -- NOT the custom loop, NOT SegFormer's LR. Distilled at alpha=0.4
    (offline pseudo-mask KD; below 0.5 so it is non-degenerate).
  - YOLO is then evaluated TWO ways on all 185 test images:
      (a) native `.predict()` argmax  -- YOLO's home turf, reproduces ~0.83 median;
      (b) custom /255 raw-module + sigmoid + val-swept threshold -- scored the same
          geometry as SegFormer.
  - Fairness across ITA skin-tone groups for every model and both YOLO paths.

To guarantee the SegFormer half cannot drift from the tested version, this script
REUSES the bruisekit module bodies verbatim out of the already-generated
bruise_colab_train_all.ipynb (data, models, losses, metrics, engine, sweep,
evaluate, postopt) and only ADDS one new module (yolo_native) plus its own
orchestration cells. Regenerate 41 first if those modules change.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NB = PROJECT_ROOT / "bruise_colab_train_all.ipynb"   # tested modules live here
OUT = PROJECT_ROOT / "bruise_colab_final.ipynb"

REUSE_MODULES = ["__init__", "data", "models", "losses", "metrics", "engine", "sweep", "evaluate", "postopt"]

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": dedent(text).strip("\n").splitlines(keepends=True)})


def raw_code(text: str) -> None:
    """Emit a code cell without dedent/strip (for reused module cells, verbatim)."""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


def writefile(path: str, body: str) -> None:
    code(f"%%writefile {path}\n" + dedent(body).strip("\n"))


def reused_module_cells() -> dict[str, str]:
    """Pull the `%%writefile bruisekit/<name>.py` cell sources out of the tested notebook."""
    if not SOURCE_NB.exists():
        raise SystemExit(f"{SOURCE_NB} not found -- run scripts/41_generate_training_notebook.py first.")
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
md(r"""
# Bruise segmentation — FINAL: SegFormer (custom loop) + native YOLO, all 185 test images

One notebook, five models, three seeds each. **SegFormer** trains with its own custom
loop at each model's best batch; **YOLO** trains with **native Ultralytics** and is
scored **two ways**. Fairness across skin tone for everything. Survives session death.

| model | trained by | batch | distill α | evaluated |
|---|---|---|---|---|
| SegFormer-B2 teacher | custom loop | per-model best (~32) | — | sigmoid + val-swept thr |
| SegFormer-B0 direct | custom loop | per-model best (~64) | — | sigmoid + val-swept thr |
| SegFormer-B0 distilled | custom loop | per-model best (~64) | **0.6** | sigmoid + val-swept thr |
| YOLO26n direct | **native Ultralytics** | native auto | — | **(a) argmax  (b) /255 swept** |
| YOLO26n distilled | **native Ultralytics** | native auto | **0.4** | **(a) argmax  (b) /255 swept** |

**Why this split (read `docs/training_notebook_explained.md` for the full story):**
- SegFormer keeps its own recipe — it's the healthy, reproducible pipeline.
- YOLO is trained **natively** (mosaic + EMA + letterbox + its own LR), *not* forced
  onto SegFormer's transformer recipe — because a CNN and a transformer don't share an
  optimal LR, and native YOLO is what reproduces its ~0.83 median.
- YOLO gets **two evaluation paths** so you see both its best case (native argmax) and
  the SegFormer-comparable geometry (custom /255). The miss rate is the honest axis.

**Data:** the zip ships **native-resolution** images. At setup we build a 640-stretch
cache **once** (for SegFormer training + the custom-YOLO path); native YOLO trains
straight off the native images so Ultralytics' letterbox reproduces its result.
""")

# ── §1 config ────────────────────────────────────────────────────────────────
md("## §1 · Configuration")

code(r'''
CFG = dict(
    img_size        = 640,
    zip_name        = "bruise_colab_final.zip",
    drive_dir       = "/content/drive/MyDrive/bruise_segmentation_gpu",
    work_dir        = "/content/bruise_final",          # local SSD, wiped on disconnect

    # ── SegFormer recipe (custom loop, unchanged) ───────────────────────────
    epochs          = 100,
    patience        = 15,
    batch_mode      = "per_model",   # each SegFormer model uses its own largest batch
    effective_batch = 8,             # fallback only; per_model probes up to max_probe_batch
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
    segformer_alpha = 0.6,           # SegFormer distillation weight (project's value)
    aux_weight      = 0.4,

    # ── YOLO recipe (NATIVE Ultralytics, its own settings) ──────────────────
    yolo_alpha      = 0.4,           # offline pseudo-mask KD (below 0.5 -> non-degenerate)
    yolo_batch      = -1,            # -1 = Ultralytics auto-batch (native "largest that fits")
    yolo_optimizer  = "auto",
    yolo_lrf        = 0.01,
    yolo_warmup_epochs = 3,
    yolo_weight_decay  = 0.0005,
    yolo_close_mosaic  = 10,
    pseudo_threshold   = 0.50,

    # ── shared ──────────────────────────────────────────────────────────────
    seeds           = (0, 1, 2),
    cut_min         = -6.0, cut_max = 6.0, cut_steps = 481,   # SegFormer logit-cut sweep
    prob_min        = 0.01, prob_max = 0.99, prob_steps = 197, # YOLO custom-path prob sweep
    drive_sync_every = 2,
)

SEGFORMER_MODELS = {
    "segformer_b2_teacher":   dict(arch="segformer", size="b2", distill=False),
    "segformer_b0_direct":    dict(arch="segformer", size="b0", distill=False),
    "segformer_b0_distilled": dict(arch="segformer", size="b0", distill=True),
}
YOLO_MODELS = {
    "yolo_sem_direct":    dict(distill=False),
    "yolo_sem_distilled": dict(distill=True),
}
print(f"{len(SEGFORMER_MODELS)} SegFormer + {len(YOLO_MODELS)} YOLO models x {len(CFG['seeds'])} seeds")
''')

# ── §2 env ───────────────────────────────────────────────────────────────────
md("## §2 · Drive, GPU, dependencies")

code(r'''
import os, sys, time
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Runtime -> Change runtime type -> GPU (A100). Refusing to run on CPU.")
print("GPU:", torch.cuda.get_device_name(0))
''')

code(r'''
%pip install -q "transformers>=4.40,<6" "ultralytics>=8.4,<9" "albumentations>=2.0,<3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.7" "pyyaml"
import transformers, ultralytics, albumentations
print("transformers", transformers.__version__, "| ultralytics", ultralytics.__version__)
''')

# ── §3 unzip + build 640 cache ───────────────────────────────────────────────
md(r"""
## §3 · Unpack (native-res) and build the 640 cache once

The zip is native resolution (~2.7 GB). We unzip to local SSD, then build a
640×640 stretch cache **once** — SegFormer trains on that (fast, no per-epoch
resize) and the custom-YOLO path reads its 640 images from it. Native YOLO uses the
native-res images directly.
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
    assert not (set(MAN[a]["stem"]) & set(MAN[b]["stem"])), f"image leak {a}/{b}"
assert (len(MAN["train"]),len(MAN["val"]),len(MAN["test"]))==(697,134,185)
print("PASS -- no leakage.")

RUNS_DIR = Path(CFG["drive_dir"]) / "runs_final"
OUT_DIR  = Path(CFG["drive_dir"]) / "results_final"
RUNS_DIR.mkdir(parents=True, exist_ok=True); OUT_DIR.mkdir(parents=True, exist_ok=True)
print("checkpoints ->", RUNS_DIR)
''')

code(r'''
# Build the 640 stretch cache once (image bilinear, mask nearest -- albumentations'
# defaults, so this is bit-exact to what the training dataloader would compute live).
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

# Manifests that point at the 640 cache (relative to CACHE640) -- SegFormer + custom-YOLO use these.
MAN640 = {}
for s, df in MAN.items():
    d = df.copy()
    d["image_path"] = d["stem"].apply(lambda x: f"{s}/images/{x}.png")
    d["mask_path"]  = d["stem"].apply(lambda x: f"{s}/masks/{x}.png")
    MAN640[s] = d
''')

# ── §4 modules ───────────────────────────────────────────────────────────────
md(r"""
## §4 · The library

SegFormer modules are **reused verbatim** from `bruise_colab_train_all.ipynb` (tested
end-to-end). One new module, `yolo_native.py`, wraps native Ultralytics training and
the two YOLO prediction paths.
""")

code(r'''
%cd /content
from pathlib import Path
Path("/content/bruisekit").mkdir(parents=True, exist_ok=True)
print("package dir ready:", Path("/content/bruisekit").is_dir())
''')

# reused modules, verbatim
_reused = reused_module_cells()
for _name in REUSE_MODULES:
    raw_code(_reused[_name])

# new native-YOLO module
writefile("bruisekit/yolo_native.py", r'''
"""Native Ultralytics YOLO training + the two prediction paths.

This module deliberately uses Ultralytics (mosaic, EMA, letterbox, its own recipe) --
that is the whole point: YOLO is trained on its home turf, not on SegFormer's
transformer recipe. It is the ONLY module that imports ultralytics at train time.

TWO PREDICTION PATHS, and why both:
  native argmax   -- YOLO.predict() letterboxes to 640, argmaxes the 2-class head,
                     returns the mask at native resolution. We bring pred and GT to
                     640 (nearest) together and score. This is YOLO's best case and
                     reproduces its ~0.83 median.
  custom /255     -- pull the raw nn.Module, feed the SAME 640 stretch tensor SegFormer
                     sees (just /255, no ImageNet norm), take sigmoid(z1 - z0), and
                     sweep the threshold on val. Same geometry as SegFormer, so it is
                     the apples-to-apples-on-eval number.
"""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from bruisekit.metrics import compute_image_row, summarize
from bruisekit.sweep import select_cut


def _ul_device() -> str:
    """Ultralytics device string: '0' on GPU (Colab), 'cpu' otherwise (local tests).

    Derived, not hardcoded, so the same module runs on the A100 in Colab and on a CPU
    box for verification without a code change -- a hardcoded '0' raises on CPU.
    """
    return "0" if torch.cuda.is_available() else "cpu"


def _read_native_mask(root: Path, rel: str) -> np.ndarray:
    m = cv2.imread(str(root / rel), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Cannot read mask: {rel}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def _to_640(arr: np.ndarray) -> np.ndarray:
    return cv2.resize(arr.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST)


# ── dataset build (native-res, Ultralytics semantic format) ──────────────────
def build_yolo_dataset(df, work_root: Path, out_dir: Path, split: str,
                       alpha=None, teacher_prob_fn=None, imgsz: int = 640) -> None:
    """images/<split>/ (symlink to native) + masks/<split>/ (0/1 class PNG, native res).

    For a distilled TRAIN split, the pseudo-mask fuses the teacher's native-resolution
    probability: class = (alpha*GT + (1-alpha)*teacher_prob >= 0.5). alpha < 0.5 keeps
    this non-degenerate (alpha > 0.5 would collapse it to plain GT). Val/test always use
    clean GT.
    """
    img_dir = out_dir / "images" / split; img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir = out_dir / "masks" / split;  msk_dir.mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        src = (work_root / r.image_path).resolve()
        dst = img_dir / Path(r.image_path).name
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        gt = _read_native_mask(work_root, r.mask_path).astype(np.float32)
        if split == "train" and teacher_prob_fn is not None:
            prob = teacher_prob_fn(work_root / r.image_path, gt.shape)
            cls = ((alpha * gt + (1.0 - alpha) * prob) >= 0.5).astype(np.uint8)
        else:
            cls = gt.astype(np.uint8)
        cv2.imwrite(str(msk_dir / f"{r.stem}.png"), cls)


def write_data_yaml(out_dir: Path, run_dir: Path) -> Path:
    import yaml
    data = {"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
            "masks_dir": "masks", "names": {0: "background", 1: "bruise"}}
    p = run_dir / "data.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return p


def train_native(weights_path: str, data_yaml: Path, run_dir: Path, cfg: dict, seed: int) -> Path:
    """Native Ultralytics training. Skips if best.pt exists; resumes from last.pt if
    interrupted (Ultralytics writes last.pt every epoch, so a disconnect costs <=1 epoch)."""
    from ultralytics import YOLO
    best = run_dir / "ultralytics_runs" / "train" / "weights" / "best.pt"
    last = run_dir / "ultralytics_runs" / "train" / "weights" / "last.pt"
    if best.exists():
        return best
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return best
    YOLO(weights_path).train(
        data=str(data_yaml), task="semantic", imgsz=cfg["img_size"], epochs=cfg["epochs"],
        patience=cfg["patience"], batch=cfg["yolo_batch"], workers=cfg["workers"],
        device=_ul_device(), project=str(run_dir / "ultralytics_runs"), name="train", exist_ok=True,
        optimizer=cfg["yolo_optimizer"], lrf=cfg["yolo_lrf"], cos_lr=True,
        warmup_epochs=cfg["yolo_warmup_epochs"], weight_decay=cfg["yolo_weight_decay"],
        close_mosaic=cfg["yolo_close_mosaic"], seed=seed,
    )
    return best


# ── path (a): native argmax ──────────────────────────────────────────────────
def predict_native_argmax(best_pt: Path, df, work_root: Path, imgsz: int = 640):
    """YOLO.predict() argmax, pred+GT brought to 640 together and scored."""
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    rows = []
    for _, r in df.iterrows():
        res = model.predict(str(work_root / r.image_path), imgsz=imgsz, device=_ul_device(), verbose=False)[0]
        if getattr(res, "semantic_mask", None) is not None:
            cm = res.semantic_mask.data
            cm = cm.cpu().numpy() if hasattr(cm, "cpu") else np.asarray(cm)
            pred = (cm == 1).astype(np.uint8)
        else:
            pred = np.zeros((imgsz, imgsz), np.uint8)
        gt = _read_native_mask(work_root, r.mask_path)
        rows.append(compute_image_row(_to_640(pred), _to_640(gt), str(r.stem)))
    return pd.DataFrame(rows), summarize(rows)


# ── path (b): custom /255, raw module, val-swept threshold ───────────────────
def _raw_module(best_pt: Path, device):
    from ultralytics import YOLO
    return copy.deepcopy(YOLO(str(best_pt)).model).to(device).eval()


@torch.no_grad()
def _probs_640(model, df640, cache_root: Path, device):
    """sigmoid(z1 - z0) at 640, feeding the 640-stretch /255 tensor (SegFormer geometry)."""
    P, G, S = [], [], []
    for _, r in df640.iterrows():
        bgr = cv2.imread(str(cache_root / r.image_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.shape[-2:] != (640, 640):
            out = F.interpolate(out.float(), size=(640, 640), mode="bilinear", align_corners=False)
        z = out[:, 1] - out[:, 0] if out.shape[1] >= 2 else out[:, 0]
        P.append(torch.sigmoid(z)[0].half().cpu())
        m = cv2.imread(str(cache_root / r.mask_path), cv2.IMREAD_GRAYSCALE)
        if m.ndim == 3: m = m[..., 0]
        G.append(torch.from_numpy((m > 0)))
        S.append(str(r.stem))
    return torch.stack(P), torch.stack(G), S


def predict_custom_255(best_pt: Path, val640, test640, cache_root: Path, device, thresholds):
    """Sweep the probability threshold on val, apply once to test. Same scoring as SegFormer."""
    from bruisekit.postopt import sweep_prob, score_prob_at
    model = _raw_module(best_pt, device)
    vp, vg, _ = _probs_640(model, val640, cache_root, device)
    grid = sweep_prob(vp, vg, thresholds)
    op = select_cut(grid)
    tp, tg, ts = _probs_640(model, test640, cache_root, device)
    per_img, summ = score_prob_at(tp, tg, ts, op["threshold"])
    return op, per_img, summ
''')

# ── §5 imports + self-test ───────────────────────────────────────────────────
md("## §5 · Import and self-test")

code(r'''
sys.path.insert(0, "/content")
import importlib
import bruisekit.data, bruisekit.models, bruisekit.losses, bruisekit.metrics
import bruisekit.engine, bruisekit.sweep, bruisekit.evaluate, bruisekit.postopt, bruisekit.yolo_native
for m in ("data","models","losses","metrics","engine","sweep","evaluate","postopt","yolo_native"):
    importlib.reload(sys.modules[f"bruisekit.{m}"])

from bruisekit.data import make_loader
from bruisekit.engine import train_run, load_teacher
from bruisekit.evaluate import evaluate_at_cut, fairness_analysis, benchmark_speed
from bruisekit.models import build_model, count_params
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts
from bruisekit.metrics import dice_np
import bruisekit.yolo_native as yn

import json, time
import numpy as np, torch, pandas as pd

PATHS = {
    "segformer_b0": str(WORK / "pretrained_weights" / "segformer_mit_b0"),
    "segformer_b2": str(WORK / "pretrained_weights" / "segformer_mit_b2"),
    "yolo":         str(WORK / "pretrained_weights" / "yolo26n-sem.pt"),
}
DEVICE = torch.device("cuda:0")
''')

code(r'''
# SegFormer sanity: emits [B,1,640,640], right param counts. (YOLO trains natively, so
# no YoloSemNet here.) Loader reads the 640 cache and emits raw [0,1] pixels.
_ld = make_loader(MAN640["val"].head(4), CACHE640, CFG["img_size"], 2, False, 0)
_x, _y, _s = next(iter(_ld))
assert _x.shape == (2,3,640,640) and _y.shape == (2,1,640,640), (_x.shape, _y.shape)
assert 0.0 <= float(_x.min()) and float(_x.max()) <= 1.0
for _a,_sz,_e in [("segformer","b0",3.71),("segformer","b2",27.35)]:
    _m = build_model(_a,_sz,PATHS).to(DEVICE).eval()
    with torch.no_grad(): _z = _m(_x.to(DEVICE))
    assert _z.shape==(2,1,640,640) and abs(count_params(_m)/1e6-_e)<0.05
    print(f"{_a}-{_sz} OK {count_params(_m)/1e6:.2f}M"); del _m
torch.cuda.empty_cache()

# native argmax + custom paths are exercised for real in §10; here just confirm imports.
assert hasattr(yn, "train_native") and hasattr(yn, "predict_native_argmax") and hasattr(yn, "predict_custom_255")
print("yolo_native OK; all imports resolved.")
''')

# ── §6 SegFormer training ────────────────────────────────────────────────────
md(r"""
## §6 · Train SegFormer (custom loop, per-model batch, 3 seeds)

B2 teacher first per seed (it calibrates the teacher its distilled student needs).
Resumable: `resume.pt` every 2 epochs, `DONE.json` skips finished runs.
""")

code(r'''
SEG_QUEUE = []
for _seed in CFG["seeds"]:
    for _name, _spec in SEGFORMER_MODELS.items():
        SEG_QUEUE.append((f"{_name}__seed{_seed}", _name, _spec, _seed))
SEG_QUEUE.sort(key=lambda r: (r[3], not r[1].startswith("segformer_b2_teacher"), r[1]))

seg_results = []
for run_id, name, spec, seed in SEG_QUEUE:
    print(f"\n{'='*70}\n{run_id}\n{'='*70}")
    cfg_run = {**CFG, "alpha": CFG["segformer_alpha"]}
    t0 = time.time()
    res = train_run(run_id, spec, seed, cfg_run, PATHS, MAN640, CACHE640, RUNS_DIR, DEVICE)
    res["minutes"] = round((time.time()-t0)/60, 1)
    seg_results.append(res)
    print(f"-> {res['status']} best_val_ap={res.get('best_val_ap', float('nan')):.4f}")
pd.DataFrame(seg_results)
''')

# ── §7 SegFormer sweep + test ────────────────────────────────────────────────
md("## §7 · SegFormer — fit threshold on val, score on test (185)")

code(r'''
CUTS = np.linspace(CFG["cut_min"], CFG["cut_max"], CFG["cut_steps"])
seg_test_rows = []; seg_per_image = {}
for run_id, name, spec, seed in SEG_QUEUE:
    rd = RUNS_DIR / run_id
    if not (rd / "best.pt").exists(): continue
    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(rd/"best.pt"), map_location=DEVICE, weights_only=True))
    logits, gts, _ = cache_logits(model, make_loader(MAN640["val"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed), DEVICE, CFG["amp"])
    grid = sweep_cuts(logits, gts, CUTS); sel = select_cut(grid)
    (rd/"operating_point.json").write_text(json.dumps(sel, indent=2))
    pi, summ = evaluate_at_cut(model, make_loader(MAN640["test"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed), DEVICE, sel["cut"], CFG["amp"])
    pi.to_csv(rd/"test_per_image.csv", index=False)
    seg_per_image[run_id] = pi
    seg_test_rows.append({"run_id": run_id, "model": name, "seed": seed, "cut": sel["cut"], **summ})
    print(f"  {run_id:<32} dice={summ['mean_dice']:.4f} miss={summ['complete_miss_rate']*100:.2f}%")
    del model, logits, gts; torch.cuda.empty_cache()
SEG_TEST = pd.DataFrame(seg_test_rows)
SEG_TEST.to_csv(OUT_DIR/"segformer_test_per_seed.csv", index=False)
SEG_TEST
''')

# ── §8 YOLO native training ──────────────────────────────────────────────────
md(r"""
## §8 · Train YOLO — **native Ultralytics** (mosaic, EMA, letterbox, its own LR)

Direct and distilled, 3 seeds. Distilled uses **offline pseudo-mask KD at α=0.4**: the
same-seed calibrated B2 teacher's native-resolution probability is fused into the hard
class mask before Ultralytics ever sees it (that's the only way its trainer can consume
a teacher — see the docs). Native auto-batch. Resumes via Ultralytics' `last.pt`.
""")

code(r'''
def make_teacher_prob_fn(seed):
    """Native-res teacher probability from the same-seed calibrated B2, for pseudo-masks."""
    teacher = load_teacher(RUNS_DIR / f"segformer_b2_teacher__seed{seed}", PATHS, DEVICE, CFG["amp"])
    def fn(img_path, native_hw):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (CFG["img_size"], CFG["img_size"]), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
        x = torch.from_numpy(((r-mean)/std).transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        prob = teacher(x)[0,0].float().cpu().numpy()   # teacher() already applies calibrated T + sigmoid
        return np.clip(cv2.resize(prob, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_LINEAR), 0, 1)
    return fn

yolo_summ = []
for seed in CFG["seeds"]:
    for name, spec in YOLO_MODELS.items():
        run_id = f"{name}__seed{seed}"; rd = RUNS_DIR / run_id; rd.mkdir(parents=True, exist_ok=True)
        best = rd / "ultralytics_runs" / "train" / "weights" / "best.pt"
        print(f"\n{'='*70}\n{run_id}  (native Ultralytics)\n{'='*70}")
        if not best.exists():
            data_dir = rd / "yolo_data"
            if not (rd / "DATASET_DONE.json").exists():
                tfn = make_teacher_prob_fn(seed) if spec["distill"] else None
                yn.build_yolo_dataset(MAN["train"], WORK, data_dir, "train",
                                      alpha=CFG["yolo_alpha"], teacher_prob_fn=tfn, imgsz=CFG["img_size"])
                yn.build_yolo_dataset(MAN["val"], WORK, data_dir, "val", imgsz=CFG["img_size"])
                (rd/"DATASET_DONE.json").write_text(json.dumps({"alpha": CFG["yolo_alpha"] if spec["distill"] else None}))
                torch.cuda.empty_cache()
            yn.write_data_yaml(data_dir, rd)
        t0 = time.time()
        best = yn.train_native(PATHS["yolo"], rd/"data.yaml", rd, CFG, seed)
        yolo_summ.append({"run_id": run_id, "trained": best.exists(), "minutes": round((time.time()-t0)/60,1)})
        print(f"-> best.pt exists: {best.exists()}")
pd.DataFrame(yolo_summ)
''')

# ── §9 YOLO two-path evaluation ──────────────────────────────────────────────
md(r"""
## §9 · Evaluate YOLO **two ways** on all 185 test images

- **native argmax** — `.predict()`, YOLO's home turf (reproduces ~0.83 median)
- **custom /255** — raw module, same 640 geometry as SegFormer, threshold swept on val
""")

code(r'''
PROB_THR = np.linspace(CFG["prob_min"], CFG["prob_max"], CFG["prob_steps"])
yolo_test_rows = []; yolo_per_image = {}
for seed in CFG["seeds"]:
    for name, spec in YOLO_MODELS.items():
        run_id = f"{name}__seed{seed}"; rd = RUNS_DIR / run_id
        best = rd / "ultralytics_runs" / "train" / "weights" / "best.pt"
        if not best.exists(): continue

        # (a) native argmax
        pi_a, summ_a = yn.predict_native_argmax(best, MAN["test"], WORK, CFG["img_size"])
        pi_a.to_csv(rd/"test_per_image_native_argmax.csv", index=False)
        yolo_per_image[f"{run_id}::native_argmax"] = pi_a
        yolo_test_rows.append({"run_id": run_id, "model": name, "seed": seed, "path": "native_argmax", **summ_a})

        # (b) custom /255 (swept on val)
        op, pi_b, summ_b = yn.predict_custom_255(best, MAN640["val"], MAN640["test"], CACHE640, DEVICE, PROB_THR)
        pi_b.to_csv(rd/"test_per_image_custom255.csv", index=False)
        yolo_per_image[f"{run_id}::custom255"] = pi_b
        yolo_test_rows.append({"run_id": run_id, "model": name, "seed": seed, "path": "custom255", "threshold": op["threshold"], **summ_b})
        print(f"  {run_id:<28} argmax dice={summ_a['mean_dice']:.4f} miss={summ_a['complete_miss_rate']*100:.1f}% | "
              f"custom dice={summ_b['mean_dice']:.4f} miss={summ_b['complete_miss_rate']*100:.1f}%")
        torch.cuda.empty_cache()
YOLO_TEST = pd.DataFrame(yolo_test_rows)
YOLO_TEST.to_csv(OUT_DIR/"yolo_test_per_seed.csv", index=False)
YOLO_TEST
''')

# ── §10 combined headline ────────────────────────────────────────────────────
md("## §10 · Combined headline table (mean ± std over 3 seeds)")

code(r'''
def agg(df, key):
    g = df.groupby(key).agg(mean_dice=("mean_dice","mean"), std_dice=("mean_dice","std"),
                            median_dice=("median_dice","mean"), miss=("complete_miss_rate","mean"),
                            miss_std=("complete_miss_rate","std")).reset_index()
    return g

seg_agg = agg(SEG_TEST, "model"); seg_agg["variant"] = seg_agg["model"]
yolo_agg = agg(YOLO_TEST.assign(mk=YOLO_TEST.model+" ("+YOLO_TEST.path+")"), "mk").rename(columns={"mk":"variant"})
FINAL = pd.concat([seg_agg[["variant","mean_dice","std_dice","median_dice","miss","miss_std"]],
                   yolo_agg[["variant","mean_dice","std_dice","median_dice","miss","miss_std"]]], ignore_index=True)
FINAL["dice"] = FINAL.apply(lambda r: f"{r.mean_dice:.4f} ± {r.std_dice:.4f}", axis=1)
FINAL["miss_%"] = FINAL.apply(lambda r: f"{r.miss*100:.2f} ± {r.miss_std*100:.2f}", axis=1)
FINAL.to_csv(OUT_DIR/"FINAL_RESULTS.csv", index=False)
print("185 test images, 3 seeds\n" + "="*70)
print(FINAL[["variant","dice","median_dice","miss_%"]].to_string(index=False))
''')

# ── §11 fairness ─────────────────────────────────────────────────────────────
md(r"""
## §11 · Fairness across skin tone (ITA) — every model, both YOLO paths

Per-image Dice averaged over seeds first, then stratified by ITA group. Kruskal–Wallis
(omnibus) + Mann–Whitney (Bonferroni over 10 pairs) + the fairness gap (best − worst
group median). A blank mask on dark skin under-documents an injury, so skin tone is a
primary result, not an ablation.
""")

code(r'''
def per_image_over_seeds(prefix_keys, store):
    frames = [store[k] for k in prefix_keys if k in store]
    if not frames: return None
    stacked = pd.concat(frames)
    return (stacked.groupby("stem", as_index=False)
            .agg({"dice":"mean","recall":"mean","pred_positive_pixels":"mean","gt_positive_pixels":"first"}))

variants = {}
for name in SEGFORMER_MODELS:
    variants[name] = [f"{name}__seed{s}" for s in CFG["seeds"]]
seg_variant_store = seg_per_image
for name in YOLO_MODELS:
    variants[f"{name} (native_argmax)"] = [f"{name}__seed{s}::native_argmax" for s in CFG["seeds"]]
    variants[f"{name} (custom255)"]     = [f"{name}__seed{s}::custom255" for s in CFG["seeds"]]

fair_group, fair_pair, fair_stats = [], [], []
for variant, keys in variants.items():
    store = seg_per_image if variant in SEGFORMER_MODELS else yolo_per_image
    mpi = per_image_over_seeds(keys, store)
    if mpi is None: continue
    out = fairness_analysis(mpi, MAN["test"], variant)
    fair_group.append(out["per_group"]); fair_pair.append(out["pairwise"]); fair_stats.append(out["stats"])
FAIR_GROUP = pd.concat(fair_group, ignore_index=True)
FAIR_STATS = pd.DataFrame(fair_stats)
FAIR_GROUP.to_csv(OUT_DIR/"fairness_per_group.csv", index=False)
pd.concat(fair_pair, ignore_index=True).to_csv(OUT_DIR/"fairness_pairwise.csv", index=False)
FAIR_STATS.to_csv(OUT_DIR/"fairness_stats.csv", index=False)
print(FAIR_STATS.to_string(index=False))
''')

code(r'''
import matplotlib.pyplot as plt
pivot = FAIR_GROUP.pivot_table(index="skin_tone_category", columns="model", values="miss_rate") * 100
fig, ax = plt.subplots(figsize=(13,5))
pivot.plot.bar(ax=ax, width=0.85, rot=20)
ax.set_ylabel("complete-miss rate (%)"); ax.set_xlabel("")
ax.set_title("Complete misses by skin tone — a blank mask is a missed injury")
ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DIR/"fairness_miss.png", dpi=140, bbox_inches="tight"); plt.show()
''')

# ── §12 annotation ceiling + finish ──────────────────────────────────────────
md("## §12 · Annotation ceiling & saved outputs")

code(r'''
IL = pd.read_csv(WORK / "interlabeler_agreement_640.csv")
human = pd.DataFrame([{"comparison": c.replace("_"," ↔ "), "mean_dice": IL[c].mean()}
    for c in ["paul_vs_gbarimah","paul_vs_erik","gbarimah_vs_erik","paul_vs_majority","gbarimah_vs_majority","erik_vs_majority"]
    if c in IL.columns])
print("HUMAN vs HUMAN (same 185 test images):")
print(human.to_string(index=False))
print("\nModel spread sits inside the human spread -> lead with complete-miss rate, not Dice.")
print("\nAll outputs ->", OUT_DIR)
for f in sorted(OUT_DIR.glob("*")): print("   ", f.name)
''')

md(r"""
### How to read this notebook's results

1. **SegFormer** is the healthy, reproducible pipeline — its numbers are the paper's backbone.
2. **YOLO native argmax** is YOLO's *best case* (~0.83 median); **YOLO custom /255** is the
   *same-geometry-as-SegFormer* number (lower). Report both and be explicit which is which.
3. **Miss rate is the honest axis** — it separates the models by more than label noise, and
   it's what matters for injury documentation. Even at its best case, YOLO blanks several %.
4. **Nothing here beats the annotation ceiling** — the model spread is inside the human spread.
""")

# ── build ────────────────────────────────────────────────────────────────────
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
