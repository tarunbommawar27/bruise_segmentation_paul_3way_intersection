#!/usr/bin/env python3
"""
scripts/44_generate_final_analysis_notebook.py  ->  bruise_colab_final_analysis.ipynb

The ANALYSIS notebook for the FINAL run (results_final / runs_final). It is the
`bruise_colab_analysis.ipynb` idea re-pointed at the final pipeline and greatly
expanded: fairness (ITA) and bruise-size are first-class sections, alongside the
annotation-ceiling / miss-rate / distribution / bootstrap analyses.

HOW IT GETS ITS NUMBERS (user's choice)
---------------------------------------
Colab **re-inference of the val-selected best seed** per model -- NOT a fresh
sweep, and NOT a pure read of the on-disk CSVs. For each model we read the seed
already chosen on val (results_final/best_seed_val_selected/...results.csv),
load exactly that seed's checkpoint from runs_final/, and predict once on all
185 test images. SegFormer is scored at its val-fitted logit cut; YOLO is scored
BOTH ways (native argmax + custom /255). Everything downstream -- tables,
figures, fairness, size, gallery -- is derived from that single best-seed
inference pass, so the whole notebook is internally consistent.

WHY IT REUSES bruisekit VERBATIM
--------------------------------
The SegFormer + YOLO inference code is pulled, byte-for-byte, out of the tested
`bruise_colab_final.ipynb` (data, models, metrics, evaluate, sweep, yolo_native
and their deps). Inference therefore cannot drift from the run that produced
results_final. Only the analysis/plotting cells are new. EDIT THIS GENERATOR,
not the .ipynb.

NO EMBEDDED TABLES
------------------
Unlike the old analysis notebook, subject IDs and ITA skin-tone labels come
straight from `manifests/test.csv` in the package, and the inter-labeler Dice
comes from `interlabeler_agreement_640.csv` in the package. Nothing is hand-
embedded, so nothing can silently go stale.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NB = PROJECT_ROOT / "bruise_colab_final.ipynb"      # tested modules live here
OUT = PROJECT_ROOT / "bruise_colab_final_analysis.ipynb"

# Reuse the full kit so every transitive import resolves (yolo_native -> sweep,
# postopt, metrics; evaluate -> metrics; engine -> data/losses/models; etc.).
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
    """Emit a code cell verbatim (for reused module cells -- no dedent/strip)."""
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.splitlines(keepends=True)})


def reused_module_cells() -> dict[str, str]:
    """Pull the `%%writefile bruisekit/<name>.py` cell sources out of the tested notebook."""
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
# Bruise segmentation — FINAL results: analysis & visualisation

Everything below is derived from **one best-seed inference pass** on the 185-image
test set. For each model we take the seed that scored highest on **validation**
(chosen in `bruise_colab_final.ipynb`, saved to
`results_final/best_seed_val_selected/`), load that seed's checkpoint from
`runs_final/`, and predict once. SegFormer is scored at its val-fitted logit cut;
YOLO is scored **two ways** (native argmax + custom /255). Nothing here is a fresh
sweep — the operating points were fixed on val already.

Every chart prints the table under it, and each model keeps the **same colour** in
every figure (the palette is colourblind-safe). Read the annotation-ceiling section
(F) first — it reframes every other number.

| # | Section | Question |
|---|---------|----------|
| A | Accuracy & distribution | How good, and how spread out? |
| B | Safety / failure modes | *How* do they fail? Which path for YOLO? |
| C | Inference statistics | Which differences are real at n=28? |
| D | ⭐ Fairness (ITA skin tone) | Is it equitable across skin tone? |
| E | ⭐ Bruise size | *When* do they fail — and is size a fairness confound? |
| F | Annotation ceiling + speed | Do they beat human–human agreement? What do they cost? |
| G | Qualitative (optional GPU) | What does it actually look like? |

**This is best-seed (val-selected) analysis.** The 3-seed-averaged tables live in
`results_final/*.csv`; this notebook recomputes the best-seed slice fresh so the
figures, fairness and gallery all share one consistent inference pass.
""")

# ══════════════════════════════════════════════════════════════════════════════
# §1 config
# ══════════════════════════════════════════════════════════════════════════════
md("## §1 · Configuration")

code(r'''
CFG = dict(
    img_size   = 640,
    zip_name   = "bruise_colab_final.zip",
    drive_dir  = "/content/drive/MyDrive/bruise_segmentation_gpu",
    work_dir   = "/content/bruise_final",          # local SSD, wiped on disconnect
    seeds      = (0, 1, 2),                          # only used to re-derive best seed if the CSV is absent
    workers    = 4,
    amp        = True,
    # sweep grids (only needed for the optional custom-255 re-derivation / fallback)
    cut_min = -6.0, cut_max = 6.0, cut_steps = 481,
    prob_min = 0.01, prob_max = 0.99, prob_steps = 197,
    render_gallery = True,   # set False to skip the optional GPU image gallery (section G)
)

SEGFORMER_MODELS = {
    "segformer_b2_teacher":   dict(arch="segformer", size="b2"),
    "segformer_b0_direct":    dict(arch="segformer", size="b0"),
    "segformer_b0_distilled": dict(arch="segformer", size="b0"),
}
YOLO_MODELS = ["yolo_sem_direct", "yolo_sem_distilled"]

# Display names + the fixed, colourblind-safe palette (same slot per model everywhere).
DISP = {
    "segformer_b2_teacher":   "SegFormer-B2 (teacher)",
    "segformer_b0_direct":    "SegFormer-B0 (direct)",
    "segformer_b0_distilled": "SegFormer-B0 (distilled)",
    "yolo_sem_direct":        "YOLO26n (direct)",
    "yolo_sem_distilled":     "YOLO26n (distilled)",
}
PALETTE = {
    "segformer_b2_teacher":   "#2a78d6",   # blue
    "segformer_b0_direct":    "#1baf7a",   # aqua
    "segformer_b0_distilled": "#eda100",   # yellow
    "yolo_sem_direct":        "#008300",   # green
    "yolo_sem_distilled":     "#4a3aa7",   # violet
}
# The 5 "core" variants used for model-to-model figures. YOLO uses its native-argmax
# best seed here (that's what the val selection picked); the custom-/255 path is added
# alongside only in the sections that explicitly compare the two YOLO paths.
CORE = list(SEGFORMER_MODELS) + YOLO_MODELS
print(f"{len(SEGFORMER_MODELS)} SegFormer + {len(YOLO_MODELS)} YOLO = {len(CORE)} core variants")
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
%pip install -q "transformers>=4.40,<6" "ultralytics>=8.4,<9" "albumentations>=2.0,<3" "scipy>=1.11" "pandas>=2.0" "matplotlib>=3.7" "pyyaml" "statsmodels>=0.14"
import transformers, ultralytics, albumentations
print("transformers", transformers.__version__, "| ultralytics", ultralytics.__version__)
''')

# ══════════════════════════════════════════════════════════════════════════════
# §3 unpack + 640 cache  (verbatim from bruise_colab_final.ipynb §3)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §3 · Unpack (native-res) and build the 640 cache once

Same setup as the training notebook: unzip the native-res package to local SSD and
build a 640-stretch cache **once**. SegFormer + the custom-YOLO path read the 640
cache; native YOLO reads the native-res images directly.
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
assert (len(MAN["train"]), len(MAN["val"]), len(MAN["test"])) == (697, 134, 185)

RUNS_DIR = Path(CFG["drive_dir"]) / "runs_final"
OUT_DIR  = Path(CFG["drive_dir"]) / "results_final"
if not RUNS_DIR.exists():
    raise FileNotFoundError(f"{RUNS_DIR} not found -- run bruise_colab_final.ipynb first (checkpoints needed).")
print("checkpoints <-", RUNS_DIR)
''')

code(r'''
# Build the 640 stretch cache once (image bilinear, mask nearest -- albumentations'
# defaults, so this is bit-exact to what the training/eval dataloader computes live).
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
    for s in ("val","test"): build_cache(MAN[s], s)   # analysis only needs val (thr) + test
    print(f"640 cache built in {time.time()-t0:.0f}s")
else:
    print("640 cache present")

MAN640 = {}
for s in ("val","test"):
    d = MAN[s].copy()
    d["image_path"] = d["stem"].apply(lambda x: f"{s}/images/{x}.png")
    d["mask_path"]  = d["stem"].apply(lambda x: f"{s}/masks/{x}.png")
    MAN640[s] = d
''')

# ══════════════════════════════════════════════════════════════════════════════
# §4 library (reused verbatim)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §4 · The library

`bruisekit` is reused **verbatim** from `bruise_colab_final.ipynb`, so inference
here is byte-identical to the run that produced `results_final/`. The mkdir comes
first so the `%%writefile` cells have a directory to land in.
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

# ══════════════════════════════════════════════════════════════════════════════
# §5 imports + PATHS + plotting style
# ══════════════════════════════════════════════════════════════════════════════
md("## §5 · Imports, model paths, and the plotting style")

code(r'''
sys.path.insert(0, "/content")
import importlib
import bruisekit.data, bruisekit.models, bruisekit.metrics
import bruisekit.sweep, bruisekit.evaluate, bruisekit.yolo_native
for m in ("data","models","metrics","sweep","evaluate","yolo_native"):
    importlib.reload(sys.modules[f"bruisekit.{m}"])

from bruisekit.data import make_loader
from bruisekit.evaluate import evaluate_at_cut, fairness_analysis
from bruisekit.models import build_model, count_params
from bruisekit.sweep import cache_logits, select_cut, sweep_cuts
import bruisekit.yolo_native as yn

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kruskal, mannwhitneyu

PATHS = {
    "segformer_b0": str(WORK / "pretrained_weights" / "segformer_mit_b0"),
    "segformer_b2": str(WORK / "pretrained_weights" / "segformer_mit_b2"),
    "yolo":         str(WORK / "pretrained_weights" / "yolo26n-sem.pt"),
}
DEVICE = torch.device("cuda:0")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor":"#fcfcfb", "axes.facecolor":"#fcfcfb",
    "axes.edgecolor":"#c3c2b7", "axes.labelcolor":INK, "text.color":INK,
    "xtick.color":MUTED, "ytick.color":MUTED, "grid.color":GRID,
    "axes.spines.top":False, "axes.spines.right":False,
    "font.size":10, "figure.dpi":120,
})
print("style set; PATHS ->", {k: Path(v).name for k,v in PATHS.items()})
''')

# ══════════════════════════════════════════════════════════════════════════════
# §6 load best seed + inference once
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
## §6 · Which seed, and one inference pass

We read the seed already chosen on **validation** from
`results_final/best_seed_val_selected/best_seed_val_selected_results.csv`. If that
file is missing (e.g. the folder was cleared) we re-derive it: score every seed on
val at its saved cut and take the best. Either way, only the winning seed is loaded
and predicted on test.
""")

code(r'''
BEST_CSV = OUT_DIR / "best_seed_val_selected" / "best_seed_val_selected_results.csv"

def _seg_val_dice(name, spec, seed):
    rd = RUNS_DIR / f"{name}__seed{seed}"
    if not (rd / "best.pt").exists() or not (rd / "operating_point.json").exists():
        return None
    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(rd/"best.pt"), map_location=DEVICE, weights_only=True))
    cut = json.loads((rd/"operating_point.json").read_text())["cut"]
    _, vs = evaluate_at_cut(model, make_loader(MAN640["val"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed),
                            DEVICE, cut, CFG["amp"])
    del model; torch.cuda.empty_cache()
    return cut, vs["mean_dice"]

def _yolo_val_dice(name, seed):
    best = RUNS_DIR / f"{name}__seed{seed}" / "ultralytics_runs" / "train" / "weights" / "best.pt"
    if not best.exists():
        return None
    _, vs = yn.predict_native_argmax(best, MAN["val"], WORK, CFG["img_size"])
    return vs["mean_dice"]

BEST = {}   # name -> dict(seed, cut)
if BEST_CSV.exists():
    bt = pd.read_csv(BEST_CSV)
    for _, r in bt.iterrows():
        BEST[r["model"]] = {"seed": int(r["seed"]), "cut": float(r["cut"]) if pd.notna(r["cut"]) else None}
    print("best seeds read from disk:")
else:
    print("best_seed CSV absent -> re-deriving on val (slower)...")
    for name, spec in SEGFORMER_MODELS.items():
        cand = {s: _seg_val_dice(name, spec, s) for s in CFG["seeds"]}
        cand = {s: v for s, v in cand.items() if v is not None}
        bs = max(cand, key=lambda s: cand[s][1]); BEST[name] = {"seed": bs, "cut": cand[bs][0]}
    for name in YOLO_MODELS:
        cand = {s: _yolo_val_dice(name, s) for s in CFG["seeds"]}
        cand = {s: v for s, v in cand.items() if v is not None}
        bs = max(cand, key=lambda s: cand[s]); BEST[name] = {"seed": bs, "cut": None}
for k, v in BEST.items():
    print(f"  {k:<26} seed {v['seed']}  cut {v['cut']}")
''')

code(r'''
# One inference pass on the val-selected best seed. Produces:
#   per_image[name]              -> SegFormer / YOLO-native per-image rows (the CORE 5)
#   per_image_custom[name]       -> YOLO custom-/255 per-image rows (extra, for path comparison)
#   SEG_MODELS / YOLO_BEST       -> handles kept for the optional gallery (section G)
def enrich(df):
    """Add the derived columns the figures need (all algebra on the stored counts)."""
    df = df.copy()
    df["complete_miss"] = (df.pred_positive_pixels == 0) & (df.gt_positive_pixels > 0)
    tp = (df.recall * df.gt_positive_pixels).round()
    df["tp_pixels"] = tp
    df["fp_pixels"] = (df.pred_positive_pixels - tp).clip(lower=0)
    df["fn_pixels"] = (df.gt_positive_pixels - tp).clip(lower=0)
    df["pred_gt_ratio"] = df.pred_positive_pixels / df.gt_positive_pixels.replace(0, np.nan)
    return df

per_image, per_image_custom = {}, {}
SEG_MODELS, YOLO_BEST = {}, {}

for name, spec in SEGFORMER_MODELS.items():
    seed, cut = BEST[name]["seed"], BEST[name]["cut"]
    rd = RUNS_DIR / f"{name}__seed{seed}"
    model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
    model.load_state_dict(torch.load(str(rd/"best.pt"), map_location=DEVICE, weights_only=True))
    pi, summ = evaluate_at_cut(model, make_loader(MAN640["test"], CACHE640, CFG["img_size"], 8, False, CFG["workers"], seed),
                              DEVICE, cut, CFG["amp"])
    per_image[name] = enrich(pi)
    SEG_MODELS[name] = (model.eval(), cut)   # kept on GPU for the gallery; A100 has room for all 3
    print(f"{DISP[name]:<26} seed {seed} | median {pi.dice.median():.3f}  miss {per_image[name].complete_miss.mean()*100:.2f}%")

for name in YOLO_MODELS:
    seed = BEST[name]["seed"]
    best = RUNS_DIR / f"{name}__seed{seed}" / "ultralytics_runs" / "train" / "weights" / "best.pt"
    YOLO_BEST[name] = best
    pi_a, _ = yn.predict_native_argmax(best, MAN["test"], WORK, CFG["img_size"])
    per_image[name] = enrich(pi_a)
    # custom /255 at the same seed (threshold swept on val) -- for the two-path comparison.
    PROB_THR = np.linspace(CFG["prob_min"], CFG["prob_max"], CFG["prob_steps"])
    _, pi_b, _ = yn.predict_custom_255(best, MAN640["val"], MAN640["test"], CACHE640, DEVICE, PROB_THR)
    per_image_custom[name] = enrich(pi_b)
    print(f"{DISP[name]:<26} seed {seed} | native median {pi_a.dice.median():.3f} miss {per_image[name].complete_miss.mean()*100:.2f}%"
          f" | custom median {pi_b.dice.median():.3f}")

# Subject + ITA labels come straight from the test manifest (no embedding, never stale).
SUBJ = MAN["test"][["stem","subject"]].copy()
ITA  = MAN["test"][["stem","skin_tone_category","ita_group_index_5"]].copy()
assert SUBJ.subject.nunique() == 28, SUBJ.subject.nunique()
print(f"\n{len(SUBJ)} images / {SUBJ.subject.nunique()} subjects | ITA groups: {sorted(ITA.skin_tone_category.unique())}")
''')

code(r'''
# Shared bootstrap helpers (subject-level cluster bootstrap; the 185 images are only
# 28 people, so we resample PEOPLE, not images). Paired where two models are compared.
RNG_SEED = 42
def _subject_index(df):
    subs = df["subject"].unique()
    return subs, {s: df[df.subject == s] for s in subs}

def cluster_ci(frame, stat, B=4000, seed=RNG_SEED):
    """95% CI of `stat(frame)` under subject resampling."""
    rng = np.random.default_rng(seed)
    subs, by = _subject_index(frame)
    vals = np.array([stat(pd.concat([by[s] for s in rng.choice(subs, len(subs), True)], ignore_index=True))
                     for _ in range(B)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_delta(frame, stat, B=4000, seed=RNG_SEED):
    """Point Δ, 95% CI, and one-sided P(Δ>0) for a paired statistic over subjects."""
    rng = np.random.default_rng(seed)
    subs, by = _subject_index(frame)
    point = stat(frame)
    vals = np.array([stat(pd.concat([by[s] for s in rng.choice(subs, len(subs), True)], ignore_index=True))
                     for _ in range(B)])
    return {"delta": float(point),
            "ci_lo": float(np.percentile(vals, 2.5)), "ci_hi": float(np.percentile(vals, 97.5)),
            "p_gt0": float((vals > 0).mean())}

def merged(a, b, cols=("dice",)):
    """Per-image join of two CORE variants on stem, with subject attached."""
    left  = per_image[a][["stem", *cols]].rename(columns={c: f"{c}_a" for c in cols})
    right = per_image[b][["stem", *cols]].rename(columns={c: f"{c}_b" for c in cols})
    return left.merge(right, on="stem").merge(SUBJ, on="stem")
print("bootstrap helpers ready")
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — accuracy & distribution
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# A · Accuracy & distribution
""")

md(r"""
## A1 · Headline table (best seed) + on-disk 3-seed spread

Left: this best-seed run. Right (printed): the 3-seed mean±std already on disk in
`results_final/FINAL_RESULTS.csv`, so you can see the best seed sitting inside the
seed spread rather than cherry-picked outside it.
""")

code(r'''
rows = []
for name in CORE:
    d = per_image[name]
    rows.append({"variant": DISP[name], "median_dice": d.dice.median(), "mean_dice": d.dice.mean(),
                 "mean_iou": d.iou.mean(), "miss_%": d.complete_miss.mean()*100,
                 "seed": BEST[name]["seed"]})
head = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9, 4.2))
y = np.arange(len(head))[::-1]
ax.barh(y, head.mean_dice, height=0.6, color=[PALETTE[n] for n in CORE], zorder=3)
for yy, m, md_ in zip(y, head.mean_dice, head.median_dice):
    ax.text(m+0.006, yy, f"mean {m:.3f} / med {md_:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels(head.variant, fontsize=9)
ax.set_xlim(0, 1.02); ax.set_xlabel("Dice (best seed, 185 test images)")
ax.set_title("Headline accuracy — best val-selected seed", fontsize=12, pad=10)
ax.grid(axis="x", lw=0.6, zorder=0); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()

try:
    disk = pd.read_csv(OUT_DIR/"FINAL_RESULTS.csv")
    print("On-disk 3-seed mean±std (results_final/FINAL_RESULTS.csv):")
    print(disk[["variant","dice","median_dice","miss_%"]].to_string(index=False))
except Exception as e:
    print("FINAL_RESULTS.csv not read:", e)
head.round(4)
''')

md(r"""
## A2 · Per-image Dice: violins + survival curves

The violin shows the spread over 185 images (white dot = median, red dash = mean).
The survival curve on the right, P(Dice ≥ x), makes the **worst-case tail** explicit:
where a curve drops early, that model has more low-Dice images — the failures a single
average number hides.
""")

code(r'''
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
data = [per_image[n].dice.values for n in CORE]
parts = a1.violinplot(data, positions=np.arange(len(CORE)), widths=0.75,
                      showmeans=False, showmedians=False, showextrema=False)
for pc, n in zip(parts["bodies"], CORE):
    pc.set_facecolor(PALETTE[n]); pc.set_alpha(0.55); pc.set_edgecolor("#fcfcfb"); pc.set_linewidth(2)
for i, v in enumerate(data):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    a1.vlines(i, q1, q3, color=INK, lw=5, zorder=3)
    a1.plot(i, med, "o", color="#fcfcfb", ms=6, zorder=4, markeredgecolor=INK, markeredgewidth=1.2)
    a1.plot(i, v.mean(), "_", color="#e34948", ms=16, mew=2.5, zorder=5)
a1.set_xticks(range(len(CORE))); a1.set_xticklabels([DISP[n] for n in CORE], rotation=20, ha="right", fontsize=7.5)
a1.set_ylabel("Per-image Dice"); a1.set_ylim(-0.02, 1.02)
a1.set_title("Distribution (white=median, red=mean)", fontsize=11); a1.grid(axis="y", lw=0.6); a1.set_axisbelow(True)

xs = np.linspace(0, 1, 200)
for n in CORE:
    v = per_image[n].dice.values
    surv = [(v >= x).mean() for x in xs]
    a2.plot(xs, surv, color=PALETTE[n], lw=2, label=DISP[n])
a2.set_xlabel("Dice threshold x"); a2.set_ylabel("P(Dice ≥ x)")
a2.set_title("Survival curves — the low-Dice tail", fontsize=11)
a2.grid(lw=0.6); a2.set_axisbelow(True)
a2.legend(fontsize=7, loc="lower left", frameon=False)
plt.tight_layout(); plt.show()
pd.DataFrame([{"model": DISP[n], "median": per_image[n].dice.median(), "mean": per_image[n].dice.mean(),
               "p25": per_image[n].dice.quantile(.25), "p05": per_image[n].dice.quantile(.05),
               "zeros": int((per_image[n].dice==0).sum())} for n in CORE]).round(4)
''')

md(r"""
## A3 · Precision vs recall, and IoU vs Dice

Left: each dot is one image, precision vs recall. It shows an operating-point
character no single number does — e.g. YOLO tends to sit high-precision / lower-recall
(it fires *cleanly* but *misses*), which is exactly the under-detection the miss-rate
and fairness sections pick up. Right: IoU vs Dice, a consistency check (they should
track monotonically; the scatter is the per-image agreement).
""")

code(r'''
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
for n in CORE:
    d = per_image[n]
    a1.scatter(d.recall, d.precision, s=14, alpha=0.45, color=PALETTE[n], edgecolors="none", label=DISP[n])
    a1.plot(d.recall.mean(), d.precision.mean(), "o", ms=11, color=PALETTE[n], markeredgecolor=INK, mew=1.2, zorder=5)
a1.set_xlabel("recall  (1 − under-segmentation)"); a1.set_ylabel("precision  (1 − over-segmentation)")
a1.set_xlim(0,1.02); a1.set_ylim(0,1.02); a1.set_title("Per-image precision vs recall (big dot = mean)", fontsize=11)
a1.grid(lw=0.6); a1.set_axisbelow(True); a1.legend(fontsize=7, loc="lower left", frameon=False)

for n in CORE:
    d = per_image[n]
    a2.scatter(d.dice, d.iou, s=13, alpha=0.4, color=PALETTE[n], edgecolors="none")
a2.plot([0,1],[0,1], color=MUTED, ls="--", lw=1)
a2.set_xlabel("Dice"); a2.set_ylabel("IoU"); a2.set_xlim(0,1.02); a2.set_ylim(0,1.02)
a2.set_title("IoU vs Dice (dashed = y=x)", fontsize=11); a2.grid(lw=0.6); a2.set_axisbelow(True)
plt.tight_layout(); plt.show()
pd.DataFrame([{"model": DISP[n], "mean_precision": per_image[n].precision.mean(),
               "mean_recall": per_image[n].recall.mean(), "mean_iou": per_image[n].iou.mean()}
              for n in CORE]).round(4)
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — safety / failure
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# B · Safety & failure modes
""")

md(r"""
## B1 · Complete-miss rate — the failure that matters, and "best Dice ≠ safest"

A **complete miss** = a real bruise, empty predicted mask. For an injury-documentation
tool that is the decisive failure: a loose outline is correctable, a blank mask is
silence about an injury that is present. The right panel plots miss vs median Dice —
watch the model with the best Dice not being the safest.
""")

code(r'''
rows = []
for n in CORE:
    f = per_image[n][["stem","complete_miss"]].merge(SUBJ, on="stem")
    f["complete_miss"] = f.complete_miss.astype(float)
    lo, hi = cluster_ci(f, lambda x: x.complete_miss.mean()*100, B=3000)
    rows.append({"model": n, "miss_pct": f.complete_miss.mean()*100, "lo": lo, "hi": hi,
                 "median_dice": per_image[n].dice.median()})
miss = pd.DataFrame(rows)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
xs = np.arange(len(miss))
a1.bar(xs, miss.miss_pct, width=0.55, color=[PALETTE[n] for n in miss.model], zorder=3)
a1.errorbar(xs, miss.miss_pct, yerr=[miss.miss_pct-miss.lo, miss.hi-miss.miss_pct],
            fmt="none", ecolor=INK, elinewidth=1.5, capsize=5, zorder=4)
for x, v, hi in zip(xs, miss.miss_pct, miss.hi):
    a1.text(x, hi+0.3, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
a1.set_xticks(xs); a1.set_xticklabels([DISP[n] for n in miss.model], rotation=20, ha="right", fontsize=7.5)
a1.set_ylabel("Complete-miss rate (%)"); a1.set_title("Blank-mask failures (lower better)", fontsize=11)
a1.grid(axis="y", lw=0.6, zorder=0); a1.set_axisbelow(True)
for _, r in miss.iterrows():
    a2.scatter(r.miss_pct, r.median_dice, s=150, color=PALETTE[r.model], edgecolors=INK, lw=0.6, zorder=3)
    a2.annotate(DISP[r.model], (r.miss_pct, r.median_dice), fontsize=7.5, textcoords="offset points", xytext=(7,5))
a2.set_xlabel("Complete-miss rate (%)  →  worse"); a2.set_ylabel("Median Dice  →  better")
a2.set_title("Best Dice ≠ safest model", fontsize=11); a2.grid(lw=0.6, zorder=0); a2.set_axisbelow(True)
plt.tight_layout(); plt.show()
miss.assign(model=lambda d: d.model.map(DISP)).round(3)
''')

md(r"""
## B2 · Over- vs under-segmentation — *how* they fail

`pred / GT area` = predicted bruise area ÷ true area. **<1** under-segments (misses
bruise — bad for evidence); **>1** over-segments (flags healthy skin — recoverable).
Which way a model leans matters more forensically than its total error.
""")

code(r'''
fig, ax = plt.subplots(figsize=(9.5, 4.4))
rng = np.random.default_rng(0)
for i, n in enumerate(CORE):
    v = per_image[n].pred_gt_ratio.replace([np.inf,-np.inf], np.nan).dropna().clip(0, 3)
    ax.scatter(rng.normal(i, 0.07, len(v)), v, s=13, alpha=0.5, color=PALETTE[n], edgecolors="none", zorder=3)
    ax.plot(i, v.median(), "_", color=INK, ms=28, mew=2.5, zorder=4)
ax.axhline(1.0, color="#e34948", ls="--", lw=1.4, zorder=2)
ax.text(len(CORE)-0.45, 1.04, "perfect size", color="#e34948", fontsize=8.5)
ax.set_xticks(range(len(CORE))); ax.set_xticklabels([DISP[n] for n in CORE], rotation=20, ha="right", fontsize=7.5)
ax.set_ylabel("pred / GT area (clipped at 3)")
ax.set_title("Under- (<1) vs over- (>1) segmentation; black dash = median", fontsize=11.5, pad=10)
ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
pd.DataFrame([{"model": DISP[n], "median_pred_gt_ratio": per_image[n].pred_gt_ratio.median(),
               "under_seg_%": (per_image[n].pred_gt_ratio < 1).mean()*100,
               "mean_FP_px": per_image[n].fp_pixels.mean(), "mean_FN_px": per_image[n].fn_pixels.mean()}
              for n in CORE]).round(2)
''')

md(r"""
## B3 · YOLO: native argmax vs custom /255 — which path, and where they differ

YOLO is evaluated two ways: **native argmax** (its home turf, letterbox + argmax) and
**custom /255** (the same 640 geometry SegFormer sees, threshold swept on val). Same
weights, different eval. The paired per-image delta shows the paths are not
interchangeable — report both and say which is which.
""")

code(r'''
fig, axes = plt.subplots(1, len(YOLO_MODELS), figsize=(11, 4.4))
comp_rows = []
for ax, n in zip(np.atleast_1d(axes), YOLO_MODELS):
    m = per_image[n][["stem","dice"]].rename(columns={"dice":"native"}).merge(
        per_image_custom[n][["stem","dice"]].rename(columns={"dice":"custom"}), on="stem")
    ax.scatter(m.native, m.custom, s=16, alpha=0.5, color=PALETTE[n], edgecolors="none")
    ax.plot([0,1],[0,1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("native argmax Dice"); ax.set_ylabel("custom /255 Dice")
    ax.set_xlim(0,1.02); ax.set_ylim(0,1.02); ax.set_title(DISP[n], fontsize=10)
    ax.grid(lw=0.6); ax.set_axisbelow(True)
    comp_rows.append({"model": DISP[n], "native_median": per_image[n].dice.median(),
                      "custom_median": per_image_custom[n].dice.median(),
                      "native_miss_%": per_image[n].complete_miss.mean()*100,
                      "custom_miss_%": per_image_custom[n].complete_miss.mean()*100,
                      "mean_abs_delta": (m.native-m.custom).abs().mean()})
fig.suptitle("YOLO evaluation paths are not interchangeable (dashed = agreement)", fontsize=12, y=1.02)
plt.tight_layout(); plt.show()
pd.DataFrame(comp_rows).round(4)
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — inference statistics
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# C · Inference statistics (which differences are real?)

The 185 images come from only **28 people**, so photos of one person are not
independent. Every interval below is a **subject-level cluster bootstrap** (resample
people, B=4000); model-vs-model contrasts are **paired** (the same resample scores
both models — correct for a shared test set and ~2× tighter than unpaired).
""")

md("## C1 · Marginal CIs — mean Dice per model")

code(r'''
rows = []
for n in CORE:
    f = per_image[n][["stem","dice"]].merge(SUBJ, on="stem")
    lo, hi = cluster_ci(f, lambda x: x.dice.mean(), B=4000)
    rows.append({"model": n, "mean": f.dice.mean(), "lo": lo, "hi": hi})
ci = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8.5, 4))
y = np.arange(len(ci))[::-1]
for yy, (_, r) in zip(y, ci.iterrows()):
    ax.plot([r.lo, r.hi], [yy, yy], color=PALETTE[r.model], lw=3, solid_capstyle="round", zorder=3)
    ax.plot(r["mean"], yy, "o", ms=9, color=PALETTE[r.model], markeredgecolor="#fcfcfb", mew=1.5, zorder=4)
    ax.text(r.hi+0.004, yy, f"{r['mean']:.3f} [{r.lo:.3f}, {r.hi:.3f}]", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([DISP[n] for n in ci.model], fontsize=9)
ax.set_xlabel("Mean Dice (95% subject cluster-bootstrap)"); ax.set_xlim(0.55, 0.92)
ax.set_title("Overlapping intervals: at 28 subjects the models are hard to separate", fontsize=11, pad=10)
ax.grid(axis="x", lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
ci.assign(model=lambda d: d.model.map(DISP)).round(4)
''')

md(r"""
## C2 · Paired contrasts — forest plot with P(Δ>0)

Each row is one paired comparison: Δ mean Dice with its 95% CI and the one-sided
**P(Δ>0)** (a 92% and a 52% are both "n.s." but mean very different things). A
contrast is resolvable only if its interval clears zero.
""")

code(r'''
CONTRASTS = [
    ("Distillation (B0-dist − B0-direct)", "segformer_b0_distilled", "segformer_b0_direct", "dice"),
    ("Student − teacher (B0-dist − B2)",   "segformer_b0_distilled", "segformer_b2_teacher", "dice"),
    ("SegFormer − YOLO (B0-dist − YOLO-d)","segformer_b0_distilled", "yolo_sem_direct",      "dice"),
    ("YOLO distill − direct",              "yolo_sem_distilled",     "yolo_sem_direct",      "dice"),
]
res = []
for label, a, b, col in CONTRASTS:
    m = merged(a, b, cols=(col,))
    r = paired_delta(m, lambda x: x[f"{col}_a"].mean() - x[f"{col}_b"].mean(), B=4000)
    r["label"] = label; res.append(r)
# add the miss-rate contrast that historically IS significant
mm = per_image["yolo_sem_direct"][["stem","complete_miss"]].rename(columns={"complete_miss":"a"}).merge(
     per_image["segformer_b0_distilled"][["stem","complete_miss"]].rename(columns={"complete_miss":"b"}), on="stem").merge(SUBJ,on="stem")
mm["a"]=mm.a.astype(float); mm["b"]=mm.b.astype(float)
rm = paired_delta(mm, lambda x: (x.a.mean()-x.b.mean())*100, B=4000); rm["label"]="MISS %: YOLO-direct − B0-dist"
FOREST = pd.DataFrame(res + [rm])

fig, ax = plt.subplots(figsize=(9.5, 4.4))
y = np.arange(len(FOREST))[::-1]
for yy, (_, r) in zip(y, FOREST.iterrows()):
    sig = r.ci_lo > 0 or r.ci_hi < 0
    c = "#c0392b" if sig else MUTED
    ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=c, lw=3, solid_capstyle="round", zorder=3)
    ax.plot(r.delta, yy, "o", ms=8, color=c, zorder=4)
    ax.text(r.ci_hi+0.002 if r.ci_hi>=0 else r.ci_hi, yy+0.28,
            f"Δ={r.delta:+.3f}  P(Δ>0)={r.p_gt0*100:.0f}%", fontsize=8, color=INK)
ax.axvline(0, color=INK, lw=1.1, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(FOREST.label, fontsize=8.5)
ax.set_xlabel("Δ (paired subject bootstrap; Dice contrasts, except the miss-% row in %)")
ax.set_title("Paired effect sizes — red = interval clears zero", fontsize=11, pad=10)
ax.grid(axis="x", lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
FOREST[["label","delta","ci_lo","ci_hi","p_gt0"]].round(4)
''')

md(r"""
## C3 · Per-subject heatmap + model agreement

Left: each of the 28 subjects × each model, mean Dice (darker = better; hardest
subjects on top). Rows dark across *every* column are hard **subjects**, not model
failures — and with only 28 rows, one or two of them move the whole average (why C1's
bars are wide). Right: Spearman correlation of per-image Dice between models — high
everywhere means the models agree on which images are hard, so an ensemble helps less
than you'd hope.
""")

code(r'''
mats = []
for n in CORE:
    s = per_image[n][["stem","dice"]].merge(SUBJ, on="stem").groupby("subject").dice.mean()
    mats.append(s.rename(DISP[n]))
H = pd.concat(mats, axis=1)
H = H.loc[H.mean(axis=1).sort_values().index]
D = pd.DataFrame({DISP[n]: per_image[n].set_index("stem").dice for n in CORE})
C = D.corr(method="spearman")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios":[1, 1.05]})
im1 = a1.imshow(H.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
a1.set_xticks(range(len(H.columns))); a1.set_xticklabels(H.columns, rotation=35, ha="right", fontsize=7.5)
a1.set_yticks(range(len(H))); a1.set_yticklabels(H.index, fontsize=7)
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]
        a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                color="#ffffff" if v > 0.55 else INK)
a1.set_title("Per-subject mean Dice (hardest on top)", fontsize=11)
fig.colorbar(im1, ax=a1, shrink=0.5, label="Mean Dice")

im2 = a2.imshow(C.values, cmap="Blues", vmin=0, vmax=1)
a2.set_xticks(range(len(C))); a2.set_xticklabels(C.columns, rotation=35, ha="right", fontsize=7.5)
a2.set_yticks(range(len(C))); a2.set_yticklabels(C.index, fontsize=7.5)
for i in range(len(C)):
    for j in range(len(C)):
        a2.text(j, i, f"{C.values[i,j]:.2f}", ha="center", va="center", fontsize=8,
                color="#ffffff" if C.values[i,j] > 0.55 else INK)
a2.set_title("Spearman corr. of per-image Dice", fontsize=11)
fig.colorbar(im2, ax=a2, shrink=0.7, label="ρ")
plt.tight_layout(); plt.show()
print("Hardest 5 subjects (mean over models):")
print(H.mean(axis=1).head(5).round(3).to_string())
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — fairness (ITA)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# D · ⭐ Fairness across skin tone (ITA)

This is a forensic tool: a model that documents bruises well on light skin and poorly
on dark skin has an **evidentiary** problem, not a metric one. Groups are ITA
(Individual Typology Angle) — an objective, pixel-computed skin-tone measure. Computed
here on the **best-seed** per-image data via the same `fairness_analysis` used to make
`results_final/fairness_*.csv` (which are 3-seed-averaged).

**Honest caveat up front:** each ITA group holds only ~9–17 *subjects*. The
group-level CIs are wide and mostly overlap — treat these as **exploratory**. The
gap's *sign* is a hypothesis, not a proven effect, at n=28.
""")

code(r'''
# Fairness fresh from best-seed per-image, for the 5 core variants + both YOLO paths.
fair_frames = {}
for n in SEGFORMER_MODELS:
    fair_frames[DISP[n]] = per_image[n]
for n in YOLO_MODELS:
    fair_frames[f"{DISP[n]} · native"] = per_image[n]
    fair_frames[f"{DISP[n]} · custom"] = per_image_custom[n]

fg, fp, fs = [], [], []
for label, pi in fair_frames.items():
    out = fairness_analysis(pi[["stem","dice","recall","pred_positive_pixels","gt_positive_pixels"]],
                            MAN["test"], label)
    fg.append(out["per_group"]); fp.append(out["pairwise"]); fs.append(out["stats"])
FAIR_GROUP = pd.concat(fg, ignore_index=True)
FAIR_PAIR  = pd.concat(fp, ignore_index=True)
FAIR_STATS = pd.DataFrame(fs)
print(FAIR_STATS[["model","kruskal_p","significant","fairness_gap","best_group","worst_group","max_miss_rate_gap"]].to_string(index=False))
''')

md("## D1 · Per-group heatmaps — median Dice, recall, miss-rate")

code(r'''
GROUP_ORDER = ["Light (II-III)","Intermediate (III-IV)","Tan (IV)","Brown (V)","Dark (VI)"]
def heat(ax, values, title, cmap, fmt, vmax):
    piv = FAIR_GROUP.pivot_table(index="model", columns="skin_tone_category", values=values)
    piv = piv.reindex(columns=[g for g in GROUP_ORDER if g in piv.columns])
    im = ax.imshow(piv.values, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, fmt(piv.values[i,j]), ha="center", va="center", fontsize=6.5,
                    color="#ffffff" if (piv.values[i,j]/vmax) > 0.55 else INK)
    ax.set_title(title, fontsize=10)
    return im

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
i1 = heat(axes[0], "median_dice", "Median Dice", "Blues", lambda v: f"{v:.2f}", 1.0)
i2 = heat(axes[1], "mean_recall", "Mean recall", "Greens", lambda v: f"{v:.2f}", 1.0)
fair_miss = FAIR_GROUP.assign(miss_pct=FAIR_GROUP.miss_rate*100)
piv = fair_miss.pivot_table(index="model", columns="skin_tone_category", values="miss_pct").reindex(columns=[g for g in GROUP_ORDER if g in fair_miss.skin_tone_category.unique()])
im3 = axes[2].imshow(piv.values, cmap="Reds", aspect="auto", vmin=0, vmax=max(1e-6, np.nanmax(piv.values)))
axes[2].set_xticks(range(piv.shape[1])); axes[2].set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=7)
axes[2].set_yticks(range(piv.shape[0])); axes[2].set_yticklabels(piv.index, fontsize=7)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        axes[2].text(j, i, f"{piv.values[i,j]:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
axes[2].set_title("Complete-miss rate (%)", fontsize=10)
for im, ax in [(i1,axes[0]),(i2,axes[1]),(im3,axes[2])]:
    fig.colorbar(im, ax=ax, shrink=0.6)
fig.suptitle("Performance by ITA skin-tone group (rows = variants)", fontsize=12, y=1.02)
plt.tight_layout(); plt.show()
FAIR_GROUP.pivot_table(index="model", columns="skin_tone_category", values="median_dice").round(3)
''')

md("## D2 · Fairness gap + the light-skin under-detection story")

code(r'''
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6))
# gap bars, coloured by omnibus significance
order = FAIR_STATS.sort_values("fairness_gap")
xs = np.arange(len(order))
cols = ["#c0392b" if s else MUTED for s in order.significant]
a1.barh(xs, order.fairness_gap, color=cols, zorder=3)
for x, g, p in zip(xs, order.fairness_gap, order.kruskal_p):
    a1.text(g+0.003, x, f"{g:.3f} (KW p={p:.2f})", va="center", fontsize=7.5)
a1.set_yticks(xs); a1.set_yticklabels(order.model, fontsize=7.5)
a1.set_xlabel("fairness gap = best − worst group median Dice")
a1.set_title("Max−min Dice gap (red = Kruskal p<0.05)", fontsize=10.5)
a1.grid(axis="x", lw=0.6); a1.set_axisbelow(True)

# precision vs recall per group for one YOLO path -> under-detection vs confusion
yl = f"{DISP['yolo_sem_direct']} · native"
sub = FAIR_GROUP[FAIR_GROUP.model == yl]
# recompute per-group precision on the fly for this panel
pi = per_image["yolo_sem_direct"].merge(ITA, on="stem")
grp = pi.groupby("skin_tone_category").agg(recall=("recall","mean"), precision=("precision","mean"),
                                           miss=("complete_miss","mean")).reindex([g for g in GROUP_ORDER if g in pi.skin_tone_category.unique()])
a2.scatter(grp.recall, grp.precision, s=140, c=range(len(grp)), cmap="viridis", edgecolors=INK, zorder=3)
for g, r in grp.iterrows():
    a2.annotate(g.split(" ")[0], (r.recall, r.precision), fontsize=7.5, textcoords="offset points", xytext=(6,4))
a2.set_xlabel("recall"); a2.set_ylabel("precision")
a2.set_title(f"{DISP['yolo_sem_direct']}: per-group precision vs recall\n(low recall + high precision = under-detection)", fontsize=9.5)
a2.grid(lw=0.6); a2.set_axisbelow(True)
plt.tight_layout(); plt.show()
grp.assign(miss_pct=lambda d: d.miss*100).round(3)
''')

md("## D3 · Pairwise significance (Bonferroni) — mostly n.s. at n=28")

code(r'''
sig = FAIR_PAIR.groupby("model").significant.sum()
tot = FAIR_PAIR.groupby("model").significant.count()
tab = pd.DataFrame({"significant_pairs": sig, "of_pairs": tot}).reset_index()
print("Bonferroni-significant group pairs per variant (out of 10):")
print(tab.to_string(index=False))
print("\nInterpretation: with ~9-17 subjects per ITA group, almost nothing survives")
print("Bonferroni. The heatmaps show DIRECTION; do not claim a proven per-group gap at n=28.")
FAIR_PAIR[FAIR_PAIR.significant][["model","group_a","group_b","bonferroni_p"]].round(4)
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — bruise size
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# E · ⭐ Bruise size

Small bruises are intrinsically harder — a few pixels of error costs proportionally
far more Dice on a small target. So a model's score partly reflects *which bruises*
were in the test set, and — crucially for section D — **if bruise size correlates with
skin tone, an apparent fairness gap can be a size effect in disguise.** This section
tests exactly that.
""")

md("## E1 · Size distribution + Dice-vs-size + miss-vs-size")

code(r'''
sizes = per_image[CORE[0]].set_index("stem").gt_positive_pixels
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
axes[0].hist(np.log10(sizes.clip(lower=1)), bins=30, color="#2a78d6", alpha=0.85)
axes[0].set_xlabel("log10(bruise size, px)"); axes[0].set_ylabel("images")
axes[0].set_title(f"Bruise-size distribution (median {int(sizes.median()):,} px)", fontsize=10)
axes[0].grid(lw=0.5); axes[0].set_axisbelow(True)

for n in CORE:
    d = per_image[n]
    axes[1].scatter(d.gt_positive_pixels, d.dice, s=12, alpha=0.35, color=PALETTE[n], edgecolors="none")
axes[1].set_xscale("log"); axes[1].set_xlabel("bruise size (px, log)"); axes[1].set_ylabel("Dice")
axes[1].set_ylim(-0.02,1.02); axes[1].set_title("Dice vs size (all models)", fontsize=10)
axes[1].grid(lw=0.5); axes[1].set_axisbelow(True)

# miss rate by size quartile, per model
q = pd.qcut(sizes, 4, labels=["Q1 smallest","Q2","Q3","Q4 largest"])
size_bin = q.to_frame("size_q")
for n in CORE:
    d = per_image[n].set_index("stem")
    mr = d.join(size_bin).groupby("size_q", observed=True).complete_miss.mean()*100
    axes[2].plot(range(len(mr)), mr.values, "-o", color=PALETTE[n], label=DISP[n], lw=2)
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(["Q1\nsmallest","Q2","Q3","Q4\nlargest"], fontsize=8)
axes[2].set_ylabel("complete-miss rate (%)"); axes[2].set_title("Misses concentrate on small bruises", fontsize=10)
axes[2].grid(lw=0.5); axes[2].set_axisbelow(True); axes[2].legend(fontsize=6.5, frameon=False)
plt.tight_layout(); plt.show()
pd.DataFrame([{"model": DISP[n],
               "spearman_size_vs_dice": spearmanr(per_image[n].gt_positive_pixels, per_image[n].dice)[0],
               "p": spearmanr(per_image[n].gt_positive_pixels, per_image[n].dice)[1]} for n in CORE]).round(4)
''')

md("## E2 · Recall & precision vs size")

code(r'''
q = pd.qcut(per_image[CORE[0]].set_index("stem").gt_positive_pixels, 5,
            labels=["Q1","Q2","Q3","Q4","Q5"])
qb = q.to_frame("size_q")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
for n in CORE:
    d = per_image[n].set_index("stem").join(qb)
    rec = d.groupby("size_q", observed=True).recall.mean()
    pre = d.groupby("size_q", observed=True).precision.mean()
    a1.plot(range(len(rec)), rec.values, "-o", color=PALETTE[n], lw=2, label=DISP[n])
    a2.plot(range(len(pre)), pre.values, "-o", color=PALETTE[n], lw=2)
for ax, t in [(a1,"Recall vs size quintile"),(a2,"Precision vs size quintile")]:
    ax.set_xticks(range(5)); ax.set_xticklabels(["Q1\nsmall","Q2","Q3","Q4","Q5\nlarge"], fontsize=8)
    ax.set_ylim(0,1.02); ax.set_title(t, fontsize=10.5); ax.grid(lw=0.5); ax.set_axisbelow(True)
a1.set_ylabel("mean recall"); a1.legend(fontsize=6.5, frameon=False, loc="lower right")
plt.tight_layout(); plt.show()
print("Recall falls fastest on the smallest bruises — that is where the misses live.")
''')

md("## E3 · ⭐ The size↔fairness confound — bruise size by ITA group")

code(r'''
sz = per_image[CORE[0]][["stem","gt_positive_pixels"]].merge(ITA, on="stem")
groups = [g for g in GROUP_ORDER if g in sz.skin_tone_category.unique()]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
data = [sz[sz.skin_tone_category==g].gt_positive_pixels.values for g in groups]
bp = ax.boxplot(data, labels=[g.split(" ")[0] for g in groups], showfliers=False, patch_artist=True)
for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0,1,len(groups)))):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_yscale("log"); ax.set_ylabel("bruise size (px, log)")
ax.set_title("Bruise size by ITA group — is the fairness gap really a size gap?", fontsize=11, pad=10)
ax.grid(axis="y", lw=0.5); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()

H, p = kruskal(*data)
tab = sz.groupby("skin_tone_category").gt_positive_pixels.agg(["count","median"]).reindex(groups)
print(f"Kruskal–Wallis (size across ITA groups): H={H:.2f}, p={p:.4f}")
print("If p is small, group-level Dice differences in section D are partly a size artefact,")
print("because smaller bruises are harder for every model (E1). Report size as a covariate.")
tab.round(0)
''')

md(r"""
## E4 · ⭐ Does the skin-tone signal survive size? (stratified + regression)

E3 showed size is confounded with ITA group. This asks the decisive question: is the
light-skin deficit *only* size? Two tests per model — (1) within size terciles, compare
light vs rest on miss-rate and recall (if the gap is pure size, it vanishes once you
hold size roughly fixed); (2) a regression that puts `log10(size)` and a light-skin flag
in together — `recall ~ log_size + light` (OLS, always defined) and
`complete_miss ~ log_size + light` (logistic, where there are enough miss events). **SEs
are clustered by subject** (the light group is only ~9 people; treating 185 images as
independent would make these p-values far too small — the same reason every CI here is a
subject bootstrap). **If the `light` term stays significant with size controlled, the
signal is not only size; if it collapses, the E3 confound explains the section-D gap.**
Still n=28-limited — diagnostic, not proof.
""")

code(r'''
try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except Exception:
    HAVE_SM = False
    print("statsmodels unavailable -> regressions skipped (stratified table still runs)")

LIGHT = "Light (II-III)"
strat_rows, coef_rows = [], []
for n in CORE:
    d = per_image[n].merge(ITA, on="stem").merge(SUBJ, on="stem").copy()
    d["light"]    = (d.skin_tone_category == LIGHT).astype(int)
    d["log_size"] = np.log10(d.gt_positive_pixels.clip(lower=1))
    d["miss"]     = d.complete_miss.astype(int)
    d["size_tercile"] = pd.qcut(d.gt_positive_pixels, 3, labels=["small","medium","large"])

    # (1) stratified light-vs-rest within each size tercile (+ overall)
    for terc, sub in list(d.groupby("size_tercile", observed=True)) + [("all", d)]:
        L, R = sub[sub.light == 1], sub[sub.light == 0]
        strat_rows.append({"model": DISP[n], "size_tercile": str(terc),
                           "n_light": len(L), "n_rest": len(R),
                           "miss_light_pct": L.miss.mean()*100 if len(L) else np.nan,
                           "miss_rest_pct":  R.miss.mean()*100 if len(R) else np.nan,
                           "recall_light":   L.recall.mean() if len(L) else np.nan,
                           "recall_rest":    R.recall.mean() if len(R) else np.nan})

    # (2) regression: light term controlling for log size. SEs are CLUSTERED BY SUBJECT --
    # the 185 images are only 28 people, and the light group is ~9 subjects, so an
    # independence assumption would make these p-values far too small (the same reason
    # every CI in this notebook is a subject bootstrap). Clustered SEs are the honest test.
    if HAVE_SM:
        clk = {"cov_type": "cluster", "cov_kwds": {"groups": d["subject"]}}
        try:
            ols = smf.ols("recall ~ log_size + light", data=d).fit(**clk)
            coef_rows.append({"model": DISP[n], "outcome": "recall (OLS, subj-clustered)",
                              "light_coef": ols.params["light"], "light_p": ols.pvalues["light"],
                              "logsize_coef": ols.params["log_size"], "logsize_p": ols.pvalues["log_size"],
                              "light_OR": np.nan})
        except Exception:
            coef_rows.append({"model": DISP[n], "outcome": "recall (OLS): failed", "light_coef": np.nan,
                              "light_p": np.nan, "logsize_coef": np.nan, "logsize_p": np.nan, "light_OR": np.nan})
        # Logistic needs enough miss EVENTS or it quasi-separates and returns garbage
        # (e.g. a model with 1 miss gives OR ~1e10). recall(OLS) carries the signal for
        # the near-zero-miss SegFormer models; the miss logit is only run where it's stable.
        if int(d.miss.sum()) >= 5:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lg = smf.logit("miss ~ log_size + light", data=d).fit(disp=0, **clk)
                coef_rows.append({"model": DISP[n], "outcome": "miss (Logit)",
                                  "light_coef": lg.params["light"], "light_p": lg.pvalues["light"],
                                  "logsize_coef": lg.params["log_size"], "logsize_p": lg.pvalues["log_size"],
                                  "light_OR": float(np.exp(lg.params["light"]))})
            except Exception:
                coef_rows.append({"model": DISP[n], "outcome": "miss (Logit: failed)",
                                  "light_coef": np.nan, "light_p": np.nan, "logsize_coef": np.nan,
                                  "logsize_p": np.nan, "light_OR": np.nan})

SIZE_COND      = pd.DataFrame(strat_rows)
SIZE_COND_COEF = pd.DataFrame(coef_rows)

# figure: miss% by size tercile, light vs rest, for YOLO-direct (where the gap lives)
focus = "yolo_sem_direct"
fd = SIZE_COND[(SIZE_COND.model == DISP[focus]) & (SIZE_COND.size_tercile != "all")]
fig, ax = plt.subplots(figsize=(8, 4.4))
x = np.arange(len(fd)); w = 0.38
ax.bar(x - w/2, fd["miss_light_pct"], w, color="#e34948", label="Light (II-III)")
ax.bar(x + w/2, fd["miss_rest_pct"],  w, color="#2a78d6", label="rest")
for xi, vl, vr in zip(x, fd["miss_light_pct"], fd["miss_rest_pct"]):
    ax.text(xi - w/2, (vl if np.isfinite(vl) else 0)+0.3, f"{vl:.0f}", ha="center", fontsize=8)
    ax.text(xi + w/2, (vr if np.isfinite(vr) else 0)+0.3, f"{vr:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(fd.size_tercile)
ax.set_ylabel("complete-miss rate (%)"); ax.set_xlabel("bruise-size tercile")
ax.set_title(f"{DISP[focus]}: is the light-skin miss gap just size?", fontsize=11, pad=10)
ax.legend(frameon=False); ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()

print("Stratified light-vs-rest by size tercile (miss% and recall):")
print(SIZE_COND.round(2).to_string(index=False))
print("\nRegression — 'light' term with log10(size) controlled:")
print(SIZE_COND_COEF.round(4).to_string(index=False))
print("\nRead: light_p<0.05 with log_size in the model => skin-tone signal is NOT only size.")
print("      light term collapsing to n.s. => the E3 size confound explains the section-D gap.")
SIZE_COND_COEF.round(4)
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — annotation ceiling + speed
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# F · Annotation ceiling & cost
""")

md(r"""
## F1 · ⭐ The annotation ceiling — do models beat human–human agreement?

Our test masks are a **majority vote of three experts**. How much do the experts even
agree with each other? We read the per-image inter-labeler Dice from
`interlabeler_agreement_640.csv` (pure mask arithmetic, no model). If experts agree at
only ~0.64, a model at ~0.79 vs their consensus is already **past the point where
humans agree with each other** — and a 0.02 Dice gap between two models is far below
the label noise. Note **Paul** (whose masks we *train on*) is the outlier: he matches
the majority least of the three.
""")

code(r'''
IL = pd.read_csv(WORK / "interlabeler_agreement_640.csv")
human_pairs = ["paul_vs_gbarimah","paul_vs_erik","gbarimah_vs_erik"]
human_maj   = ["paul_vs_majority","gbarimah_vs_majority","erik_vs_majority"]
rows = []
for c in human_pairs:
    a, b = c.split("_vs_"); rows.append({"label": f"{a.capitalize()} vs {b.capitalize()}", "kind":"human vs human", "dice": IL[c].mean()})
for c in human_maj:
    a = c.split("_vs_")[0]; rows.append({"label": f"{a.capitalize()} vs majority", "kind":"human vs majority", "dice": IL[c].mean()})
for n in CORE:
    rows.append({"label": DISP[n], "kind":"model vs majority", "dice": per_image[n].dice.mean()})
ceiling = pd.DataFrame(rows)
HH = IL[human_pairs].values.mean()

KIND_C = {"human vs human":"#e34948", "human vs majority":"#eb6834", "model vs majority":"#2a78d6"}
fig, ax = plt.subplots(figsize=(9, 5.4))
y = np.arange(len(ceiling))[::-1]
ax.barh(y, ceiling.dice, height=0.62, color=[KIND_C[k] for k in ceiling.kind], zorder=3)
for yy, v in zip(y, ceiling.dice):
    ax.text(v+0.008, yy, f"{v:.3f}", va="center", fontsize=8.5)
ax.axvline(HH, color=MUTED, ls="--", lw=1.5, zorder=2)
ax.text(HH, len(ceiling)-0.2, f"  avg human-human = {HH:.3f}", color=MUTED, fontsize=9, va="top")
ax.set_yticks(y); ax.set_yticklabels(ceiling.label, fontsize=8.5)
ax.set_xlabel("Mean Dice"); ax.set_xlim(0, 1.05)
ax.set_title("Annotation ceiling: every model beats human–human agreement", fontsize=12, pad=10)
ax.grid(axis="x", lw=0.6, zorder=0); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([0],[0],marker="s",lw=0,markerfacecolor=c,markeredgecolor="none",markersize=9,label=k)
                   for k,c in KIND_C.items()], loc="upper center", bbox_to_anchor=(0.5,-0.09), ncol=3, fontsize=8.5, frameon=False)
plt.tight_layout(); plt.show()
ceiling.round(4)
''')

md(r"""
## F2 · Does the model beat the annotator it learned from?

Models train on **Paul's** masks only, scored vs the 3-expert majority. Paul himself
matches that majority at ~0.70. Paired subject-bootstrap Δ (model − Paul, vs the same
majority): if the interval clears zero, the model trained on Paul agrees with consensus
**better than Paul does** — training averages away one annotator's idiosyncrasies.
""")

code(r'''
base = IL[["stem","paul_vs_majority"]].merge(SUBJ, on="stem")
rows = []
for n in CORE:
    m = base.merge(per_image[n][["stem","dice"]].rename(columns={"dice":"model"}), on="stem")
    r = paired_delta(m, lambda x: x.model.mean() - x.paul_vs_majority.mean(), B=4000)
    rows.append({"model": n, "model_vs_maj": m.model.mean(), "paul_vs_maj": m.paul_vs_majority.mean(),
                 "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p_gt0": r["p_gt0"],
                 "beats_paul": r["ci_lo"] > 0})
beat = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(8.8, 4.2))
xs = np.arange(len(beat))
ax.bar(xs, beat.delta, width=0.55, color=[PALETTE[n] for n in beat.model], zorder=3)
ax.errorbar(xs, beat.delta, yerr=[beat.delta-beat.ci_lo, beat.ci_hi-beat.delta],
            fmt="none", ecolor=INK, elinewidth=1.6, capsize=5, zorder=4)
ax.axhline(0, color="#c3c2b7", lw=1.2, zorder=2)
for x, d_, hi, p in zip(xs, beat.delta, beat.ci_hi, beat.p_gt0):
    ax.text(x, hi+0.005, f"{d_:+.3f}\nP={p*100:.0f}%", ha="center", fontsize=7.5)
ax.set_xticks(xs); ax.set_xticklabels([DISP[n] for n in beat.model], rotation=18, ha="right", fontsize=7.5)
ax.set_ylabel("Δ mean Dice (model − Paul, vs majority)")
ax.set_title("Every model matches expert consensus better than the annotator it trained on", fontsize=10.5, pad=10)
ax.grid(axis="y", lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()
beat.assign(model=lambda d: d.model.map(DISP)).round(4)
''')

md(r"""
## F3 · Speed / cost — accuracy vs latency

From `results_final/benchmark_640.csv` (640-tensor-on-GPU → mask-on-GPU, seed 0, the
architectural number). Pareto view: up-and-left is better. YOLO is far faster but pays
in miss rate; SegFormer-B0 is the balance point.
""")

code(r'''
try:
    B = pd.read_csv(OUT_DIR/"benchmark_640.csv")
    name_map = {"segformer_b2_teacher":"segformer_b2_teacher","segformer_b0_direct":"segformer_b0_direct",
                "segformer_b0_distilled":"segformer_b0_distilled","yolo_sem_direct":"yolo_sem_direct",
                "yolo_sem_distilled":"yolo_sem_distilled"}
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for _, r in B.iterrows():
        n = r["model"]
        if n not in per_image: continue
        med = per_image[n].dice.median()
        ax.scatter(r.fps, med, s=180, color=PALETTE[n], edgecolors=INK, lw=0.7, zorder=3)
        ax.annotate(f"{DISP[n]}\n{r.params_M:.1f}M · p95 {r.p95_ms:.1f}ms",
                    (r.fps, med), fontsize=7.5, textcoords="offset points", xytext=(8,-4))
    ax.set_xlabel("throughput (FPS, higher better)"); ax.set_ylabel("median Dice (higher better)")
    ax.set_title("Accuracy vs speed — best seed Dice × benchmarked FPS", fontsize=11, pad=10)
    ax.grid(lw=0.6); ax.set_axisbelow(True)
    plt.tight_layout(); plt.show()
    print(B[["model","median_ms","p95_ms","fps","params_M"]].to_string(index=False))
except Exception as e:
    print("benchmark_640.csv not read:", e)
''')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — qualitative (optional GPU)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
# G · Qualitative (optional GPU)

Reloads native images + re-predicts a handful of masks to render overlays. Uses the
best-seed model handles already in memory, so it's cheap. Set `CFG["render_gallery"]
= False` in §1 to skip. Self-skips if the images aren't present.
""")

code(r'''
if not CFG.get("render_gallery", True):
    print("gallery skipped (CFG['render_gallery'] = False)")
else:
    IMG_H = IMG_W = CFG["img_size"]
    def load_rgb640(p):
        im = cv2.imread(str(p));
        return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H)) if im is not None else None
    def to640(mask):
        m = np.asarray(mask)
        if m.ndim == 3: m = m[...,0]
        return (cv2.resize(m.astype("uint8"), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 0).astype("uint8")
    def overlay(img, mask, color=(230,60,60), alpha=0.45):
        lay = np.zeros_like(img); lay[mask.astype(bool)] = color
        return cv2.addWeighted(lay, alpha, img, 1-alpha, 0)

    def predict_mask(name, img_path):
        if name in SEG_MODELS:
            model, cut = SEG_MODELS[name]
            rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            r = cv2.resize(rgb, (IMG_W, IMG_H)).astype(np.float32)/255.0
            t = torch.from_numpy(r.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                return (model(t)[:,0] >= cut).to(torch.uint8)[0].cpu().numpy()
        from ultralytics import YOLO
        res = YOLO(str(YOLO_BEST[name])).predict(str(img_path), imgsz=IMG_H, device="0", verbose=False)[0]
        cm = res.semantic_mask.data if getattr(res,"semantic_mask",None) is not None else np.zeros((IMG_H,IMG_W))
        cm = cm.cpu().numpy() if hasattr(cm,"cpu") else np.asarray(cm)
        return to640((cm==1).astype("uint8"))

    D = pd.DataFrame({n: per_image[n].set_index("stem").dice for n in CORE})
    mean_d = D.mean(axis=1).sort_values()
    picks = [mean_d.index[i] for i in [len(mean_d)-1, int(len(mean_d)*0.6), int(len(mean_d)*0.25), 0]]
    labels = ["easiest","typical","hard","hardest"]
    tdf = MAN["test"].set_index("stem")
    fig, axes = plt.subplots(len(picks), len(CORE)+1, figsize=(3*(len(CORE)+1), 3*len(picks)))
    for i, (stem, lab) in enumerate(zip(picks, labels)):
        row = tdf.loc[stem]
        img = load_rgb640(WORK/row.image_path)
        gt  = to640(cv2.imread(str(WORK/row.mask_path), cv2.IMREAD_GRAYSCALE))
        axes[i,0].imshow(overlay(img, gt, (40,190,40))); axes[i,0].set_ylabel(f"{lab}\n{stem}", fontsize=7)
        if i==0: axes[i,0].set_title("Ground truth", fontsize=9)
        for j, n in enumerate(CORE, start=1):
            m = predict_mask(n, WORK/row.image_path)
            axes[i,j].imshow(overlay(img, m))
            axes[i,j].set_xlabel(f"Dice {per_image[n].set_index('stem').dice[stem]:.2f}", fontsize=7.5)
            if i==0: axes[i,j].set_title(DISP[n], fontsize=8)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Predictions across easy → hard (green = truth, red = model)", fontsize=12, y=1.01)
    plt.tight_layout(); plt.show()
''')

code(r'''
if CFG.get("render_gallery", True):
    # Failure gallery: images where YOLO-direct returns EMPTY but SegFormer-B0-distilled finds it.
    yd = per_image["yolo_sem_direct"]
    misses = yd[yd.complete_miss].stem.tolist()[:3]
    tdf = MAN["test"].set_index("stem")
    if not misses:
        print("No complete misses for YOLO-direct in this best seed.")
    else:
        fig, axes = plt.subplots(len(misses), 3, figsize=(9, 3*len(misses))); axes = np.atleast_2d(axes)
        for i, stem in enumerate(misses):
            row = tdf.loc[stem]
            img = load_rgb640(WORK/row.image_path)
            gt  = to640(cv2.imread(str(WORK/row.mask_path), cv2.IMREAD_GRAYSCALE))
            y_m = predict_mask("yolo_sem_direct", WORK/row.image_path)
            s_m = predict_mask("segformer_b0_distilled", WORK/row.image_path)
            axes[i,0].imshow(overlay(img, gt, (40,190,40))); axes[i,0].set_ylabel(stem, fontsize=7)
            axes[i,1].imshow(overlay(img, y_m)); axes[i,2].imshow(overlay(img, s_m))
            axes[i,1].set_xlabel(f"{int(y_m.sum())} px", fontsize=7.5)
            axes[i,2].set_xlabel(f"Dice {per_image['segformer_b0_distilled'].set_index('stem').dice[stem]:.2f}", fontsize=7.5)
            if i==0:
                axes[i,0].set_title("Ground truth", fontsize=9)
                axes[i,1].set_title("YOLO-direct — EMPTY", fontsize=9, color="#d03b3b")
                axes[i,2].set_title("SegFormer-B0 distilled", fontsize=9)
        for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
        fig.suptitle("Complete misses: a real bruise, model returns nothing", fontsize=12, y=1.01)
        plt.tight_layout(); plt.show()
''')

md(r"""
## G2 · ⭐ Overlays by skin-tone (ITA) group

One representative image **per ITA group** — the image whose SegFormer-B0-distilled Dice
is closest to that group's median (the "typical" case, not a cherry-picked best/worst).
Columns: ground truth, the safe model (B0-distilled), and the miss-prone one
(YOLO-direct). This is the visual companion to sections D/E: eyeball how the two models
behave as skin tone darkens (and remember from E3 that light-skin bruises are also the
smallest, so tone and size move together here).
""")

code(r'''
if CFG.get("render_gallery", True):
    ITA_ORDER = ["Light (II-III)","Intermediate (III-IV)","Tan (IV)","Brown (V)","Dark (VI)"]
    groups = [g for g in ITA_ORDER if g in ITA.skin_tone_category.unique()]
    ref = "segformer_b0_distilled"
    dref = per_image[ref].set_index("stem").dice
    tdf  = MAN["test"].merge(ITA, on="stem").set_index("stem")
    COLS = [("Ground truth", None),
            (DISP["segformer_b0_distilled"], "segformer_b0_distilled"),
            (DISP["yolo_sem_direct"], "yolo_sem_direct")]
    picks = []
    for g in groups:
        stems = ITA[ITA.skin_tone_category == g].stem
        dd = dref.loc[dref.index.isin(stems)]
        picks.append(dd.sub(dd.median()).abs().idxmin())   # image closest to the group's median Dice
    fig, axes = plt.subplots(len(picks), 3, figsize=(9.5, 3*len(picks))); axes = np.atleast_2d(axes)
    for i, (g, stem) in enumerate(zip(groups, picks)):
        row = tdf.loc[stem]; img = load_rgb640(WORK/row.image_path)
        gt = to640(cv2.imread(str(WORK/row.mask_path), cv2.IMREAD_GRAYSCALE))
        for j, (title, mdl) in enumerate(COLS):
            if mdl is None:
                axes[i,j].imshow(overlay(img, gt, (40,190,40)))
            else:
                m = predict_mask(mdl, WORK/row.image_path)
                axes[i,j].imshow(overlay(img, m))
                axes[i,j].set_xlabel(f"Dice {per_image[mdl].set_index('stem').dice[stem]:.2f}", fontsize=7.5)
            if i == 0: axes[i,j].set_title(title, fontsize=8.5)
        axes[i,0].set_ylabel(f"{g.split(' ')[0]}\n{int(row.gt_positive_pixels) if 'gt_positive_pixels' in row else ''}", fontsize=7)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Typical prediction per ITA skin-tone group (green = truth, red = model)", fontsize=11.5, y=1.01)
    plt.tight_layout(); plt.show()
''')

md(r"""
## G3 · ⭐ Overlays by bruise-size category

Same idea, binned by **bruise size** instead of skin tone: one typical image per size
quartile (Q1 smallest → Q4 largest), same three columns. This makes the E1/E2 finding
concrete — watch the small-bruise row, where YOLO is most likely to blank while
SegFormer still finds something.
""")

code(r'''
if CFG.get("render_gallery", True):
    ref = "segformer_b0_distilled"
    sz = per_image[ref][["stem","gt_positive_pixels"]].copy()
    sz["bin"] = pd.qcut(sz.gt_positive_pixels, 4, labels=["Q1 smallest","Q2","Q3","Q4 largest"])
    tdf = MAN["test"].set_index("stem")
    COLS = [("Ground truth", None),
            (DISP["segformer_b0_distilled"], "segformer_b0_distilled"),
            (DISP["yolo_sem_direct"], "yolo_sem_direct")]
    bins = ["Q1 smallest","Q2","Q3","Q4 largest"]
    picks = []
    for b in bins:
        s = sz[sz.bin == b].set_index("stem").gt_positive_pixels
        picks.append(s.sub(s.median()).abs().idxmin())     # image with the bin's median size
    fig, axes = plt.subplots(len(picks), 3, figsize=(9.5, 3*len(picks))); axes = np.atleast_2d(axes)
    for i, (b, stem) in enumerate(zip(bins, picks)):
        row = tdf.loc[stem]; img = load_rgb640(WORK/row.image_path)
        gt = to640(cv2.imread(str(WORK/row.mask_path), cv2.IMREAD_GRAYSCALE))
        px = int(sz.set_index("stem").gt_positive_pixels[stem])
        for j, (title, mdl) in enumerate(COLS):
            if mdl is None:
                axes[i,j].imshow(overlay(img, gt, (40,190,40)))
            else:
                m = predict_mask(mdl, WORK/row.image_path)
                axes[i,j].imshow(overlay(img, m))
                axes[i,j].set_xlabel(f"Dice {per_image[mdl].set_index('stem').dice[stem]:.2f}", fontsize=7.5)
            if i == 0: axes[i,j].set_title(title, fontsize=8.5)
        axes[i,0].set_ylabel(f"{b}\n{px:,} px", fontsize=7)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Typical prediction per bruise-size quartile (green = truth, red = model)", fontsize=11.5, y=1.01)
    plt.tight_layout(); plt.show()
''')

# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════
md("## Save every table + figure to Drive")

code(r'''
import datetime, shutil
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ADIR = Path(CFG["drive_dir"]) / f"final_analysis_{stamp}"
ADIR.mkdir(parents=True, exist_ok=True)
for n in CORE:
    per_image[n].merge(SUBJ, on="stem").merge(ITA, on="stem").to_csv(ADIR/f"per_image_{n}.csv", index=False)
for n in YOLO_MODELS:
    per_image_custom[n].to_csv(ADIR/f"per_image_{n}_custom255.csv", index=False)
head.to_csv(ADIR/"headline.csv", index=False)
FOREST.to_csv(ADIR/"paired_contrasts.csv", index=False)
miss.to_csv(ADIR/"miss_rates.csv", index=False)
beat.to_csv(ADIR/"model_vs_paul.csv", index=False)
ceiling.to_csv(ADIR/"annotation_ceiling.csv", index=False)
FAIR_GROUP.to_csv(ADIR/"fairness_per_group_bestseed.csv", index=False)
FAIR_STATS.to_csv(ADIR/"fairness_stats_bestseed.csv", index=False)
FAIR_PAIR.to_csv(ADIR/"fairness_pairwise_bestseed.csv", index=False)
SIZE_COND.to_csv(ADIR/"size_conditioned_fairness_stratified.csv", index=False)
SIZE_COND_COEF.to_csv(ADIR/"size_conditioned_fairness_regression.csv", index=False)
print("saved ->", ADIR)
for f in sorted(ADIR.glob("*")): print("   ", f.name)
''')

md(r"""
### How to read this analysis

1. **Best-seed, one inference pass** — every figure shares it; 3-seed spread is on disk.
2. **Miss rate is the honest axis** (B1): it separates models by more than label noise.
3. **Fairness (D) is exploratory** at n=28 — direction, not proof; and **size (E3) is a
   confound** you must report as a covariate before claiming a skin-tone gap.
4. **The ceiling (F1) reframes everything**: model-to-model Dice gaps live below the
   ~0.36 of annotation noise, and a model trained on Paul beats Paul vs consensus (F2).
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
