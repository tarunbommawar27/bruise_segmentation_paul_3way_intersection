#!/usr/bin/env python3
"""
scripts/47_generate_saved_analysis_notebook.py  ->  bruise_colab_saved_analysis.ipynb

The figure-rich ANALYSIS notebook (like bruise_colab_final_analysis.ipynb) but for the 5
SAVED models in the local top-level dirs -- the ones bruise_colab_saved_eval.ipynb /
bruise_colab_inference_demo.ipynb load and the user presented. No training.

HOW IT IS BUILT (maximally reuse tested code)
---------------------------------------------
1. FRONT END (saved-model inference): reuse bruise_colab_inference_demo.ipynb's proven cells
   (mount, GPU, unzip, `pipeline` imports, load val/test + guards, stage on GPU, load the 5
   saved models, 1-D threshold sweep on val, benchmark, per-image accuracy). Then add: a
   YOLO native-argmax pass, embedded ITA + inter-labeler CSVs, and an ADAPTER cell that maps
   these structures onto the exact variable names bruise_colab_final_analysis.ipynb's figures
   expect (CORE / DISP / PALETTE / per_image / per_image_custom / SUBJ / ITA / MAN /
   SEG_MODELS / YOLO_BEST / cluster_ci / paired_delta / merged / fairness_analysis / BENCH_DF).
2. FIGURES: splice bruise_colab_final_analysis.ipynb's statistical figure cells (sections
   A..F) VERBATIM -- only 3 path patches (interlabeler read -> embedded, benchmark read ->
   computed, output folder name). The gallery is NOT reused (final_analysis feeds /255 to a
   model that self-normalises; the saved pipeline SegFormer expects pre-normalised input), so
   fresh gallery cells are written that use inference_demo's correct `bruise_mask_640` path.

EDIT THIS GENERATOR, not the .ipynb.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEMO_NB = ROOT / "bruise_colab_inference_demo.ipynb"
FINAL_ANALYSIS_NB = ROOT / "bruise_colab_final_analysis.ipynb"
OUT = ROOT / "bruise_colab_saved_analysis.ipynb"
ITA_CSV = ROOT / "ita_labels" / "wl_test_per_image_ita.csv"
IL_CSV = ROOT / "interlabeler_agreement_640.csv"

mid_cells: list[dict] = []


def md(text: str) -> None:
    mid_cells.append({"cell_type": "markdown", "metadata": {},
                      "source": dedent(text).strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    mid_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": dedent(text).strip("\n").splitlines(keepends=True)})


def load_cells(p: Path) -> list[dict]:
    if not p.exists():
        raise SystemExit(f"{p} not found.")
    return json.loads(p.read_text(encoding="utf-8"))["cells"]


def embed_csvs() -> tuple[str, str]:
    ita = pd.read_csv(ITA_CSV)[["stem", "subject", "skin_tone_category", "ita_group_index_5"]]
    il = pd.read_csv(IL_CSV)
    assert len(ita) == 185 and len(il) == 185
    return ita.to_csv(index=False), il.to_csv(index=False)


ITA_TEXT, IL_TEXT = embed_csvs()

# ══════════════════════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
# Bruise segmentation — SAVED MODELS: analysis & visualisation

The same figure suite as `bruise_colab_final_analysis.ipynb`, but for the **5 saved
models** in the project's local dirs (loaded via `pipeline`, exactly as
`bruise_colab_inference_demo.ipynb`). **No training** — checkpoints are loaded and scored.

Front end (below) loads the models, sweeps thresholds on val, and scores test (SegFormer +
YOLO custom /255 + YOLO native argmax). Then the analysis sections build every chart:
accuracy distributions, safety/miss, paired subject-bootstrap contrasts, per-subject
heatmap, fairness (ITA), bruise size + the size-conditioned regression, the annotation
ceiling + "model beats Paul", speed, and qualitative galleries (easy→hard, by skin tone,
by bruise size). Colourblind palette; every chart prints its table.
""")

# ── middle cells (after the reused demo cells): native argmax, embeds, adapter ──
md(r"""
---
## YOLO native argmax pass

The demo scored YOLO through custom /255. We also score native `.predict()` argmax (YOLO's
home turf) so the analysis can use it as the core YOLO number and compare the two paths.
""")

code(r'''
from ultralytics import YOLO as _YOLO_native
def _to640_nn(mask):
    m = np.asarray(mask)
    if m.ndim == 3: m = m[..., 0]
    return (cv2.resize(m.astype('uint8'), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 0).astype('uint8')
native_per_image = {}
for run in ['yolo_sem_direct', 'yolo_sem_distilled']:
    w = _YOLO_native(MODELS[run]['ckpt']); rows = []
    for i, (_, r) in enumerate(test_df.iterrows()):
        res = w.predict(source=str(r.image_path), imgsz=IMG_H, device=0, verbose=False)[0]
        if getattr(res, 'semantic_mask', None) is not None:
            cm = res.semantic_mask.data; cm = cm.cpu().numpy() if hasattr(cm, 'cpu') else np.asarray(cm)
            pred = _to640_nn((cm == 1).astype('uint8'))
        else:
            pred = np.zeros((IMG_H, IMG_W), np.uint8)
        gt = (GT_640[i, 0].numpy() > 0.5).astype('uint8')
        rows.append(compute_image_row(pred, gt, STEMS[i]))
    df = pd.DataFrame(rows); df['complete_miss'] = (df.pred_positive_pixels == 0) & (df.gt_positive_pixels > 0)
    native_per_image[run] = df
    print(f"{MODELS[run]['display']:26s} native argmax median={df.dice.median():.3f} miss={df.complete_miss.mean()*100:.2f}%")
''')

md("## Skin-tone (ITA) + inter-labeler labels (embedded — not in the zip)")

code('ITA_CSV_TEXT = """' + ITA_TEXT.strip() + '\n"""\n' + dedent(r'''
import io as _io
ita = pd.read_csv(_io.StringIO(ITA_CSV_TEXT))
assert len(ita) == 185 and ita.subject.nunique() == 28
print("ITA:", ita.shape, "| subjects:", ita.subject.nunique())
'''))

code('IL_CSV_TEXT = """' + IL_TEXT.strip() + '\n"""\n' + dedent(r'''
il = pd.read_csv(_io.StringIO(IL_CSV_TEXT))
assert len(il) == 185
print("inter-labeler:", il.shape, "| human-human avg:",
      round(il[['paul_vs_gbarimah','paul_vs_erik','gbarimah_vs_erik']].values.mean(), 4))
'''))

md(r"""
## Adapter — map the saved-model structures onto the analysis figures' variables

Everything below this cell is the tested `bruise_colab_final_analysis.ipynb` figure code,
which expects `CORE / DISP / PALETTE / per_image / per_image_custom / SUBJ / ITA / MAN /
SEG_MODELS / YOLO_BEST / cluster_ci / paired_delta / merged / fairness_analysis / BENCH_DF`.
This cell builds all of them from the saved-model run above. The **core YOLO path is native
argmax**; the custom /255 path is kept as `per_image_custom` for the two-path comparison.
""")

code(r'''
!pip install -q statsmodels

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kruskal, mannwhitneyu

# --- names / palette (fixed slot per model, colourblind-safe) ---
SEGFORMER_MODELS = ['segformer_b2_teacher', 'segformer_b0_direct', 'segformer_b0_distilled']
YOLO_MODELS      = ['yolo_sem_direct', 'yolo_sem_distilled']
CORE = SEGFORMER_MODELS + YOLO_MODELS
DISP = {r: MODELS[r]['display'] for r in CORE}
PALETTE = {'segformer_b2_teacher':'#2a78d6', 'segformer_b0_direct':'#1baf7a',
           'segformer_b0_distilled':'#eda100', 'yolo_sem_direct':'#008300', 'yolo_sem_distilled':'#4a3aa7'}
INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
plt.rcParams.update({'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','axes.edgecolor':'#c3c2b7',
    'axes.labelcolor':INK,'text.color':INK,'xtick.color':MUTED,'ytick.color':MUTED,'grid.color':GRID,
    'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'figure.dpi':120})

# --- per-image tables in the shape the figures want (enriched) ---
def enrich(df):
    df = df.copy()
    df['complete_miss'] = (df.pred_positive_pixels == 0) & (df.gt_positive_pixels > 0)
    tp = (df.recall * df.gt_positive_pixels).round()
    df['tp_pixels'] = tp
    df['fp_pixels'] = (df.pred_positive_pixels - tp).clip(lower=0)
    df['fn_pixels'] = (df.gt_positive_pixels - tp).clip(lower=0)
    df['pred_gt_ratio'] = df.pred_positive_pixels / df.gt_positive_pixels.replace(0, np.nan)
    return df

_saved_pi = dict(per_image)                     # demo's per_image: SegFormer + YOLO custom /255
per_image = {}
for s in SEGFORMER_MODELS: per_image[s] = enrich(_saved_pi[s])
for y in YOLO_MODELS:      per_image[y] = enrich(native_per_image[y])       # core YOLO = native argmax
per_image_custom = {y: enrich(_saved_pi[y]) for y in YOLO_MODELS}           # custom /255 (two-path fig)

# --- subject + ITA tables ---
SUBJ = ita[['stem', 'subject']].copy()
ITA  = ita[['stem', 'skin_tone_category', 'ita_group_index_5']].copy()
MAN  = {'test': test_df.merge(ITA, on='stem', how='left')}                  # image_path/mask_path are absolute

# --- handles for the gallery ---
SEG_MODELS = {s: (MODELS[s]['model'], MODELS[s]['cut']) for s in SEGFORMER_MODELS}
YOLO_BEST  = {y: MODELS[y]['ckpt'] for y in YOLO_MODELS}

# --- config + placeholders the spliced figures reference ---
CFG = {'drive_dir': DRIVE_DIR, 'img_size': IMG_H, 'render_gallery': True}
OUT_DIR = Path(DRIVE_DIR)                        # A1 tries OUT_DIR/FINAL_RESULTS.csv (absent -> handled)
BEST = {r: {'seed': 'saved', 'cut': MODELS[r].get('cut')} for r in CORE}

# --- benchmark table for the speed figure (from the demo's `bench` dict) ---
BENCH_DF = pd.DataFrame([{'model': run, 'median_ms': bench[run]['median_ms'],
                          'p95_ms': bench[run]['p95_ms'], 'fps': bench[run]['fps'],
                          'params_M': MODELS[run]['params_m']} for run in CORE])

# --- subject-level bootstrap helpers (verbatim from the final-analysis notebook) ---
RNG_SEED = 42
def _subject_index(df):
    subs = df['subject'].unique(); return subs, {s: df[df.subject == s] for s in subs}
def cluster_ci(frame, stat, B=4000, seed=RNG_SEED):
    rng = np.random.default_rng(seed); subs, by = _subject_index(frame)
    vals = np.array([stat(pd.concat([by[s] for s in rng.choice(subs, len(subs), True)], ignore_index=True)) for _ in range(B)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
def paired_delta(frame, stat, B=4000, seed=RNG_SEED):
    rng = np.random.default_rng(seed); subs, by = _subject_index(frame); point = stat(frame)
    vals = np.array([stat(pd.concat([by[s] for s in rng.choice(subs, len(subs), True)], ignore_index=True)) for _ in range(B)])
    return {'delta': float(point), 'ci_lo': float(np.percentile(vals, 2.5)),
            'ci_hi': float(np.percentile(vals, 97.5)), 'p_gt0': float((vals > 0).mean())}
def merged(a, b, cols=('dice',)):
    left = per_image[a][['stem', *cols]].rename(columns={c: f'{c}_a' for c in cols})
    right = per_image[b][['stem', *cols]].rename(columns={c: f'{c}_b' for c in cols})
    return left.merge(right, on='stem').merge(SUBJ, on='stem')

# --- fairness_analysis (ported from bruisekit.evaluate; same output schema) ---
from scipy import stats as _st
def _bootstrap_ci(values, n=2000, seed=0):
    if len(values) < 2: return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))
def fairness_analysis(per_image_df, manifest, model_name):
    df = per_image_df.merge(manifest[['stem', 'skin_tone_category', 'ita_group_index_5']],
                            on='stem', how='left', validate='one_to_one')
    per_group, samples = [], []
    for gidx, g in sorted(df.groupby('ita_group_index_5'), key=lambda kv: kv[0]):
        vals = g['dice'].to_numpy(); lo, hi = _bootstrap_ci(vals)
        per_group.append({'model': model_name, 'ita_group_index_5': int(gidx),
                          'skin_tone_category': g['skin_tone_category'].iloc[0], 'n_images': len(g),
                          'median_dice': float(np.median(vals)),
                          'iqr_dice': float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                          'ci95_lo': lo, 'ci95_hi': hi, 'mean_recall': float(g['recall'].mean()),
                          'miss_rate': float(((g['pred_positive_pixels'] == 0) & (g['gt_positive_pixels'] > 0)).mean())})
        samples.append(vals)
    H, p = _st.kruskal(*samples)
    pairs = [(i, j) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    pairwise = []
    for i, j in pairs:
        pv = _st.mannwhitneyu(samples[i], samples[j], alternative='two-sided').pvalue
        adj = min(1.0, pv * len(pairs))
        pairwise.append({'model': model_name, 'group_a': per_group[i]['skin_tone_category'],
                         'group_b': per_group[j]['skin_tone_category'], 'pvalue': pv,
                         'bonferroni_p': adj, 'significant': bool(adj < 0.05)})
    pg = pd.DataFrame(per_group)
    best, worst = pg.loc[pg['median_dice'].idxmax()], pg.loc[pg['median_dice'].idxmin()]
    stat = {'model': model_name, 'kruskal_H': float(H), 'kruskal_p': float(p), 'significant': bool(p < 0.05),
            'fairness_gap': float(best['median_dice'] - worst['median_dice']),
            'best_group': best['skin_tone_category'], 'worst_group': worst['skin_tone_category'],
            'max_miss_rate_gap': float(pg['miss_rate'].max() - pg['miss_rate'].min())}
    return {'per_group': pg, 'pairwise': pd.DataFrame(pairwise), 'stats': stat}

print("adapter ready:", len(CORE), "core variants |", {k: len(v) for k, v in per_image.items()})
''')

# ── fresh gallery cells (correct SegFormer normalisation via bruise_mask_640) ──
md(r"""
---
## Qualitative galleries (saved-model inference path)

These use inference_demo's `bruise_mask_640` for SegFormer/YOLO-custom (correct ImageNet
normalisation for the pipeline SegFormer) and native `.predict()` for the core YOLO number.
Three views: easy→hard, by ITA skin-tone group, and by bruise-size quartile.
""")

code(r'''
def _rgb640(p):
    im = cv2.imread(str(p)); return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H)) if im is not None else None
def _overlay(img, mask, color=(230, 60, 60), a=0.45):
    lay = np.zeros_like(img); lay[mask.astype(bool)] = color; return cv2.addWeighted(lay, a, img, 1 - a, 0)
_STEM2IDX = {s: i for i, s in enumerate(STEMS)}
def saved_predict_mask(run, stem, img_path):
    if run in SEG_MODELS:                                   # SegFormer: correct via staged /ImageNet tensor
        idx = _STEM2IDX[stem]
        return bruise_mask_640(MODELS[run], X_TEST[idx:idx+1])[0].to(torch.uint8).cpu().numpy()
    res = _YOLO_native(YOLO_BEST[run]).predict(str(img_path), imgsz=IMG_H, device=0, verbose=False)[0]
    cm = res.semantic_mask.data if getattr(res, 'semantic_mask', None) is not None else np.zeros((IMG_H, IMG_W))
    cm = cm.cpu().numpy() if hasattr(cm, 'cpu') else np.asarray(cm)
    return _to640_nn((cm == 1).astype('uint8'))

def gallery(picks, row_labels, title):
    tdf = MAN['test'].set_index('stem')
    fig, axes = plt.subplots(len(picks), len(CORE)+1, figsize=(2.7*(len(CORE)+1), 2.7*len(picks)))
    axes = np.atleast_2d(axes)
    for i, (stem, lab) in enumerate(zip(picks, row_labels)):
        row = tdf.loc[stem]; img = _rgb640(row.image_path)
        gt = (GT_640[_STEM2IDX[stem], 0].numpy() > 0.5).astype('uint8')
        axes[i, 0].imshow(_overlay(img, gt, (40, 190, 40))); axes[i, 0].set_ylabel(lab, fontsize=7)
        if i == 0: axes[i, 0].set_title('Ground truth', fontsize=9)
        for j, run in enumerate(CORE, start=1):
            m = saved_predict_mask(run, stem, row.image_path)
            axes[i, j].imshow(_overlay(img, m))
            axes[i, j].set_xlabel(f"Dice {per_image[run].set_index('stem').dice[stem]:.2f}", fontsize=7)
            if i == 0: axes[i, j].set_title(DISP[run], fontsize=7)
    for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=12, y=1.01); plt.tight_layout(); plt.show()

# easy -> hard (by mean Dice across core models)
_D = pd.DataFrame({r: per_image[r].set_index('stem').dice for r in CORE})
_ord = _D.mean(axis=1).sort_values()
_picks = [_ord.index[k] for k in [len(_ord)-1, int(len(_ord)*0.6), int(len(_ord)*0.25), 0]]
gallery(_picks, [f'{lab}\n{s}' for lab, s in zip(['easiest','typical','hard','hardest'], _picks)],
        'Predictions across easy → hard (green = truth, red = model)')
''')

code(r'''
# by ITA skin-tone group (typical image per group = closest to that group's median Dice)
_ref = 'segformer_b0_distilled'; _dref = per_image[_ref].set_index('stem').dice
_ORDER = ["Light (II-III)","Intermediate (III-IV)","Tan (IV)","Brown (V)","Dark (VI)"]
_groups = [g for g in _ORDER if g in ITA.skin_tone_category.unique()]
_picks = []
for g in _groups:
    st = ITA[ITA.skin_tone_category == g].stem; dd = _dref.loc[_dref.index.isin(st)]
    _picks.append(dd.sub(dd.median()).abs().idxmin())
gallery(_picks, [f"{g.split(' ')[0]}\n{s}" for g, s in zip(_groups, _picks)],
        'Typical prediction per ITA skin-tone group')
''')

code(r'''
# by bruise-size quartile (typical image per quartile = closest to that quartile's median size)
_sz = per_image[_ref][['stem','gt_positive_pixels']].copy()
_sz['q'] = pd.qcut(_sz.gt_positive_pixels, 4, labels=['Q1 smallest','Q2','Q3','Q4 largest'])
_picks, _labs = [], []
for b in ['Q1 smallest','Q2','Q3','Q4 largest']:
    s = _sz[_sz.q == b].set_index('stem').gt_positive_pixels
    stem = s.sub(s.median()).abs().idxmin(); _picks.append(stem)
    _labs.append(f"{b}\n{int(s[stem]):,} px")
gallery(_picks, _labs, 'Typical prediction per bruise-size quartile')
''')

# ══════════════════════════════════════════════════════════════════════════════
# build: [title] + demo cells [1..results] + [native, ita, il, adapter, galleries] + final-analysis figures A..F + save
# ══════════════════════════════════════════════════════════════════════════════
demo = load_cells(DEMO_NB)
# demo cells 1..(before "Save everything to Drive")
cut = next((i for i, c in enumerate(demo)
            if c["cell_type"] == "markdown" and "Save everything to Drive" in "".join(c["source"])), None)
if cut is None:
    raise SystemExit("could not find demo save marker")
demo_reused = demo[1:cut]

fa = load_cells(FINAL_ANALYSIS_NB)
a_start = next((i for i, c in enumerate(fa)
               if c["cell_type"] == "markdown" and "# A · Accuracy" in "".join(c["source"])), None)
gal_start = next((i for i, c in enumerate(fa)
                 if c["cell_type"] == "markdown" and "# G · Qualitative" in "".join(c["source"])), None)
save_start = next((i for i, c in enumerate(fa)
                  if c["cell_type"] == "markdown" and "Save every table" in "".join(c["source"])), None)
if None in (a_start, gal_start, save_start):
    raise SystemExit(f"could not locate final-analysis splice points: {a_start}, {gal_start}, {save_start}")

PATCHES = [
    ('IL = pd.read_csv(WORK / "interlabeler_agreement_640.csv")', 'IL = il.copy()'),
    ('B = pd.read_csv(OUT_DIR/"benchmark_640.csv")', 'B = BENCH_DF.copy()'),
    ('final_analysis_{stamp}', 'saved_analysis_{stamp}'),
]
def patch(cell):
    c = dict(cell)
    src = "".join(cell["source"])
    for a, b in PATCHES:
        src = src.replace(a, b)
    c["source"] = src.splitlines(keepends=True)
    return c

figures = [patch(c) for c in fa[a_start:gal_start]]      # sections A..F (skip final's gallery)
save_cells = [patch(c) for c in fa[save_start:]]         # save + "how to read"

cells = [mid_cells[0]] + demo_reused + mid_cells[1:] + figures + save_cells

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
print(f"  reused demo cells 1..{cut-1}; final-analysis figures {a_start}..{gal_start-1} + save {save_start}..{len(fa)-1}")
