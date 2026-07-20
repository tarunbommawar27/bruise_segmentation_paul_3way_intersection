#!/usr/bin/env python3
"""
scripts/46_generate_saved_eval_notebook.py  ->  bruise_colab_saved_eval.ipynb

An EVAL-ONLY notebook for the 5 already-trained models saved in the LOCAL top-level
dirs (segformer_b2_teacher/, segformer_b0_direct/, segformer_b0_distilled/,
yolo_sem_direct/, yolo_sem_distilled/) -- the exact models bruise_colab_inference_demo.ipynb
loads and that the user has presented. It TRAINS NOTHING: it loads the saved checkpoints
and reproduces the full bruise_colab_final.ipynb-style evaluation.

STYLE (user's choice): the inference_demo style -- logic is IMPORTED from the `pipeline/`
package inside the uploaded zip (bruise_colab_gpu_full.zip). Those pipeline/*.py files are
real, openable source in Colab (`??load_segformer_model`, file browser), so "click a
function -> see the source" still works; the source just lives in the package rather than
in %%writefile cells. This maximises reuse of the proven loaders that read the original
best_model.pt checkpoints correctly (transformers-version handling via
pipeline.models._pick_matching_checkpoint).

HOW IT IS BUILT
---------------
It reuses bruise_colab_inference_demo.ipynb's cells verbatim (mount, GPU, unzip, imports,
val/test load + leak/mask guards, GPU staging, load 5 models, 1-D threshold sweep on val,
benchmark, per-image accuracy, results table) and APPENDS the bruise_colab_final sections
that the demo lacks:
  * YOLO NATIVE argmax path (a second YOLO evaluation, its home turf)
  * combined headline table (all variants)
  * fairness across ITA skin tone
  * annotation ceiling + "model beats Paul"
  * an inference / overlay demo
ITA per-image labels and the inter-labeler Dice are EMBEDDED as literal CSVs (they are not
in bruise_colab_gpu_full.zip), so no new upload is needed. EDIT THIS GENERATOR, not the ipynb.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from textwrap import dedent

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_NB = PROJECT_ROOT / "bruise_colab_inference_demo.ipynb"
OUT = PROJECT_ROOT / "bruise_colab_saved_eval.ipynb"
ITA_CSV = PROJECT_ROOT / "ita_labels" / "wl_test_per_image_ita.csv"
IL_CSV = PROJECT_ROOT / "interlabeler_agreement_640.csv"

new_cells: list[dict] = []


def md(text: str) -> None:
    new_cells.append({"cell_type": "markdown", "metadata": {},
                      "source": dedent(text).strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    new_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": dedent(text).strip("\n").splitlines(keepends=True)})


def load_demo_cells() -> list[dict]:
    if not DEMO_NB.exists():
        raise SystemExit(f"{DEMO_NB} not found -- it supplies the reused eval cells.")
    return json.loads(DEMO_NB.read_text(encoding="utf-8"))["cells"]


def embed_csvs() -> tuple[str, str]:
    """Read the two local CSVs and return compact literal strings to embed.

    Only the columns the notebook needs are kept, so the embedded blocks stay small.
    """
    ita = pd.read_csv(ITA_CSV)[["stem", "subject", "skin_tone_category", "ita_group_index_5"]]
    ita_txt = ita.to_csv(index=False)
    il = pd.read_csv(IL_CSV)   # stem, subject, paul_vs_*, *_vs_majority
    il_txt = il.to_csv(index=False)
    assert len(ita) == 185 and len(il) == 185, (len(ita), len(il))
    return ita_txt, il_txt


ITA_TEXT, IL_TEXT = embed_csvs()

# ══════════════════════════════════════════════════════════════════════════════
# New title (prepended; replaces the demo's title cell)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
# Bruise segmentation — SAVED MODELS, evaluation & inference (no training)

The 5 already-trained models from the project's saved directories, run end to end with
**no training** — a `bruise_colab_final.ipynb`-style evaluation on the models you've
presented. Everything is loaded from `bruise_colab_gpu_full.zip` and scored on the fixed
185-image test set (thresholds fitted on the 134-image val set).

**What this notebook does (all eval):**
1. Load the 5 saved models as raw PyTorch modules (SegFormer via `pipeline`, YOLO via Ultralytics).
2. Fit each operating point with a 1-D threshold sweep **on val**.
3. Score on **test**: mean/median Dice, IoU, complete-miss rate.
4. Evaluate **YOLO two ways** — native argmax (its home turf) and custom /255 (SegFormer geometry).
5. Fairness across **ITA skin tone**; the **annotation ceiling** + "model beats Paul".
6. **Benchmark** (640→640 on GPU) and an **inference / overlay demo**.

Implementations live in the `pipeline/` package inside the zip (openable source: file
browser or `??func`), plus the inline functions in this notebook. Nothing is retrained —
the checkpoints are loaded and used as-is.
""")

# ── the reused inference_demo cells get spliced in here by the builder (see bottom) ──

# ══════════════════════════════════════════════════════════════════════════════
# §A · YOLO native argmax (second YOLO path)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §A · YOLO — native Ultralytics argmax (home-turf path)

The sweep above scored YOLO through the custom /255 raw-module path (same 640 geometry as
SegFormer). Here we also score it the way it was **trained** — native `YOLO.predict()`
letterbox + argmax — and bring prediction and GT to 640 together (nearest) to compare on
the same grid. Reported alongside the custom path so both YOLO numbers are visible.
""")

code(r'''
from ultralytics import YOLO as _YOLO_native

def _to640_nn(mask):
    m = np.asarray(mask)
    if m.ndim == 3: m = m[..., 0]
    return (cv2.resize(m.astype('uint8'), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST) > 0).astype('uint8')

native_per_image = {}
for run in ['yolo_sem_direct', 'yolo_sem_distilled']:
    w = _YOLO_native(MODELS[run]['ckpt'])
    rows = []
    for i, (_, r) in enumerate(test_df.iterrows()):
        res = w.predict(source=str(r.image_path), imgsz=IMG_H, device=0, verbose=False)[0]
        if getattr(res, 'semantic_mask', None) is not None:
            cm = res.semantic_mask.data
            cm = cm.cpu().numpy() if hasattr(cm, 'cpu') else np.asarray(cm)
            pred = _to640_nn((cm == 1).astype('uint8'))
        else:
            pred = np.zeros((IMG_H, IMG_W), np.uint8)
        gt = (GT_640[i, 0].numpy() > 0.5).astype('uint8')     # GT_640 order == STEMS order == test_df order
        rows.append(compute_image_row(pred, gt, STEMS[i]))
    df = pd.DataFrame(rows)
    df['complete_miss'] = (df.pred_positive_pixels == 0) & (df.gt_positive_pixels > 0)
    native_per_image[run] = df
    print(f"{MODELS[run]['display']:26s} native argmax  median={df.dice.median():.3f}  "
          f"mean={df.dice.mean():.3f}  miss={df.complete_miss.mean()*100:.2f}%")
''')

# ══════════════════════════════════════════════════════════════════════════════
# §B · Combined headline table
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §B · Combined headline table (all variants)

SegFormer (val-swept threshold) + YOLO custom /255 (val-swept) + YOLO native argmax.
Read `complete_miss_%` next to `median_Dice`: the best-Dice model is not necessarily the
safest.
""")

code(r'''
def _summ(df):
    return dict(median_Dice=round(df.dice.median(), 4), mean_Dice=round(df.dice.mean(), 4),
                median_IoU=round(df.iou.median(), 4),
                complete_miss_pct=round(df.complete_miss.mean()*100, 2))

rows = []
# SegFormer + YOLO custom path come from the demo's per_image dict
for run, m in MODELS.items():
    label = m['display'] + (' · custom/255' if run.startswith('yolo') else '')
    rows.append({'variant': label, 'params_M': round(m['params_m'], 2),
                 'threshold': round(m['threshold'], 4), **_summ(per_image[run])})
# YOLO native argmax path
for run in ['yolo_sem_direct', 'yolo_sem_distilled']:
    rows.append({'variant': MODELS[run]['display'] + ' · native argmax',
                 'params_M': round(MODELS[run]['params_m'], 2), 'threshold': float('nan'),
                 **_summ(native_per_image[run])})
HEADLINE = pd.DataFrame(rows)
print("185 test images | thresholds fitted on 134 val images\n" + "="*72)
print(HEADLINE.to_string(index=False))
HEADLINE
''')

# ══════════════════════════════════════════════════════════════════════════════
# §C · Subject + ITA labels (embedded)
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §C · Skin-tone (ITA) labels

Per-image ITA skin-tone group for the 185 test images, embedded here because it is not in
`bruise_colab_gpu_full.zip` (pure metadata, no model). Subject IDs come with it — needed
for the subject-level bootstrap in §E. The test set contains both `TAM009` and `TAM0009`
as **distinct** subjects, which is why IDs are loaded from a table, never parsed from stems.
""")

code('ITA_CSV_TEXT = """' + ITA_TEXT.strip() + '\n"""\n' + dedent(r'''
import io as _io
ita = pd.read_csv(_io.StringIO(ITA_CSV_TEXT))
assert len(ita) == 185 and ita.subject.nunique() == 28, (len(ita), ita.subject.nunique())
subjects = ita[['stem', 'subject']].copy()
print(f"{len(ita)} images / {ita.subject.nunique()} subjects | groups: {sorted(ita.skin_tone_category.unique())}")
'''))

# ══════════════════════════════════════════════════════════════════════════════
# §D · Fairness across ITA skin tone
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §D · Fairness across ITA skin tone (exploratory)

For a forensic tool, subgroup behaviour across skin tone is a primary concern. Groups are
ITA (objective, pixel-computed). **Exploratory at n=28** — each group has only ~9–17
*subjects*, so read direction, not proven significance. Omnibus Kruskal–Wallis + the
best−worst-group median-Dice gap + per-group complete-miss rate.
""")

code(r'''
from scipy.stats import kruskal

GROUP_ORDER = ["Light (II-III)", "Intermediate (III-IV)", "Tan (IV)", "Brown (V)", "Dark (VI)"]

def fairness_for(df, label):
    d = df.merge(ita[['stem', 'skin_tone_category']], on='stem', how='left')
    assert not d.skin_tone_category.isna().any()
    per_group, samples = [], []
    for g in [g for g in GROUP_ORDER if g in d.skin_tone_category.unique()]:
        s = d[d.skin_tone_category == g]
        per_group.append({'variant': label, 'group': g, 'n': len(s),
                          'median_dice': round(s.dice.median(), 4),
                          'mean_recall': round(s.recall.mean(), 4),
                          'miss_pct': round(((s.pred_positive_pixels == 0) & (s.gt_positive_pixels > 0)).mean()*100, 2)})
        samples.append(s.dice.values)
    H, p = kruskal(*samples)
    pg = pd.DataFrame(per_group)
    stat = {'variant': label, 'kruskal_p': round(p, 4), 'significant': bool(p < 0.05),
            'dice_gap': round(pg.median_dice.max() - pg.median_dice.min(), 4),
            'worst_group': pg.loc[pg.median_dice.idxmin(), 'group'],
            'max_miss_gap_pct': round(pg.miss_pct.max() - pg.miss_pct.min(), 2)}
    return pg, stat

variants = {MODELS[r]['display'] + (' · custom' if r.startswith('yolo') else ''): per_image[r] for r in MODELS}
for r in ['yolo_sem_direct', 'yolo_sem_distilled']:
    variants[MODELS[r]['display'] + ' · native'] = native_per_image[r]

fair_groups, fair_stats = [], []
for label, df in variants.items():
    pg, stat = fairness_for(df, label); fair_groups.append(pg); fair_stats.append(stat)
FAIR_GROUP = pd.concat(fair_groups, ignore_index=True)
FAIR_STATS = pd.DataFrame(fair_stats)
print(FAIR_STATS.to_string(index=False))

# per-group median-Dice heatmap
import numpy as _np
piv = FAIR_GROUP.pivot_table(index='group', columns='variant', values='median_dice').reindex(
    [g for g in GROUP_ORDER if g in FAIR_GROUP.group.unique()])
fig, ax = plt.subplots(figsize=(min(13, 2 + 1.5*piv.shape[1]), 4.2))
im = ax.imshow(piv.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha='right', fontsize=7)
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                color='#ffffff' if v > 0.55 else '#0b0b0b')
ax.set_title('Median Dice by ITA group (exploratory, n=28)', fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.7); plt.tight_layout(); plt.show()
FAIR_GROUP
''')

# ══════════════════════════════════════════════════════════════════════════════
# §E · Annotation ceiling + model beats Paul
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §E · Annotation ceiling & "does the model beat the annotator it learned from?"

The test masks are a 2-of-3 expert majority. How much do the experts agree with each
*other*? (Inter-labeler Dice embedded below — pure mask arithmetic.) If they agree at only
~0.64 and a model scores ~0.77 vs the consensus, the model is already past human–human
agreement. Then, with a paired subject-level bootstrap, we ask whether each model — trained
only on **Paul** — matches the 3-expert majority **better than Paul himself does**.
""")

code('IL_CSV_TEXT = """' + IL_TEXT.strip() + '\n"""\n' + dedent(r'''
import io as _io2
il = pd.read_csv(_io2.StringIO(IL_CSV_TEXT))
assert len(il) == 185

hp = ['paul_vs_gbarimah', 'paul_vs_erik', 'gbarimah_vs_erik']
hm = ['paul_vs_majority', 'gbarimah_vs_majority', 'erik_vs_majority']
HH = float(il[hp].values.mean())
print("Inter-annotator agreement (mean Dice, 185 images):")
for c in hp + hm:
    print(f"  {c:24s} {il[c].mean():.4f}")
print(f"  {'AVERAGE human-human':24s} {HH:.4f}\n")

# model vs Paul, paired subject bootstrap
RNG = np.random.default_rng(42)
def paired_beats_paul(model_df):
    m = il[['stem', 'subject', 'paul_vs_majority']].merge(
        model_df[['stem', 'dice']].rename(columns={'dice': 'model'}), on='stem')
    subs = m.subject.unique(); by = {s: m[m.subject == s] for s in subs}
    point = m.model.mean() - m.paul_vs_majority.mean()
    vals = np.array([(lambda x: x.model.mean() - x.paul_vs_majority.mean())(
        pd.concat([by[s] for s in RNG.choice(subs, len(subs), True)], ignore_index=True)) for _ in range(4000)])
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float((vals > 0).mean()), m.model.mean()

rows = []
for label, df in variants.items():
    d, lo, hi, pg0, mvm = paired_beats_paul(df)
    rows.append({'variant': label, 'model_vs_maj': round(mvm, 4), 'paul_vs_maj': round(il.paul_vs_majority.mean(), 4),
                 'delta': round(d, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                 'P(delta>0)': round(pg0, 4), 'beats_paul': lo > 0})
CEILING = pd.DataFrame(rows)
print("Model vs Paul (paired subject bootstrap, B=4000):")
print(CEILING.to_string(index=False))
CEILING
'''))

# ══════════════════════════════════════════════════════════════════════════════
# §F · Inference / overlay demo
# ══════════════════════════════════════════════════════════════════════════════
md(r"""
---
## §F · Inference demo — predictions on real test images

Ground truth (green) then each SegFormer/YOLO-custom prediction (red), on a spread of
easy → hard test images (ranked by mean Dice across models). This uses the exact
`bruise_mask_640` inference path defined above.
""")

code(r'''
def _load_rgb640(p):
    im = cv2.imread(str(p))
    return cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (IMG_W, IMG_H)) if im is not None else None
def _overlay(img, mask, color=(230, 60, 60), alpha=0.45):
    lay = np.zeros_like(img); lay[mask.astype(bool)] = color
    return cv2.addWeighted(lay, alpha, img, 1 - alpha, 0)

D = pd.DataFrame({run: per_image[run].set_index('stem').dice for run in MODELS})
order = D.mean(axis=1).sort_values()
picks = [order.index[i] for i in [len(order)-1, int(len(order)*0.6), int(len(order)*0.25), 0]]
labels = ['easiest', 'typical', 'hard', 'hardest']
tdf = test_df.set_index('stem')

runs = list(MODELS)
fig, axes = plt.subplots(len(picks), len(runs)+1, figsize=(2.7*(len(runs)+1), 2.7*len(picks)))
with torch.inference_mode():
    for i, (stem, lab) in enumerate(zip(picks, labels)):
        idx = STEMS.index(stem)
        img = _load_rgb640(tdf.loc[stem].image_path)
        gt = (GT_640[idx, 0].numpy() > 0.5).astype('uint8')
        axes[i, 0].imshow(_overlay(img, gt, (40, 190, 40))); axes[i, 0].set_ylabel(f'{lab}\n{stem}', fontsize=7)
        if i == 0: axes[i, 0].set_title('Ground truth', fontsize=9)
        for j, run in enumerate(runs, start=1):
            m = bruise_mask_640(MODELS[run], X_TEST[idx:idx+1])[0].to(torch.uint8).cpu().numpy()
            axes[i, j].imshow(_overlay(img, m))
            axes[i, j].set_xlabel(f"Dice {per_image[run].set_index('stem').dice[stem]:.2f}", fontsize=7.5)
            if i == 0: axes[i, j].set_title(MODELS[run]['display'], fontsize=7.5)
for a in axes.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle('Predictions across easy → hard (green = truth, red = model)', fontsize=12, y=1.01)
plt.tight_layout(); plt.show()
''')

# ══════════════════════════════════════════════════════════════════════════════
# §G · Save
# ══════════════════════════════════════════════════════════════════════════════
md("---\n## §G · Save everything to Drive")

code(r'''
import datetime, shutil
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = f'{LOCAL_DIR}/saved_eval'; os.makedirs(OUT, exist_ok=True)
for run, df in per_image.items():
    df.merge(subjects, on='stem', how='left').to_csv(f'{OUT}/per_image_{run}_custom.csv', index=False)
for run, df in native_per_image.items():
    df.merge(subjects, on='stem', how='left').to_csv(f'{OUT}/per_image_{run}_native.csv', index=False)
HEADLINE.to_csv(f'{OUT}/headline.csv', index=False)
FAIR_GROUP.to_csv(f'{OUT}/fairness_per_group.csv', index=False)
FAIR_STATS.to_csv(f'{OUT}/fairness_stats.csv', index=False)
CEILING.to_csv(f'{OUT}/annotation_ceiling_model_vs_paul.csv', index=False)
results.to_csv(f'{OUT}/results_accuracy_and_speed.csv', index=False)
dest = f'{DRIVE_DIR}/saved_eval_{stamp}'
shutil.copytree(OUT, dest)
print('Saved to:', dest)
print(f"\nKEY NUMBERS\n  human-human ceiling : {HH:.4f}\n  best model vs maj   : {CEILING.model_vs_maj.max():.4f}"
      f"\n  models beating Paul : {int(CEILING.beats_paul.sum())}/{len(CEILING)}")
''')

# ══════════════════════════════════════════════════════════════════════════════
# build: [new title] + demo cells [1:29] + new sections
# ══════════════════════════════════════════════════════════════════════════════
demo = load_demo_cells()
# demo[0] is the demo's title (replaced by ours). Keep everything from mount through the
# results table, i.e. up to (but not including) the demo's "Save everything to Drive" cell.
cut = None
for i, c in enumerate(demo):
    if c["cell_type"] == "markdown" and "Save everything to Drive" in "".join(c["source"]):
        cut = i
        break
if cut is None:
    # fallback: cut before the last code cell (the demo's save cell)
    for i in range(len(demo) - 1, -1, -1):
        if demo[i]["cell_type"] == "code":
            cut = i
            break
if cut is None:
    raise SystemExit("Could not locate where to splice the demo cells.")

title_cell = new_cells[0]
appended = new_cells[1:]
cells = [title_cell] + demo[1:cut] + appended

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
print(f"wrote {OUT}  ({len(cells)} cells: {n_code} code, {len(cells)-n_code} markdown; "
      f"reused demo cells 1..{cut-1}, appended {len(appended)} new)")
