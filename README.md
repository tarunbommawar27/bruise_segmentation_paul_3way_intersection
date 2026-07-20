# Forensic White-Light Bruise Segmentation — a 3-annotator intersection study

Code for a controlled comparison of semantic-segmentation models for **white-light (WL) bruise
segmentation** on a multi-annotator forensic dataset, with knowledge distillation and a strong
emphasis on **honest statistics** (subject-level clustering, failure-mode analysis, annotation-noise
ceiling).

> **Headline finding:** at the maximum sample size for which 3-annotator consensus labels exist
> (**28 subjects**), model-vs-model Dice differences are smaller than annotation noise — and a model
> trained on a *single* annotator agrees with the 3-expert consensus **better than that annotator
> does**. The binding constraint is label quality, not model capacity.

---

## 1. What this project is

The data are white-light photographs from the NIJ bruise dataset, labelled by multiple annotators.
The three highest-volume annotators — **Paul, Gbarimah, Erik** — were identified.

- **Test set** = the strict intersection of all three annotators: **185 images / 28 subjects**.
- **Target** = the per-pixel **2-of-3 majority vote** (not any single annotator's mask).
- **Training** = **Paul's** non-intersection subjects only (831 imgs / 115 subjects → **697 train /
  134 val**, subject-level split). Test subjects are structurally disjoint from training (0 overlap).

This makes the evaluation a **cross-annotator generalisation test**, not an ordinary split.
`n = 28` is a **ceiling, not a choice**: a 29th consensus subject cannot exist without new annotation.

## 2. Models compared (5 core + 2 baselines)

| Model | Role | Params | Input norm | Notes |
|---|---|---|---|---|
| SegFormer-B2 | teacher | 27.35 M | ImageNet | MiT-B2 encoder, 1-class head |
| SegFormer-B0 direct | student | 3.71 M | ImageNet | trained on GT only |
| SegFormer-B0 distilled | student | 3.71 M | ImageNet | calibrated soft-target KD, α=0.6 |
| YOLO26n-sem direct | detector | 1.63 M | /255 | native Ultralytics recipe |
| YOLO26n-sem distilled | detector | 1.63 M | /255 | offline teacher-fused pseudo-mask, α=0.4 |
| U-Net (ResNet-50) | baseline | 32.52 M | ImageNet | `segmentation_models_pytorch` |
| DeepLabV3+ (ResNet-50) | baseline | 26.68 M | ImageNet | `segmentation_models_pytorch` |

All models emit a **single bruise logit at full resolution**, so the loss, threshold sweep, metric,
and benchmark are architecture-blind.

**Two important architecture notes**
- YOLO must be fed **/255, not ImageNet-normalised** pixels — its BatchNorms carry running stats for
  the /255 distribution; ImageNet norm silently caps it at Dice ≈ 0.479.
- YOLO's head runs at **stride 8** (80×80 for a 640 input) vs SegFormer's **stride 4** (160×160) — a
  4× coarser grid that is an architectural ceiling on boundary precision.

## 3. Distillation

- **SegFormer (online, α=0.6):** calibrated soft-target distillation
  `L = α·DiceBCE(z_s, y) + (1−α)·BCE(z_s, σ(z_t / T_cal))`, where `T_cal` is a **calibration**
  temperature fitted by minimising validation NLL (temperature scaling, Guo et al. 2017). This is
  **not** the Hinton T² formulation — cite **Menon et al. 2021** (calibrated soft-target
  distillation) for the formula; Hinton et al. 2015 is the conceptual ancestor.
- **YOLO (offline, α=0.4):** the teacher can't be plugged into Ultralytics' trainer, so distillation
  is done **before training** by fusing the teacher's probability into the label:
  `class = (α·GT + (1−α)·teacher_prob ≥ 0.5)` (α<0.5 keeps it non-degenerate). YOLO then trains
  normally on the resulting hard **pseudo-mask**.

## 4. Evaluation & statistics

- Everything is scored on a common **640×640** grid; masks resized with nearest-neighbour.
- **Model selection** is on threshold-free validation AP; the decision threshold is **swept on
  validation and applied once to test** (never fitted on test).
- YOLO is reported **two ways**: **native argmax** (`.predict()`, its home turf, ~0.83 median) and
  **custom /255** (same 640 geometry as SegFormer). The gap between them is a *preprocessing* effect,
  not a threshold or resolution effect.
- **Primary safety metric = complete-miss rate**: fraction of bruise-containing images that receive a
  wholly empty predicted mask. A loose outline is correctable; a blank mask is a missed injury.
- Because 185 images come from only **28 subjects**, all CIs are **subject-level cluster bootstraps**
  (B=4000, resample subjects), and model-vs-model contrasts are **paired** (same resample scores both
  models). `P(Δ>0)` is reported alongside each interval.
- Fairness across skin tone uses **ITA** groups with Kruskal–Wallis (omnibus) + Mann–Whitney
  (Bonferroni). It is **exploratory** at 9–17 subjects/group.

## 5. Key results (val-selected best seed; see `docs/final_run_report.pdf`)

| Model | Mean Dice | Median Dice | Miss % | FPS (A100) |
|---|---|---|---|---|
| SegFormer-B2 teacher | 0.7692 | 0.8192 | 0.00 | 29.7 |
| SegFormer-B0 distilled | 0.7680 | 0.8167 | 0.00 | 60.1 |
| SegFormer-B0 direct | 0.7663 | 0.8129 | 0.54 | 59.9 |
| DeepLabV3+ (ResNet-50) | 0.7584 | 0.8183 | 2.16 | — |
| U-Net (ResNet-50) | 0.7570 | 0.8329 | 3.78 | — |
| YOLO26n distilled (native) | 0.7261 | 0.8012 | 2.16 | 121.7 |
| YOLO26n direct (native) | 0.7021 | 0.8061 | 6.49 | 122.2 |

**What's resolvable at n=28 (paired bootstrap):** distillation (+0.002) and student−teacher (−0.001)
are **null**; SegFormer−YOLO (+0.066) and the **miss-rate gap** (YOLO-direct − B0-distilled, +6.49 pp)
are **significant**. **Annotation ceiling:** the 3 experts agree with each other at only **0.639**
mean Dice, and every model beats that — a model trained on Paul alone beats **Paul** against the
consensus (+0.07 Dice, P≈0.98).

**Recommendation:** ship **SegFormer-B0 distilled** — teacher-level accuracy at 2× speed / 7× fewer
params, and the lowest miss rate. YOLO is 2× faster again but its 2–6 % blank-mask rate is
disqualifying for injury documentation.

## 6. Repository layout

This repository is intentionally scoped to the **final run + its analysis + the baselines**.
The three Colab notebooks are self-contained (they embed the library inline), so they run without
the rest of the tree; `pipeline/` and the three generator scripts are included as the reference
implementation.

```
bruise_colab_final.ipynb            The final run (5 models × 3 seeds, native YOLO, fairness).
bruise_colab_final_analysis.ipynb   Best-seed re-inference, annotation ceiling, paired contrasts,
                                    size confound, ~25 figures.
bruise_colab_baselines.ipynb        U-Net / DeepLabV3+ through the identical recipe.

pipeline/          Core library: data, models, losses, metrics, trainer, benchmark,
                   YOLO stage + threshold/temperature, shared stage runners.
scripts/           The three notebook generators (43 final, 44 analysis, 45 baselines)
                   that emit the notebooks above from pipeline/.
EXTRA/             Baseline trainers: train_smp_baselines.py (U-Net / DeepLabV3+),
                   train_nnunet_baseline.py.
configs/           paths.yaml, common_train.yaml, benchmark / model-registry yamls.
tests/             Unit tests for pipeline/.
docs/              LaTeX + PDF reports (final_run_report, pipeline_reference, …),
                   figures, and explanatory markdown.
results_final/     Aggregate result CSVs (per-seed, benchmark, fairness) — no raw data.
```

### The Colab notebooks
| Notebook | Purpose |
|---|---|
| `bruise_colab_final.ipynb` | The final run: 5 models × 3 seeds, native YOLO (two eval paths), fairness. |
| `bruise_colab_final_analysis.ipynb` | Best-seed re-inference, annotation ceiling, paired contrasts, size confound, ~25 figures. |
| `bruise_colab_baselines.ipynb` | U-Net / DeepLabV3+ through the identical recipe. |

## 7. What is NOT in this repo (by design)

The **dataset** (images, masks, individual-annotator labels), **model checkpoints/weights**,
**pretrained backbones**, **run/evaluation output directories**, and **large archives** are *not*
committed (see `.gitignore`). The NIJ bruise dataset is access-controlled; obtain it through the
appropriate channel and point `configs/paths.yaml` at your local copy.

## 8. Environment

```bash
conda env create -f environment.yml   # CPU-only torch; see the file header for the GPU pip step
# core deps: torch, transformers, ultralytics, segmentation-models-pytorch,
#            albumentations, opencv-python-headless, pandas, scipy, optuna, pytest
```

## 9. Reproducing the headline numbers

1. Obtain the dataset and set paths in `configs/paths.yaml`.
2. Run `bruise_colab_final.ipynb` (train 5 models × 3 seeds → `results_final/`).
3. Run `bruise_colab_baselines.ipynb` for U-Net / DeepLabV3+.
4. Run `bruise_colab_final_analysis.ipynb` for the annotation ceiling, paired contrasts, and figures.
5. All report numbers are read directly from the run CSVs — nothing is hand-transcribed.

## 10. Licensing note

For a non-commercial forensic tool both families are usable, but licenses differ: SegFormer's
pretrained weights carry an **NVIDIA non-commercial** license, and Ultralytics/YOLO is **AGPL-3.0**
(network copyleft). On both accuracy-safety (misses) and license risk, SegFormer is the preferable
deployment target, with YOLO retained as a fast baseline.

## Key references
- Xie et al., *SegFormer*, NeurIPS 2021.
- Jocher et al., *YOLO26*, arXiv:2606.03748, 2026.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 (temperature scaling).
- Menon et al., *A Statistical Perspective on Distillation*, ICML 2021 (calibrated soft-target KD).
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 (KD, conceptual origin).
- Milletari et al., *V-Net* (Dice loss), 3DV 2016.
- Chardon et al., *Individual Typology Angle*, 1991 (objective skin tone).
- Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993.
