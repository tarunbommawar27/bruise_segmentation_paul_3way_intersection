# Final Results — Bruise Segmentation

**Source:** `bruise_colab_final.ipynb` + `results_final/`
**Test set:** 185 white-light images / 28 subjects (strict 3-labeler intersection), scored against the **2-of-3 majority vote** on a common 640×640 grid.
**YOLO evaluation:** native Ultralytics `.predict()` argmax.
**Metric note:** Dice is strongly bimodal (a model either localizes the bruise or blanks). **Median** = typical-case quality; **mean** is dragged down by complete misses; **miss %** = the safety axis (images with a bruise where the model predicts *zero* pixels).

---

## 1. Accuracy — test results (185 images)

| Model | Params (M) | Mean Dice | Median Dice | Complete-miss % |
|---|---|---|---|---|
| **SegFormer-B2 (teacher)** | 27.35 | **0.7692** | 0.8192 | **0.00** |
| **SegFormer-B0 (distilled, α=0.6)** | 3.71 | **0.7680** | 0.8167 | **0.00** |
| SegFormer-B0 (direct) | 3.71 | 0.7663 | 0.8129 | 0.54 |
| YOLO26n (distilled, α=0.4) | 1.63 | 0.7261 | 0.8012 | 2.16 |
| YOLO26n (direct) | 1.63 | 0.7021 | 0.8061 | 6.49 |

The three SegFormers are statistically indistinguishable on Dice (~0.77 mean / ~0.81 median) but separate sharply on **miss rate**: the teacher never blanks, the distilled student never blanks. YOLO's *median* is competitive (~0.80) but its *mean* is lower because it blanks on 2–6 % of images.

---

## 2. External baselines (val-selected best seed — same policy as the 5 models above)

| Model | Params (M) | Mean Dice | Median Dice | Complete-miss % |
|---|---|---|---|---|
| DeepLabV3+ (ResNet-50) | 26.68 | 0.7584 | 0.8183 | 2.16 |
| U-Net (ResNet-50) | 32.52 | 0.7570 | 0.8329 | 3.78 |

Both baselines are now selected the **same way as the five core models** — highest **validation** Dice across 3 seeds, then that seed's test result — so the comparison is symmetric (best-of-3 for everyone). They land just below the three SegFormers on mean Dice and blank more often; the SegFormer distilled student remains the best of everything tested and, at 3.71 M params, one of the smallest. (U-Net posts the highest *median* Dice of any model, 0.8329, but its larger mean-vs-median gap and 3.78 % miss rate reflect more complete failures.)

---

## 3. Speed / latency (640-tensor-on-GPU → mask, per image)

Timed inside a double `cuda.synchronize()`; disk / decode / resize / copy excluded (identical across models, I/O-dominated).

| Model | Median (ms) | p95 (ms) | FPS | Params (M) | Peak activation (MB) |
|---|---|---|---|---|---|
| SegFormer-B2 (teacher) | 33.67 | 34.95 | 29.7 | 27.35 | 1779 |
| SegFormer-B0 (direct) | 16.68 | 16.94 | 59.9 | 3.71 | 1204 |
| SegFormer-B0 (distilled) | 16.64 | 17.07 | 60.1 | 3.71 | 1204 |
| YOLO26n (direct) | 8.18 | 8.36 | 122.2 | 1.63 | 967 |
| YOLO26n (distilled) | 8.22 | 8.51 | 121.7 | 1.63 | 967 |

YOLO is ~2× faster and ~7× smaller than B2; the B0 student matches the B2 teacher's accuracy at 2× the teacher's speed.

---

## 4. Paired contrasts (subject-level cluster bootstrap)

Δ = model A − model B on the same resample. `P(Δ>0)` is one-sided.

| Contrast | Δ | 95% CI | P(Δ>0) | Verdict |
|---|---|---|---|---|
| Distillation (B0-dist − B0-direct), Dice | +0.0017 | [−0.009, +0.013] | 0.58 | n.s. (coin flip) |
| Student − teacher (B0-dist − B2), Dice | −0.0012 | [−0.020, +0.018] | 0.44 | n.s. (student = teacher) |
| YOLO distilled − direct, Dice | +0.0239 | [−0.014, +0.059] | 0.90 | n.s. |
| **SegFormer − YOLO (B0-dist − YOLO-dir), Dice** | **+0.0659** | **[+0.011, +0.121]** | **0.994** | **SIGNIFICANT** |
| **Miss %: YOLO-direct − B0-distilled** | **+6.49** | **[+2.39, +11.63]** | **1.00** | **SIGNIFICANT** |

**The takeaway:** at n=28 subjects, Dice differences between healthy models (distillation, student-vs-teacher) are smaller than the resolvable effect — they are null. What *is* resolvable lives in the **miss rate** and in the SegFormer-vs-YOLO gap.

---

## 5. Fairness across skin tone (ITA groups)

Per-image Dice averaged over runs, stratified by 5 ITA groups. Kruskal–Wallis omnibus + Mann–Whitney (Bonferroni over 10 pairs). Fairness gap = best − worst group median.

| Model | Omnibus p | Significant? | Dice gap | Worst group | Miss-rate gap |
|---|---|---|---|---|---|
| SegFormer-B2 (teacher) | 0.011 | **yes** | 0.112 | Tan (IV) | 0.00 |
| SegFormer-B0 (direct) | 0.638 | no | 0.052 | Dark (VI) | 0.00 |
| SegFormer-B0 (distilled) | 0.470 | no | 0.050 | Dark (VI) | 0.00 |
| YOLO26n (direct) | 0.230 | no | 0.104 | Light (II-III) | 0.154 |
| YOLO26n (distilled) | 0.151 | no | 0.090 | Light (II-III) | 0.051 |

**Only one pairwise comparison survives Bonferroni:** SegFormer-B2 Intermediate (III-IV) vs Dark (VI), p=0.0079. Everything else is n.s. after correction. Note YOLO's worst group is **Light**, not Dark — driven by a size confound (light-skin bruises are smaller, so more misses), and the intervals are too wide at 9–17 subjects/group to report as a fairness finding. At n=28, subgroup gaps are not reliably resolvable.

---

## Bottom line

1. **Ship SegFormer-B0 distilled** — matches the B2 teacher's accuracy (Δ n.s.) at 2× the speed and 7× fewer params, and never blanks (0/185).
2. **Keep B2 teacher** as the distillation source and a model that also never blanks (0/185).
3. **YOLO** is 2× faster / 7× smaller but blanks on 2–6 % of images → risky for injury documentation despite a competitive median. If reported, lead with **miss rate**, not Dice.
4. **Model differences are label-limited:** at n=28 subjects, the only resolvable accuracy signals are the miss rate and the SegFormer-vs-YOLO gap; healthy-model Dice differences are null.
