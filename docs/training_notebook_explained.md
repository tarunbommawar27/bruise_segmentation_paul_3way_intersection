# `bruise_colab_train_all.ipynb` — what it does, why, and where each choice comes from

*Written 2026-07-16. Companion to `docs/inference_demo_explained.md`, which covers
the inference/eval notebook. This one covers **training**. Updated 2026-07-16 with
the completed matched run (Part V) and the per-model-batch variant (Part VI).*

---

## TL;DR

One notebook trains all five models from scratch on Colab, three seeds each, then
fits operating points on val, scores on test, runs the fairness analysis, benchmarks
speed, and writes every table to Drive. It survives session death: every run
checkpoints to Drive and Run All resumes exactly where it stopped.

It exists because an audit of the previous training pipeline found **four
correctness bugs**, two of which invalidate claims in the current draft:

1. 🔴 **The B2-vs-B0 comparison was confounded.** A VRAM probe silently gave B0 a
   4× larger batch and a quarter of the gradient updates, at the same LR.
2. 🔴 **YOLO's "distillation" was algebraically a no-op** for α > 0.5 — and the
   Optuna log proves it, with six trials returning bit-identical numbers.
3. 🔴 **Both Optuna alpha searches selected noise**, at ~1/7 the scale of their own
   seed-to-seed spread.
4. 🔴 **`pipeline/yolo_threshold_temp.py` feeds YOLO ImageNet-normalised pixels**,
   which caps it at Dice 0.479.

Sections 1–4 document those. Sections 5–14 document what the new notebook does
instead and which paper each decision comes from.

**The matched run has now been done** (all 15 runs, A100). **Part V** has the real
numbers — and the headline flipped: with the batch confound removed, **B2 is now
clearly on top and "the student beats the teacher" is gone**, distillation is inside
seed noise, and YOLO's weakness is a seed-unstable miss rate. **Part VI** documents a
second notebook, `bruise_colab_train_per_model_batch.ipynb`, that trains each model at
its own best batch for speed — deliberately *not* recipe-matched, pending its own run.

### The two notebooks

| notebook | batch policy | use it for |
|---|---|---|
| `bruise_colab_train_all.ipynb` | effective 8 for **every** model | the fair B0-vs-B2 comparison — the paper |
| `bruise_colab_train_per_model_batch.ipynb` | each model's **own** largest batch | faster training / GPU utilisation — not recipe-matched |

Both are generated from one source (`scripts/41_generate_training_notebook.py`,
`--per-model` for the second) and are byte-identical except the batch logic in
`engine.py::resolve_micro_batch`. They write to separate Drive folders
(`runs_v2` vs `runs_v2_per_model_batch`) so they never collide.

---

## Part I — What was wrong

### 1. 🔴 The effective batch size was never what the config said

`configs/common_train.yaml` says `effective_batch: 8`. The trainer computed:

```python
micro_batch, accum_steps, _ = find_optimal_micro_batch(..., max_probe=32)
accum_steps = max(1, effective_batch // micro_batch)   # 8 // 32 -> 0 -> max(1,0) = 1
```

The moment the VRAM probe returned anything ≥ 8, `accum_steps` collapsed to 1 and
the effective batch became *whatever the probe found*. Your own `run_config.json`
files record the result:

| run | micro_batch | accum | **effective** | steps/epoch | **total steps** |
|---|---|---|---|---|---|
| `segformer_b2_teacher` | 8 | 1 | **8** | 87 | **8700** |
| `segformer_b0_direct` | 32 | 1 | **32** | 21 | **2100** |
| `segformer_b0_distilled` | 32 | 1 | **32** | 21 | **2100** |

B0 trained with **4× the batch and a quarter of the gradient updates** as B2, at an
identical learning rate. B2 is bigger, so the probe found a smaller batch for it —
meaning the batch size was set by *model size*, precisely the variable the
comparison is supposed to isolate.

**What this does and does not invalidate:**

- ❌ **Invalidated**: any claim that B0 and B2 were trained under "identical
  hyperparameters" — including `03_train_segformer_b0_direct.py`'s docstring
  ("Why use the SAME trainer as B2… Any difference in training recipe would make
  the comparison confounded"). The comparison *was* confounded.
- ✅ **Survives**: B0-direct vs B0-distilled. Both ran at 32/2100, so they are
  matched to each other and the distillation comparison is clean.

The fix is one line — cap the probe at `effective_batch` and let accumulation cover
the rest — but it changes the schedule for 2 of 5 models, which is why retraining is
the only way to get a defensible table.

### 2. 🔴 YOLO distillation was a no-op above α = 0.5 — and the trials prove it

`scripts/08_train_yolo_sem_distilled.py` built pseudo-masks:

```python
fused      = alpha * gt + (1.0 - alpha) * prob        # gt ∈ {0,1}, prob ∈ [0,1]
class_mask = (fused >= cfg["pseudo_threshold"]).astype("uint8")   # threshold 0.5
```

Take α > 0.5 and consider the two cases:

- **gt = 1** → `fused = α + (1−α)·prob ≥ α > 0.5` → always **1**
- **gt = 0** → `fused = (1−α)·prob ≤ 1−α < 0.5` → always **0**

So `class_mask ≡ gt`, **exactly**, for every α > 0.5. The teacher is loaded, 697
forward passes are computed, and the result is discarded — the model trains on the
plain ground truth. "YOLO distilled" at α > 0.5 *is* "YOLO direct".

This is not a theoretical worry. It is visible in `optuna_alpha_search/yolo_sem_trials.csv`:

| trial | α | val Dice |
|---|---|---|
| 1 | 0.9 | `0.5992961273817916` |
| 2 | 0.7 | `0.5992961273817916` |
| 3 | 0.6 | `0.5992961273817916` |
| 7 | 0.8 | `0.5992961273817916` |
| 8 | 0.6 | `0.5992961273817916` |
| 9 | 0.7 | `0.5992961273817916` |

Six trials, four distinct α values, **bit-identical to 16 significant figures**.
That is not a coincidence or a flat optimum — it is the same computation run six
times. Roughly 3 GPU-hours (6 × 30 min) recomputing one number, and **40% of the
search space was degenerate**.

The shipped model used α = 0.4, which is *below* 0.5 and so does do something:
GT-positive pixels are dropped where the teacher is confident-negative
(`prob < 1/6`), and GT-negative pixels are added where the teacher is
confident-positive (`prob > 5/6`). But note what that is: **hard label refinement**,
not knowledge distillation. Ultralytics' semantic trainer can only consume a binary
class-index PNG, so every bit of the teacher's soft, graded uncertainty — the entire
"dark knowledge" that distillation is *defined* by (Hinton et al. 2015) — is
thresholded away before training starts. The paper should not call this KD.

### 3. 🔴 Both Optuna searches selected noise

Optuna repeated some α values, which accidentally gives us a direct read on
run-to-run noise. Compare the noise to the effect:

**SegFormer-B0** (`segformer_b0_trials.csv`):

| α | trials | val Dice |
|---|---|---|
| 0.6 | #3, #8 | 0.7375, 0.7279 → **spread 0.0096** |
| 0.4 | #0, #12 | 0.7296, 0.7362 → spread 0.0066 |
| 0.5 | #10, #11 | 0.7324, 0.7331 → spread 0.0007 |
| 0.2 | #4, #5 | 0.7249, 0.7276 → spread 0.0027 |

The search picked **α = 0.6** on the strength of trial #3's 0.7375. But trial #8 ran
the *same α* and got 0.7279 — and 0.7375 beats the best α = 0.4 trial (0.7362) by
**0.0013**, which is **7× smaller than α = 0.6's own seed-to-seed spread**. The
"optimal alpha" is the lucky member of a noisy pair.

**YOLO** (`yolo_sem_trials.csv`): α = 0.4 → 0.6172 and 0.6074 (spread **0.0098**).
Winner α = 0.4 beat α = 0.3 by 0.0019 — 5× smaller than that spread. And, per §2,
every trial above 0.5 was the same run.

Neither search measured α. Both measured seeds. This is the classic failure mode
Bergstra & Bengio (2012) and Henderson et al. (2018, "Deep RL that Matters")
describe: **hyperparameter selection on a single noisy run**, where the selection
procedure fits the noise and the reported optimum does not replicate.

**Consequence for the notebook:** α is fixed at 0.5 and the budget is spent on
**3 seeds per model** instead. Three seeds cannot prove a small effect, but they can
tell you *whether you are entitled to claim one* — which the old setup could not.

### 4. 🔴 The YOLO threshold sweep feeds ImageNet-normalised pixels

`pipeline/yolo_threshold_temp.py::run_threshold_search()` builds
`BruiseDataset(val_df, ...)`, whose `get_augmentation()` hardcodes
`A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`, and feeds that
straight to the raw YOLO `nn.Module`. Per your own Bug 1 (inference doc §5), that
caps YOLO at **Dice 0.479 at any threshold**.

But `yolo_sem_direct/threshold_search.csv` records its best at **0.7375**, at
`T=1.0, threshold=0.15` — a cut of `logit(0.15) = −1.73`, comfortably inside the
±2.94 range that T=1 already reaches. Both cannot be true:

- If the CSV came from this code path, 0.7375 > 0.479 is impossible.
- If 0.479 is the real ImageNet ceiling, the CSV came from **something other than
  the code currently in the repo**.

**Therefore `docs/inference_demo_explained.md` §7 — "RESOLVED: the old
`threshold_search.csv` files were right all along" — is not established.** Its
evidence is that a fresh `/255` sweep reproduces the CSV to 4×10⁻⁵; but if the CSV
was produced with ImageNet normalisation, that agreement is impossible, and if it
was produced with `/255`, then the repo's sweep code did not produce it. One of the
two is wrong and the doc currently asserts both. **Either re-derive the provenance
of those CSVs or retract §7.** It does not affect the new notebook, which sweeps its
own thresholds from its own checkpoints — but it does affect any historical number
still quoted in the paper.

### 5. 🟠 Smaller things, fixed in passing

| | issue | fix |
|---|---|---|
| **Probe mutates the model** | `find_optimal_micro_batch` runs real `optimizer.step()` on the live model, perturbing pretrained weights a model-size-dependent number of times before training | probe a `deepcopy` |
| **Selection threshold ≠ reported threshold** | `best_model.pt` = best val Dice **at 0.5**, but the reported operating points are 0.18–0.67 | select on threshold-free **val AP** |
| **argmax on a flat curve** | the sweep took `argmax` of a curve that moves 0.009 across a 6× threshold range — that fits val's sampling error | tie band of ±1 SE |
| **Tie broken on median cut** | band cuts are Dice-equivalent but *not* miss-equivalent | break ties on **complete-miss rate** (inference doc §10's "free win") |
| **Native-res decode every epoch** | 4022×6024 JPEG → 640, ~287 ms/image/epoch, CPU-bound on Colab | pre-resize once, **bit-exact** (verified) |

---

## Part II — What the notebook does, and why

### 6. One recipe, five models — the point of the whole design

Nothing below the model wrapper branches on a model's name. One dataloader, one
resize, one augmentation pipeline, one loss, one optimizer, one LR schedule, one
sweep, one metric. The **only** thing that differs between the five runs is the
architecture and the pixel scale its weights require — and both live inside the
model wrapper.

This is what makes the comparison an actual comparison. The old pipeline trained
SegFormer with a custom loop and YOLO with Ultralytics' trainer, which differ in
optimizer (AdamW vs auto-selected SGD/MuSGD), schedule (poly vs cosine), loss
(Dice+BCE vs CE+Dice+aux), augmentation (albumentations vs mosaic), and batch size
— so "SegFormer beats YOLO" was never separable from "our recipe beats theirs".

**What we give up:** YOLO no longer gets Ultralytics' mosaic augmentation or EMA, so
its absolute Dice may come out below the historical 0.699. That is a real cost and
should be stated. What we buy is that the number *means something* relative to the
other four.

### 7. Why `nc=1` for YOLO

The pretrained `yolo26n-sem.pt` has `nc=19` (Cityscapes). Two options for a binary
task:

- `nc=2` → output `[B,2,H,W]`, bruise logit = `z₁ − z₀`. (2-class softmax and
  sigmoid-of-the-difference are the same function, so this is exact, not an
  approximation.)
- `nc=1` → output `[B,1,H,W]`, which **is** the bruise logit.

We use `nc=1`. It is structurally identical to SegFormer's 1-channel head, it removes
a transformation where the sign could be flipped, and Ultralytics' own loss supports
it (`nn.BCEWithLogitsLoss` when `nc == 1`). Verified: `.load()` transfers **360/364
tensors** — only the head is new, exactly mirroring SegFormer's randomly-initialised
1-class head on a pretrained backbone.

**An architectural fact worth putting in the paper:** YOLO's semantic head predicts
at **stride 8** (80×80 for a 640 input); SegFormer's decode head is at **stride 4**
(160×160). YOLO predicts at a quarter of SegFormer's spatial resolution *in area*.
That is a ceiling on boundary precision that no amount of training removes, and it
is a more honest explanation of any Dice gap than "YOLO is worse".

### 8. Pixel scale belongs to the model, not the loader

> Ultralytics trains on plain `/255`; its BatchNorms carry **frozen** running
> statistics for that distribution and cannot adapt. Fed ImageNet-normalised pixels,
> YOLO under-fires by 4× and no threshold recovers it (0.479 ceiling).
> — inference doc §5, Bug 1

So the dataloader emits **raw `[0,1]` pixels**, and each wrapper applies its own
scale: `SegFormerNet` does `(x − mean)/std`; `YoloSemNet` uses `x` as-is. Every model
shares one disk read, one resize and one augmentation — the geometry is *identical by
construction* — while each still sees the distribution its weights were trained for.

This is cleaner than the inference notebook's approach (normalise in the loader, then
un-normalise for YOLO with `x*STD + MEAN`). That roundtrip is accurate to 6×10⁻⁸ and
works, but it only exists to undo something the loader should not have done.

### 9. The training recipe, and its sources

| choice | value | source |
|---|---|---|
| Layer-wise LR: backbone 6e-5, head 6e-4 (10×) | both architectures | **Xie et al. 2021**, *SegFormer* (NeurIPS) — pretrained encoder gets a conservative LR, randomly-init decoder gets 10× to catch up |
| AdamW, β=(0.9,0.999), wd=0.01, **no decay on norms/biases** | | Xie et al. 2021; **Loshchilov & Hutter 2019**, *Decoupled Weight Decay Regularization* (ICLR) |
| Linear warmup 1% of steps → poly decay (power=1.0) | | Xie et al. 2021 §4.1 |
| Loss = BCE + (1 − per-image soft Dice) | | **Milletari et al. 2016**, *V-Net* (Dice); **Drozdzal et al. 2016** (the combination). Also what Ultralytics' own `SemanticSegmentationLoss` computes (BCEWithLogits + binary Dice), so it does not disadvantage YOLO vs its native recipe |
| Aux head weight 0.4 (YOLO only) | | Ultralytics' own `SemanticSegmentationLoss` — their value, kept |
| Teacher temperature calibration | fitted by NLL on val, L-BFGS | **Guo et al. 2017**, *On Calibration of Modern Neural Networks* (ICML) |
| Distillation fusion, α = 0.5 | fixed, not searched | see §10 |
| Subject-level split | 697/134/185 | one subject contributes several photos; an image-level split leaks skin tone and bruise appearance |
| ITA skin-tone groups | 5 groups | **Chardon et al. 1991**, *Skin colour typology and suntanning pathways* |
| Bootstrap CI of the median | 2000 resamples | **Efron & Tibshirani 1993** |
| Kruskal–Wallis + Mann–Whitney + Bonferroni | 5 groups, 10 pairs | non-parametric because per-image Dice is bimodal and bounded |

**Why effective batch = 8 for everything.** It is the value the config always claimed,
and it is the one B2 can actually fit at 640×640. Every model now takes the same
number of optimizer steps (8700) on the same LR schedule, and a smaller GPU just uses
more gradient accumulation to reach the identical result. The product
`micro_batch × accum_steps == effective_batch` is asserted, not approximated.

### 10. What the distillation loss is — and what it is not

```
loss = α · [BCE(z_s, y) + (1 − Dice(σ(z_s), y))]        ← hard, supervised
     + (1 − α) · BCE(z_s, σ(z_t / T_cal))               ← soft, from the teacher
```

**This is calibrated soft-target distillation. It is *not* Hinton et al. (2015) KD**,
and the paper must not cite Hinton for this formula as written:

- Hinton's KD divides **both** the student's and the teacher's logits by a shared
  temperature `T` and multiplies the soft term by **T²** (so the soft gradient stays
  comparable in magnitude to the hard term as T varies).
- Here the **student's logits are not temperature-scaled** and there is no T².
  `T_cal` is not a KD knob at all — it is the temperature fitted by NLL on val
  (Guo et al. 2017) that makes the *teacher's probabilities calibrated*.

The correct citation is **Menon et al. 2021**, *A statistical perspective on
distillation* (ICML): distillation helps to the extent the teacher approximates the
Bayes class-probability `P(y|x)` — and calibration is exactly what improves that
approximation. The teacher is being used as a better probability estimate, and the
student regresses onto it. That is a coherent, defensible method with a real
citation; it is just a different one from the paper's current claim.

**Why calibrate at all:** BCE drives logits toward ±∞ (σ(±∞) is a perfect loss), so
the teacher's probability histogram ends up nearly binary — and `σ(z_t)` as a soft
label becomes indistinguishable from the hard GT, defeating the entire point.
Dividing by `T > 1` pulls saturated logits back into sigmoid's responsive region so
the student can see where the teacher is *uncertain*. Your B2 previously calibrated
to **T = 1.84** (val NLL 0.0634 → 0.0452), which is a real, useful amount of
softening. The notebook warns loudly if T lands outside [0.5, 10], since that means
the teacher is under-trained and would teach a constant 0.5.

**Why α = 0.5 and not a search:** see §3. The old search's effect size was ~1/7 of
its own noise. Three seeds at a fixed, defensible α is a better use of the same GPU
hours, and it produces the error bars needed to say anything at all.

**Why each distilled run uses its own seed's teacher** rather than one fixed teacher:
the reported spread then includes the teacher's own variance, which is part of the
pipeline being measured. Pinning one teacher would make distillation look more
reproducible than it is.

### 11. Model selection: threshold-free val AP

The old pipeline saved `best_model.pt` by val Dice **at a fixed 0.5** — but the
threshold is re-fitted afterwards anyway, and the fitted operating points are nowhere
near 0.5 (YOLO's lands at ~0.18). So 0.5-Dice selection answers "which epoch is best
at an operating point we will not use?", and can pick the wrong epoch for any model
whose calibration drifts during training — which is exactly what YOLO does.

**Average precision** integrates over all thresholds, so the epoch choice cannot be
biased by one arbitrary cut. Implemented as a **histogram-binned AP** (4096 bins) on
the GPU: 134 val images × 640² = 55M pixels, and sorting 55M floats every epoch costs
seconds and ~450 MB, while binning is O(bins) and effectively free. Quantisation error
~1/4096 in probability — three orders of magnitude below the epoch-to-epoch
differences it has to rank.

### 12. The operating point: fitted on val, tie-broken on misses

**val (134) chooses the threshold and is never reported. test (185) is reported and
never chooses anything.** A model emits a probability per pixel; a mask needs a
threshold; different thresholds give different Dice — so the threshold is a parameter,
and fitting it on test means reporting a score tuned on the exam.

The sweep is over the **raw-logit cut** `c` (mask = `z ≥ c`), reported as `σ(c)`.
Temperature is not swept, because for a **hard mask** it is redundant:
`σ(z/T) ≥ thr ⟺ z ≥ T·logit(thr)`, so the decision depends only on the product
`c = T·logit(thr)`. The old 2-D `(T, thr)` grid was a redundant parameterisation of a
1-D family — of 152 grid points, only 116 were distinct outcomes (inference doc §6).

**The tie band.** These sweeps are extraordinarily flat: B2's val Dice moved by
**0.009** across thresholds from **0.154 to 0.959**. That is not a peak, it is noise on
a plateau, and `argmax` of it fits val's sampling error. Every cut within **one
standard error** of the peak is statistically tied.

**Breaking the tie on miss rate.** Band cuts are Dice-equivalent but **not**
miss-equivalent: a lower cut predicts more pixels → fewer blank masks → statistically
identical Dice. The old rule took the band's *median* cut, optimising for stability.
But a blank mask is a missed injury, and the miss rate is the one metric that
separates these models by more than label noise. So: minimise miss rate within the
band → break remaining ties on Dice → break what is left on the median cut, for
reproducibility. This implements the "free win" the inference doc §10 identified and
never got round to.

### 13. Fairness is a primary result, not an ablation

This is a forensic injury-documentation tool. A model that segments bruises well on
light skin and poorly on dark skin does not have a metric problem, it has an
**evidentiary** one — it would under-document injuries on exactly the population most
likely to need the documentation. The previous run found `yolo_sem_direct` had a
**0.75 fairness gap** with a Kruskal p of 5×10⁻⁷, and an 18% miss rate on Light (II-III).

Method, and why each piece:

- **ITA** (Chardon et al. 1991), computed from image pixels, not a rater's Fitzpatrick
  guess — objective and reproducible. Test distribution: Dark (VI) 55, Light (II-III)
  39, Intermediate (III-IV) 38, Brown (V) 29, Tan (IV) 24.
- **Kruskal–Wallis** omnibus across all 5 groups: non-parametric, because per-image
  Dice is bimodal and bounded, so ANOVA's normality assumption fails.
- **Pairwise Mann–Whitney U, Bonferroni-corrected** over the 10 pairs: with 5 groups,
  uncorrected pairwise testing finds a "significant" pair ~40% of the time on pure noise.
- **`fairness_gap` = best group's median Dice − worst group's**: the effect size. The
  p-value says the gap is real; only the gap says whether it matters.
- Per-image Dice is **averaged over seeds first**: one run's seed noise is not a
  property of the model, and the fairness question is about the model. n stays 185.

### 14. Speed: what is and is not timed

**640 tensor already on the GPU → 640 mask still on the GPU.** Nothing else.

| ✅ timed | ❌ not timed |
|---|---|
| forward pass | disk read / JPEG decode |
| logit upsample to 640 | resize, normalisation |
| threshold | host→GPU copy (staged once) |
| | **GPU→host copy — the mask never leaves the GPU** |
| | model load, warmup |

Disk/decode/copy are identical for all five models and dominated by I/O, so including
them would compress the real architectural differences into measurement noise.
`cuda.synchronize()` around every call is not optional: CUDA is asynchronous, so
without it you measure how long it takes to *queue* the work, not do it — and every
model reports as equally, impossibly fast.

### 14b. Reducing complete misses **without retraining** (notebook §8b)

The models are frozen, so this is a post-processing question fitted on val and applied
to test — it changes no weight. The point: lowering the global threshold slides *along*
the miss-vs-Dice curve (fewer misses bought with more false positives, one-for-one).
What we want is to move the *curve* — fewer misses at the same Dice. Three no-retrain
techniques, each fitted on val with the same miss-tie-break as the baseline:

- **3-seed ensemble** — average the three seeds' probability maps. A miss now needs all
  three seeds to blank the *same* image, and the per-seed misses are different images
  (direct blanked 1 / 13 / 10). Free, and the honest recommended lever.
- **Ensemble + TTA** — also average over H/V flips; raises borderline probabilities so
  they clear the threshold without lowering it.
- **Ensemble + no-blank floor** — if a mask is still empty, recover the most-confident
  region. This **games the miss metric**, so it is reported separately as a floor.

Outputs go to a **new** folder, `results_v2/miss_reduction/` — nothing existing is
overwritten. The verdict is read off a miss-vs-Dice Pareto plot: a real win sits
down-and-right of the single-seed baseline; a point that slid down-and-left is the
threshold in disguise. The durable fix — a recall-weighting loss (**Focal-Tversky**,
Salehi 2017 / Abraham 2018, β>α) — needs a retrain and is deliberately **out of scope**
here because these weights are frozen; it belongs to the next training run.

Verified locally on the real `runs_v2/` checkpoints (CPU): probability maps in [0,1];
the seed ensemble aligns by stem even under scrambled loader order; the prob-space sweep
equals `dice_np` exactly (<1e-9); **the val-fit threshold and the test-apply score share
the same float32 boundary** (a first draft compared fp16 on val and float32 on test —
caught and fixed); the no-blank floor never increases misses.

---

## Part III — Things retraining cannot fix

### 15. 🔴 The annotation ceiling

From `interlabeler_agreement_640.csv`, on the same 185 test images:

| comparison | mean Dice |
|---|---|
| paul ↔ gbarimah | **0.5809** |
| paul ↔ erik | **0.5812** |
| gbarimah ↔ erik | 0.7549 |
| paul → majority GT | **0.6998** |
| gbarimah → majority GT | 0.8729 |
| erik → majority GT | 0.8657 |

The previous five models spanned **0.692–0.786** against the consensus. The three
humans span **0.700–0.873** against the *same* consensus. **The models are closer to
each other than the annotators are to each other.** The best model agreed with the
consensus better than one of the three experts who *defined* it.

No amount of retraining changes this. It is why:

- the sweeps are flat — you cannot resolve a threshold more finely than the labels
  are self-consistent;
- the α search was measuring noise — the noise floor *is* the label noise;
- **complete-miss rate, not Dice, should be the paper's primary axis** — it is the one
  metric that separates the models by more than label disagreement (0.00% vs 9.19%).

The honest framing: *these models have reached human-level agreement on this dataset,
and the remaining variance is in the labels, not the architecture.* That is a stronger
result than a 0.01 Dice win, and it is defensible.

### 16. 🟠 The label-standard mismatch

Train and val masks come from `train_paul_wl_minus_test_subjects/masks/` — **Paul's**
labels. Test masks are `masks_majority_vote` — the **consensus**. Per §15, Paul is the
outlier annotator (≈0.58 with each of the other two, who agree 0.755 with each other).

So the pipeline **trains** on one labelling standard, **fits the threshold** on that
same standard (val is also Paul's), and **scores** against a different one. That is a
real distribution shift and likely explains part of the val→test gap.

**It cannot be fixed by retraining**: consensus labels exist only for the test
subjects, and those subjects must stay out of training. It should be stated explicitly
in the paper. And it makes §15's headline *more* striking, not less: a model trained
on Paul's labels agrees with the consensus better than Paul does (0.786 vs 0.700) — so
it is not merely copying its annotator.

---

## Part V — Results of the completed matched run (2026-07-16, A100)

All 15 runs finished (`runs_v2/` on Drive, `DONE.json` in every folder). Numbers
below are computed directly from each run's `test_per_image.csv`.

### The run is clean — the fixes landed

Every `config.json` records **`micro_batch=8 × accum_steps=1`, effective batch 8**,
for all five models — the confound is gone, B0 and B2 genuinely share the recipe.
Both distilled runs used **α = 0.5** and teacher temperature **T = 1.82** (right in
the plausible range, and almost exactly the historical 1.84 — calibration is
behaving). Early stopping fired everywhere (26–88 epochs, never the 100 ceiling).

### Headline table — 3 seeds, 185 test images

| model | params | mean Dice | median | complete-miss % |
|---|---|---|---|---|
| SegFormer-B2 teacher | 27.35M | **0.786 ± 0.020** | 0.830 | **0.00** |
| SegFormer-B0 distilled | 3.71M | 0.767 ± 0.005 | 0.820 | 0.72 |
| SegFormer-B0 direct | 3.71M | 0.761 ± 0.012 | 0.812 | 0.36 |
| YOLO26n distilled | 1.63M | 0.661 ± 0.029 | 0.747 | 3.42 |
| YOLO26n direct | 1.63M | 0.644 ± 0.030 | 0.723 | 4.32 |

### What the numbers say

**1. 🔴 "The student beats the teacher" is GONE — and this version is the honest
one.** The previous (confounded) run had B0 distilled at 0.786 *above* B2 at 0.773.
That was partly an artefact of B0 silently getting a 4× batch. With the batch equal,
**B2 is clearly on top (0.786 vs 0.767)** and B0 no longer over-trains relative to
it. The old headline does not survive the fix; the new ordering is defensible.

**2. Distillation helps both times, but n=3 cannot prove it.** Paired over seeds:
SegFormer **+0.0066 (p = 0.45)**, YOLO **+0.0167 (p = 0.31)**. Both point the right
way, both inside seed noise — the correct phrase is "not proven," not "no effect"
(three seeds has almost no power against a 0.012 seed spread). Where distillation
*does* show a cleaner signal is **variance and misses**: B0 distilled's std is 0.005
vs direct's 0.012, and it cut YOLO's miss rate 4.32% → 3.42%. It looks more like a
**stabiliser** than a Dice booster.

**3. 🔴 YOLO's problem is seed-unstable misses, not Dice.** Per-seed blanks:
YOLO-direct misses **1, 13, 10** of 185 across the three seeds; distilled misses
**3, 6, 10**. The miss rate swings 0.5%–7% purely by seed. That instability — not
the absolute Dice — is the disqualifying property for injury documentation, and it is
consistent with the stride-8 architectural note (§7). **B2 never blanks (0/185 × 3).**

**4. The sweeps are still extremely flat — the annotation ceiling holds (§15).**
On seed 0, B2's Dice moves only 0.30 across the entire cut range, with **251 of 481
cuts statistically tied** (band spanning threshold 0.01 → 0.85). The chosen thresholds
vary wildly by seed (B2: 0.15, 0.11, 0.88) — not instability, just three equivalent
points on one flat plateau.

### What this means for the paper

The defensible headline is the one §15 already pointed to: **model ranking by Dice is
inside label noise; lead with complete-miss rate.** On that axis the ordering is
unambiguous and stable — B2 (0.00%) ≫ B0 (~0.5%) ≫ YOLO (3–4%, wildly seed-dependent).
That is a cleaner and more clinically meaningful headline than any 0.01 Dice
comparison. (Fairness across skin tone was not re-examined in this pass; the
`results_v2/fairness_stats.csv` the notebook wrote is the place to look next, and
YOLO's problems are likely sharpest there.)

---

## Part VI — The per-model-batch variant (`bruise_colab_train_per_model_batch.ipynb`)

### What it is

A second notebook, identical to the matched one **except** each model trains at the
largest batch its own size allows (`resolve_micro_batch` probes up to 64, `accum=1`,
no cap). On an A100 expect roughly **B2 ≈ 8–16, B0 ≈ 32–64, YOLO ≈ 64** — the exact
number is probed at run time and recorded in each `config.json`. Verified: of the
eight library modules, seven are byte-identical between the two notebooks; only
`engine.py::resolve_micro_batch` differs.

### Why it is NOT the paper notebook

Bigger batch + **same LR** means the small models take **fewer, larger gradient
steps** — which is exactly the confound the matched notebook removed, just now on
purpose. So B0-vs-B2 is no longer apples-to-apples here. The notebook carries a ⚠️
banner in §1 and a note in §6 saying so, and writes to its own Drive folders
(`runs_v2_per_model_batch` / `results_v2_per_model_batch`) so it cannot overwrite or
skip-load the matched run.

### The open question: should the higher-batch models get a higher LR?

**Decision: run it once at the same LR first, look at the results, then decide.**
The theory and the trade-off, for the record:

- **The linear scaling rule** (Goyal et al. 2017) says LR ∝ batch size — so a model
  going 8 → 64 would nominally want more LR, and at the *same* LR it is under-stepping.
- **But three things weaken that here:** (a) we use **AdamW**, whose update is roughly
  invariant to gradient scale, so the honest rule is closer to **√(batch ratio)**, not
  linear; (b) these are **fine-tuning** runs with tiny LRs on a pretrained backbone,
  where too-high LR damages the features you are keeping; (c) we are at the
  **annotation ceiling** (§15), so an LR tweak moving val AP by a fraction of a percent
  cannot change any conclusion.
- **The bigger risk at high batch is fewer optimizer steps, not LR magnitude** — and
  early stopping on val AP already handles "has it converged?", so a model that needs
  more updates just trains more epochs before stopping.

So same-LR is the safe default for a *speed* variant, and if the results come back
close to the matched run it settles the question. If the per-model models come back
clearly worse, √-scaling the LR (with lengthened warmup) is the principled next step —
but at that point it is becoming a controlled experiment again, which is what the
matched notebook already is.

---

## Part IV — Running it

### 17. Build and upload

```bash
C:/Users/91962/miniconda3/envs/bruise_local/python.exe scripts/40_build_colab_training_package.py
```

Writes `bruise_colab_train.zip` (**992 MB**). First build takes ~6 min (decoding 1016
native JPEGs); rebuilds are seconds, because the resized PNGs are cached in
`.colab_train_cache/`.

| file | destination |
|---|---|
| `bruise_colab_train.zip` | Drive → `MyDrive/bruise_segmentation_gpu/` (same zip for both notebooks) |
| `bruise_colab_train_all.ipynb` | Colab → File → Upload notebook (the fair, matched run) |
| `bruise_colab_train_per_model_batch.ipynb` | Colab → File → Upload notebook (the fast, per-model-batch run) |

Then **Runtime → Change runtime type → A100**, and Run All. The two notebooks write to
separate Drive folders, so you can run both without them interfering.

To regenerate the notebooks after editing their source:
```bash
C:/Users/91962/miniconda3/envs/bruise_local/python.exe scripts/41_generate_training_notebook.py             # matched
C:/Users/91962/miniconda3/envs/bruise_local/python.exe scripts/41_generate_training_notebook.py --per-model # per-model batch
```
(The notebooks are build artifacts; `scripts/41_...` is the real source. Edit that. The
two differ only in the batch cells — one generator guarantees they can't otherwise drift.)

### 18. Budget, and surviving disconnects

15 runs. Roughly **12–20 GPU-hours** total on an A100 — early stopping (patience 15)
usually ends runs well before the 100-epoch worst case. This will not finish in one
Colab session, and it does not need to:

- every run writes `resume.pt` to Drive every 2 epochs (atomically: temp file + rename,
  so a kill mid-write cannot leave a truncated checkpoint that poisons the next session);
- finished runs write `DONE.json` and are skipped for free;
- **if Colab dies: reconnect, Run All.** It continues from the last synced epoch.

Verified locally: a run killed after epoch 2 resumed at epoch 3 and finished with one
continuous history and no repeated epochs.

Raise `drive_sync_every` to reduce Drive traffic (a B2 resume checkpoint is ~330 MB:
weights plus AdamW's two moment buffers), lower it to 1 if sessions are dying often.
Correctness is identical either way; only the amount of work at risk changes.

### 19. What it refuses to do

The notebook fails fast rather than producing bad numbers. It stops if:

- there is no GPU (a CPU run would take days and every timing would be meaningless);
- the zip is under 0.8 GB (that is the *inference* package, which ships no train images);
- any two splits share a subject or an image;
- the GT mask arrives as `[B,1,H,W,1]` (ultralytics' `cv2.imread` patch — a
  pixel-perfect prediction scores **63.9** instead of 1.0 when that axis broadcasts);
- the loader emits anything outside `[0,1]`, or a model's parameter count is wrong;
- binned AP disagrees with a perfect ranking, or the vectorised sweep disagrees with
  the numpy Dice.

Those last two are real self-tests in §5 of the notebook, not decoration. The sweep
assertion is at **1e-9** and it earned its place: it caught an fp16 reduction in the
first draft that drifted from the numpy Dice by 1.5×10⁻⁴ — a tenth of the signal the
tie band uses to rank cuts.

---

## 20. Verification status

**Verified locally against the real package (CPU, `bruise_local` env):**

- the pre-resize is **bit-exact** — max image-tensor difference **0.0**, **0** mask
  pixels differing, over 25 val images vs the old native-JPEG path;
- the loader emits `x[2,3,640,640]` in `[0,1]` and `y[2,1,640,640]` binary;
- all three architectures emit `[B,1,640,640]`; params 3.71M / 27.35M / 1.63M as
  expected; YOLO's aux head is present in `train()` mode and absent in `eval()`;
- YOLO `nc=1` `.load()` transfers **360/364** tensors;
- binned AP = 1.0 on a perfect ranking, = the positive rate on a random one;
- the vectorised sweep equals `dice_np` **exactly** (< 1e-9);
- the tie-break prefers a 0%-miss cut over the argmax cut when they are tied;
- end-to-end train → best.pt/DONE.json/history.csv, idempotent re-run skips;
- **crash at epoch 2 → resume at epoch 3 → continuous history [1,2,3,4,5]**;
- the distillation path: calibration → `teacher_fn` (no grad, prob ∈ [0,1]) → KD run
  with α and T recorded in `config.json`;
- the run queue orders each seed's B2 teacher before its distilled students.

**Verified by the completed matched Colab run (2026-07-16, A100 — Part V):**

- all 15 runs finished; every `config.json` records effective batch 8 (the fix held);
- teacher calibration landed at T = 1.82 (plausible range, matches historical 1.84);
- absolute accuracy now known (Part V table); with the confound gone, B2 leads and the
  old "student beats teacher" result does not reproduce;
- distillation is inside seed noise (SegFormer p = 0.45, YOLO p = 0.31);
- YOLO's weakness is a seed-unstable miss rate (1/13/10 and 3/6/10 blanks), not Dice;
- YOLO under our recipe does come out below its Ultralytics-trained historical 0.699
  (0.64 direct) — expected, the price of the fair comparison, not a bug.

**Not verified (requires the per-model-batch Colab run):**

- Part VI — accuracy of the per-model-batch variant, and therefore whether the
  same-LR-at-higher-batch decision needs revisiting (the plan is: run once, then decide);
- whether the miss-rate tie-break measurably reduces YOLO's blank rate vs a median-cut
  tie-break (predicted by inference doc §10, still not isolated);
- the fairness/skin-tone breakdown of the new checkpoints (`results_v2/fairness_stats.csv`
  written but not yet examined in this doc).

**Open, and needs a decision before the paper:**

- §4 — `docs/inference_demo_explained.md` §7 asserts two things that cannot both be
  true. Re-derive the provenance of the historical `threshold_search.csv` files, or
  retract §7.

---

## References

- Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., Luo, P. (2021).
  *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.* NeurIPS.
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* NIPS Deep Learning Workshop.
- Menon, A. K., Rawat, A. S., Reddi, S., Kim, S., Kumar, S. (2021).
  *A statistical perspective on distillation.* ICML. ← the correct citation for our KD formula
- Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML.
- Milletari, F., Navab, N., Ahmadi, S.-A. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV.
- Drozdzal, M., Vorontsov, E., Chartrand, G., Kadoury, S., Pal, C. (2016). *The Importance of Skip Connections in Biomedical Image Segmentation.* DLMIA.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Bergstra, J., Bengio, Y. (2012). *Random Search for Hyper-Parameter Optimization.* JMLR.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018).
  *Deep Reinforcement Learning that Matters.* AAAI. ← on selecting hyperparameters from single noisy runs
- Chardon, A., Cretois, I., Hourseau, C. (1991). *Skin colour typology and suntanning pathways.* Int. J. Cosmetic Science.
- Efron, B., Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall.
- Jocher, G., et al. *Ultralytics YOLO.* (`SemanticSegmentationLoss`: BCEWithLogits + Dice + 0.4·aux)
