# What `bruise_colab_inference_demo.ipynb` does, and what it found

*Written 2026-07-15. Updated 2026-07-16 with the first real Colab run
(`analysis/inference_demo_20260716_015433`, A100-SXM4-40GB).*

---

## TL;DR

The notebook answers three questions about our 5 trained models, on Colab, in pure PyTorch:

1. **What threshold should each model use?** → swept on the **134 val images**
2. **How fast is each model?** → timed on all **185 test images**
3. **How accurate is each model?** → scored on those same **185 test images**

The zip grew **868 MB → 1206 MB** because it now contains the 134 validation images.
Without them there is nowhere honest to fit a threshold.

**It has now been run.** Three things came out of it:

- **The student beats the teacher.** SegFormer-B0 distilled (3.71M params) scores a higher
  mean Dice than its own 27.35M-param B2 teacher, at 2× the speed. → **ship this one**
- **YOLO's problem is blanks, not quality.** It predicts *nothing at all* on 9.19% of test
  images. Its quality on the rest is competitive.
- **🔴 The models have hit the annotation ceiling.** Human annotators agree with each other
  at Dice 0.58–0.75. Our best model agrees with the consensus GT *better than one of the
  three experts who defined it*. The differences between our 5 models are smaller than the
  disagreement between our humans.

---

## 1. Why the zip had to change

### The old zip was a trap

The old package contained **only the 185 test images**, but `configs/paths.yaml` inside it
said this:

```yaml
train_manifest:      .../fixed_consensus_test/manifest.csv
fixed_test_manifest: .../fixed_consensus_test/manifest.csv    # ← the SAME file
```

Both keys pointed at the test set. Harmless while nothing read `train_manifest` — but the
moment you add a threshold sweep, the sweep reads a "training" manifest, gets the **test
set**, and fits the threshold on the very data it then reports scores on.

**That would not have crashed.** It would have printed slightly better-looking numbers and
nothing would have told you they were fake.

### What the new zip contains

| | old (868 MB) | new (1206 MB) |
|---|---|---|
| test images + masks | 185 ✅ | 185 ✅ |
| **val images + masks** | ❌ **none** | **134 ✅** |
| `splits/val_split.csv` | ❌ | ✅ |
| `val_manifest` key in paths.yaml | ❌ | ✅ (a *different* file from test) |
| `train_manifest` → test manifest | ⚠️ yes | ✅ no longer |
| model weights, `pipeline/`, configs | ✅ | ✅ |

The 134 val images come from `train_paul_wl_minus_test_subjects/`, a folder that also holds
697 **train** images we deliberately don't package (not needed for a sweep; would add
~1.9 GB). The build script picks the 134 out file-by-file, and **refuses to build** a
package where val and test share any image or subject:

```
Verified: 185 test + 134 val manifest rows all resolve inside the zip;
no image or subject overlap between them; threshold_search.csv present for all 5 models.
```

---

## 2. Why val and test must be different sets

A model outputs a **probability per pixel**. To get a mask you must pick a **threshold**.
Different thresholds give different Dice. So the threshold is a *parameter you choose* —
and if you choose it using the test set, your test score is no longer an honest estimate.
You've tuned on your exam.

- **val (134)** → used to **choose** the threshold. Never reported.
- **test (185)** → used to **report** Dice/IoU/speed. Never used to choose anything.

No shared images, no shared subjects (subject-level split — the same person never appears
in both). The notebook asserts all of this and stops if it isn't true.

---

## 3. What the notebook does, section by section

```
 §1-5   mount Drive, check GPU, unzip, install, imports
 §6     load val (134) + test (185)      ── leak guard + mask-shape guard
 §7     stage the 185 test images on the GPU (~0.9 GB), once
 §8     load all 5 models as raw PyTorch nn.Modules   (no thresholds yet)
 §9     define THE inference path        ── one function, used by everything below
 §10    ► THRESHOLD SWEEP on val (134)   ── this sets each model's threshold
 §11    ► BENCHMARK on test (185)        ── 640 in → 640 out, GPU only
 §12    ► ACCURACY on test (185)         ── Dice / IoU / complete-miss
 §13-14 results table, save everything to Drive
```

### §9 — the one inference path

Everything — every Dice, every millisecond — comes out of one function with exactly two
branches. **Neither branches on a model's name.** Both are facts about the checkpoint:

| | SegFormer B2/B0 | YOLO26n-sem |
|---|---|---|
| pixel scale it was trained on | ImageNet norm | **`/255`** |
| raw head output | `[1, 1, 160, 160]` | `[1, 2, 80, 80]` |
| → single bruise logit | `out[:, 0]` | `out[:,1] − out[:,0]` |

**Why `z₁ − z₀` for YOLO.** For a 2-class head, softmax over channels is
`P(bruise) = e^z1/(e^z0+e^z1) = σ(z₁−z₀)`. Not an approximation — 2-class softmax and
sigmoid-of-the-difference are *the same function*. So `z₁−z₀` is YOLO's bruise logit,
exactly analogous to SegFormer's, and both go through the identical `sigmoid → threshold`.
Ultralytics' `argmax` is that rule frozen at 0.5.

### §10 — the threshold sweep

Sweeps **one** number: the cut `c` on the raw logit (`mask = z >= c`), reported as `σ(c)`.
Three improvements over the old sweep:

1. **Temperature is gone** — it was mathematically redundant (§6).
2. **134 forward passes instead of 20,368.** Logits are computed once per val image and
   cached; all 481 cuts are then pure tensor math.
3. **It doesn't pick `argmax`.** Every cut within 1 standard error of the peak is
   *statistically tied*; it picks the median of that band and prints the band width. (See
   §9 — this rule turned out to have a flaw worth fixing.)

### §11 — what the benchmark measures

**640 tensor already on the GPU → 640 mask still on the GPU.** Nothing else.

| ✅ timed | ❌ not timed |
|---|---|
| forward pass | disk read / decode |
| logit upsample to 640 | resize to 640, normalisation |
| logit difference / channel select | host→GPU copy (done once in §7) |
| sigmoid + threshold | **GPU→host copy — the mask never leaves the GPU** |
| | model loading, warmup |

185 images × 3 repeats = 555 timed calls per model, `cuda.synchronize()` around each
(without it you'd time how long it takes to *queue* GPU work, not do it).

---

## 4. "Same resizing across all models" — precisely what is and isn't shared

| stage | identical across all 5? |
|---|---|
| disk read | ✅ one read, `BruiseDataset` |
| **resize to 640** | ✅ **identical** — cv2 stretch, image bilinear / mask nearest |
| logits → 640 upsample | ✅ identical bilinear |
| bruise logit → mask | ✅ identical `sigmoid ≥ threshold` |
| threshold sweep method | ✅ identical |
| **pixel scale** | ❌ **ImageNet for SegFormer, `/255` for YOLO** |

The **resizing is identical** — one dataloader, one resize, one disk read, no letterbox, no
`cv2.resize` on any model output. The **pixel scale is not, and cannot be** (§5, Bug 1).

---

## 5. The three bugs found by running the code

### 🔴 Bug 1 — ImageNet normalisation destroys YOLO

Measured on real val images, identical resize in both arms:

| YOLO input | Dice @ argmax | **best Dice at *any* threshold** |
|---|---|---|
| ImageNet norm | 0.3092 | **0.4791** |
| `/255` | 0.7507 | **0.7940** |

Ultralytics trains on plain `/255`; its BatchNorms carry **frozen** running statistics for
that distribution and cannot adapt. Fed ImageNet-normalised pixels YOLO **under-fires by
4×** (1.2% of pixels predicted bruise vs a 4.7% GT rate), and **no threshold fixes it** —
0.479 is the ceiling.

**The fix:** the dataloader still produces one ImageNet-normalised tensor with identical
geometry. For YOLO only, `model_input()` undoes it — `x*STD + MEAN` — recovering the exact
`/255` image it was trained on, verified accurate to **6×10⁻⁸**. No second dataloader, no
second disk read, no different resize. Pixel scale is a property of the weights.

### 🔴 Bug 2 — the `cv2` / `ultralytics` comment was a myth

The notebook always carried `import cv2  # before ultralytics, on purpose`. Measured:

| | `imread(IMREAD_GRAYSCALE)` |
|---|---|
| cv2 alone, no ultralytics | `(4022, 6024)` ✅ |
| ultralytics imported first | `(4022, 6024, 1)` ❌ |
| **cv2 imported first** | `(4022, 6024, 1)` ❌ |

**Import order is irrelevant** — ultralytics monkey-patches `cv2.imread` at import; once
it's imported anywhere, the patch is live. That extra axis reaches `y` as
`[B,1,640,640,1]`, and `dice_np(pred[640,640], gt[640,640,1])` **broadcasts** to
`[640,640,640]` and returns garbage: a **pixel-perfect prediction scores Dice 63.9**.

Fixed at source in `pipeline/data.py`. The notebook asserts GT shape on both splits.

### 🔴 Bug 3 — the old zip's `train_manifest` → test manifest

See §1. Fixed in the build script; the notebook also asserts it independently.

---

## 6. Temperature: why it's gone

For a **hard mask**, `σ(z/T) ≥ thr ⟺ z ≥ T·logit(thr)`. The decision depends **only on the
product** `c = T·logit(thr)`. The old 2-D `(T, thr)` grid was a redundant parameterisation
of a **1-D** family. Verified on the real model — two different grid points, same cut,
**bit-identical** result:

| T | thr | `c = T·logit(thr)` | measured Dice |
|---|---|---|---|
| 2.0 | 0.25 | −2.197225 | **0.40995198** |
| 1.0 | 0.10 | −2.197225 | **0.40995198** |

Of 152 old grid points only **116** were distinct outcomes. Temperature's only real effect
was widening the reachable cut range (±23.6 vs ±2.94 at `T=1`), and both optima sat well
inside what `T=1` already reached.

---

## 7. ✅ RESOLVED: the old `threshold_search.csv` files were right all along

An earlier version of this document claimed the YOLO `threshold_search.csv` files were
"unreproducible" and that `pipeline/benchmark_640.py` and `scripts/27_...` were therefore
suspect. **That claim was wrong and has been retracted.**

The fresh in-notebook val sweep reproduces them almost exactly:

| run | historical CSV | fresh val sweep | delta |
|---|---|---|---|
| `yolo_sem_direct` | 0.7374864 | **0.7375224** | 3.6e-5 |
| `yolo_sem_distilled` | 0.7387694 | **0.7387877** | 1.8e-5 |

The earlier failure to reproduce was an artifact of the *diagnostic* feeding YOLO
ImageNet-normalised pixels (Bug 1) — not a problem with your CSVs. Agreement to ~4×10⁻⁵ on
an independent reimplementation is strong mutual validation of both.

Selected thresholds also land close to the historical ones, all within the tie band:

| model | historical equiv | ours |
|---|---|---|
| B2 teacher | 0.75 | 0.674 |
| B0 direct | 0.60 | 0.574 |
| B0 distilled | 0.55 | 0.562 |
| YOLO direct | 0.15 | 0.182 |
| YOLO distilled | 0.197 | 0.214 |

---

## 8. The results

### Headline table (185 test images, A100)

| model | params | cut | thr | val peak | **median Dice** | **mean Dice** | median IoU | **miss %** | median ms | FPS | act. MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SegFormer-B2 (teacher) | 27.35M | +0.725 | 0.674 | 0.7872 | **0.8341** | 0.7728 | 0.7154 | **0.00** | 34.05 | 29.3 | 693.0 |
| SegFormer-B0 (direct) | 3.71M | +0.300 | 0.574 | 0.7744 | 0.8132 | 0.7768 | 0.6851 | 0.54 | 16.75 | 59.4 | 234.1 |
| **SegFormer-B0 (distilled)** | **3.71M** | +0.250 | 0.562 | 0.7702 | 0.8276 | **0.7863** | 0.7059 | 0.54 | 16.67 | **60.0** | 234.1 |
| YOLO26n-sem (direct) | 1.63M | −1.500 | 0.182 | 0.7375 | 0.8251 | 0.6989 | 0.7023 | **9.19** | 7.90 | **121.6** | 29.7 |
| YOLO26n-sem (distilled) | 1.63M | −1.300 | 0.214 | 0.7388 | 0.8053 | 0.6921 | 0.6741 | 5.95 | 8.96 | 111.3 | 29.7 |

### Distillation works — the student beats its teacher

**SegFormer-B0 distilled beats the B2 teacher on mean Dice (0.7863 vs 0.7728)** with 7.4×
fewer parameters, 2× the throughput, and a third of the activation memory. It also beats
B0 direct (+0.0095 mean, +0.0144 median), so the gain is distillation, not just the
architecture.

For YOLO, distillation is a **wash-to-negative on Dice** (0.6989 → 0.6921) but **improves
the miss rate** (9.19% → 5.95%). It taught the model to fire more often, but less
precisely — non-miss quality drops 0.7696 → 0.7359. Not a clean win.

### YOLO's deficit is entirely complete misses

| model | mean | median | gap | Dice=0 | <0.5 | >0.8 | **mean Dice excluding blanks** |
|---|---|---|---|---|---|---|---|
| B2 teacher | 0.7728 | 0.8341 | +0.061 | 0 | 19 | 113 | 0.7728 (185 imgs) |
| B0 direct | 0.7768 | 0.8132 | +0.036 | 1 | 9 | 101 | 0.7811 (184) |
| B0 distilled | 0.7863 | 0.8276 | +0.041 | 1 | 9 | 113 | **0.7906** (184) |
| YOLO direct | 0.6989 | 0.8251 | **+0.126** | 17 | 37 | 101 | **0.7696** (168) |
| YOLO distilled | 0.6921 | 0.8053 | +0.113 | 13 | 33 | 94 | 0.7359 (174) |

Read the last column: **when YOLO fires, it's competitive** (0.7696 vs B0 direct's 0.7811).
Its whole deficit is **17/185 images where it outputs literally zero pixels**. The large
mean-median gap is that bimodality — it either does well or fails completely. For an
injury-documentation tool a blank mask is a missed injury, so 9.19% is disqualifying
regardless of 121 FPS.

Notes on those failures:
- Missed bruises are **smaller but not tiny**: median GT area 6,603 px vs 11,869 px on hits
  (range 822–17,214 px). At 640×640, 6,603 px is ~1.6% of the frame.
- **No image is missed by all 5 models** — every bruise is detectable; the failures are
  model-specific, not impossible cases.
- Only **6 of 17** YOLO-direct misses are also missed by YOLO-distilled.
- **B2 teacher never blanks (0/185)**, though it has 19 images below Dice 0.5 — it always
  fires *something*.

---

## 9. 🔴 The most important finding: we are at the annotation ceiling

`interlabeler_agreement_640.csv`, same 185 test images:

| comparison | mean Dice | median |
|---|---|---|
| paul ↔ gbarimah | **0.5809** | 0.6325 |
| paul ↔ erik | **0.5812** | 0.6160 |
| gbarimah ↔ erik | 0.7549 | 0.8087 |
| | | |
| paul → majority GT | **0.6998** | 0.7501 |
| gbarimah → majority GT | 0.8729 | 0.9263 |
| erik → majority GT | 0.8657 | 0.8922 |
| | | |
| **our B0 distilled → majority GT** | **0.7863** | 0.8276 |

Three consequences, and they matter more than any row of the results table:

**1. Our model agrees with the consensus better than one of the three experts who defined
it.** B0 distilled scores 0.7863 against the majority GT; Paul scores 0.6998 against the
same GT.

**2. The spread among our 5 models (0.692–0.786) sits *inside* the spread among our humans
(0.700–0.873).** The differences we're comparing are smaller than the disagreement between
annotators. Chasing +0.01 Dice is chasing label noise. This is also why retraining to
recover a hypothetical fraction of a point is not where the value is.

**3. It explains the absurdly flat sweeps.** You cannot resolve a threshold more finely
than the labels are self-consistent:

| model | val Dice at cut −3.0 / −1.0 / 0.0 / +1.0 / +2.0 / +3.0 | tie band (threshold) | tied cuts |
|---|---|---|---|
| B2 teacher | 0.7574 · 0.7785 · 0.7847 · 0.7871 · 0.7852 · 0.7752 | **0.154 → 0.959** | 98/481 |
| B0 direct | 0.7342 · 0.7675 · 0.7740 · 0.7728 · 0.7638 · 0.7418 | 0.168 → 0.900 | 77/481 |
| B0 distilled | 0.7306 · 0.7638 · 0.7700 · 0.7684 · 0.7584 · 0.7341 | 0.154 → 0.900 | 79/481 |
| YOLO direct | 0.7299 · 0.7349 · 0.7261 · 0.7120 · 0.6918 · 0.6521 | 0.028 → 0.634 | 83/481 |
| YOLO distilled | 0.7299 · 0.7384 · 0.7336 · 0.7230 · 0.7025 · 0.6633 | 0.024 → 0.750 | 97/481 |

B2's Dice moves by **0.009** across thresholds from **0.154 to 0.959**. The threshold is
essentially arbitrary over a 6× range.

**The one metric that separates the models by more than noise is the complete-miss rate**
(0.00% vs 9.19%). That arguably belongs as the primary axis of the paper, with Dice as
supporting evidence.

**Suggested framing:** these models have reached human-level agreement on this dataset, and
the remaining variance is in the labels, not the architecture.

---

## 10. Two open items worth acting on

### A free win the notebook is leaving on the table

Within a tie band, all cuts are Dice-equivalent — but they are **not miss-equivalent**.
YOLO direct's band runs down to cut −3.55 (threshold 0.028); the notebook picked the
*median*, −1.50 (0.182). A lower cut in the same band predicts more pixels → fewer blanks →
statistically identical Dice.

The current selection rule optimises for **stability**. The miss rate is what actually
matters clinically. **Breaking ties on complete-miss rate instead of the median would
likely cut that 9.19% for free.** Not yet implemented.

### A label-standard mismatch

Training and val masks come from `train_paul_wl_minus_test_subjects/masks/` — **Paul's**
labels. Test masks are `masks_majority_vote` — the **consensus**.

Per §9, Paul is the outlier annotator (0.58 agreement with each of the other two, who agree
0.755 with each other; 0.70 with the majority). So we:

- **train** on one labelling standard,
- **sweep the threshold** on that same standard (val is also Paul's),
- **score** against a different one.

That's a real distribution shift and may explain part of the val(0.787) → test(0.773) gap.
It also makes the §9 headline more striking, not less: a model trained on Paul's labels
agrees with the consensus **better than Paul does** (0.786 vs 0.700) — so it isn't merely
copying its annotator. This should be stated explicitly in the paper rather than left
implicit in folder names.

---

## 11. Recommendation

**Ship SegFormer-B0 distilled.** Best mean Dice of all five (0.7863), 0.54% blanks, 60 FPS,
3.71M params, 234 MB activation. It dominates the B2 teacher on every axis except a 0.006
median and B2's perfect miss rate.

Keep **B2 teacher** in the paper as the distillation source and as the only model that
never blanks (0/185) — that property is worth reporting.

**YOLO** is 2× faster and 7× smaller, but 9.19% blanks rules it out for injury
documentation. If it stays in the paper, its story is *speed at the cost of catastrophic
recall failures*, and it should be evaluated on miss rate first, Dice second.

---

## 12. How to run it

**Step 1 — rebuild the zip** (~70 s; the `pipeline/data.py` mask fix rides inside it, so
this is not optional):

```
C:/Users/91962/miniconda3/envs/bruise_local/python.exe scripts/32_build_colab_gpu_package.py
```

Writes `C:\BRUISE_SEGMENTATION_PROJECT\bruise_colab_gpu_full.zip` (~1206 MB).

**Step 2 — upload two files:**

| file | destination |
|---|---|
| `bruise_colab_gpu_full.zip` (new, 1.2 GB) | Drive → `MyDrive/bruise_segmentation_gpu/` — **replace** the old 868 MB file |
| `bruise_colab_inference_demo.ipynb` | Colab → File → Upload notebook |

**Step 3 — Runtime → Change runtime type → A100 GPU**, then Run All.

The notebook fails fast rather than producing bad numbers. It stops if: the zip is under
1 GB (old package), there's no GPU, val and test overlap, or the GT mask has the wrong shape.

---

## 13. Verification status

**Verified by the 2026-07-16 Colab run (A100-SXM4-40GB), `analysis/inference_demo_20260716_015433`:**
- all 5 models load and run end-to-end
- the val sweep reproduces the historical `threshold_search.csv` to ~4×10⁻⁵ (§7)
- GPU benchmark timings (p95 ≈ median for every model — very stable)
- all guards passed (leak, mask shape, GPU, package size)

**Verified locally against the real checkpoints (CPU):**
- the GPU Dice implementation matches `pipeline.metrics.dice_np` exactly (0.844443)
- the temperature-redundancy claim (bit-identical Dice)
- the ImageNet-vs-`/255` measurement
- the de-normalisation roundtrip (6×10⁻⁸)
- the `cv2` patch behaviour and its Dice-63.9 consequence
- val/test have no image or subject overlap

**Not verified:**
- whether the tie-break-on-miss-rate change (§10) actually reduces YOLO's blank rate —
  predicted, not measured
