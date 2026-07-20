# Final results vs. U-Net / DeepLab baselines — what I think

Source: `results_final/` (final notebook, 3 seeds, inference done) vs
`EXTRA/smp_baselines.zip` (U-Net R50, DeepLabV3+ R50, **seed 42 only**).
All scored on the same 185-image test set at 640, threshold fitted on val.

## The full table (mean over available seeds)

| Model | Training | seeds | params | mean Dice | median | complete-miss |
|---|---|---|---|---|---|---|
| SegFormer-B2 teacher | custom loop | 3 | 27.3 M | **0.765 ± 0.004** | 0.811 | **0.00%** |
| SegFormer-B0 distilled | custom loop | 3 | **3.7 M** | 0.764 ± 0.005 | 0.811 | 0.18% |
| SegFormer-B0 direct | custom loop | 3 | 3.7 M | 0.760 ± 0.006 | 0.811 | 0.54% |
| **U-Net R50** | custom loop | 1 (42) | 32.5 M | 0.741 | **0.828** | 5.41% |
| **DeepLabV3+ R50** | custom loop | 1 (42) | 26.7 M | 0.736 | 0.800 | 2.16% |
| YOLO26n direct (native) | Ultralytics | 3 | 1.6 M | 0.691 ± 0.020 | 0.802 | 7.57% |
| YOLO26n distilled (native) | Ultralytics | 3 | 1.6 M | 0.668 ± 0.065 | 0.766 | 7.57% |

Ordering by mean Dice: **SegFormer (0.760–0.765) > U-Net (0.741) > DeepLab (0.736) > YOLO (0.668–0.691).**

## What stands out

1. **SegFormer-B0 is the efficiency winner — it beats BOTH classic baselines at ~1/8 the
   parameters.** 3.7 M params vs U-Net's 32.5 M and DeepLab's 26.7 M, and higher mean Dice
   (0.764 vs 0.741 / 0.736) with a far lower miss rate. This is a clean accuracy-vs-size
   Pareto point and probably the strongest single sentence the baselines buy us.

2. **The three strong architectures cluster tightly (0.736–0.765 mean, 0.80–0.83 median).**
   SegFormer, U-Net, DeepLab all land in the same band → reinforces the annotation-ceiling
   thesis: performance is **label-limited, not capacity-limited**. Adding two more
   architectures that also plateau here is good evidence, not a null result to apologise for.

3. **YOLO is the clear outlier below the pack this run — and the most unstable.** Native argmax
   0.691 (direct) / 0.668 (distilled), with big seed variance (std up to 0.065; one custom255
   seed collapsed to 0.52). U-Net and DeepLab both beat YOLO comfortably. So the story is no
   longer "YOLO ties the baselines" — here **the classic baselines outperform YOLO.**

4. **Mean-vs-median tells the failure-mode story.** YOLO's median (~0.80) is close to everyone
   else, but its mean (~0.69) is dragged down by a fat left tail of complete misses. SegFormer's
   mean and median are both high and close → consistent, few catastrophic misses. This is the
   bimodality point the paper already makes, now visible across all models.

## Safety (complete-miss rate) — a separate ranking

**SegFormer (0–0.5%) < DeepLab (2.2%) < U-Net (5.4%) < YOLO (7.6%).** For a forensic tool a
blank mask is a missed injury, so this ordering matters as much as Dice — and SegFormer wins it
decisively. Note U-Net has the **best median Dice (0.828)** yet a 5.4% miss rate: "best Dice ≠
safest model" holds for the baselines too.

## Distillation verdict (unchanged / reinforced)

- **SegFormer:** B0-distilled 0.764 vs B0-direct 0.760 → **+0.004 mean, essentially null**
  (smaller than the ±0.005 seed noise). Miss slightly better (0.18% vs 0.54%).
- **YOLO:** distillation **hurt** it (native 0.668 distilled vs 0.691 direct) and blew up the
  variance. So distillation is null-to-negative across the board — consistent with the paper's
  thesis that at n=28 the distillation effect is inside the noise.

## Fairness (skin tone) — keep light, per project policy

Only SegFormer-B2 teacher shows a "significant" skin-tone gap (Kruskal p=0.011); B0 direct/
distilled and all YOLO paths are non-significant. Given the documented n=28 unreliability of the
extremum-of-extrema gap, treat this as **exploratory, not a claim** (already the paper's stance).

## Important caveats before using baseline numbers

1. **Baselines are single-seed (42); final models are 3-seed averaged.** The 0.741 / 0.736 have
   **no error bars** and could be a lucky/unlucky init. Not rigorously comparable yet.
2. **Different train/val split** (baselines 693/138 vs final 697/134). Same 185 test, but the
   val-fitted threshold differs.
3. → **Re-run the baselines with the identical 697/134 split and seeds 0/1/2**, then run the
   paired subject-level cluster bootstrap (28 subjects) so SegFormer-vs-baseline gaps get CIs.
   The B0-distilled − baseline gap (~0.023–0.028) is near the ~0.04 MDE at n=28, so
   **do not claim "SegFormer beats U-Net" without the interval.** (See `baselines_paper_todo.md`.)

## Bottom line for the paper

The baselines make the paper stronger in three concrete ways:
- **External-baseline coverage** (a previously-named reviewer weakness) — now U-Net + DeepLab,
  soon nnU-Net.
- **A crisp efficiency headline**: SegFormer-B0 matches/beats 9×-larger classic segmenters and
  is far safer.
- **Stronger annotation-ceiling evidence**: five architectures, one 0.74–0.77 band; the only one
  that falls out is YOLO, and it falls *below*, not above.

Frame it as *"every capable architecture sits at the label ceiling; the differences that remain
are failure-rate and stability, not headline Dice"* — not as "SegFormer is the winner."
