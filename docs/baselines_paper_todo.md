# Baselines → paper: scheme of things to do

Status: **planned, not done yet.** Captures the agreed work for folding the
U-Net / DeepLabV3+ (and later nnU-Net) baselines into the paper's evidence.

## Context
- `EXTRA/` has `train_smp_baselines.py`, `train_nnunet_baseline.py`, and
  `smp_baselines.zip` (results for **U-Net ResNet50** and **DeepLabV3+ ResNet50**,
  **seed 42 only**).
- The SMP baselines already use the **same SegFormer custom-loop recipe** as the
  final notebook (640 stretch, /255 loader + ImageNet-normalising model, Dice+BCE,
  AdamW encoder/head LR split 6e-5/6e-4, poly decay, val threshold sweep, test@640).
  This is correct — U-Net/DeepLab are plain nn.Modules, so the shared recipe is fair.

## Fixes to make on the re-run (consistency with the 5 core models)
1. **Use the identical train/val split** as the final notebook (**697 train / 134 val**).
   The current baselines used 693/138 — same 831 pool + same 185 test, but a different
   partition, so the val-fitted threshold isn't strictly comparable.
2. **Train 3 seeds (0, 1, 2)**, not the single seed 42 — so multi-seed averaging and
   the paired cluster bootstrap work like they do for SegFormer/YOLO.
3. Score against the same **2-of-3 majority** target at **640**.
4. Report **complete-miss rate** alongside Dice for both baselines.
5. For **nnU-Net**: train in its OWN native self-configuring pipeline (that's the point
   of nnU-Net) — this is an "each-at-its-strongest" datapoint, distinct from the shared
   recipe used for U-Net/DeepLab.

## The analysis cell to write (when ready)
- Drop U-Net + DeepLabV3+ into the **same paired subject-level cluster bootstrap
  (28 subjects, B=4000, paired)** as the 5 core models. For each baseline-vs-model
  comparison report **Δ mean/median Dice, 95% CI, and P(Δ>0)**, plus the miss-rate gap.
- Caution: B0-distilled (0.787) vs baselines (U-Net 0.741 / DeepLab 0.736) is ~0.046 mean —
  near the ~0.04 minimum detectable effect at n=28. **Do NOT claim "SegFormer beats U-Net"
  without the CI**; it may not survive.

## Current single-seed numbers (reference only, seed 42)
| Model | mean Dice | median | complete-miss |
|---|---|---|---|
| U-Net R50 | 0.741 | 0.828 | 5.4% (10/185) |
| DeepLabV3+ R50 | 0.736 | 0.800 | 2.2% (4/185) |

## Why this strengthens the paper
1. Adds the **external baselines** reviewers expect (a previously-named weakness).
2. Reinforces the **annotation-ceiling thesis**: 5+ architectures all cluster at
   0.74–0.79 → label-limited, not capacity-limited.
3. Extends the **miss-rate story**: SegFormer (0–0.5%) / DeepLab (2.2%) safe vs YOLO (7–10%).
   Frame as "all architectures sit at the ceiling," not "SegFormer wins."
