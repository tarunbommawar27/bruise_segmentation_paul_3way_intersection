#!/usr/bin/env python3
"""
scripts/35_generate_paper_tables.py

Emits the LaTeX result tables for docs/aaai_paper.tex directly from the
evaluation CSVs on disk, so no number in the paper is ever hand-transcribed.

WHY GENERATE RATHER THAN TYPE
-------------------------------
Every number in the paper must be traceable to a file in this repository.
Hand-copying values from CSVs into LaTeX is exactly where silent transcription
errors (and stale numbers that no longer match a re-run) enter a paper. This
script reads the authoritative CSV for each claim and writes
docs/tables_generated.tex, which the paper \\input{}s. Re-run it after any
re-evaluation and the paper's tables update with it.

SOURCE OF TRUTH PER TABLE (and why that source, not another)
--------------------------------------------------------------
* SegFormer accuracy -> test_eval_640_no_upscale/<run>/test_summary.csv.
  Chosen over track_b_evaluation/ because the latter's FPS columns were
  overwritten by a CPU-only re-run on a laptop (see SESSION_HANDOFF.md);
  its accuracy columns agree with test_eval_640_no_upscale to <1e-3, so the
  untouched directory is used for everything.
* YOLO accuracy -> yolo_native_640_evaluation/<run>/test_summary.csv, i.e.
  Ultralytics' native inference scored on the same 640 grid as SegFormer
  (scripts/34_evaluate_yolo_native_at_640.py). The custom raw-logit
  threshold+temperature path is reported ONLY in the evaluation-protocol
  ablation, because its ImageNet-normalized input does not match how YOLO
  was trained -- see the paper's Section on evaluation protocol.
* Speed -> results/benchmark_640_gpu_a100_fp32/benchmark_640_summary.json,
  the only benchmark in which all five models were timed on one GPU (A100)
  under one protocol.

DELIBERATELY NOT EMITTED
--------------------------
* Skin-tone/ITA fairness tables. The max-min group gap is an extremum-of-
  extrema statistic and, at this test set's effective sample size (28
  subjects, 9-17 per ITA group), a subject-level cluster bootstrap puts its
  95% CI at roughly [0.04, 0.89] -- too wide to support any claim. Held back
  until the analysis can be reported with proper cluster-bootstrap intervals.
* Any evaluation scored at native camera resolution. Every number in the
  paper is scored on the same 640x640 grid, so no cross-resolution
  comparison arises and none is reported.

Run from project root:
    python scripts/35_generate_paper_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_TEX = PROJECT_ROOT / "docs" / "tables_generated.tex"

# Subject IDs for the cluster bootstrap. Taken from the ITA file's own
# `subject` column -- NEVER regex-derived from stems, because the test set
# contains both TAM009 and TAM0009 as distinct subjects and a prefix regex
# silently merges them.
SUBJECT_CSV = PROJECT_ROOT / "ita_labels" / "wl_test_per_image_ita.csv"

N_BOOT = 4000
BOOT_SEED = 42

A100_JSON = PROJECT_ROOT / "results" / "benchmark_640_gpu_a100_fp32" / "benchmark_640_summary.json"

# (run_name, display, source_dir, benchmark_key, params_millions_source)
MODELS = [
    ("segformer_b2_teacher", "SegFormer-B2 (teacher)", "test_eval_640_no_upscale", "SegFormer-B2 Teacher"),
    ("segformer_b0_direct", "SegFormer-B0 (direct)", "test_eval_640_no_upscale", "SegFormer-B0 Direct"),
    ("segformer_b0_distilled", "SegFormer-B0 (distilled)", "test_eval_640_no_upscale", "SegFormer-B0 Distilled"),
    ("yolo_sem_direct", "YOLO26n-sem (direct)", "yolo_native_640_evaluation", "YOLO26n-sem Direct"),
    ("yolo_sem_distilled", "YOLO26n-sem (distilled)", "yolo_native_640_evaluation", "YOLO26n-sem Distilled"),
]



def _fps_and_params() -> tuple[dict, dict]:
    rows = json.loads(A100_JSON.read_text())
    fps = {r["model"]: r["fps_from_mean"] for r in rows}
    params = {r["model"]: r["parameters_millions"] for r in rows}
    return fps, params


def _summary(run: str, src: str) -> pd.Series:
    return pd.read_csv(PROJECT_ROOT / src / run / "test_summary.csv").iloc[0]


def _per_image(run: str, src: str) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / src / run / "test_per_image.csv")


# ---------------------------------------------------------------------------
# Subject-level (cluster) bootstrap.
#
# WHY CLUSTER, NOT IMAGE. The 185 test images come from only 28 subjects;
# images of one subject share skin, lighting and injury and are not
# independent. An image-level bootstrap treats n=185 and reports intervals
# that are far too narrow. Everything below resamples SUBJECTS.
#
# WHY PAIRED FOR COMPARISONS. Every model is evaluated on the same 185
# images, so a model-vs-model difference is a paired quantity. Comparing two
# marginal CIs for overlap is both wrong and needlessly conservative; the
# bootstrap below resamples subjects once per iteration and evaluates BOTH
# models on that same resample, which is the correct paired test.
# ---------------------------------------------------------------------------

def _with_subjects(run: str, src: str) -> pd.DataFrame:
    subj = pd.read_csv(SUBJECT_CSV)[["stem", "subject"]]
    d = _per_image(run, src)
    d["miss"] = ((d.pred_positive_pixels == 0) & (d.gt_positive_pixels > 0)).astype(float)
    m = d.merge(subj, on="stem", how="inner")
    if len(m) != 185:
        raise RuntimeError(f"{run}: subject merge yielded {len(m)} rows, expected 185")
    if m.subject.nunique() != 28:
        raise RuntimeError(f"{run}: {m.subject.nunique()} subjects, expected 28")
    return m


def _boot_ci(frame: pd.DataFrame, stat, n_boot: int = N_BOOT) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    subs = frame.subject.unique()
    by = {s: frame[frame.subject == s] for s in subs}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        vals.append(stat(pd.concat([by[s] for s in pick], ignore_index=True)))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def _paired_boot(run_a: str, src_a: str, run_b: str, src_b: str, stat,
                  n_boot: int = N_BOOT) -> tuple[float, float, float]:
    a, b = _with_subjects(run_a, src_a), _with_subjects(run_b, src_b)
    m = a.merge(b, on=["stem", "subject"], suffixes=("_a", "_b"))
    if len(m) != 185:
        raise RuntimeError(f"paired merge {run_a}/{run_b}: {len(m)} rows")
    rng = np.random.default_rng(BOOT_SEED)
    subs = m.subject.unique()
    by = {s: m[m.subject == s] for s in subs}
    obs = stat(m)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        vals.append(stat(pd.concat([by[s] for s in pick], ignore_index=True)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return obs, lo, hi


def table_main() -> str:
    fps, params = _fps_and_params()
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{Fixed-test results (185 images from 28 subjects, all metrics on the same "
        r"$640\times640$ grid). Brackets are 95\% subject-level cluster-bootstrap intervals "
        r"($B{=}4000$, resampling subjects). SegFormer operating points are validation-selected "
        r"thresholds; YOLO uses Ultralytics' native \texttt{argmax} (no tunable threshold). "
        r"Throughput is fp32 raw-forward on one A100. \emph{The intervals are wide: at 28 subjects "
        r"most between-model accuracy differences here are not resolvable} "
        r"(Table~\ref{tab:paired}).}",
        r"\label{tab:main}",
        r"\begin{tabular}{llrrccrr}",
        r"\toprule",
        r"Model & Training & Params & $\tau$ & Med.\ Dice [95\% CI] & Mean Dice [95\% CI] & Miss\% & FPS \\",
        r"\midrule",
    ]
    for run, disp, src, bench in MODELS:
        s = _summary(run, src)
        thr = s.get("best_threshold")
        thr_s = "argmax" if (pd.isna(thr) or src.startswith("yolo_native")) else f"{float(thr):.2f}"
        train = "distilled" if "distill" in run else ("teacher" if "teacher" in run else "direct")
        f = _with_subjects(run, src)
        md_lo, md_hi = _boot_ci(f, lambda d: d.dice.median())
        mn_lo, mn_hi = _boot_ci(f, lambda d: d.dice.mean())
        ms_lo, ms_hi = _boot_ci(f, lambda d: d.miss.mean() * 100)
        lines.append(
            f"{disp} & {train} & {params[bench]:.2f}M & {thr_s} & "
            f"{float(s['median_dice']):.3f} [{md_lo:.3f}, {md_hi:.3f}] & "
            f"{float(s['mean_dice']):.3f} [{mn_lo:.3f}, {mn_hi:.3f}] & "
            f"{float(s['complete_miss_rate'])*100:.2f} [{ms_lo:.2f}, {ms_hi:.2f}] & "
            f"{fps[bench]:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_paired() -> str:
    """Paired subject-level bootstrap for the comparisons the paper claims."""
    SEG = "test_eval_640_no_upscale"
    YOL = "yolo_native_640_evaluation"
    med = lambda m: m.dice_a.median() - m.dice_b.median()
    mean = lambda m: m.dice_a.mean() - m.dice_b.mean()
    miss = lambda m: (m.miss_a.mean() - m.miss_b.mean()) * 100.0

    tests = [
        ("segformer_b0_distilled", SEG, "segformer_b0_direct", SEG,
         r"Distillation: B0-distilled $-$ B0-direct"),
        ("segformer_b0_distilled", SEG, "segformer_b2_teacher", SEG,
         r"Student $-$ teacher: B0-distilled $-$ B2"),
        ("yolo_sem_direct", YOL, "segformer_b0_distilled", SEG,
         r"YOLO-direct $-$ B0-distilled"),
        ("yolo_sem_distilled", YOL, "yolo_sem_direct", YOL,
         r"YOLO distillation: distilled $-$ direct"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\caption{\textbf{Paired} subject-level cluster bootstrap ($B{=}4000$) for each comparison the "
        r"paper makes. Both models are evaluated on the same 185 images, so differences are paired; each "
        r"bootstrap iteration resamples subjects once and scores both models on that resample. Intervals "
        r"excluding zero are marked \textbf{sig}. Only the \emph{failure-mode} differences are resolvable "
        r"at this sample size---no Dice difference is.}",
        r"\label{tab:paired}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Comparison & $\Delta$ Med.\ Dice [95\% CI] & $\Delta$ Mean Dice [95\% CI] & $\Delta$ Miss\% [95\% CI] \\",
        r"\midrule",
    ]
    for a, sa, b, sb, label in tests:
        cells = []
        for stat, fmt in ((med, ".3f"), (mean, ".3f"), (miss, ".2f")):
            o, lo, hi = _paired_boot(a, sa, b, sb, stat)
            sig = (lo > 0) or (hi < 0)
            txt = f"{o:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}]"
            cells.append(rf"\textbf{{{txt}}}\,sig" if sig else txt)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def table_protocol_ablation() -> str:
    """YOLO evaluation-protocol ablation: identical checkpoints, two input
    pipelines, both scored on the same 640 grid so the scoring grid cannot
    contribute to the difference."""
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Evaluation-protocol ablation. \emph{Identical checkpoints and identical $640\times640$ "
        r"scoring grid in every row}; only the input pipeline changes. The custom path applies ImageNet "
        r"normalisation and a stretch-resize; the native path applies Ultralytics' $[0,1]$ scaling and "
        r"letterbox, matching how the weights were trained. Bypassing a framework's native preprocessing "
        r"to gain threshold control is not free.}",
        r"\label{tab:protocol}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Input pipeline & Mean Dice & Med.\ Dice & Miss\% \\",
        r"\midrule",
    ]
    for run, disp in [("yolo_sem_direct", "YOLO-direct"), ("yolo_sem_distilled", "YOLO-distilled")]:
        cs = _summary(run, "test_eval_640_no_upscale")          # custom: ImageNet norm + stretch
        vs = _summary(run, "yolo_native_640_evaluation")        # native: /255 + letterbox
        lines.append(rf"\multirow{{2}}{{*}}{{{disp}}}")
        lines.append(
            f"  & custom (ImageNet norm) & {float(cs['mean_dice']):.3f} & "
            f"{float(cs['median_dice']):.3f} & {float(cs['complete_miss_rate'])*100:.2f} \\\\")
        lines.append(
            f"  & native (Ultralytics) & {float(vs['mean_dice']):.3f} & "
            f"{float(vs['median_dice']):.3f} & {float(vs['complete_miss_rate'])*100:.2f} \\\\")
        lines.append(r"\midrule" if run == "yolo_sem_direct" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_hparams() -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Training configuration. SegFormer follows the encoder/decoder learning-rate recipe of "
        r"\citet{xie2021segformer}; YOLO follows Ultralytics' own recipe \citep{jocher2026yolo26}. "
        r"Batch size is not hand-set: a VRAM probe raises it per model up to a cap of 32, which B0 fits "
        r"and the larger B2 does not. Distillation weights $\alpha$ were selected by Optuna.}",
        r"\label{tab:hparams}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Setting & SegFormer & YOLO26n-sem \\",
        r"\midrule",
        r"Resolution & \multicolumn{2}{c}{$640\times640$} \\",
        r"Optimiser & AdamW & auto (Ultralytics) \\",
        r"Backbone LR & $6\times10^{-5}$ & --- \\",
        r"Decoder LR & $6\times10^{-4}$ ($10\times$) & --- \\",
        r"Weight decay & 0.01 & 0.0005 \\",
        r"LR schedule & poly, $p{=}1.0$ & cosine, $\mathrm{lr}_f{=}0.01$ \\",
        r"Warmup & 1\% of steps & 3 epochs \\",
        r"Batch (effective) & 8 (B2) / 32 (B0) & 32 \\",
        r"Steps/epoch & 87 (B2) / 21 (B0) & --- \\",
        r"Max epochs / patience & \multicolumn{2}{c}{100 / 15} \\",
        r"Grad.\ clip & 1.0 & --- \\",
        r"AMP & \multicolumn{2}{c}{enabled} \\",
        r"Seed & \multicolumn{2}{c}{42} \\",
        r"\midrule",
        r"Teacher temperature $T$ & \multicolumn{2}{c}{1.840 (NLL-fitted on val)} \\",
        r"Distillation $\alpha$ & 0.60 (Optuna) & 0.40 (Optuna) \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    missing = [p for p in (A100_JSON, SUBJECT_CSV) if not p.exists()]
    for run, _, src, _ in MODELS:
        p = PROJECT_ROOT / src / run / "test_summary.csv"
        if not p.exists():
            missing.append(p)
    if missing:
        print("ERROR: missing required inputs:", file=sys.stderr)
        for m in missing:
            print("  ", m, file=sys.stderr)
        sys.exit(1)

    parts = [
        "% AUTO-GENERATED by scripts/35_generate_paper_tables.py -- DO NOT EDIT BY HAND.",
        "% Re-run that script after any re-evaluation to refresh these tables.",
        "",
        table_main(),
        table_paired(),
        table_hparams(),
        table_protocol_ablation(),
    ]
    OUT_TEX.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
