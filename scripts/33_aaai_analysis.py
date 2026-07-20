#!/usr/bin/env python3
"""
scripts/33_aaai_analysis.py

AAAI results-section analysis, assembled entirely from already-computed data:
Track B test-set evaluation (`track_b_evaluation/`), CPU/GPU latency
benchmarks (`results/benchmark_640_*`), ITA/skin-tone labels (`ita_labels/`),
Optuna alpha-search trial histories (`optuna_alpha_search/`), and the 3
already-evaluated YOLO normalization protocols (`fixed_test_evaluation/`).
No retraining, no new model inference -- see scripts/34_aaai_qualitative.py
for the two items that need a fresh inference pass (qualitative prediction
grid, failure-case gallery).

PREREQUISITE: track_b_evaluation/ must reflect the CURRENT threshold_search.csv
for all 5 models (re-run via `python scripts/11_track_b_evaluate.py --force`
if the YOLO threshold/temperature registry has changed since it last ran --
see this script's own README/session-handoff note for why that matters).

Run from project root:
    python scripts/33_aaai_analysis.py

Writes everything under aaai_analysis/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "aaai_analysis"

# Canonical 5-model registry: ties together the three different naming
# conventions used across track_b_evaluation/ (run_name), the CPU/GPU
# benchmark CSVs (display "model" string), and this script's own labels.
MODELS = [
    {"run_name": "segformer_b2_teacher", "display": "SegFormer-B2 Teacher",
     "family": "SegFormer", "kd_family": "Teacher (no KD)"},
    {"run_name": "segformer_b0_direct", "display": "SegFormer-B0 Direct",
     "family": "SegFormer", "kd_family": "Direct (no KD)"},
    {"run_name": "segformer_b0_distilled", "display": "SegFormer-B0 Distilled",
     "family": "SegFormer", "kd_family": "WL->WL Distilled"},
    {"run_name": "yolo_sem_direct", "display": "YOLO26n-sem Direct",
     "family": "YOLO", "kd_family": "Direct (no KD)"},
    {"run_name": "yolo_sem_distilled", "display": "YOLO26n-sem Distilled",
     "family": "YOLO", "kd_family": "WL->WL Distilled"},
]


def _load_benchmark_fps(summary_path: Path) -> dict[str, float]:
    df = pd.read_csv(summary_path) if summary_path.suffix == ".csv" else pd.DataFrame(
        json.loads(summary_path.read_text()))
    return dict(zip(df["model"], df["fps_from_mean"]))


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 + 2: master comparison table + accuracy-efficiency Pareto figure
# ─────────────────────────────────────────────────────────────────────────────

def build_master_table() -> pd.DataFrame:
    cpu_raw_fps = _load_benchmark_fps(
        PROJECT_ROOT / "results/benchmark_640_resize_NOT_TIMED_185img/benchmark_640_summary.csv")
    cpu_full_fps = _load_benchmark_fps(
        PROJECT_ROOT / "results/benchmark_640_resize_TIMED_185img/benchmark_640_summary.csv")
    gpu_fp32_fps = _load_benchmark_fps(
        PROJECT_ROOT / "results/benchmark_640_gpu_a100_fp32/benchmark_640_summary.json")

    rows = []
    for spec in MODELS:
        summary = pd.read_csv(
            PROJECT_ROOT / "track_b_evaluation" / spec["run_name"] / "test_summary.csv").iloc[0]
        rows.append({
            "model": spec["display"],
            "family": spec["family"],
            "kd_family": spec["kd_family"],
            "n_images": int(summary["n_images"]),
            "median_dice": summary["median_dice"],
            "mean_dice": summary["mean_dice"],
            "complete_miss_rate": summary["complete_miss_rate"],
            "mean_precision": summary["mean_precision"],
            "mean_recall": summary["mean_recall"],
            "median_hd95_px": summary.get("median_hd95_px"),
            "threshold_used": summary["best_threshold"],
            "temperature_used": summary["best_temperature"],
            "params_millions": None,  # filled from CPU benchmark below
            "cpu_raw_fwd_fps": cpu_raw_fps.get(spec["display"]),
            "cpu_full_pipeline_fps": cpu_full_fps.get(spec["display"]),
            "gpu_a100_fp32_fps": gpu_fp32_fps.get(spec["display"]),
        })

    table = pd.DataFrame(rows)
    # params_millions: read once from the CPU raw-forward benchmark CSV
    # (it's the same trained checkpoint, so param count is precision/device-
    # independent -- no need to also pull it from the GPU JSON).
    params_df = pd.read_csv(
        PROJECT_ROOT / "results/benchmark_640_resize_NOT_TIMED_185img/benchmark_640_summary.csv")
    params_lookup = dict(zip(params_df["model"], params_df["parameters_millions"]))
    table["params_millions"] = table["model"].map(params_lookup)
    return table


def plot_pareto(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"Teacher (no KD)": "^", "Direct (no KD)": "o", "WL->WL Distilled": "s"}
    colors = {"SegFormer": "#1f77b4", "YOLO": "#d62728"}
    for _, row in table.iterrows():
        ax.scatter(
            row["gpu_a100_fp32_fps"], row["median_dice"],
            marker=markers[row["kd_family"]], color=colors[row["family"]],
            s=120, edgecolor="black", linewidth=0.5, zorder=3,
        )
        ax.annotate(
            row["model"], (row["gpu_a100_fp32_fps"], row["median_dice"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax.set_xlabel("A100 GPU throughput, fp32, raw-forward (FPS)")
    ax.set_ylabel("Median Dice (185-image fixed test set)")
    ax.set_title("Accuracy vs. GPU throughput -- 5 core models")
    ax.grid(True, linestyle="--", alpha=0.4)

    from matplotlib.lines import Line2D
    family_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                              markeredgecolor="black", markersize=10, label=fam)
                       for fam, c in colors.items()]
    kd_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=10, label=k)
                  for k, m in markers.items()]
    ax.legend(handles=family_handles + kd_handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Item 3: normalization-bug ablation table (YOLO, 3 already-evaluated protocols)
# ─────────────────────────────────────────────────────────────────────────────

PROTOCOLS = [
    ("", "native (no temp. scaling)"),
    ("_temp_scaled", "temp-scaled, buggy 0-1 scale (scripts/29-era)"),
    ("_temp_scaled_corrected", "temp-scaled, corrected ImageNet-norm preprocessing"),
]


def build_normalization_ablation_table() -> pd.DataFrame:
    rows = []
    for run_name in ("yolo_sem_direct", "yolo_sem_distilled"):
        for suffix, label in PROTOCOLS:
            path = PROJECT_ROOT / "fixed_test_evaluation" / f"{run_name}{suffix}" / "test_summary.csv"
            if not path.exists():
                continue
            s = pd.read_csv(path).iloc[0]
            rows.append({
                "model": run_name,
                "protocol": label,
                "mean_dice": s["mean_dice"],
                "median_dice": s["median_dice"],
                "complete_miss_rate": s["complete_miss_rate"],
                "mean_precision": s["mean_precision"],
                "mean_recall": s["mean_recall"],
                "mean_pred_gt_ratio": s.get("mean_pred_gt_ratio"),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Item 8: multi-metric fairness table (ITA skin-tone groups; DEO/DP analogs)
# ─────────────────────────────────────────────────────────────────────────────

def build_fairness_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    ita = pd.read_csv(PROJECT_ROOT / "ita_labels/wl_test_per_image_ita.csv")[
        ["stem", "skin_tone_category"]]

    per_group_rows = []
    for spec in MODELS:
        per_image = pd.read_csv(
            PROJECT_ROOT / "track_b_evaluation" / spec["run_name"] / "test_per_image.csv")
        merged = per_image.merge(ita, on="stem", how="inner")
        if len(merged) != len(per_image):
            raise RuntimeError(
                f"{spec['run_name']}: {len(per_image) - len(merged)} images failed to join "
                "against ita_labels/wl_test_per_image_ita.csv on 'stem' -- check for a stem "
                "mismatch before trusting this model's fairness numbers.")
        grouped = merged.groupby("skin_tone_category").agg(
            n_images=("stem", "count"),
            mean_dice=("dice", "mean"),
            mean_recall=("recall", "mean"),
            complete_miss_rate=("complete_miss", "mean"),
        ).reset_index()
        grouped.insert(0, "model", spec["display"])
        per_group_rows.append(grouped)
    per_group = pd.concat(per_group_rows, ignore_index=True)

    summary_rows = []
    for model, grp in per_group.groupby("model"):
        summary_rows.append({
            "model": model,
            # DEO analog: max-min True-Positive-Rate (recall) across groups.
            "deo_recall_range": grp["mean_recall"].max() - grp["mean_recall"].min(),
            "worst_recall_group": grp.loc[grp["mean_recall"].idxmin(), "skin_tone_category"],
            "best_recall_group": grp.loc[grp["mean_recall"].idxmax(), "skin_tone_category"],
            # Demographic-parity analog: max-min complete-miss (non-detection) rate.
            "dp_miss_rate_range": grp["complete_miss_rate"].max() - grp["complete_miss_rate"].min(),
            "worst_miss_group": grp.loc[grp["complete_miss_rate"].idxmax(), "skin_tone_category"],
        })
    summary = pd.DataFrame(summary_rows)
    return per_group, summary


# ─────────────────────────────────────────────────────────────────────────────
# Item 9: alpha sweep curves (KD weight vs. validation Dice, full trial history)
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_sweep(trials_csv: Path, title: str, out_path: Path) -> None:
    df = pd.read_csv(trials_csv).sort_values("params_alpha")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["params_alpha"], df["value"], marker="o", linewidth=1.5)
    best = df.loc[df["value"].idxmax()]
    ax.scatter([best["params_alpha"]], [best["value"]], color="red", zorder=5,
               label=f"best: alpha={best['params_alpha']:.2f}, dice={best['value']:.4f}")
    ax.set_xlabel("KD alpha (distillation loss weight)")
    ax.set_ylabel("Validation mean Dice")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("[1/5] Master comparison table...")
    master = build_master_table()
    master.to_csv(OUT_DIR / "master_comparison_table.csv", index=False)
    print(master.to_string(index=False))

    print("\n[2/5] Pareto figure...")
    plot_pareto(master, OUT_DIR / "pareto_fps_vs_dice.png")

    print("\n[3/5] Normalization-bug ablation table...")
    ablation = build_normalization_ablation_table()
    ablation.to_csv(OUT_DIR / "normalization_ablation_table.csv", index=False)
    print(ablation.to_string(index=False))

    print("\n[4/5] Fairness table (DEO/DP analogs across ITA skin-tone groups)...")
    per_group, fairness_summary = build_fairness_table()
    per_group.to_csv(OUT_DIR / "fairness_per_group_table.csv", index=False)
    fairness_summary.to_csv(OUT_DIR / "fairness_summary_table.csv", index=False)
    print(fairness_summary.to_string(index=False))

    print("\n[5/5] Alpha sweep curves...")
    plot_alpha_sweep(
        PROJECT_ROOT / "optuna_alpha_search/segformer_b0_trials.csv",
        "SegFormer-B0: KD alpha vs. validation Dice",
        OUT_DIR / "alpha_sweep_segformer_b0.png",
    )
    plot_alpha_sweep(
        PROJECT_ROOT / "optuna_alpha_search/yolo_sem_trials.csv",
        "YOLO26n-sem: KD alpha vs. validation Dice",
        OUT_DIR / "alpha_sweep_yolo_sem.png",
    )

    print(f"\nAll outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
