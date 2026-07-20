#!/usr/bin/env python3
"""Step 10 — single consolidated CSV: one row per model (5 total), with the
LR-schedule/recipe used, the Optuna-found alpha (for distilled models),
val metrics, and test metrics side by side."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml

RUN_NAMES = [
    "segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled",
    "yolo_sem_direct", "yolo_sem_distilled",
]

IDENTITY_COLS = {
    "run_name", "model_name", "model_family", "yolo_model", "training_type", "alpha",
    "lrf", "cos_lr", "warmup_epochs", "optimizer", "backbone_lr", "head_lr",
    "warmup_steps", "total_steps", "poly_power",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="configs/paths.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    project_root = Path(paths["project_root"])

    rows = []
    for run_name in RUN_NAMES:
        run_dir = project_root / run_name
        val_csv = run_dir / "val_summary.csv"
        test_csv = project_root / "fixed_test_evaluation" / run_name / "test_summary.csv"
        if not val_csv.exists():
            print(f"SKIP {run_name}: no val_summary.csv yet")
            continue

        val_row = pd.read_csv(val_csv).iloc[0].to_dict()
        identity = {k: v for k, v in val_row.items() if k in IDENTITY_COLS}
        identity.setdefault("run_name", run_name)
        val_metrics = {f"val_{k}": v for k, v in val_row.items() if k not in IDENTITY_COLS}

        test_metrics = {}
        if test_csv.exists():
            test_row = pd.read_csv(test_csv).iloc[0].to_dict()
            test_metrics = {f"test_{k}": v for k, v in test_row.items() if k not in IDENTITY_COLS}
        else:
            print(f"  (no test results yet for {run_name})")

        rows.append({**identity, **val_metrics, **test_metrics})

    if not rows:
        print("No completed runs found yet.")
        return

    out_df = pd.DataFrame(rows)
    out_path = project_root / "ALL_MODELS_VAL_TEST_LR_CONSOLIDATED.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows -> {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
