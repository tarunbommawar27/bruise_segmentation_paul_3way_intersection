#!/usr/bin/env python3
"""
Step 2 — Temperature-calibrate the SegFormer-B2 teacher on val logits.

Finds the scalar temperature T that minimises the negative log-likelihood
(binary cross-entropy) of p(y | x/T) over the val set, via L-BFGS
(Guo et al. 2017, "On Calibration of Modern Neural Networks").

════════════════════════════════════════════════════════════════════════════
WHY TEMPERATURE CALIBRATION IS NEEDED
════════════════════════════════════════════════════════════════════════════
BCE loss pushes the final-layer logits toward ±∞ because sigmoid(±∞) = 0/1
gives perfect loss. After training, the teacher's logit histogram is nearly
bimodal: most pixels are either very large positive (foreground) or very
large negative (background). When a student tries to learn from
sigmoid(z_teacher), the soft labels are almost binary — indistinguishable
from hard GT labels, defeating the purpose of soft-label distillation.

Dividing logits by T > 1 (temperature scaling) spreads the distribution.
At T=4, a logit of 8.0 becomes 2.0, giving sigmoid(2.0) ≈ 0.88 instead of
sigmoid(8.0) ≈ 0.9997. The student can now see the teacher's uncertainty
near decision boundaries and learn richer representations from soft labels.

════════════════════════════════════════════════════════════════════════════
WHY L-BFGS (NOT SGD)
════════════════════════════════════════════════════════════════════════════
We optimise a single scalar (log_t) over the entire val set. L-BFGS
converges in ~10 steps vs hundreds for SGD. The second-order curvature
information makes it ideal for low-dimensional calibration problems.

Why optimise log(T) instead of T directly:
  Constrains T > 0 automatically (exp is always positive), avoids the
  singularity at T=0, and makes the loss landscape more symmetric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data import BruiseDataset, load_train_val_split
from pipeline.io_utils import load_yaml, setup_logging, validate_cfg, validate_paths
from pipeline.models import SegformerWrapper, build_segformer

logger = setup_logging()


def _collect_val_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass over the val set and collect all logits and targets.

    Why collect ALL logits before calibration:
      L-BFGS needs the full dataset loss gradient on every step. Processing
      mini-batches would require multiple forward passes per L-BFGS step,
      making each closure call expensive. Collecting logits once (no_grad,
      no backward) and keeping them in RAM is cheaper because: (a) logits
      are much smaller than activations, (b) the val set is small (~18% of
      train), (c) we only do this once per training run.

    Returns:
        (logits [N,1,H,W], targets [N,1,H,W]) on CPU (calibration is cheap on CPU).
    """
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            x = x.to(device)
            all_logits.append(model(x).cpu())
            all_targets.append(y)    # y already on CPU
    return torch.cat(all_logits), torch.cat(all_targets)


def _calibrate(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Find optimal temperature T via L-BFGS on the full collected val set.

    Why L-BFGS closure pattern: L-BFGS is a line-search optimiser that may
    call the function multiple times per step. The `closure` callable allows
    it to recompute loss (and gradients via backward()) on demand.

    Returns:
        Optimal temperature T (≥ 1.0 expected if model is over-confident).
    """
    # log_t: optimise log(T) so T = exp(log_t) is always > 0
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        t    = torch.exp(log_t)          # ensure T > 0
        loss = F.binary_cross_entropy_with_logits(logits / t, targets)
        loss.backward()                  # gradient of loss w.r.t. log_t
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).item())


def main() -> None:
    ap = argparse.ArgumentParser(description="Temperature-calibrate SegFormer-B2 teacher")
    ap.add_argument("--paths",  default="configs/paths.yaml")
    ap.add_argument("--common", default="configs/common_train.yaml")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    cfg   = load_yaml(args.common)
    validate_paths(paths)
    validate_cfg(cfg)

    device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = Path(paths["project_root"]) / "segformer_b2_teacher"
    out_path = run_dir / "temperature.json"

    if out_path.exists():
        logger.info("Already calibrated: %s", out_path)
        return

    # Teacher must exist — raise clearly if not (don't proceed with uninitialised model)
    best_pt = run_dir / "best_model.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {best_pt}\n"
            "Run 01_train_segformer_b2_teacher.py first.")

    _, val_df = load_train_val_split(
        Path(paths["project_root"]) / "splits" / "train_val_split.csv")

    # batch_size=4: small batch is fine; we never backprop through the model here
    loader = DataLoader(
        BruiseDataset(val_df, cfg["img_h"], cfg["img_w"], training=False),
        batch_size=4, shuffle=False, num_workers=4,
    )

    # Load the trained teacher (best val checkpoint)
    model = SegformerWrapper(
        build_segformer(paths["segformer_b2_pretrained"], num_labels=1)
    ).to(device)
    model.load_state_dict(
        torch.load(str(best_pt), map_location=device, weights_only=True))

    logger.info("Collecting val logits for calibration (%d images)...", len(val_df))
    logits, targets = _collect_val_logits(model, loader, device)

    # Baseline NLL before calibration — used to verify calibration actually helped
    nll_before = F.binary_cross_entropy_with_logits(logits, targets).item()

    temperature = _calibrate(logits, targets)

    # Verify calibration improved NLL (it should; if not, T≈1.0 and model was well-calibrated)
    nll_after = F.binary_cross_entropy_with_logits(logits / temperature, targets).item()

    out_path.write_text(json.dumps({
        "temperature": temperature,
        "nll_before":  nll_before,
        "nll_after":   nll_after,
    }, indent=2))

    logger.info(
        "Calibration complete: T=%.4f | NLL %.4f → %.4f (delta=%.4f)",
        temperature, nll_before, nll_after, nll_after - nll_before,
    )
    if temperature < 1.0:
        logger.warning(
            "T=%.4f < 1.0 — model is UNDER-confident. "
            "This is unusual for BCE-trained models; check training.", temperature)


if __name__ == "__main__":
    main()
