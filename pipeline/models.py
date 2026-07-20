from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_segformer(pretrained: str, num_labels: int = 1) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(
        pretrained, num_labels=num_labels, ignore_mismatched_sizes=True,
    )


class SegformerWrapper(nn.Module):
    """Wraps HF SegformerForSemanticSegmentation; upsamples logits to input resolution
    and exposes the backbone (.segformer) and head (.decode_head) separately so the
    trainer can give them different learning rates (pipeline/scheduler.py)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def backbone(self) -> nn.Module:
        return self.model.segformer

    @property
    def decode_head(self) -> nn.Module:
        return self.model.decode_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def gradient_checkpointing_enable(self) -> None:
        try:
            self.model.segformer.encoder.gradient_checkpointing = True
        except AttributeError:
            pass


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint + threshold loading (moved here from
# scripts/29_benchmark_inference_all_models.py, which originally duplicated
# this logic inline). Any script that needs a trained SegFormer run --
# benchmarking, further evaluation, a demo -- should import these instead of
# re-deriving the checkpoint/threshold-file layout again.
# ─────────────────────────────────────────────────────────────────────────────

def load_segformer_threshold(run_dir: Path) -> float:
    """Read the best val-set threshold for one SegFormer run.

    threshold_search.csv (written by pipeline/trainer.py's val-set sweep) has
    one row per candidate threshold with its resulting mean_dice; the best
    threshold is whichever row has the highest mean_dice. We deliberately
    read this from val, never recompute it on test -- fitting a threshold on
    the same data you evaluate accuracy on would leak information and inflate
    the reported score.
    """
    csv = run_dir / "threshold_search.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing threshold_search.csv: {csv}")

    row = pd.read_csv(csv).sort_values("mean_dice", ascending=False).iloc[0]
    return float(row["threshold"])


def _pick_matching_checkpoint(run_dir: Path, model: nn.Module) -> tuple[Path, dict]:
    """Return whichever of best_model.pt / best_model.pre_migration_backup.pt
    actually matches this environment's transformers version.

    Why this exists: scripts/26_migrate_legacy_segformer_checkpoints.py
    rewrote some checkpoints' internal parameter names (e.g.
    encoder.block.0.0... -> stages.0.blocks.0...) to match the transformers
    version ORC had at the time, keeping the original under
    best_model.pre_migration_backup.pt as a safety copy. Whichever file's
    keys match the CURRENTLY installed transformers version's naming is the
    one that should be loaded -- that isn't necessarily best_model.pt, e.g.
    on a machine whose transformers install matches the pre-migration
    naming instead. Comparing key sets (not tensor values) is enough to
    tell which file is compatible, and is cheap since state_dicts are
    already in memory once loaded.
    """
    primary = run_dir / "best_model.pt"
    backup = run_dir / "best_model.pre_migration_backup.pt"

    if not primary.exists():
        raise FileNotFoundError(f"Missing checkpoint: {primary}")

    model_keys = set(model.state_dict().keys())
    candidates = [primary] + ([backup] if backup.exists() else [])
    tried = []
    for candidate in candidates:
        state_dict = torch.load(str(candidate), map_location="cpu", weights_only=True)
        if set(state_dict.keys()) == model_keys:
            return candidate, state_dict
        tried.append(candidate)

    raise RuntimeError(
        f"None of {tried} have parameter names matching the currently installed "
        "transformers version's SegFormer implementation. See "
        "scripts/26_migrate_legacy_segformer_checkpoints.py's docstring for the "
        "root cause (an upstream transformers internal rename) and how to add a "
        "new key-remapping rule if neither existing file matches."
    )


def load_segformer_model(
    model_name: str,
    pretrained_key: str,
    paths: dict,
    device: torch.device,
) -> tuple[nn.Module, float, Path]:
    """Load a trained SegFormer run's weights + its calibrated val threshold.

    Returns (model, threshold, checkpoint_path) so callers can both run
    inference and report which exact checkpoint/threshold produced a given
    result -- important for benchmark/eval CSVs where provenance matters.
    """
    run_dir = Path(paths["project_root"]) / model_name

    thr = load_segformer_threshold(run_dir)

    model = SegformerWrapper(build_segformer(paths[pretrained_key], num_labels=1)).to(device)
    ckpt, state_dict = _pick_matching_checkpoint(run_dir, model)
    model.load_state_dict(state_dict)
    model.eval()

    return model, thr, ckpt
