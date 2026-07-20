"""
Layer-wise LR + warmup/poly-decay schedule for SegFormer fine-tuning --
the full recipe from the original SegFormer training config (Xie et al.
2021): backbone gets a conservative LR (it's ImageNet-pretrained), the
randomly-initialized decode head gets 10x that LR so it catches up, weight
decay is skipped on norm/bias parameters, and the LR itself ramps up
linearly for a short warmup before decaying linearly (poly, power=1.0)
back toward zero over the rest of training.

This is a different design from the constant-LR convention used elsewhere
in this project -- a deliberate, separate experiment at 640x640 with this
fuller recipe, not a contradiction of the earlier finding (that finding
was specific to the bruise dataset's own ablation; this folder tests the
paper's own recipe on its own terms).
"""
import torch.nn as nn


def _is_no_decay(name: str, param) -> bool:
    return "norm" in name.lower() or "bias" in name.lower() or param.ndim <= 1


def build_param_groups(model, backbone_lr: float, head_lr: float, weight_decay: float) -> list[dict]:
    """model must expose .backbone and .decode_head (SegformerWrapper)."""
    backbone_names = {id(p) for p in model.backbone.parameters()}

    groups = {
        "backbone_decay": {"params": [], "lr": backbone_lr, "weight_decay": weight_decay},
        "backbone_no_decay": {"params": [], "lr": backbone_lr, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": head_lr, "weight_decay": weight_decay},
        "head_no_decay": {"params": [], "lr": head_lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = id(param) in backbone_names
        no_decay = _is_no_decay(name, param)
        key = ("backbone" if is_backbone else "head") + ("_no_decay" if no_decay else "_decay")
        groups[key]["params"].append(param)

    return [g for g in groups.values() if g["params"]]


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, power: float = 1.0) -> float:
    """1-indexed step. Linear warmup 0 -> 1 over warmup_steps, then poly decay
    (power=1.0 == linear) from 1 -> 0 over the remaining steps."""
    if step <= warmup_steps:
        return step / max(1, warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / remaining)
    return (1.0 - progress) ** power


def apply_lr(optimizer, peak_lrs: list[float], multiplier: float) -> None:
    for group, peak_lr in zip(optimizer.param_groups, peak_lrs):
        group["lr"] = peak_lr * multiplier
