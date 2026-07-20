"""
pipeline/trainer.py

Training and evaluation engine for SegFormer models (B2 teacher, B0 direct,
B0 distilled). Implements the full layer-wise LR + warmup/poly schedule from
the SegFormer paper (Xie et al. 2021) and the VRAM-probing batch finder.

All functions here are SegFormer-specific. YOLO training is handled by
Ultralytics' own trainer (scripts/06, 08) and is intentionally kept separate
because Ultralytics uses its own optimizer, LR schedule, and data format.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipeline.batch_finder import find_optimal_micro_batch
from pipeline.data import BruiseDataset
from pipeline.losses import DiceBCELoss, DistillSegLoss
from pipeline.metrics import compute_image_row, summarize
from pipeline.models import SegformerWrapper, build_segformer, count_params
from pipeline.scheduler import apply_lr, build_param_groups, lr_multiplier

# All pipeline modules use the same named logger so setup_logging() in main()
# configures both file and stdout handlers for all modules at once
logger = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# Teacher loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_temperature(teacher_dir: Path) -> float:
    """Read the calibrated temperature scalar from the teacher's run directory.

    Why temperature calibration: the B2 teacher's raw logits are over-confident
    (BCE loss pushes them toward ±∞). Dividing logits by T > 1 before sigmoid
    spreads the probability distribution, giving more useful soft labels for
    knowledge distillation. T was found by step 02_calibrate_teacher.py.

    Why default T=1.0: without calibration the teacher is still usable —
    T=1.0 means no temperature scaling. We raise only if the file is expected
    and cannot be parsed (a genuine error), not if the file is absent.

    Args:
        teacher_dir: directory where temperature.json was written by step 02.

    Returns:
        float temperature value (≥ 1.0 expected; 1.0 if file absent).
    """
    temp_file = teacher_dir / "temperature.json"
    if not temp_file.exists():
        # Temperature file is written by a separate calibration step.
        # If missing, T=1.0 is the safe default (uncalibrated teacher).
        logger.warning("temperature.json not found in %s — using T=1.0 (uncalibrated)", teacher_dir)
        return 1.0
    # Parse temperature from JSON; get() guards against empty/malformed files
    return float(json.loads(temp_file.read_text()).get("temperature", 1.0))


def load_teacher(
    teacher_dir: Path,
    pretrained: str,
    device: torch.device,
    amp: bool = True,
) -> Callable:
    """Load the calibrated SegFormer-B2 teacher as a callable soft-label generator.

    Returns a function that takes an input tensor x and returns the teacher's
    temperature-scaled sigmoid probability map. The function runs under no_grad
    and amp (matching the conditions used during training) so it never computes
    gradients and does not inflate VRAM usage.

    Why return a callable (not the model directly): the trainer's batch loop
    calls teacher_fn(x) without knowing anything about the teacher architecture.
    This abstraction lets us swap in any teacher (different architecture, cached
    logits, etc.) without changing the training loop.

    Args:
        teacher_dir: path to the teacher run directory (contains best_model.pt).
        pretrained:  path to HuggingFace SegFormer-B2 pretrained checkpoint.
        device:      GPU device to load the teacher onto.
        amp:         whether to use autocast for teacher forward passes.

    Returns:
        A callable: (x: Tensor) → soft_prob: Tensor of same spatial size.

    Raises:
        FileNotFoundError: if best_model.pt is missing in teacher_dir.
    """
    teacher_dir = Path(teacher_dir)
    best_model_path = teacher_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(
            f"Teacher weights not found: {best_model_path}\n"
            "Run 01_train_segformer_b2_teacher.py first.")

    # Build teacher model and load best-val checkpoint
    model = SegformerWrapper(build_segformer(pretrained, num_labels=1)).to(device)
    # weights_only=True: load only state dict — avoids arbitrary pickle execution
    model.load_state_dict(
        torch.load(str(best_model_path), map_location=device, weights_only=True)
    )
    model.eval()    # disable dropout / BN train mode for inference

    temperature = _load_temperature(teacher_dir)
    logger.info("Teacher loaded from %s (T=%.3f)", teacher_dir, temperature)

    def teacher_fn(x: torch.Tensor) -> torch.Tensor:
        """Generate soft probability labels from teacher for input x.

        Why no_grad: teacher is frozen — computing gradients through it wastes
        memory and compute. This is the largest memory saving in the distillation
        setup.
        Why autocast: matches the student's training conditions so both
        models see the same numerical environment within each batch.
        """
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
        # Temperature-scale before sigmoid: spreads over-confident teacher probabilities
        return torch.sigmoid(logits / temperature)

    # Attach model reference so Approach B's FeatureHook can register on it
    teacher_fn._model = model
    return teacher_fn


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    amp: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate a SegFormer model on a DataLoader and return per-image rows + summary.

    Why evaluate at threshold 0.50 during training (not the best threshold):
      The best threshold is found by the post-training sweep. During training we
      use 0.50 to track progress consistently — if we swept thresholds every epoch
      it would add 19× the evaluation time and introduce noise.

    Args:
        model:     SegformerWrapper in any state (will be set to eval() internally).
        loader:    DataLoader over val or test set.
        device:    GPU device.
        threshold: binary classification threshold (0.0–1.0).
        amp:       whether to use autocast.

    Returns:
        (per_image_df, summary_dict)
    """
    model.eval()    # always set to eval — caller may have left model in train mode
    rows = []

    with torch.no_grad():
        for x, y, stems, *_ in loader:
            x = x.to(device, non_blocking=True)    # non_blocking: overlaps H2D transfer with CPU work
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
            # Convert to fp32 before sigmoid: avoids fp16 overflow near ±65504
            prob = torch.sigmoid(logits).float().cpu().numpy()
            gt   = y.cpu().numpy()

            for i, stem in enumerate(stems):
                pred = (prob[i, 0] >= threshold).astype("uint8")   # threshold to binary mask
                g    = (gt[i, 0] > 0.5).astype("uint8")             # binarise GT (handles float masks)
                rows.append(compute_image_row(pred, g, str(stem)))

    return pd.DataFrame(rows), summarize(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
    amp: bool,
) -> tuple[pd.DataFrame, float]:
    """Sweep candidate thresholds on the val set and return the best one.

    Why sweep on val (not test): we must never tune any hyperparameter on the
    test set. The threshold is a hyperparameter — selecting it on test would
    inflate reported Dice. The test set is only evaluated once, with the
    threshold already fixed.

    Args:
        thresholds: list of candidate thresholds (from common_train.yaml).

    Returns:
        (threshold_df sorted best-first, best_threshold float).
    """
    rows = []
    for thr in thresholds:
        _, s = evaluate(model, loader, device, thr, amp)   # run full val pass per threshold
        rows.append({"threshold": thr, **s})               # store summary for this threshold

    df = pd.DataFrame(rows).sort_values("mean_dice", ascending=False)
    best_thr = float(df.iloc[0]["threshold"])    # top row has highest val Dice
    return df, best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    sup_loss: nn.Module,
    dist_loss: Optional[nn.Module],
    scaler: Optional[torch.amp.GradScaler],
    accum_steps: int,
    device: torch.device,
    clip_grad: float,
    teacher_fn: Optional[Callable],
    total_steps: int,
    warmup_steps: int,
    peak_lrs: list[float],
    poly_power: float,
    global_step: int,
    desc: str,
) -> tuple[float, int]:
    """Run one training epoch with gradient accumulation and AMP.

    Why teacher runs BEFORE student forward:
      Running teacher first (under its own no_grad context which is released
      immediately) means the teacher's activations are freed before the student's
      backward graph is built. Running student first would keep both the student's
      retained backward graph AND the teacher's forward activations alive simultaneously —
      doubling peak VRAM. The batch-size probe uses this same ordering, so the
      probe's VRAM measurement accurately predicts actual training VRAM.

    Args:
        accum_steps: number of micro-batches to accumulate before a weight update.
                     Allows effective batch size > micro-batch size.
        clip_grad:   max gradient norm for gradient clipping (prevents exploding gradients).
        global_step: current total step count (used for LR schedule, updated here).

    Returns:
        (mean_train_loss_per_batch, updated_global_step)
    """
    model.train()       # enable dropout and BN train mode
    optimizer.zero_grad()
    total_loss = 0.0
    n_batches  = len(loader)
    amp_enabled = (scaler is not None)

    for step, (x, y, *_) in enumerate(tqdm(loader, desc=desc, leave=False)):
        global_step += 1    # increment before LR update so step 1 uses warmup LR

        # Update LR using the warmup/poly schedule (SegFormer paper recipe)
        mult = lr_multiplier(global_step, total_steps, warmup_steps, poly_power)
        apply_lr(optimizer, peak_lrs, mult)

        x = x.to(device, non_blocking=True)    # non_blocking: overlaps H2D with CPU work
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            # Teacher runs first (under its own no_grad) to minimise peak VRAM
            tprob  = teacher_fn(x) if teacher_fn is not None else None
            logits = model(x)
            # Distillation loss if teacher present, otherwise pure supervised loss
            if teacher_fn is not None:
                loss = dist_loss(logits, y, tprob) / accum_steps
            else:
                loss = sup_loss(logits, y) / accum_steps

        # Backward pass (with AMP scaling to prevent fp16 underflow)
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Accumulate to running total (undo the /accum_steps to track true loss magnitude)
        total_loss += loss.item() * accum_steps

        # Weight update: only on accumulation boundaries and at end of epoch
        is_last = ((step + 1) == n_batches)
        if (step + 1) % accum_steps == 0 or is_last:
            if amp_enabled:
                scaler.unscale_(optimizer)           # restore true gradients before clipping
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)               # update weights (skips if grads are inf/nan)
                scaler.update()                      # adjust scaler scale for next iteration
            else:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad()    # clear gradients before next accumulation window

    return total_loss / n_batches, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Training setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer_and_schedule(
    model: nn.Module, cfg: dict
) -> tuple[torch.optim.Optimizer, list[float], int, int, float]:
    """Build AdamW optimizer with layer-wise LRs and compute schedule parameters.

    Why layer-wise LRs (SegFormer paper):
      The backbone (MiT encoder) is pretrained on ImageNet — it already has good
      representations. A conservative backbone_lr preserves those features.
      The decode head is randomly initialised — it needs head_lr = 10× backbone_lr
      to learn quickly and catch up with the pretrained backbone.

    Returns:
        (optimizer, peak_lrs, total_steps, warmup_steps, poly_power)
    """
    # build_param_groups: separates backbone and head params + no weight decay on bias/norm
    param_groups = build_param_groups(
        model, cfg["backbone_lr"], cfg["head_lr"], cfg["weight_decay"]
    )
    optimizer = torch.optim.AdamW(
        param_groups, betas=tuple(cfg.get("betas", [0.9, 0.999]))
    )
    # peak_lrs: maximum LR per group (used by the warmup/poly schedule)
    peak_lrs = [g["lr"] for g in param_groups]
    return optimizer, peak_lrs


def _compute_schedule_params(
    n_batches: int, accum_steps: int, cfg: dict
) -> tuple[int, int, float]:
    """Compute total_steps, warmup_steps, poly_power for the LR schedule.

    Why steps (not epochs): the warmup/poly schedule (SegFormer paper) is
    defined in terms of gradient-update steps, not epochs. Using epochs
    would be correct but less precise for runs with non-integer accum_steps.

    Args:
        n_batches:   number of batches per epoch (from DataLoader).
        accum_steps: gradient accumulation steps.
        cfg:         common_train.yaml config.

    Returns:
        (total_steps, warmup_steps, poly_power)
    """
    # Effective updates per epoch: each accum_steps micro-batches = 1 update
    steps_per_epoch = max(1, n_batches // accum_steps)
    total_steps     = steps_per_epoch * cfg["epochs"]
    # warmup_fraction of total_steps: 1% by default (SegFormer paper)
    warmup_steps    = max(1, int(total_steps * cfg.get("warmup_fraction", 0.01)))
    poly_power      = cfg.get("poly_power", 1.0)
    return total_steps, warmup_steps, poly_power


def _save_run_config(
    run_dir: Path, model_name: str, training_type: str, alpha: Optional[float],
    micro_batch: int, accum_steps: int, vram_frac: float,
    warmup_steps: int, total_steps: int, poly_power: float,
    total_params: int, trainable_params: int, cfg: dict,
) -> None:
    """Save run configuration to run_config.json in the run directory.

    Why persist run config: evaluation scripts (09, 10, 11) need to know
    exactly how each model was trained (micro_batch, LR, etc.) to interpret
    results correctly. Writing config to disk at training time means the
    evaluation script doesn't need to re-parse CLI args.
    """
    run_cfg = {
        "model_name":       model_name,
        "training_type":    training_type,
        "alpha":            alpha,
        "micro_batch":      micro_batch,
        "accum_steps":      accum_steps,
        "effective_batch":  micro_batch * accum_steps,
        "vram_fraction_at_probe": round(vram_frac, 4),
        "backbone_lr":      cfg["backbone_lr"],
        "head_lr":          cfg["head_lr"],
        "weight_decay":     cfg["weight_decay"],
        "warmup_steps":     warmup_steps,
        "total_steps":      total_steps,
        "poly_power":       poly_power,
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "img_h":            cfg["img_h"],
        "img_w":            cfg["img_w"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Resume checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_resume_state(
    resume_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
) -> tuple[int, float, int, list[dict], int, int, int, float]:
    """Load training state from a resume checkpoint.

    Why resume checkpoints: GPU sessions frequently disconnect mid-training.
    Without resume support every crash restarts from epoch 0. The resume
    checkpoint is written every epoch so at most 1 epoch of work is lost.

    Args:
        resume_path: path to resume_checkpoint.pt.

    Returns:
        (start_epoch, best_score, patience_counter, history, global_step,
         micro_batch, accum_steps, vram_frac)
    """
    saved = torch.load(str(resume_path), map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"])
    optimizer.load_state_dict(saved["optimizer_state"])
    if scaler is not None and saved.get("scaler_state") is not None:
        scaler.load_state_dict(saved["scaler_state"])

    start_epoch      = saved["epoch"] + 1        # resume from NEXT epoch
    best_score       = saved["best_score"]
    patience_counter = saved["patience_counter"]
    history          = saved.get("history", [])
    global_step      = saved.get("global_step", 0)
    micro_batch      = saved["micro_batch"]
    accum_steps      = saved["accum_steps"]
    vram_frac        = saved.get("vram_frac", 0.0)
    del saved    # free the large checkpoint dict from RAM

    logger.info(
        "Resumed from epoch %d (best_score=%.4f patience=%d)",
        start_epoch - 1, best_score, patience_counter,
    )
    return (start_epoch, best_score, patience_counter, history,
            global_step, micro_batch, accum_steps, vram_frac)


def _save_resume_state(
    resume_path: Path, epoch: int, model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    best_score: float, patience_counter: int, history: list,
    global_step: int, micro_batch: int, accum_steps: int, vram_frac: float,
) -> None:
    """Save a resume checkpoint at the end of each epoch.

    Why save every epoch (not just on improvement): if training crashes on
    an epoch where val Dice did NOT improve, we still want to resume from that
    epoch rather than from the best epoch. The best model weights are saved
    separately in best_model.pt.
    """
    torch.save({
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scaler_state":     scaler.state_dict() if scaler else None,
        "best_score":       best_score,
        "patience_counter": patience_counter,
        "history":          history,
        "global_step":      global_step,
        "micro_batch":      micro_batch,
        "accum_steps":      accum_steps,
        "vram_frac":        vram_frac,
    }, resume_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_pytorch(
    model: nn.Module,
    model_name: str,
    run_dir: Path,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict,
    device: torch.device,
    teacher_fn: Optional[Callable] = None,
    alpha: float = 0.75,
) -> dict:
    """Full training loop for a SegFormer model with optional distillation.

    Handles: batch-size probing, optimizer + schedule setup, resume from
    checkpoint, best-model tracking, early stopping, threshold sweep, and
    final val summary.

    Args:
        model:      SegformerWrapper (uninitialised weights for normal run).
        model_name: used for logging and saved JSON.
        run_dir:    all outputs (checkpoints, CSVs, logs) are written here.
        train_df:   training split DataFrame (from load_train_val_split).
        val_df:     validation split DataFrame.
        cfg:        common_train.yaml config dict.
        device:     GPU device.
        teacher_fn: callable from load_teacher(), or None for direct training.
        alpha:      GT loss weight (only used when teacher_fn is not None).

    Returns:
        Summary dict with model metrics and hyperparameters.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    amp          = cfg.get("amp", True)
    resume_path  = run_dir / "resume_checkpoint.pt"
    model        = model.to(device)

    # ── Batch size probe or resume ────────────────────────────────────────────
    if resume_path.exists():
        # Skip the probe — reuse the batch size found before the crash
        # (the probe modifies VRAM state and should not run mid-training)
        saved_tmp = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        micro_batch = saved_tmp["micro_batch"]
        accum_steps = saved_tmp["accum_steps"]
        vram_frac   = saved_tmp.get("vram_frac", 0.0)
        del saved_tmp
        logger.info("Resume found: micro_batch=%d accum_steps=%d", micro_batch, accum_steps)
        resuming = True
    else:
        micro_batch, accum_steps, vram_frac = find_optimal_micro_batch(
            model=model, img_h=cfg["img_h"], img_w=cfg["img_w"], device=device,
            effective_batch=cfg.get("effective_batch", 8),
            target_hi=cfg.get("vram_target_fraction", 0.75),
            amp=amp, max_probe=cfg.get("max_probe_batch", 32),
            teacher_fn=teacher_fn,
        )
        resuming = False

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = BruiseDataset(train_df, cfg["img_h"], cfg["img_w"], training=True)
    val_ds   = BruiseDataset(val_df,   cfg["img_h"], cfg["img_w"], training=False)
    dl_kw    = dict(num_workers=cfg.get("workers", 8), pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,  drop_last=True, **dl_kw)
    val_loader   = DataLoader(val_ds,   batch_size=micro_batch, shuffle=False, **dl_kw)

    # ── Optimizer, schedule, losses ───────────────────────────────────────────
    optimizer, peak_lrs = _build_optimizer_and_schedule(model, cfg)
    total_steps, warmup_steps, poly_power = _compute_schedule_params(
        len(train_loader), accum_steps, cfg)

    # GradScaler for AMP: scales loss to prevent fp16 underflow in backward pass
    scaler       = torch.amp.GradScaler("cuda") if amp else None
    sup_loss     = DiceBCELoss()
    dist_loss_fn = DistillSegLoss(alpha=alpha) if teacher_fn is not None else None

    # ── Save run config to disk ───────────────────────────────────────────────
    total_params, trainable_params = count_params(model)
    training_type = "distill" if teacher_fn is not None else "direct"
    _save_run_config(
        run_dir, model_name, training_type,
        alpha if teacher_fn is not None else None,
        micro_batch, accum_steps, vram_frac,
        warmup_steps, total_steps, poly_power,
        total_params, trainable_params, cfg,
    )

    # ── Initialise training state ──────────────────────────────────────────────
    best_score       = float("-inf")
    patience_counter = 0
    history: list    = []
    global_step      = 0
    start_epoch      = 1

    if resuming:
        (start_epoch, best_score, patience_counter, history,
         global_step, micro_batch, accum_steps, vram_frac) = _load_resume_state(
             resume_path, model, optimizer, scaler)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        train_loss, global_step = _train_one_epoch(
            model, train_loader, optimizer, sup_loss, dist_loss_fn, scaler,
            accum_steps, device, cfg["gradient_clip"], teacher_fn,
            total_steps, warmup_steps, peak_lrs, poly_power, global_step,
            desc=f"{model_name} {epoch}/{cfg['epochs']}",
        )
        _, val_summary = evaluate(model, val_loader, device, 0.50, amp)
        score          = val_summary["mean_dice"]

        # LR at this step for the history CSV
        current_lr = peak_lrs[0] * lr_multiplier(global_step, total_steps, warmup_steps, poly_power)
        row = {"epoch": epoch, "train_loss": round(train_loss, 6),
               "backbone_lr": current_lr, **val_summary}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        logger.info("epoch %3d | loss=%.4f | dice=%.4f | lr=%.2e",
                    epoch, train_loss, score, current_lr)

        # Save best model when val Dice improves
        if score > best_score:
            best_score, patience_counter = score, 0
            # Save only the state_dict (not the wrapper) so it can be loaded
            # independently of the SegformerWrapper class definition
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Resume checkpoint written every epoch — at most 1 epoch lost if crash
        _save_resume_state(resume_path, epoch, model, optimizer, scaler,
                           best_score, patience_counter, history, global_step,
                           micro_batch, accum_steps, vram_frac)

        if patience_counter >= cfg["patience"]:
            logger.info("Early stopping at epoch %d (patience=%d).", epoch, cfg["patience"])
            break

    # ── Post-training: load best model, threshold sweep, final summary ────────
    # Delete resume checkpoint — next run of this script will start fresh
    if resume_path.exists():
        resume_path.unlink()

    # Reload best checkpoint — not the last epoch weights, which may have overfit
    model.load_state_dict(
        torch.load(str(run_dir / "best_model.pt"), map_location=device, weights_only=True)
    )

    # Threshold sweep on val to find the best operating point
    thr_df, best_thr = _threshold_sweep(model, val_loader, device, cfg["thresholds"], amp)
    thr_df.to_csv(run_dir / "threshold_search.csv", index=False)
    _, val_final = evaluate(model, val_loader, device, best_thr, amp)

    summary = {
        "model_name":       model_name,
        "training_type":    training_type,
        "alpha":            alpha if teacher_fn is not None else None,
        "best_threshold":   best_thr,
        "micro_batch":      micro_batch,
        "accum_steps":      accum_steps,
        "effective_batch":  micro_batch * accum_steps,
        "vram_fraction":    round(vram_frac, 4),
        "backbone_lr":      cfg["backbone_lr"],
        "head_lr":          cfg["head_lr"],
        "warmup_steps":     warmup_steps,
        "total_steps":      total_steps,
        "poly_power":       poly_power,
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "best_val_mean_dice": best_score,
        **val_final,
    }
    pd.DataFrame([summary]).to_csv(run_dir / "val_summary.csv", index=False)
    logger.info("Training complete. best_val_dice=%.4f  best_threshold=%.2f",
                best_score, best_thr)
    return summary
