import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_optimal_micro_batch(
    model: nn.Module,
    img_h: int,
    img_w: int,
    device: torch.device,
    effective_batch: int = 8,
    target_hi: float = 0.75,
    amp: bool = True,
    max_probe: int = 32,
    teacher_fn: Optional[Callable] = None,
) -> tuple[int, int, float]:
    """Probe increasing micro-batch sizes with a real forward+backward+optimizer.step(),
    measuring PEAK reserved memory via max_memory_reserved() (reset before each probe) --
    not memory_reserved(), which only reads *current* reserved memory after autograd has
    already freed forward-pass activations and under-reports true peak usage.

    If teacher_fn is given (distillation runs), it's also run forward on every
    probe batch -- real training calls teacher_fn(x) every step too, and its
    memory footprint must be included in the probe or the chosen batch size
    will OOM the moment real (student + teacher) training starts.

    Returns (micro_batch, accum_steps, vram_fraction_at_chosen_batch).
    """
    if not torch.cuda.is_available():
        micro_batch = min(effective_batch, max_probe)
        return micro_batch, max(1, effective_batch // micro_batch), 0.0

    total_vram = torch.cuda.get_device_properties(device).total_memory
    scaler = torch.amp.GradScaler("cuda") if amp else None
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)  # throwaway, just to exercise .step()

    chosen_batch = 1
    chosen_frac = 0.0
    batch = 1
    while batch <= max_probe:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.randn(batch, 3, img_h, img_w, device=device)
            y = torch.randint(0, 2, (batch, 1, img_h, img_w), device=device).float()

            model.train()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                if teacher_fn is not None:
                    _ = teacher_fn(x)  # same forward call real training makes every step
                logits = model(x)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            peak = torch.cuda.max_memory_reserved(device)
            frac = peak / total_vram
            del x, y, logits, loss
            torch.cuda.empty_cache()

            if frac > target_hi:
                break
            chosen_batch, chosen_frac = batch, frac
            batch *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

    micro_batch = max(1, chosen_batch)
    accum_steps = max(1, effective_batch // micro_batch)
    logger.info(
        "Batch probe: micro_batch=%d accum_steps=%d vram_fraction=%.3f",
        micro_batch, accum_steps, chosen_frac,
    )
    return micro_batch, accum_steps, chosen_frac
