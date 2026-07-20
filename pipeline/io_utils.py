"""
pipeline/io_utils.py

Foundation utilities: YAML loading, directory creation, JSON persistence,
structured logging, and pre-flight config validation.

Why this module exists: Every one of the 12+ pipeline scripts needs to load
configs, create output directories, write JSON summaries, and log progress.
Centralising those operations here means:
  (a) All scripts use identical error semantics — no script silently swallows
      a missing config or a bad path.
  (b) Changing the log format or validation rules affects all scripts at once.
  (c) New scripts added for ALS experiments automatically inherit all checks.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# YAML / JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file and return its contents as a dict.

    Why YAML: configs use YAML (not JSON) because YAML supports inline
    comments that explain *why* each hyperparameter was chosen. Those
    comments survive round-trips and are visible when auditing the run.

    Why raise on missing file: returning an empty dict for a missing config
    would let every downstream cfg.get(...) silently use wrong defaults —
    the error would surface hours into training with a cryptic KeyError.
    Raising immediately gives a clear message before any GPU work starts.

    Args:
        path: Absolute or relative path to a .yaml file.

    Returns:
        Dict of parsed YAML content. Empty files coerce to {}.

    Raises:
        FileNotFoundError: if the file does not exist on disk. We raise
            explicitly (not inside open()) so the error message is clear.
    """
    path = Path(path)
    if not path.exists():
        # Raise with a helpful message rather than a bare FileNotFoundError
        # from open() — the user needs to know which config is missing and
        # where to look for it.
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Run scripts from the project root so that configs/ is reachable, "
            "or pass --paths / --common with absolute paths.")
    with path.open() as f:
        # yaml.safe_load returns None for empty files — coerce to {} so
        # all callers can unconditionally use cfg.get() without AttributeError
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and all parents) if it does not already exist.

    Why centralise mkdir: every training script creates run-specific output
    directories. Putting mkdir here means parents=True and exist_ok=True are
    always set — forgetting either flag is a common source of crashes on the
    first run of a new experiment.

    Args:
        path: Target directory (str or Path).

    Returns:
        Path object to the (now-existing) directory — lets callers chain:
            csv_path = ensure_dir(run_dir / "eval") / "results.csv"
    """
    path = Path(path)
    # parents=True: creates all intermediate directories (e.g. a/b/c) in one call
    # exist_ok=True: makes the call idempotent — safe to call multiple times
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, payload: dict) -> None:
    """Serialise a dict to a pretty-printed JSON file, creating parents.

    Why: Training scripts save run_config.json (hyperparameters) and
    temperature.json (calibration result) so that downstream evaluation and
    benchmark scripts can read back exact parameters without re-parsing CLI
    args or re-loading the model. A plain text JSON is human-auditable and
    version-controllable.

    Args:
        path:    Target .json file path (created if it doesn't exist).
        payload: Dict to serialise. Must contain only JSON-serialisable types.
    """
    path = Path(path)
    # Parent may not exist if this is the first write in a new run directory
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2: human-readable layout — easy to inspect in a text editor
    path.write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Structured logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    run_dir: Path | str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return the shared pipeline logger.

    Writes to stdout AND (if run_dir is given) to <run_dir>/run.log.

    Why two sinks:
      - stdout: lets the user watch per-epoch progress live in the terminal.
      - run.log: persists after the terminal session ends (GPU sessions often
        disconnect mid-run). The supervisor needs to inspect the log to verify
        that specific decisions (e.g. micro_batch chosen, temperature loaded)
        happened correctly — a plain print() statement is gone when the
        session ends.

    Log format: [YYYY-MM-DD HH:MM:SS] [LEVEL] message
    The timestamp makes it easy to correlate log lines with GPU utilisation
    metrics from nvidia-smi or wandb.

    Args:
        run_dir: If given, appends to <run_dir>/run.log (mode='a' preserves
                 the log across resumed runs). Pass None for stdout only.
        level:   Logging verbosity; INFO by default.

    Returns:
        Logger named "pipeline". All pipeline modules that call
        logging.getLogger("pipeline") receive this same configured instance.
    """
    # Consistent format across every module — timestamp + level + message
    fmt     = "[%(asctime)s] [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    logger = logging.getLogger("pipeline")
    # Clear handlers from any previous call in the same process — without
    # this, calling setup_logging() twice (e.g. in tests or notebooks)
    # duplicates every log line
    logger.handlers.clear()
    logger.setLevel(level)

    # Handler 1: stdout — always present so progress is visible in terminal
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if run_dir is not None:
        # Handler 2: file — appended so resumed runs keep full history
        ensure_dir(run_dir)
        # mode='a': append, not overwrite — resuming a crashed run keeps
        # the original log context, making it possible to see the full picture
        file_handler = logging.FileHandler(Path(run_dir) / "run.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Config validation — fail fast before any GPU work begins
# ─────────────────────────────────────────────────────────────────────────────

# Keys that MUST be present in paths.yaml.
# Defined at module level (not inside the function) so any script can import
# this list for documentation or testing without calling validate_paths().
_REQUIRED_PATHS_KEYS: list[str] = [
    "dataset_root",            # Labelbox export root (00_build_split.py input)
    "train_manifest",          # subject-level split CSV (output of 00)
    "fixed_test_manifest",     # held-out test manifest (never touched during training)
    "project_root",            # parent dir for all training outputs
    "pretrained_weights_root", # single folder containing ALL pretrained weights
    "segformer_b2_pretrained", # HuggingFace MiT-B2 checkpoint directory
    "segformer_b0_pretrained", # HuggingFace MiT-B0 checkpoint directory
    "yolo_sem_weights",        # YOLO26n-sem .pt file
]

# Keys that MUST be present in common_train.yaml.
_REQUIRED_CFG_KEYS: list[str] = [
    "seed",                   # global RNG seed for reproducibility
    "img_h", "img_w",         # model input resolution
    "epochs", "patience",     # training loop termination criteria
    "workers",                # DataLoader num_workers
    "backbone_lr", "head_lr", # layer-wise learning rates (SegFormer paper recipe)
    "betas",                  # AdamW momentum parameters
    "weight_decay",           # L2 regularisation coefficient
    "warmup_fraction",        # fraction of total steps used for LR warm-up
    "poly_power",             # polynomial LR decay exponent
    "effective_batch",        # logical batch size (micro_batch × accum_steps)
    "vram_target_fraction",   # VRAM occupancy target for batch probe
    "max_probe_batch",        # maximum micro-batch to probe
    "amp",                    # whether to use automatic mixed precision
    "gradient_clip",          # max gradient norm for clipping
    "thresholds",             # list of candidate thresholds for val sweep
    "pseudo_threshold",       # threshold used to binarise teacher soft labels
    "device",                 # default device string ("cuda" or "cpu")
]


def validate_paths(paths: dict) -> None:
    """Pre-flight check: assert all required paths.yaml keys exist and that
    critical INPUT files/directories are accessible on disk.

    Why: A typo in paths.yaml (e.g. a missing character in a weight path)
    causes a crash deep inside training — after hours of GPU use. Running
    this check at startup costs < 1 second and prevents wasted compute. This
    embodies the "fail loudly, fail early" principle: every required input is
    verified before any work begins.

    Inputs verified on disk (must exist BEFORE training):
      - train_manifest          — needed by every training script
      - fixed_test_manifest     — needed by every evaluation script
      - pretrained_weights_root — parent of all weight files/dirs
      - segformer_b2_pretrained — teacher/B2 initialisation
      - segformer_b0_pretrained — student/B0 initialisation
      - yolo_sem_weights        — YOLO student starting point

    NOT verified (will be created by scripts):
      - project_root            — mkdir'd by each training script
      - dataset_root            — only accessed by 00_build_split.py, which
                                  validates it independently

    Args:
        paths: Dict loaded from paths.yaml via load_yaml().

    Raises:
        KeyError:     if any required key is absent. Raises with a complete
                      list of all missing keys so the user fixes them in one pass.
        RuntimeError: if any required input path does not exist on disk.
                      Lists all missing paths at once, not just the first.
    """
    # Collect all missing keys at once — fix one-and-re-run cycles are
    # frustrating when there are multiple typos in paths.yaml
    missing_keys = [k for k in _REQUIRED_PATHS_KEYS if k not in paths]
    if missing_keys:
        raise KeyError(
            f"paths.yaml is missing required keys: {missing_keys}\n"
            "Add the missing entries to configs/paths.yaml and re-run.")

    # Check input file/dir existence — only inputs, not outputs
    # (outputs are created by the scripts themselves)
    must_exist: dict[str, str] = {
        "train_manifest":          paths["train_manifest"],
        "fixed_test_manifest":     paths["fixed_test_manifest"],
        "pretrained_weights_root": paths["pretrained_weights_root"],
        "segformer_b2_pretrained": paths["segformer_b2_pretrained"],
        "segformer_b0_pretrained": paths["segformer_b0_pretrained"],
        "yolo_sem_weights":        paths["yolo_sem_weights"],
    }
    errors: list[str] = []
    for label, p in must_exist.items():
        if not Path(p).exists():
            # Accumulate all errors before raising — one complete message
            # is better than six separate crashes
            errors.append(f"  [{label}] {p}")

    if errors:
        raise RuntimeError(
            "The following required paths do not exist on disk.\n"
            "Update configs/paths.yaml or create the missing files:\n"
            + "\n".join(errors))


def validate_cfg(cfg: dict) -> None:
    """Pre-flight check: assert all required common_train.yaml keys exist
    with sensible numeric values.

    Why: A missing 'amp' key silently disables mixed precision (2–3× slower
    training with no warning). A typo in 'patience' could mean the model
    trains for 100 epochs instead of stopping at 15. We catch all problems
    before the training loop starts.

    Args:
        cfg: Dict loaded from common_train.yaml via load_yaml().

    Raises:
        KeyError:   if any required key is absent.
        ValueError: if any value is outside its valid range.
    """
    # Collect all missing keys at once
    missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
    if missing:
        raise KeyError(
            f"common_train.yaml is missing required keys: {missing}")

    # Numeric range checks — wrong values cause silent under-training or crashes
    errors: list[str] = []
    if not (1 <= cfg["epochs"] <= 10_000):
        # epochs=0 would skip training entirely; >10000 is almost certainly a typo
        errors.append(f"  epochs={cfg['epochs']} — expected in [1, 10000]")
    if not (1 <= cfg["patience"] <= cfg["epochs"]):
        # patience > epochs means early stopping never triggers — a silent bug
        errors.append(
            f"  patience={cfg['patience']} — expected in [1, epochs={cfg['epochs']}]")
    if not (0.0 < cfg["backbone_lr"] < 1.0):
        # backbone_lr outside (0,1) is almost certainly a unit error (e.g. 6 instead of 6e-5)
        errors.append(
            f"  backbone_lr={cfg['backbone_lr']} — expected in (0, 1)")
    if not (0.0 < cfg["head_lr"] < 1.0):
        errors.append(
            f"  head_lr={cfg['head_lr']} — expected in (0, 1)")
    if not isinstance(cfg["thresholds"], list) or len(cfg["thresholds"]) == 0:
        # An empty threshold list means the sweep produces no candidates —
        # the training script would crash when trying to pick the best threshold
        errors.append("  thresholds — must be a non-empty list of floats in (0, 1)")

    if errors:
        raise ValueError(
            "common_train.yaml has invalid values:\n" + "\n".join(errors))
