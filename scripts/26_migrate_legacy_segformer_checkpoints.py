#!/usr/bin/env python3
"""
scripts/26_migrate_legacy_segformer_checkpoints.py

One-time migration for SegFormer checkpoints trained under an OLDER
`transformers` version than the one currently installed in the `bruise_orc`
conda env.

Root cause (discovered 2026-07-07 while running scripts 25 and 12):
  HuggingFace refactored SegformerForSemanticSegmentation's internal module
  layout between transformers versions. `segformer_b2_teacher`'s and
  `segformer_b0_distilled`'s best_model.pt (both trained 2026-07-04) use the
  OLD layout; `build_segformer()` under the CURRENTLY installed transformers
  version builds the NEW layout. Loading the old checkpoint into the new
  skeleton raises RuntimeError (missing/unexpected keys) -- see the
  Missing/Unexpected key dumps captured while debugging scripts 25/12 for the
  exact before/after key lists this mapping is derived from.

Old layout -> New layout (confirmed against TWO independent checkpoints:
segformer_b0_distilled and segformer_b2_teacher, both showing the identical
rename pattern):
  segformer.encoder.patch_embeddings.{i}          -> segformer.stages.{i}.patch_embeddings
  segformer.encoder.layer_norm.{i}                 -> segformer.stages.{i}.layer_norm
  segformer.encoder.block.{i}.{j}.layer_norm_1     -> segformer.stages.{i}.blocks.{j}.layernorm_before
  segformer.encoder.block.{i}.{j}.layer_norm_2     -> segformer.stages.{i}.blocks.{j}.layernorm_after
  ...attention.self.query/key/value                -> ...attention.q_proj/k_proj/v_proj
  ...attention.self.sr                             -> ...attention.sequence_reduction.sequence_reduction
  ...attention.self.layer_norm (the post-sr norm)  -> ...attention.sequence_reduction.layer_norm
  ...attention.output.dense                        -> ...attention.o_proj
  ...mlp.dense1 / mlp.dense2                       -> ...mlp.fc1 / mlp.fc2
  ...mlp.dwconv.dwconv                             -> unchanged
  decode_head.linear_c.{i}                         -> decode_head.linear_projections.{i}
  decode_head.batch_norm / linear_fuse / classifier -> unchanged (already matched in both diffs)

WHY REWRITE THE CHECKPOINT FILE (not patch loading code in every script):
  pipeline.trainer.load_teacher() and every evaluation script (09, 10, 11,
  12, 17, 25) call the exact same SegformerWrapper(build_segformer(...)) +
  load_state_dict(...) pattern. Patching each call site individually would
  mean touching pipeline/trainer.py -- forbidden by this project's ground
  rule (see PHASE_3_4_5_PLAN.md: "existing files 00-12 and pipeline/ must
  never be modified"). Rewriting the .pt file's own keys fixes every caller
  at once without touching a single .py file.

SAFETY:
  - Defaults to --dry-run: prints the migration diff and verifies the
    remapped state_dict loads cleanly (strict=True, zero missing/unexpected
    keys) into a freshly-built model, WITHOUT writing anything.
  - --apply is required to actually write. The original file is always
    copied to <name>.pre_migration_backup.pt first (never overwritten
    in-place blind) so this is fully reversible.

Usage:
    # Preview only (no files touched)
    python scripts/26_migrate_legacy_segformer_checkpoints.py --paths configs/paths.yaml

    # Actually migrate the three known-affected Phase 1/2 checkpoints
    python scripts/26_migrate_legacy_segformer_checkpoints.py --paths configs/paths.yaml --apply

    # Migrate a specific checkpoint not in the default list
    python scripts/26_migrate_legacy_segformer_checkpoints.py --paths configs/paths.yaml \\
        --model-name segformer_b0_fairness_distill_approach_a --apply
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.io_utils import load_yaml, setup_logging, validate_paths
from pipeline.models import SegformerWrapper, build_segformer

logger = setup_logging()

# Known checkpoints trained 2026-07-04, before the transformers version
# changed on ORC -- confirmed affected by direct inspection (script 25/12
# debugging). Other models (segformer_b2_als_teacher, and anything script
# 12/16 produces from today onward) were/will be trained under the CURRENT
# transformers version and do not need migration.
DEFAULT_MODEL_NAMES = ["segformer_b2_teacher", "segformer_b0_direct", "segformer_b0_distilled"]

# Applied in order via re.sub; patterns are mutually exclusive (each matches
# a distinct literal suffix), so order does not matter for correctness.
_KEY_RULES: list[tuple[str, str]] = [
    (r"encoder\.patch_embeddings\.(\d+)\.", r"stages.\1.patch_embeddings."),
    (r"encoder\.layer_norm\.(\d+)\.", r"stages.\1.layer_norm."),
    (r"encoder\.block\.(\d+)\.(\d+)\.layer_norm_1\.", r"stages.\1.blocks.\2.layernorm_before."),
    (r"encoder\.block\.(\d+)\.(\d+)\.layer_norm_2\.", r"stages.\1.blocks.\2.layernorm_after."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.query\.", r"stages.\1.blocks.\2.attention.q_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.key\.", r"stages.\1.blocks.\2.attention.k_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.value\.", r"stages.\1.blocks.\2.attention.v_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.sr\.",
     r"stages.\1.blocks.\2.attention.sequence_reduction.sequence_reduction."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.self\.layer_norm\.",
     r"stages.\1.blocks.\2.attention.sequence_reduction.layer_norm."),
    (r"encoder\.block\.(\d+)\.(\d+)\.attention\.output\.dense\.", r"stages.\1.blocks.\2.attention.o_proj."),
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dense1\.", r"stages.\1.blocks.\2.mlp.fc1."),
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dense2\.", r"stages.\1.blocks.\2.mlp.fc2."),
    (r"encoder\.block\.(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.", r"stages.\1.blocks.\2.mlp.dwconv.dwconv."),
    (r"decode_head\.linear_c\.(\d+)\.", r"decode_head.linear_projections.\1."),
]


def _remap_key(key: str) -> str:
    for pattern, repl in _KEY_RULES:
        new_key = re.sub(pattern, repl, key)
        if new_key != key:
            return new_key
    return key    # unchanged: already-new-style key (batch_norm, linear_fuse, classifier, ...)


def _is_legacy_checkpoint(state_dict: dict) -> bool:
    """A checkpoint is legacy iff any key still contains 'encoder.block.' or
    'linear_c.' -- both substrings are exclusive to the old layout."""
    return any("encoder.block." in k or "linear_c." in k for k in state_dict)


def _migrate_one(model_name: str, run_dir: Path, pretrained: str, apply: bool) -> None:
    best_pt = run_dir / "best_model.pt"
    if not best_pt.exists():
        logger.warning("SKIP %s: no checkpoint at %s", model_name, best_pt)
        return

    old_sd = torch.load(str(best_pt), map_location="cpu", weights_only=True)
    if not _is_legacy_checkpoint(old_sd):
        logger.info("SKIP %s: already new-style keys, nothing to migrate.", model_name)
        return

    new_sd = {_remap_key(k): v for k, v in old_sd.items()}
    n_changed = sum(1 for k in old_sd if _remap_key(k) != k)
    logger.info("[%s] %d/%d keys remapped", model_name, n_changed, len(old_sd))

    # ── verify: freshly-built model must load the remapped state_dict with
    # strict=True and zero missing/unexpected keys before we trust it ──────
    model = SegformerWrapper(build_segformer(pretrained, num_labels=1))
    fresh_keys = set(model.state_dict().keys())
    new_keys = set(new_sd.keys())
    missing = sorted(fresh_keys - new_keys)
    unexpected = sorted(new_keys - fresh_keys)
    if missing or unexpected:
        logger.error("[%s] MIGRATION VERIFICATION FAILED -- missing=%s unexpected=%s",
                      model_name, missing, unexpected)
        raise RuntimeError(
            f"{model_name}: remapped state_dict does not match the current model "
            "architecture. Do NOT apply -- the key-rename rules need revising.")

    model.load_state_dict(new_sd, strict=True)    # will raise if any shape mismatches slipped through
    logger.info("[%s] verification passed: remapped state_dict loads cleanly (strict=True).", model_name)

    if not apply:
        logger.info("[%s] DRY RUN -- no files written. Re-run with --apply to write.", model_name)
        return

    backup_path = run_dir / "best_model.pre_migration_backup.pt"
    if backup_path.exists():
        logger.info("[%s] backup already exists at %s, not overwriting it.", model_name, backup_path)
    else:
        shutil.copy2(best_pt, backup_path)
        logger.info("[%s] original backed up -> %s", model_name, backup_path)

    torch.save(new_sd, best_pt)
    logger.info("[%s] MIGRATED -> %s now uses new-style keys.", model_name, best_pt)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate legacy (pre-transformers-refactor) SegFormer checkpoint keys to the current layout")
    ap.add_argument("--paths", default="configs/paths.yaml")
    ap.add_argument("--model-name", action="append", default=None,
                     help="Specific model_name(s) under project_root to migrate "
                          "(repeatable). Defaults to the 3 known-affected Phase 1/2 models.")
    ap.add_argument("--apply", action="store_true",
                     help="Actually write the migrated checkpoint (default: dry-run only).")
    args = ap.parse_args()

    paths = load_yaml(args.paths)
    validate_paths(paths)
    project_root = Path(paths["project_root"])
    model_names = args.model_name if args.model_name else DEFAULT_MODEL_NAMES

    for model_name in model_names:
        # B2 models use the B2 pretrained skeleton, everything else B0 --
        # matches every other script's pretrained_key convention (script 17)
        pretrained_key = "segformer_b2_pretrained" if "_b2_" in model_name else "segformer_b0_pretrained"
        _migrate_one(model_name, project_root / model_name, paths[pretrained_key], args.apply)

    if not args.apply:
        logger.info("Dry run complete. Re-run with --apply once the verification above looks correct.")


if __name__ == "__main__":
    main()
