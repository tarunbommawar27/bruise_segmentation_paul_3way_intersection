"""
pipeline/benchmark_640.py

Controlled 640x640 deployment-latency benchmark. Works on CUDA (ORC) and on a
CPU-only machine (this laptop) -- the exact thing scripts/29 can't do.

WHY THIS FILE EXISTS
---------------------
scripts/29_benchmark_inference_all_models.py raises immediately if
torch.cuda.is_available() is False -- it was written to produce GPU FPS
numbers on the ORC cluster and treats CPU as meaningless. That's a reasonable
stance for "final" FPS numbers, but it also means the benchmark code cannot
be exercised at all while ORC is down. This module makes every timing helper
device-agnostic (a no-op sync off CUDA, NaN peak-memory off CUDA, fp16
skipped rather than attempted off CUDA) so the exact same code path runs on
this laptop right now as a correctness check, and on ORC's GPU later for
real FPS numbers -- same script, same code, just a different --device.

Modeled on the benchmark_640_integrated/ reference package you pointed me at
(a benchmark module written for a different, more modular sibling codebase,
"bruise_repro_v3") -- its adapter-class design, dataclasses, and
device-agnostic timing are ported here, rewired to this repo's actual
pipeline/ modules, configs/paths.yaml layout, and the 8 models that actually
exist under this project's project_root.

WHY YOLO IS PREPROCESSED WITH IMAGENET NORMALIZATION, NOT A 0-1 UNIT SCALE
------------------------------------------------------------------------------
scripts/27_evaluate_test_640_models_01_to_11.py (the script that produced
this project's actual reported YOLO test-set Dice numbers) and
pipeline/yolo_threshold_temp.py::run_threshold_search (which produced every
threshold_search.csv -- i.e. the threshold + temperature this module reads
back) both feed YOLO's raw nn.Module through pipeline.data.BruiseDataset,
which applies ImageNet mean/std normalization; script 27 even comments on
this explicitly ("matching BruiseDataset's GT"). By contrast,
scripts/29_..._all_models.py's make_yolo_input() scales YOLO input to plain
[0, 1] with no normalization -- a different input distribution than the one
the loaded threshold/temperature were actually calibrated against.
This module uses pipeline.data.get_augmentation() -- the literal function
BruiseDataset uses -- for BOTH SegFormer and YOLO, so the operating point
(threshold, temperature) loaded from threshold_search.csv is always applied
to the same input distribution it was calibrated on.

CONTROLLED TIMED SCOPE
------------------------
benchmark_adapter() has two timing modes, toggled by commenting/uncommenting
one block inside its per-image loop (see the "TIMING MODE TOGGLE" comment
there). Only one mode is active at a time.

ACTIVE MODE ("raw forward" -- excludes resize/normalize):

    RGB image already decoded in RAM
    -> [NOT TIMED] resize to 640x640 + normalize (pipeline.data.get_augmentation)
    -> [NOT TIMED] CPU-to-device transfer
    -> [TIMED] model forward
    -> [TIMED] resize logits to 640x640 if the model didn't already (SegFormer's
       wrapper does this itself; yolo_raw_class_logits does it for YOLO)
    -> [TIMED] temperature scaling + sigmoid + threshold
    -> [TIMED] copy final binary uint8 mask to CPU

This isolates model + postprocessing speed from preprocessing cost -- useful
for comparing architectures independent of how expensive each one's input
pipeline happens to be.

OTHER MODE ("full pipeline" -- includes resize/normalize, commented out by
default): the same steps as above, but resize/normalize/H2D-transfer are
INSIDE the timed section too. This is the deployment-realistic number: a live
camera photo is never pre-resized to 640x640 ahead of time, so a real system
pays that resize cost on every single frame, not just once -- see the
"TIMING MODE TOGGLE" comment in benchmark_adapter() for exactly what to
comment/uncomment to switch to this mode.

Excluded from timing in BOTH modes: disk image decode, loading the
ground-truth mask, Dice computation, resizing the mask back to native camera
resolution, model loading, and warmup iterations -- a real camera frame is
already decoded in RAM before your production pipeline would see it, and you
only load/warm up a model once, not once per frame.
"""
from __future__ import annotations

import gc
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from pipeline.data import get_augmentation, load_fixed_test
from pipeline.models import count_params, load_segformer_model
from pipeline.yolo_threshold_temp import bruise_prob_from_logits, load_yolo_model, yolo_raw_class_logits

IMAGE_SIZE = 640


@dataclass(frozen=True)
class SourceImage:
    """One benchmark image reference (stem + path only, no decoded pixels).

    Deliberately NOT preloaded into RAM: these are full camera-resolution
    images (4022x6024x3 = ~72MB each as a decoded RGB array). Holding all
    ~185 of them in memory at once needs ~13GB, which OOM'd on a laptop
    (cv2.imread raising "Failed to allocate 72685584 bytes" -- exactly one
    image's worth -- partway through preloading). Each image is instead
    decoded on demand, once per use, immediately before it's timed (see
    read_source_image() and benchmark_adapter() below) -- this keeps peak
    memory to about one image at a time while still keeping disk decode
    out of the timed section, just as before.
    """
    stem: str
    path: str


def read_source_image(source: SourceImage) -> np.ndarray:
    """Decode one benchmark image from disk. Called right before a timed
    inference, never cached across calls -- see SourceImage's docstring for
    why (avoiding an OOM from holding the whole fixed-test set in RAM)."""
    img_bgr = cv2.imread(source.path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {source.path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


@dataclass
class BenchmarkSummary:
    """One row of the benchmark's output CSV -- every field a downstream
    comparison table might want, named so the CSV is self-explanatory
    without reading this file."""
    model: str
    family: str
    checkpoint: str
    device: str
    precision: str
    threshold: float
    temperature: float
    parameters_total: int
    parameters_millions: float
    n_images: int
    repeats: int
    timed_images: int
    warmup_iterations: int
    batch_size: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    std_latency_ms: float
    fps_from_mean: float
    fps_from_median: float
    peak_gpu_memory_mb: float
    timing_scope: str


class Fixed640Adapter:
    """Common interface for the controlled 640x640 benchmark.

    infer_full() takes an RGB image already decoded into RAM (a real camera
    frame would already be in RAM too) and returns a CPU uint8 binary mask
    of shape [640, 640]. Subclasses implement the one line that actually
    differs between architectures: how raw model output becomes a
    per-pixel bruise logit.
    """

    family = "unknown"

    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        threshold: float,
        temperature: float,
        checkpoint: Path,
        device: torch.device,
        precision: str,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.temperature = temperature
        self.checkpoint = checkpoint
        self.device = device
        self.precision = precision
        # fp16 is only a meaningful *speed* benchmark on a CUDA device in
        # this codebase (no fp16 CPU kernels to measure) -- see
        # benchmark_adapter()'s guard, which refuses fp16 off-CUDA outright
        # rather than silently running fp32 timings under an "fp16" label.
        self.use_half = precision == "fp16" and device.type == "cuda"
        # Real half-precision weights, not torch.autocast. autocast recasts
        # fp32 weights to fp16 on every op and every call (no cache persists
        # across separate `with autocast(...)` blocks, and this class opens
        # a fresh one per image) -- pure overhead with no reuse, which is
        # why the original autocast-based fp16 numbers were SLOWER than
        # fp32. Casting the weights once here means the timed forward pass
        # actually runs in fp16 with no per-call cast tax.
        self.model = model.half() if self.use_half else model
        # training=False: no flips/color-jitter, just the deterministic
        # resize + normalize + to-tensor every eval/threshold-search/test
        # script in this repo already uses for inference.
        self.transform = get_augmentation(training=False, img_h=IMAGE_SIZE, img_w=IMAGE_SIZE)

    @property
    def parameter_count(self) -> int:
        return count_params(self.model)[0]

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Resize + normalize exactly like BruiseDataset's eval-time
        transform, add the batch dimension, move to device, and cast to the
        model's actual dtype (fp16 weights need fp16 input, not fp32 input
        implicitly upcast op-by-op like autocast used to do)."""
        aug = self.transform(image=image_rgb)
        dtype = torch.float16 if self.use_half else torch.float32
        tensor = aug["image"].to(dtype=dtype).unsqueeze(0)
        return tensor.to(self.device, non_blocking=False)

    def _forward_and_threshold(self, tensor: torch.Tensor) -> np.ndarray:
        """Everything infer_full() does AFTER preprocessing: forward pass +
        temperature/sigmoid + threshold. Split out from infer_full() as its
        own method specifically so benchmark_adapter() can time this part
        alone, excluding _preprocess()'s resize/normalize -- see that
        function for the toggle between the two timing modes.

        tensor may hold a batch of >1 images (see benchmark_adapter()'s
        batch_size) -- returns one uint8 mask per input image, shape
        [B, 640, 640], never squeezed here."""
        raise NotImplementedError

    def infer_full(self, image_rgb: np.ndarray) -> np.ndarray:
        """Full pipeline: preprocess (resize+normalize) then forward+threshold
        for a single image. Used by warmup (timing doesn't matter there) and
        by anyone who just wants one mask without caring about the timing
        split or about batching."""
        tensor = self._preprocess(image_rgb)
        return self._forward_and_threshold(tensor)[0]


class SegFormer640Adapter(Fixed640Adapter):
    family = "segformer"

    @torch.inference_mode()
    def _forward_and_threshold(self, tensor: torch.Tensor) -> np.ndarray:
        logits = self.model(tensor)  # SegformerWrapper already upsamples to input res (640x640)
        probability = torch.sigmoid(logits[:, 0].float() / self.temperature)
        mask = probability >= self.threshold
        return mask.to(dtype=torch.uint8, device="cpu").numpy()


class YOLOSemantic640Adapter(Fixed640Adapter):
    """Uses the raw YOLO nn.Module + yolo_raw_class_logits/bruise_prob_from_logits
    (pipeline/yolo_threshold_temp.py), never Ultralytics' own .predict()
    postprocessing -- see that module's docstring for why: .predict() bakes
    in an argmax with no threshold/temperature control."""

    family = "yolo_semantic"

    @torch.inference_mode()
    def _forward_and_threshold(self, tensor: torch.Tensor) -> np.ndarray:
        class_logits = yolo_raw_class_logits(self.model, tensor, out_hw=(IMAGE_SIZE, IMAGE_SIZE))
        probability = bruise_prob_from_logits(class_logits.float(), self.temperature)
        mask = probability >= self.threshold
        return mask.to(dtype=torch.uint8, device="cpu").numpy()


def build_adapter(
    spec: dict[str, Any],
    paths: dict,
    device: torch.device,
    precision: str,
) -> Fixed640Adapter:
    """Build one adapter from a configs/benchmark_640_models.yaml entry.

    Deliberately reuses load_segformer_model()/load_yolo_model()
    (pipeline/models.py, pipeline/yolo_threshold_temp.py) -- the same
    checkpoint + threshold_search.csv resolution logic
    scripts/29_..._all_models.py uses -- so "which checkpoint" and "which
    threshold" can never drift between the two benchmark scripts.
    """
    name = str(spec["name"])
    family = str(spec["family"]).lower()
    run_name = str(spec["run_name"])

    if family == "segformer":
        pretrained_key = str(spec["pretrained_key"])
        model, threshold, checkpoint = load_segformer_model(run_name, pretrained_key, paths, device)
        model = model.to(device)
        return SegFormer640Adapter(
            name=name, model=model, threshold=threshold, temperature=1.0,
            checkpoint=checkpoint, device=device, precision=precision,
        )

    if family in {"yolo", "yolo_semantic"}:
        model, threshold, temperature, checkpoint = load_yolo_model(run_name, paths, device)
        return YOLOSemantic640Adapter(
            name=name, model=model, threshold=threshold, temperature=temperature,
            checkpoint=checkpoint, device=device, precision=precision,
        )

    raise ValueError(f"Unsupported model family: {family!r} (expected 'segformer' or 'yolo_semantic')")


def load_fixed_test_images(paths: dict, max_images: int | None) -> list[SourceImage]:
    """List the fixed test-set images to benchmark (stems + paths only --
    see SourceImage's docstring for why the pixels themselves are not
    decoded here)."""
    frame = load_fixed_test(paths["fixed_test_manifest"])
    if max_images is not None and max_images > 0:
        frame = frame.head(max_images)

    images = [SourceImage(stem=str(row.stem), path=str(row.image_path))
              for row in frame.itertuples(index=False)]

    if not images:
        raise RuntimeError("The fixed test manifest produced zero benchmark images.")
    return images


def _synchronize(device: torch.device) -> None:
    """No-op off CUDA -- there is no async queue to drain on CPU, so timing
    a CPU-only run doesn't need (and can't use) torch.cuda.synchronize()."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(values_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No timed values were collected.")
    mean_ms = float(statistics.fmean(values_ms))
    median_ms = float(statistics.median(values_ms))
    return {
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "p90_latency_ms": float(np.quantile(values, 0.90)),
        "p95_latency_ms": float(np.quantile(values, 0.95)),
        "std_latency_ms": float(np.std(values, ddof=0)),
        "fps_from_mean": float(1000.0 / mean_ms),
        "fps_from_median": float(1000.0 / median_ms),
    }


def benchmark_adapter(
    adapter: Fixed640Adapter,
    images: Sequence[SourceImage],
    *,
    warmup: int,
    repeats: int,
    batch_size: int = 1,
) -> tuple[BenchmarkSummary, pd.DataFrame]:
    if adapter.precision == "fp16" and adapter.device.type != "cuda":
        # Refuse outright rather than silently timing fp32 under an "fp16"
        # label -- a mislabeled number is worse than a missing one.
        raise ValueError("fp16 benchmarking is only supported on a CUDA device.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    for index in range(warmup):
        warmup_rgb = read_source_image(images[index % len(images)])
        mask = adapter.infer_full(warmup_rgb)
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.dtype != np.uint8:
            raise RuntimeError(
                f"{adapter.name} returned shape={mask.shape}, dtype={mask.dtype}; "
                f"expected uint8 [{IMAGE_SIZE},{IMAGE_SIZE}]."
            )
    _synchronize(adapter.device)

    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)

    # Chunked into batch_size-sized groups. batch_size=1 (the default)
    # reproduces the original one-image-at-a-time loop exactly, unchanged --
    # see Fixed640Adapter's docstring for why single-frame latency is the
    # deployment-realistic default. batch_size>1 measures the amortized
    # per-image cost of a batched forward pass instead (each image's
    # "latency" below is that batch's elapsed time / images in the batch) --
    # a different, but meaningful, number: fp16/Tensor Cores only pay off
    # once there's enough work per kernel launch, which batch=1 often can't
    # provide (see benchmark output's timing_scope note).
    batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for repeat_index in range(repeats):
        image_index = 0
        for batch_sources in batches:
            # Decoded fresh each iteration (not cached) -- see SourceImage's
            # docstring. Disk decode is always excluded from the timed
            # measurement -- a real camera frame is already decoded in RAM.
            image_rgbs = [read_source_image(source) for source in batch_sources]

            # ============================================================
            # TIMING MODE TOGGLE -- exactly one of the two blocks below
            # should be active at a time.
            #
            # ACTIVE NOW: "raw forward" -- resize/normalize (_preprocess)
            # happens BEFORE the timer starts, so mean_latency_ms/fps_from_mean
            # below measure ONLY forward pass + temperature/sigmoid/threshold,
            # isolating model+postprocessing speed from preprocessing cost.
            tensors = [adapter._preprocess(rgb) for rgb in image_rgbs]
            batched_tensor = torch.cat(tensors, dim=0)
            _synchronize(adapter.device)
            start = time.perf_counter()
            masks = adapter._forward_and_threshold(batched_tensor)
            _synchronize(adapter.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # COMMENTED OUT: "full pipeline" -- resize/normalize happens
            # INSIDE the timed section, matching the original deployment-
            # realistic design (a live camera photo is never pre-resized,
            # so a real system pays this cost on every frame). To switch
            # back to this mode: comment out the active lines above (from
            # "tensors = [adapter._preprocess(rgb) ..." down to
            # "elapsed_ms = ...") and uncomment the lines below. Only
            # meaningful at batch_size=1 -- infer_full() returns one mask.
            #
            # _synchronize(adapter.device)
            # start = time.perf_counter()
            # masks = adapter.infer_full(image_rgbs[0])[None]
            # _synchronize(adapter.device)
            # elapsed_ms = (time.perf_counter() - start) * 1000.0
            # ============================================================

            per_image_ms = elapsed_ms / len(batch_sources)
            for offset, source in enumerate(batch_sources):
                mask = masks[offset]
                if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.dtype != np.uint8:
                    raise RuntimeError(
                        f"{adapter.name} returned shape={mask.shape}, dtype={mask.dtype}; "
                        f"expected uint8 [{IMAGE_SIZE},{IMAGE_SIZE}]."
                    )
                latencies_ms.append(per_image_ms)
                records.append({
                    "model": adapter.name,
                    "family": adapter.family,
                    "precision": adapter.precision,
                    "repeat": repeat_index,
                    "image_index": image_index,
                    "stem": source.stem,
                    "image_path": source.path,
                    "latency_ms": per_image_ms,
                })
                image_index += 1

    stats = _summarize(latencies_ms)
    peak_memory = (
        float(torch.cuda.max_memory_allocated(adapter.device) / (1024 ** 2))
        if adapter.device.type == "cuda"
        else math.nan
    )
    summary = BenchmarkSummary(
        model=adapter.name,
        family=adapter.family,
        checkpoint=str(adapter.checkpoint),
        device=str(adapter.device),
        precision=adapter.precision,
        threshold=adapter.threshold,
        temperature=adapter.temperature,
        parameters_total=adapter.parameter_count,
        parameters_millions=adapter.parameter_count / 1_000_000.0,
        n_images=len(images),
        repeats=repeats,
        timed_images=len(latencies_ms),
        warmup_iterations=warmup,
        batch_size=batch_size,
        peak_gpu_memory_mb=peak_memory,
        # NOTE: keep this string in sync with whichever block is active in
        # the "TIMING MODE TOGGLE" section above -- it is not derived
        # automatically from which lines are commented out.
        timing_scope=(
            "resize+normalize+H2D NOT timed -> [TIMED] forward -> 640 logits -> "
            "temperature/sigmoid/threshold -> CPU uint8 mask (raw-forward mode; "
            "see TIMING MODE TOGGLE comment to switch to full-pipeline mode)"
            + (
                f"; batch_size={batch_size} -- latency/FPS below are per-image "
                "AMORTIZED across the batch (batch elapsed / batch_size), not "
                "isolated single-frame latency"
                if batch_size > 1 else ""
            )
        ),
        **stats,
    )
    return summary, pd.DataFrame.from_records(records)


def run_benchmark_640(
    *,
    paths_config: str | Path,
    common_config: str | Path,
    models_config: str | Path,
    output_dir: str | Path,
    device_name: str,
    precisions: Iterable[str],
    max_images: int | None,
    warmup: int,
    repeats: int,
    batch_size: int = 1,
) -> pd.DataFrame:
    from pipeline.io_utils import load_yaml, validate_cfg, validate_paths

    paths = load_yaml(paths_config)
    cfg = load_yaml(common_config)
    validate_paths(paths)
    validate_cfg(cfg)

    # This module is specifically the "640 protocol" benchmark -- fail loudly
    # if common_train.yaml ever stops matching that, instead of silently
    # benchmarking at the wrong resolution.
    if cfg["img_h"] != IMAGE_SIZE or cfg["img_w"] != IMAGE_SIZE:
        raise ValueError(
            f"configs/common_train.yaml has img_h={cfg['img_h']}, img_w={cfg['img_w']}; "
            f"pipeline/benchmark_640.py is hardcoded to {IMAGE_SIZE}x{IMAGE_SIZE}."
        )

    models_path = Path(models_config).expanduser().resolve()
    payload = yaml.safe_load(models_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Benchmark models YAML must contain a top-level 'models' list.")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    if device.type == "cuda":
        # Lets cuDNN benchmark multiple conv algorithms on the first call per
        # (shape, dtype) and cache the fastest -- off by default in PyTorch.
        # Without this, fp16 convs in particular can silently fall back to a
        # slower, non-autotuned algorithm even after warmup. Safe here since
        # every input this run ever sees is a fixed 640x640 shape (no
        # variable input sizes that would otherwise thrash the cache).
        torch.backends.cudnn.benchmark = True

    images = load_fixed_test_images(paths, max_images=max_images)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    per_image_frames: list[pd.DataFrame] = []

    print(f"Project root: {paths['project_root']}")
    print(f"Fixed test images preloaded: {len(images)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("No CUDA device -- timings below are a CPU correctness/latency "
              "reference only, not comparable to ORC's GPU FPS numbers.")

    for precision in precisions:
        precision = str(precision).lower()
        if precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision: {precision}")
        if precision == "fp16" and device.type != "cuda":
            print(f"Skipping fp16 ({device.type} has no fp16 speed benefit to measure here).")
            continue

        for spec in payload["models"]:
            if not bool(spec.get("enabled", True)):
                continue
            print(f"[{precision}] Loading {spec['name']}...")

            # Not every model in the registry is guaranteed to load on this
            # particular machine -- e.g. this laptop's download is missing
            # the fairness/ALS-distill run folders entirely, and separately
            # a too-new `transformers` install can fail to load an older
            # SegFormer checkpoint's state_dict (internal parameter names
            # were renamed upstream between library versions -- see
            # README/troubleshooting note). Catching broad Exception here
            # (not just FileNotFoundError) matches
            # scripts/29_..._all_models.py's own SKIP behavior exactly, so
            # one bad model never aborts the whole benchmark run.
            try:
                adapter = build_adapter(spec, paths, device, precision)
            except Exception as e:
                print(f"  SKIP {spec['name']}: {e}")
                continue

            try:
                summary, per_image = benchmark_adapter(
                    adapter, images, warmup=warmup, repeats=repeats, batch_size=batch_size)
                summaries.append(asdict(summary))
                per_image_frames.append(per_image)
                print(
                    f"  {summary.mean_latency_ms:.3f} ms/image | {summary.fps_from_mean:.2f} FPS | "
                    f"p95 {summary.p95_latency_ms:.3f} ms | {summary.parameters_millions:.3f} M params"
                )
            finally:
                del adapter
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summaries)
    per_image_frame = pd.concat(per_image_frames, ignore_index=True) if per_image_frames else pd.DataFrame()

    summary_csv = destination / "benchmark_640_summary.csv"
    summary_json = destination / "benchmark_640_summary.json"
    per_image_csv = destination / "benchmark_640_per_image_latency.csv"
    metadata_json = destination / "benchmark_640_metadata.json"

    summary_frame.to_csv(summary_csv, index=False)
    per_image_frame.to_csv(per_image_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2, allow_nan=True))
    metadata_json.write_text(json.dumps({
        "paths_config": str(Path(paths_config).expanduser().resolve()),
        "common_config": str(Path(common_config).expanduser().resolve()),
        "models_config": str(models_path),
        "project_root": str(paths["project_root"]),
        "fixed_test_manifest": str(paths["fixed_test_manifest"]),
        "device": str(device),
        "precisions": list(precisions),
        "image_size": IMAGE_SIZE,
        "n_preloaded_images": len(images),
        "warmup": warmup,
        "repeats": repeats,
        "batch_size": batch_size,
    }, indent=2))

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {metadata_json}")
    return summary_frame
