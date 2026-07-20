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

from .config import Config, load_config
from .data import _read_rgb, _to_tensor, normalize_manifest
from .models import count_parameters, load_model_checkpoint
from .yolo import raw_yolo_model, yolo_bruise_logit

IMAGE_SIZE = 640


@dataclass(frozen=True)
class SourceImage:
    stem: str
    path: str
    rgb: np.ndarray


@dataclass(frozen=True)
class OperatingPoint:
    temperature: float
    threshold: float


@dataclass
class BenchmarkSummary:
    model: str
    family: str
    mode: str
    checkpoint: str
    device: str
    precision: str
    input_height: int
    input_width: int
    output_height: int
    output_width: int
    temperature: float
    threshold: float
    parameters_total: int
    parameters_millions: float
    images_per_repeat: int
    repeats: int
    timed_images: int
    warmup_iterations: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    std_latency_ms: float
    fps_from_mean: float
    fps_from_median: float
    peak_gpu_memory_mb: float
    timing_scope: str
    disk_io_timed: bool
    native_resolution_resize_timed: bool


class Fixed640Adapter:
    """Common interface for the controlled 640x640 deployment benchmark.

    The input to ``infer_full`` is an RGB image already decoded into RAM.
    Each adapter must return a CPU uint8 binary mask with shape [640, 640].
    """

    family = "unknown"
    mode = "unknown"
    normalization = "unknown"

    def __init__(
        self,
        *,
        name: str,
        checkpoint: Path,
        device: torch.device,
        precision: str,
        operating_point: OperatingPoint,
    ) -> None:
        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self.precision = precision
        self.operating_point = operating_point

    @property
    def autocast_enabled(self) -> bool:
        return self.precision == "fp16" and self.device.type == "cuda"

    @property
    def parameter_count(self) -> int:
        raise NotImplementedError

    def infer_full(self, image_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SegFormer640Adapter(Fixed640Adapter):
    family = "segformer"
    mode = "pytorch_logits"
    normalization = "imagenet"

    def __init__(
        self,
        *,
        name: str,
        pretrained: str,
        checkpoint: Path,
        device: torch.device,
        precision: str,
        operating_point: OperatingPoint,
    ) -> None:
        super().__init__(
            name=name,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            operating_point=operating_point,
        )
        self.model = load_model_checkpoint(pretrained, checkpoint, device).eval()
        self._parameter_count = count_parameters(self.model)[0]

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @torch.inference_mode()
    def infer_full(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image_rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = _to_tensor(resized, self.normalization).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=False)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.autocast_enabled,
        ):
            logits = self.model(tensor)

        if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            logits = F.interpolate(
                logits.float(),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        else:
            logits = logits.float()

        probability = torch.sigmoid(
            logits[:, 0] / float(self.operating_point.temperature)
        )
        mask = probability >= float(self.operating_point.threshold)
        return mask[0].to(dtype=torch.uint8, device="cpu").numpy()

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


class YOLOSemantic640Adapter(Fixed640Adapter):
    """Controlled YOLO semantic-logit path used by the v3 evaluator.

    This deliberately does not call the high-level Ultralytics predictor. It
    uses the same raw semantic model and ``yolo_bruise_logit`` helper as the
    project evaluation code, so input/output handling is directly comparable
    with SegFormer.
    """

    family = "yolo_semantic"
    mode = "pytorch_semantic_logits"
    normalization = "unit"

    def __init__(
        self,
        *,
        name: str,
        checkpoint: Path,
        device: torch.device,
        precision: str,
        operating_point: OperatingPoint,
    ) -> None:
        super().__init__(
            name=name,
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            operating_point=operating_point,
        )
        self.model = raw_yolo_model(checkpoint, device).eval()
        self._parameter_count = sum(p.numel() for p in self.model.parameters())

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @torch.inference_mode()
    def infer_full(self, image_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image_rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = _to_tensor(resized, self.normalization).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=False)

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.autocast_enabled,
        ):
            logits = yolo_bruise_logit(self.model, tensor)

        # yolo_bruise_logit returns [B,H,W] and already aligns to the input
        # tensor size, but we enforce 640x640 again as a hard benchmark guard.
        if logits.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            logits = F.interpolate(
                logits.unsqueeze(1).float(),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        else:
            logits = logits.float()

        probability = torch.sigmoid(
            logits / float(self.operating_point.temperature)
        )
        mask = probability >= float(self.operating_point.threshold)
        return mask[0].to(dtype=torch.uint8, device="cpu").numpy()


def _format_path(value: str, cfg: Config, seed: int) -> Path:
    rendered = value.format(
        root=str(cfg.root),
        seed=seed,
        seed_root=str(cfg.root / f"seed_{seed}"),
    )
    path = Path(rendered).expanduser()
    if not path.is_absolute():
        path = cfg.root / path
    return path.resolve()


def _read_first_numeric(csv_path: Path, column: str) -> float:
    frame = pd.read_csv(csv_path)
    if column not in frame.columns or frame.empty:
        raise ValueError(f"Column '{column}' not found in non-empty CSV: {csv_path}")
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    if value.empty:
        raise ValueError(f"Column '{column}' has no numeric values: {csv_path}")
    return float(value.iloc[0])


def resolve_checkpoint(spec: dict[str, Any], cfg: Config, seed: int) -> Path:
    value = spec.get("checkpoint", "auto")
    if value != "auto":
        path = _format_path(str(value), cfg, seed)
    else:
        run_name = str(spec["run_name"])
        filename = str(spec.get("checkpoint_filename", "best_checkpoint.pt"))
        path = cfg.root / f"seed_{seed}" / run_name / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {spec.get('name')}: {path}")
    return path.resolve()


def resolve_operating_point(
    spec: dict[str, Any], cfg: Config, seed: int
) -> OperatingPoint:
    threshold_value = spec.get("threshold", 0.5)
    if isinstance(threshold_value, str) and threshold_value.lower() == "auto":
        run_name = str(spec["run_name"])
        threshold_csv_value = spec.get(
            "threshold_csv",
            f"{{seed_root}}/{run_name}/val_summary.csv",
        )
        threshold_csv = _format_path(str(threshold_csv_value), cfg, seed)
        threshold_column = str(spec.get("threshold_column", "best_threshold"))
        threshold = _read_first_numeric(threshold_csv, threshold_column)
    else:
        threshold = float(threshold_value)

    temperature_value = spec.get("temperature", 1.0)
    if isinstance(temperature_value, str) and temperature_value.lower() == "auto":
        temperature_csv = _format_path(str(spec["temperature_csv"]), cfg, seed)
        temperature_column = str(
            spec.get("temperature_column", "calibrated_temperature")
        )
        temperature = _read_first_numeric(temperature_csv, temperature_column)
    else:
        temperature = float(temperature_value)

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    if not 0 < threshold < 1:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return OperatingPoint(temperature=temperature, threshold=threshold)


def build_adapter(
    spec: dict[str, Any],
    cfg: Config,
    seed: int,
    device: torch.device,
    precision: str,
) -> Fixed640Adapter:
    checkpoint = resolve_checkpoint(spec, cfg, seed)
    operating_point = resolve_operating_point(spec, cfg, seed)
    family = str(spec["family"]).lower()

    if family == "segformer":
        pretrained_key = str(spec["pretrained_key"])
        return SegFormer640Adapter(
            name=str(spec["name"]),
            pretrained=cfg.pretrained(pretrained_key),
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            operating_point=operating_point,
        )
    if family in {"yolo", "yolo_semantic"}:
        return YOLOSemantic640Adapter(
            name=str(spec["name"]),
            checkpoint=checkpoint,
            device=device,
            precision=precision,
            operating_point=operating_point,
        )
    raise ValueError(f"Unsupported model family: {family}")


def load_fixed_test_images(
    cfg: Config,
    *,
    max_images: int | None,
) -> list[SourceImage]:
    frame = normalize_manifest(pd.read_csv(cfg.path("wl_test_manifest")))
    if max_images is not None and max_images > 0:
        frame = frame.head(max_images)

    images: list[SourceImage] = []
    for row in frame.itertuples(index=False):
        images.append(
            SourceImage(
                stem=str(row.stem),
                path=str(row.image_path),
                rgb=_read_rgb(str(row.image_path)),
            )
        )
    if not images:
        raise RuntimeError("The fixed test manifest produced zero benchmark images.")
    return images


def _synchronize(device: torch.device) -> None:
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
) -> tuple[BenchmarkSummary, pd.DataFrame]:
    if adapter.precision == "fp16" and adapter.device.type != "cuda":
        raise ValueError("FP16 benchmark is supported only on CUDA in this script.")

    for index in range(warmup):
        mask = adapter.infer_full(images[index % len(images)].rgb)
        if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.dtype != np.uint8:
            raise RuntimeError(
                f"{adapter.name} returned shape={mask.shape}, dtype={mask.dtype}; "
                "expected uint8 [640,640]."
            )
    _synchronize(adapter.device)

    if adapter.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(adapter.device)

    records: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    for repeat_index in range(repeats):
        for image_index, source in enumerate(images):
            _synchronize(adapter.device)
            start = time.perf_counter()
            mask = adapter.infer_full(source.rgb)
            _synchronize(adapter.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE) or mask.dtype != np.uint8:
                raise RuntimeError(
                    f"{adapter.name} returned shape={mask.shape}, dtype={mask.dtype}; "
                    "expected uint8 [640,640]."
                )
            latencies_ms.append(elapsed_ms)
            records.append(
                {
                    "model": adapter.name,
                    "family": adapter.family,
                    "mode": adapter.mode,
                    "precision": adapter.precision,
                    "repeat": repeat_index,
                    "image_index": image_index,
                    "stem": source.stem,
                    "image_path": source.path,
                    "latency_ms": elapsed_ms,
                }
            )

    stats = _summarize(latencies_ms)
    peak_memory = (
        float(torch.cuda.max_memory_allocated(adapter.device) / (1024**2))
        if adapter.device.type == "cuda"
        else math.nan
    )
    summary = BenchmarkSummary(
        model=adapter.name,
        family=adapter.family,
        mode=adapter.mode,
        checkpoint=str(adapter.checkpoint),
        device=str(adapter.device),
        precision=adapter.precision,
        input_height=IMAGE_SIZE,
        input_width=IMAGE_SIZE,
        output_height=IMAGE_SIZE,
        output_width=IMAGE_SIZE,
        temperature=adapter.operating_point.temperature,
        threshold=adapter.operating_point.threshold,
        parameters_total=adapter.parameter_count,
        parameters_millions=adapter.parameter_count / 1_000_000.0,
        images_per_repeat=len(images),
        repeats=repeats,
        timed_images=len(latencies_ms),
        warmup_iterations=warmup,
        peak_gpu_memory_mb=peak_memory,
        timing_scope=(
            "decoded RGB in RAM -> resize 640 -> model normalization -> H2D -> "
            "forward -> logits resized to 640 -> sigmoid/temperature/threshold -> "
            "CPU uint8 640 mask"
        ),
        disk_io_timed=False,
        native_resolution_resize_timed=False,
        **stats,
    )
    return summary, pd.DataFrame.from_records(records)


def run_benchmark_640(
    *,
    project_config: str | Path,
    benchmark_config: str | Path,
    output_dir: str | Path,
    device_name: str,
    precisions: Iterable[str],
    seed: int,
    max_images: int | None,
    warmup: int,
    repeats: int,
) -> pd.DataFrame:
    cfg = load_config(project_config)
    bench_cfg_path = Path(benchmark_config).expanduser().resolve()
    payload = yaml.safe_load(bench_cfg_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Benchmark YAML must contain a top-level models list.")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    images = load_fixed_test_images(cfg, max_images=max_images)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    per_image_frames: list[pd.DataFrame] = []

    print(f"Project root: {cfg.root}")
    print(f"Fixed test images preloaded: {len(images)}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("Timing excludes disk decode and excludes resize back to native camera size.")
    print("Every returned binary mask is exactly 640x640.\n")

    for precision in precisions:
        precision = str(precision).lower()
        if precision not in {"fp32", "fp16"}:
            raise ValueError(f"Unsupported precision: {precision}")
        if precision == "fp16" and device.type != "cuda":
            print("Skipping FP16 on CPU.")
            continue

        for spec in payload["models"]:
            if not bool(spec.get("enabled", True)):
                continue
            print(f"[{precision}] Loading {spec['name']}...")
            adapter = build_adapter(spec, cfg, seed, device, precision)
            try:
                summary, per_image = benchmark_adapter(
                    adapter,
                    images,
                    warmup=warmup,
                    repeats=repeats,
                )
                summaries.append(asdict(summary))
                per_image_frames.append(per_image)
                print(
                    f"  {summary.mean_latency_ms:.3f} ms/image | "
                    f"{summary.fps_from_mean:.2f} FPS | "
                    f"p95 {summary.p95_latency_ms:.3f} ms | "
                    f"{summary.parameters_millions:.3f} M params"
                )
            finally:
                adapter.close()
                del adapter
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_frame = pd.DataFrame(summaries)
    per_image_frame = (
        pd.concat(per_image_frames, ignore_index=True)
        if per_image_frames
        else pd.DataFrame()
    )

    summary_csv = destination / "benchmark_640_summary.csv"
    summary_json = destination / "benchmark_640_summary.json"
    per_image_csv = destination / "benchmark_640_per_image_latency.csv"
    metadata_json = destination / "benchmark_640_metadata.json"

    summary_frame.to_csv(summary_csv, index=False)
    per_image_frame.to_csv(per_image_csv, index=False)
    summary_json.write_text(json.dumps(summaries, indent=2, allow_nan=True))
    metadata_json.write_text(
        json.dumps(
            {
                "project_config": str(Path(project_config).expanduser().resolve()),
                "benchmark_config": str(bench_cfg_path),
                "project_root": str(cfg.root),
                "fixed_test_manifest": str(cfg.path("wl_test_manifest")),
                "seed": seed,
                "device": str(device),
                "precisions": list(precisions),
                "image_size": IMAGE_SIZE,
                "n_preloaded_images": len(images),
                "warmup": warmup,
                "repeats": repeats,
                "timing_scope": (
                    "decoded RGB in RAM -> resize 640 -> model-specific normalization -> "
                    "H2D -> forward -> 640 logits/mask -> threshold -> CPU uint8 mask"
                ),
                "disk_io_timed": False,
                "native_resolution_resize_timed": False,
            },
            indent=2,
        )
    )

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {per_image_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {metadata_json}")
    return summary_frame
