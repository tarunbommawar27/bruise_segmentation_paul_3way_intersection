# Integrated 640×640 Segmentation Benchmark

This module connects directly to the existing `bruise_repro_v3` pipeline. It reuses:

- `Config` and the existing `configs/project.yaml`
- the fixed white-light test manifest
- the pipeline's RGB reading and normalization helpers
- `load_model_checkpoint()` for SegFormer
- `raw_yolo_model()` and `yolo_bruise_logit()` for YOLO semantic segmentation

## Controlled timed scope

For every model, the timed section is:

```text
RGB image already decoded in RAM
→ resize image to 640×640
→ model-specific normalization
→ CPU-to-GPU transfer
→ model forward
→ resize logits to 640×640 when needed
→ temperature scaling
→ sigmoid + threshold
→ copy final binary uint8 mask to CPU
```

The final mask is always `640×640`. The benchmark excludes:

- disk image decoding;
- loading the ground-truth mask;
- Dice computation;
- resizing the mask back to the original ~6000-pixel camera resolution;
- saving masks or overlays.

This is the recommended apples-to-apples deployment benchmark. Ground-truth masks are required for accuracy evaluation, but not for inference latency.

## Install into the existing repository

Copy the three folders from this package into the repository root:

```bash
cd /home/tbommawa/bruise_repro_v3
unzip -o /PATH/TO/benchmark_640_integrated.zip
python -m pip install -e .
```

The files should end up at:

```text
src/bruise_repro/benchmark_640.py
scripts/run_benchmark_640.py
configs/benchmark_640.yaml
tests/test_benchmark_640.py
```

## Validate before the full run

```bash
cd /home/tbommawa/bruise_repro_v3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bruise_repro_v3
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python -m py_compile \
  src/bruise_repro/benchmark_640.py \
  scripts/run_benchmark_640.py

pytest -q tests/test_benchmark_640.py

python scripts/run_benchmark_640.py \
  --project-config configs/project.yaml \
  --benchmark-config configs/benchmark_640.yaml \
  --device cuda:0 \
  --precision fp32 \
  --max-images 5 \
  --warmup 3 \
  --repeats 1 \
  --output-dir /home/tbommawa/bruise_repro_v3_runs/protocol_v3_distillation_suite/benchmarks_640_smoke
```

## Final FP32 benchmark

```bash
python scripts/run_benchmark_640.py \
  --project-config configs/project.yaml \
  --benchmark-config configs/benchmark_640.yaml \
  --device cuda:0 \
  --precision fp32 \
  --max-images 185 \
  --warmup 30 \
  --repeats 3 \
  --output-dir /home/tbommawa/bruise_repro_v3_runs/protocol_v3_distillation_suite/benchmarks_640_fp32
```

## Final FP16 benchmark

```bash
python scripts/run_benchmark_640.py \
  --project-config configs/project.yaml \
  --benchmark-config configs/benchmark_640.yaml \
  --device cuda:0 \
  --precision fp16 \
  --max-images 185 \
  --warmup 30 \
  --repeats 3 \
  --output-dir /home/tbommawa/bruise_repro_v3_runs/protocol_v3_distillation_suite/benchmarks_640_fp16
```

## Outputs

```text
benchmark_640_summary.csv
benchmark_640_summary.json
benchmark_640_per_image_latency.csv
benchmark_640_metadata.json
```

Use these columns for the primary comparison:

- `mean_latency_ms`
- `fps_from_mean`
- `p95_latency_ms`
- `peak_gpu_memory_mb`
- `parameters_millions`

## Important operating-point distinction

For YOLO Soft Response KD:

```text
KD training temperature = 2.0
Post-training calibration temperature = 1.639499...
Validated inference threshold = 0.20
```

The benchmark uses the post-training calibration temperature, because it is reproducing inference, not KD training.
