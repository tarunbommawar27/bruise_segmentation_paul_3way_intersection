# Speed / inference-latency benchmark — how it works

This explains the **Speed** cell added to `bruise_colab_final.ipynb` (§ between
fairness and the annotation ceiling). It writes `benchmark_640.csv`.

## The one-line idea

We want to answer: **"once a 640×640 image is sitting on the GPU, how many
milliseconds does each model take to turn it into a mask?"** Nothing else — not
disk, not decode, not resize. Just the model's compute.

---

## 1. Load the images with the *same* dataloader as training

```python
_bench_loader = make_loader(MAN640["test"], CACHE640, CFG["img_size"], 8, False, CFG["workers"])
GPU_IMAGES = torch.cat([x for x, _, _ in _bench_loader]).to(DEVICE)
```

- `make_loader` is the identical function training and test-scoring use. It reads
  the 185 pre-resized 640×640 PNGs from the cache, applies `/255` (the loader's
  only normalization), and returns tensors.
- We loop the whole loader once, concatenate every batch into **one big tensor**
  `[185, 3, 640, 640]`, and `.to(DEVICE)` — push **all 185 images onto the GPU at
  once** (~0.9 GB).
- **Why up front:** so that when the timer runs later, the image is *already* on
  the GPU. No disk read, no JPEG decode, no resize, no CPU→GPU copy happens inside
  the timed region. Those steps are the same for every model and are I/O-bound, so
  timing them would hide the real differences between architectures.

Every image here is **640×640** — that's what's baked into the cache, and it's
what all the accuracy numbers are scored at too, so speed and accuracy are
measured on the same grid.

---

## 2. Load each model from its saved checkpoint

For SegFormer:

```python
model = build_model(spec["arch"], spec["size"], PATHS).to(DEVICE)
model.load_state_dict(torch.load(rd/"best.pt", ...))
```

Rebuilds the architecture, loads the trained weights (`best.pt`), puts it on the
GPU, `.eval()`.

For YOLO it's `yn._raw_module(best)` — pulls the raw torch network out of the
native Ultralytics checkpoint.

We only load **seed 0** of each model, because speed is a property of the
*architecture*, not of which random seed it was trained with — all 3 seeds of B0
run at the same speed.

---

## 3. Warm up (untimed)

```python
for _ in range(warmup):
    _ = model(images[:1])
torch.cuda.synchronize()
```

The first few GPU calls are slow — cuDNN is picking algorithms, CUDA kernels are
being compiled/cached, memory is being allocated. We run ~10 throwaway forward
passes **before** starting the clock so those one-time costs don't pollute the
measurement.

---

## 4. Time one image at a time

```python
for _ in range(repeats):            # repeats = 3
    for i in range(len(images)):    # all 185 images
        x = images[i:i+1]           # one image: [1, 3, 640, 640]
        torch.cuda.synchronize()
        t0 = time.perf_counter()    # start
        z = model(x)                # forward pass
        _ = z >= cut                # threshold logit -> mask
        torch.cuda.synchronize()
        times.append((perf_counter() - t0) * 1000)   # stop, record ms
```

Three important details:

1. **One image per call** (`images[i:i+1]`, batch of 1). This is real deployment
   latency — one camera frame at a time — not amortized throughput.
2. **`torch.cuda.synchronize()` on both sides is mandatory.** CUDA is
   *asynchronous*: `model(x)` returns almost instantly because it only *queues*
   the work on the GPU, it doesn't wait for it to finish. Without the syncs you'd
   be timing "how fast can Python hand off the job," which reports every model as
   impossibly fast. The first sync makes sure the GPU is idle before we start; the
   second forces us to wait until the GPU has actually *finished* before we stop
   the clock.
3. **We do it 185 images × 3 repeats = 555 timings** per model, so we get a
   distribution, not a single lucky/unlucky number.

The SegFormer path also does `z >= cut` — thresholding the logit into a binary
mask — so the timed work is the *full* image→mask path, not just the backbone.
For YOLO the raw module returns a detection tuple instead of a clean logit map, so
that path times just the forward pass; that's the one difference, and it's labeled
`yolo_native_raw_forward` in the output.

---

## 5. Summarize into stats

```python
return {"median_ms": np.median(arr), "p95_ms": np.percentile(arr, 95),
        "fps": 1000/median, ...}
```

From those 555 numbers per model:

- **median_ms** — the typical per-image latency (median, not mean, because a few
  outliers shouldn't move it).
- **p95_ms** — the slow tail: 95% of frames finish faster than this. Matters for
  "worst-case" behavior.
- **fps** — just `1000 / median_ms`, frames per second.
- **params_M** and **peak_activation_MB** — model size and peak GPU memory during
  the run.

All rows go into the `BENCH` dataframe and `benchmark_640.csv`.

---

## The mental model

Think of it as a stopwatch race where **all the runners start already standing on
the track** (images pre-staged on GPU). We're not timing them walking to the
stadium (disk/decode/resize). We fire the gun (`synchronize` + `t0`), the model
runs 640×640 → mask, and we stop the clock only when the GPU truly crosses the
finish line (second `synchronize`). We run the race 555 times per model and report
the typical time and the slow-tail time.

## One caveat when reading the results

SegFormer's number is the true image→mask path, while YOLO's is raw-forward-only —
comparable as architecture speed, but not the exact same postprocessing. That's why
the two are labeled differently in the `path` column (`segformer` vs
`yolo_native_raw_forward`) rather than pretending they're identical.
