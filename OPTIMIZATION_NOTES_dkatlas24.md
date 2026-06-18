# Optimizing the 24ch / 104-class DKatlas MeshNet for browser WebGPU

How the deep, gridding-free DK-atlas model (24 channels, 13 conv layers, affine
GroupNorm + GELU, full 256³ volume, dilations ramping to 31) went from
**unusably slow to ~12 s** in the browser, and why. Measurements are on an Apple
M-series GPU (Dawn → Metal) running the full 256³ volume.

| variant | inference | notes |
|---|---|---|
| fp16, no fixes | ~47 s | also **slower than fp32** (see fix 2) |
| fp16, `--beam 2` | ~16 s | after fixes 1 + 2 |
| **fp16, `--beam 3`** | **~12 s** | current best |
| fp32, `--beam 2` | ~22 s | |

The export script is `examples/export_meshnet_webgpu.py`. The two model-side
fixes are in `brainchop/tiny_meshnet.py`. A third fix is browser-side
(`brainchop-test/main.js`).

## Why this model is different

The other browser models either **crop** to a brain bounding box (smaller
volume, e.g. the 30-channel atlas models) or have **foldable / no normalization**
(conv→relu stacks). This one cannot crop — its receptive field (dilations
1,3,5,7,13,19,31,… RF=255) is matched to the full 256³ cube, and cropping makes
the large-dilation layers read mostly zero-padding and collapse the output. It
also has per-channel **affine GroupNorm that can't be folded**. So it runs the
full volume with real normalization, like the omnimodal skull stripper (mindgrab,
15ch) rather than the cropped atlas models.

## Fix 1 — backbone was recomputed once per output chunk (5×)

`SequentialConvArgmax._chunked_argmax` computes the final 24→104 classifier conv
in chunks (FUSE_CHUNK) fused with argmax, so the 104·256³ logits volume is never
materialized (memory win). With `--chunk 24` and 104 classes that's 5 chunks.

The bug: `export_model()` traces the chunk loop as one lazy graph. Each chunk
reads `x` (the backbone output), and without a realization boundary the scheduler
re-traced the **entire 13-layer backbone in all 5 chunk branches**. The exported
runner therefore recomputed the whole network 5×. (The CLI was fine — eager
execution realizes `x` on the first chunk and reuses it.)

Evidence: the input conv kernel appeared **5×** in the runner, and the runner had
**539 compute passes**.

Fix: `x = x.realize()` once before the chunk loop. Peak memory is unchanged (one
backbone activation + one chunk), but the backbone computes once.
Result: **539 → 123 passes**, ~5× faster inference and ~5× faster BEAM/export.

## Fix 2 — "fp16" was fake mixed precision (f32 activations)

tinygrad upcasts reduction accumulators to f32 for half inputs
(`sum_acc_dtype`: `half → least_upper_dtype(half, float32) = float32`). So a conv
or GroupNorm on f16 data **returns f32**, and nothing cast it back — the dominant
24·256³ activation lived as **f32 (1.6 GiB)** through the whole stack. The fp16
export only halved the (tiny) weights while adding ~1500 element-wise f16↔f32
conversions. On GPUs without 2:1 f16 throughput (Apple Silicon), that made fp16
**slower than fp32** (47 s vs 22 s).

Fix: in `MeshNet.__call__`, when the model is half precision, cast each layer's
**output** back to f16. This is numerically safe — the reduction still
accumulates in f32 (the cast changes only the *stored* result), so there is no
f16 variance-sum overflow — and the cast folds into the producing kernel's store
(no extra pass). The 24·256³ activation is now stored as f16 (0.8 GiB), halving
the dominant memory traffic.
Result: fp16 dropped **47 s → 16 s** and is now faster than fp32.

Why not accumulate in f16 too (`SUM_DTYPE=float16`)? Summing 16.7M values per
channel in f16 is broken regardless of arrangement: naive sums overflow f16's
65504 ceiling and lose precision after ~2k terms; scaling by 1/N first underflows
~38% of terms to zero; `x²` overflows once activations leave the unit range. The
f32 accumulator is an in-register upcast and essentially free — the bytes in
memory are already f16, which is where the cost was.

## Fix 3 — browser device limits (in brainchop-test, not this repo)

`requestDevice()` must opt into the adapter's `maxComputeInvocationsPerWorkgroup`
(and the other compute-workgroup limits). BEAM emits workgroups up to 1024
invocations, but the default device limit is 256, so a BEAM-tuned runner fails
`createComputePipeline` with *"total number of workgroup invocations (512/1024)
exceeds the maximum allowed (256)"* unless the higher limit is requested.

## Where the time actually goes (profile)

`DEBUG=2` on `FP16=1 WEBGPU=1 brainchop -m DKatlas …`, aggregated by op:

| op | share |
|---|---|
| **conv2d** (12× 24→24, 3³, 256³) | **~90%** |
| GroupNorm reduce + fused GELU | ~6% |
| argmax (chunked final conv) | ~1% |
| f16 store casts (fix 2) | ~3% |

Two things this corrected: GroupNorm is *not* the bottleneck (it's cheap
streaming reductions), and **dilation barely matters** — per-conv times across
dilations 3…31 vary only ~1.7× (d31 is no worse than d7), because the conv is a
fused gather (no im2col buffer) and isn't strongly cache-bound. The cost is
simply twelve full 24→24-channel 3³ convolutions over 256³, running well below
the GPU's peak (~250 GFLOPS un-tuned), which is why BEAM helps so much.

## Recipe

The export script is general-purpose (any MeshNet); output names default to the
`--model-dir` basename and are overridable with `--runner-name`/`--web-model-dir`.
For the DK-atlas entry the runner name must stay `dkatlas24`, so pass it
explicitly:

```bash
source ~/venv/torch/bin/activate
cd brainchop-cli
IGNORE_BEAM_CACHE=0 python examples/export_meshnet_webgpu.py \
    --model-dir   ../brainchop-models/meshnet/model24chan104cls \
    --bct         ../brainchop-test \
    --runner-name dkatlas24 \
    --chunk 24 --beam 3 --which both
```

For another model (e.g. `model24chan1t`) just point `--model-dir` at it and omit
`--runner-name`; it derives `model24chan1t_runner.js` +
`public/models/model24chan1t/model.safetensors`.

- `--beam 3` is the sweet spot (~12 s vs ~16 s at beam 2). First run is slow
  (tens of minutes); it caches, so re-runs at the same beam are fast.
- **Do not** use `WINO=1` — 3D Winograd is much slower here.
- Sanity checks on the generated fp16 runner:
  - `grep -c addComputePass …/dkatlas24_runner.js` ≈ 123 (not ~539 → fix 1 regressed)
  - `grep -c onSubmittedWorkDone …/dkatlas24_runner.js` = 0 (single submit)
  - the 24·256³ buffer (`data*_402653184`) should be mostly `array<f16>` (fix 2)

## Next lever

Convs are ~90% of runtime, so after BEAM the remaining knob is **working
resolution** — 256³ is the dominant cost. A lower-resolution pass (e.g. 160³
then upsample labels) would scale roughly linearly with voxel count, at some
accuracy cost. Channel count / depth reductions need retraining (conv cost ∝
Cin·Cout).
