#!/usr/bin/env python3
"""Export a brainchop MeshNet to optimized WebGPU runners for brainchop-test.

General-purpose: works for any MeshNet checkpoint (the DK-atlas 24ch/104-class
model is just the worked example). Output naming defaults to the --model-dir
basename and can be overridden with --runner-name / --web-model-dir, so one
script serves every model. Produces fp16 and/or fp32 runners and drops them
straight into a brainchop-test checkout:

    webgpu_runners/<runner>_runner.js      + public/models/<web-model-dir>/model.safetensors      (fp16)
    webgpu_runners/<runner>_f32_runner.js  + public/models/<web-model-dir>/model_f32.safetensors  (fp32)

e.g. for the DK-atlas model (--runner-name dkatlas24):
    webgpu_runners/dkatlas24_runner.js     + public/models/model24chan104cls/model.safetensors

brainchop-test selects fp16 vs fp32 at runtime via the model entry's
`forceFP32` flag (see inference-webgpu.js); its `webgpu_runner` must equal
--runner-name.

Two optimizations vs a plain `bc.export(...)`:

  * MEMORY: the final 24->104 classifier conv is computed in chunks of
    --chunk channels (FUSE_CHUNK), fused with the running arg-max. The full
    104-channel logits volume (104 * 256^3 * 2B = 3.5 GB at fp16) is never
    materialized; peak buffer stays at one chunk (<= one 24-channel hidden
    activation, ~768 MiB fp16). This is baked into the exported kernel graph.

  * SPEED: --beam runs tinygrad's BEAM kernel search so each generated WGSL
    compute shader gets tuned workgroup/local sizes. BEAM times kernels on the
    real device, so this step needs a working WebGPU (Dawn) backend.

Run on a machine with the torch/tinygrad env AND a working WebGPU device:

    source ~/venv/torch/bin/activate
    cd brainchop-cli
    IGNORE_BEAM_CACHE=0 python examples/export_dkatlas24_webgpu.py \
        --model-dir   ../brainchop-models/meshnet/model24chan104cls \
        --bct         ../brainchop-test \
        --runner-name dkatlas24 \
        --chunk 24 --beam 3 --which both

(--runner-name defaults to the --model-dir basename; pass it explicitly only
when the runner name must differ from the folder, as for dkatlas24. --bct is
required. --beam takes precedence over any BEAM= env var.)

Notes
-----
* --model-dir must contain model.json + model.pth (the brainchop weights, e.g.
  produced by tools/convert_catalyst_gn_hdc_deep.py). You can also pass a model
  NAME registered in brainchop (e.g. "DKatlas") instead of a path.
* fp32 needs ~2x the device memory of fp16 during export/BEAM. If BEAM OOMs at
  fp32, drop --beam for the fp32 pass or raise --chunk granularity (lower
  --chunk value).

Performance history (Apple M-series, full 256^3, browser WebGPU) -- see
OPTIMIZATION_NOTES_dkatlas24.md for the full write-up:

  fp16, before any fix ........ ~47 s   (and SLOWER than fp32 -- see below)
  fp16, --beam 2 .............. ~16 s
  fp16, --beam 3 .............. ~12 s   <- current best
  fp32, --beam 2 .............. ~22 s

Three fixes were required to get here; the first two live in
brainchop/tiny_meshnet.py and are essential -- without them this script
produces a correct but pathologically slow runner:

  1. Backbone recompute (tiny_meshnet.SequentialConvArgmax._chunked_argmax):
     the lazy backbone output is now realize()d once before the chunk loop.
     export_model() traces the chunked final conv, and without a realization
     boundary every chunk re-traced the whole 13-layer backbone -> the runner
     recomputed the network once PER CHUNK (104/chunk24 = 5x). Fixing it cut
     the fp16 runner from 539 -> 123 compute passes (5x inference AND ~5x
     export/BEAM time).

  2. True fp16 activations (tiny_meshnet.MeshNet.__call__): each layer's output
     is cast back to f16 so the 24*256^3 activation is stored as f16 (0.8 GiB)
     not f32 (1.6 GiB). tinygrad upcasts reduction accumulators to f32
     regardless (sum_acc_dtype: half->float), so this is safe (no f16
     variance-sum overflow) and the cast folds into the producing kernel. Before
     this, the "fp16" export kept activations in f32 and only halved the (tiny)
     weights, so it added ~1500 f16<->f32 conversions and ran SLOWER than fp32
     on GPUs without 2:1 f16 throughput (Apple Silicon).

  3. (browser side, in brainchop-test/main.js, NOT this repo) requestDevice must
     opt into the adapter's maxComputeInvocationsPerWorkgroup. BEAM emits
     workgroups up to 1024 invocations; the default device limit is 256, so the
     BEAM-tuned runner fails ComputePipeline creation without the higher limit.

Tuning notes:
* --beam 3 is the sweet spot here; the FIRST run is very slow (BEAM benchmarks
  many variants per kernel at 256^3 -- tens of minutes), then caches in
  ~/.cache/tinygrad so re-runs at the same beam are fast. Keep IGNORE_BEAM_CACHE=0.
* Do NOT use WINO=1: 3D Winograd's transform overhead and large intermediates
  make these convs much slower, not faster.
* Profile shows the convolutions are ~90% of runtime (GroupNorm ~6%, argmax ~1%)
  and dilation has only a mild (~1.7x) effect, so kernel tuning (BEAM) is the
  main lever; the next one is working resolution (256^3 is the cost).
"""
import argparse
import os
import shutil
from pathlib import Path


class _ChunkedExport:
    """Wrap a MeshNet so the traced forward chunks the final conv+argmax.

    export_model() calls `.forward(*inputs)` and `get_state_dict()` on this
    object; delegating to the wrapped model exposes the conv/GroupNorm weights
    under `m.*` keys (the generated runner references weights by these names, and
    we save the matching safetensors, so they stay consistent).
    """

    def __init__(self, model, chunk):
        self.m = model
        self.chunk = int(chunk)
        self.n_classes = model.n_classes

    def forward(self, x):
        return self.m(x, fuse_chunk=self.chunk)


def _export_one(model_dir, name, fp16, chunk, beam, out_dir):
    # FP16 must be set BEFORE importing/loading so load_meshnet() halves weights.
    if fp16:
        os.environ["FP16"] = "1"
    else:
        os.environ.pop("FP16", None)
    os.environ["WEBGPU"] = "1"
    if beam and beam > 0:
        os.environ["BEAM"] = str(beam)
    else:
        os.environ.pop("BEAM", None)

    import time
    from brainchop.api import _load_model
    import brainchop.export_model as _em
    from brainchop.export_model import export_model
    from tinygrad import Tensor, Device
    from tinygrad.nn.state import safe_save

    # Diagnostic: the export traces on the WEBGPU (Dawn) backend. On a headless
    # CUDA server Dawn often has no Vulkan device and falls back to SwiftShader
    # (CPU), which is 10-100x slower than the GPU-backed WebGPU your browser uses.
    # If this is slow, run the export on a machine with GPU-accelerated WebGPU
    # (e.g. the M1 Mac, where Dawn uses Metal). Device.DEFAULT should be WEBGPU.
    print(f"[device] tinygrad Device.DEFAULT = {Device.DEFAULT}")

    # CRITICAL for speed: export_model() calls export_model_webgpu() without a
    # batch_size, so it defaults to 1. With >1 kernel that forces a
    # queue.submit() + await onSubmittedWorkDone() between EVERY compute pass --
    # hundreds of CPU<->GPU round-trips that serialize the GPU and make inference
    # 1-2 orders of magnitude slower. The working runners (mindgrab, model30*)
    # batch all passes into ONE command buffer / ONE submit. Inject a large
    # batch_size so num_statements <= batch_size -> single batch -> one submit.
    if getattr(_em.export_model_webgpu, "_single_batch_patched", None) is None:
        _orig_webgpu = _em.export_model_webgpu

        def _single_batch_webgpu(*a, **k):
            k.setdefault("batch_size", 1_000_000)
            return _orig_webgpu(*a, **k)

        _single_batch_webgpu._single_batch_patched = True
        _em.export_model_webgpu = _single_batch_webgpu

    m = _load_model(str(model_dir))
    # Re-bind the fused conv+argmax to the (possibly halved) final conv so the
    # classifier runs in the same precision as the rest of the network.
    if hasattr(m, "init_seq_conv_argmax"):
        m.init_seq_conv_argmax()

    wrapper = _ChunkedExport(m, chunk)
    # Feed an fp16 dummy input for the fp16 export so tinygrad keeps the whole
    # activation stack in f16 (real fp16 compute + half memory bandwidth, ~2x on
    # most GPUs), with f32 only for the GroupNorm reduction accumulators -- exactly
    # like the working model30chan50cls runner. Feeding an fp32 input (the previous
    # behavior) left every activation in f32, so only the weights were f16 and there
    # was no compute speedup. The browser still passes a Float32Array; the runner's
    # Float16Array.set() converts it.
    in_dtype = "float16" if fp16 else "float32"
    dummy = Tensor.randn(1, 1, 256, 256, 256, dtype=in_dtype)

    print(f"[export] {name}: fp16={fp16} chunk={chunk} beam={beam} ...")
    print("[export] tracing on WEBGPU (runs the 256^3 model twice to capture the JIT)...")
    _t0 = time.time()
    prg, in_sizes, out_sizes, state = export_model(wrapper, "webgpu", dummy, model_name=name)
    print(f"[export] trace+codegen took {time.time() - _t0:.1f}s "
          f"(if this is minutes, the WEBGPU backend is likely CPU/SwiftShader -- see header notes)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js_path = out_dir / f"{name}.js"
    st_path = out_dir / f"{name}.safetensors"
    js_path.write_text(prg)
    safe_save(state, str(st_path))
    print(f"[export]   wrote {js_path} ({js_path.stat().st_size} bytes)")
    print(f"[export]   wrote {st_path} ({st_path.stat().st_size} bytes)")
    return js_path, st_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="dir with model.json+model.pth, or a registered model name")
    ap.add_argument("--bct", required=True, help="path to brainchop-test checkout")
    ap.add_argument("--chunk", type=int, default=24,
                    help="FUSE_CHUNK: final-conv channels per chunk (<=24 keeps peak at one hidden activation)")
    ap.add_argument("--beam", type=int, default=0,
                    help="BEAM kernel search level (0=off; 3 recommended -- best fp16 result here, "
                         "~12s vs ~16s at beam 2. The FIRST export is very slow at 256^3 (BEAM "
                         "benchmarks many variants per kernel -- tens of minutes), then caches in "
                         "~/.cache/tinygrad, so a later re-run with the same --beam is fast. Keep "
                         "IGNORE_BEAM_CACHE=0.)")
    ap.add_argument("--which", choices=["fp16", "fp32", "both"], default="both")
    ap.add_argument("--runner-name", default=None,
                    help="base name of the emitted runner: <name>_runner.js (+ <name>_f32_runner.js). "
                         "The brainchop-test model entry's `webgpu_runner` must equal this. "
                         "Default: basename of --model-dir (e.g. model24chan104cls). For the deployed "
                         "DK-atlas entry (id 14) pass --runner-name dkatlas24 to keep that name.")
    ap.add_argument("--web-model-dir", default=None,
                    help="subfolder under <bct>/public/models where the safetensors are written "
                         "(usually already holds the tfjs model.json + colormap.json). "
                         "Default: basename of --model-dir.")
    ap.add_argument("--staging", default=None,
                    help="scratch dir for the raw export (default: /tmp/<runner-name>_export)")
    args = ap.parse_args()

    # Universal: derive output naming from --model-dir unless overridden. Nothing
    # here is DK-atlas-specific; this exports any brainchop MeshNet to WebGPU.
    base = Path(args.model_dir).name
    runner_name = args.runner_name or base
    web_model_dir = args.web_model_dir or base
    staging = args.staging or f"/tmp/{runner_name}_export"

    bct = Path(args.bct)
    runners = bct / "webgpu_runners"
    models = bct / "public" / "models" / web_model_dir
    runners.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.which in ("fp16", "both"):
        jobs.append((runner_name, True, "model.safetensors"))
    if args.which in ("fp32", "both"):
        jobs.append((f"{runner_name}_f32", False, "model_f32.safetensors"))

    for name, fp16, st_name in jobs:
        js_path, st_path = _export_one(args.model_dir, name, fp16, args.chunk, args.beam, staging)
        # Runner JS -> webgpu_runners/<name>_runner.js (auto-discovered by import.meta.glob)
        dst_js = runners / f"{name}_runner.js"
        shutil.copyfile(js_path, dst_js)
        # Weights -> public/models/<web_model_dir>/<st_name>
        dst_st = models / st_name
        shutil.copyfile(st_path, dst_st)
        print(f"[place]  {dst_js}")
        print(f"[place]  {dst_st}\n")

    print(f"Done. Set the brainchop-test model entry's webgpu_runner to '{runner_name}'.")
    print(f"  forceFP32:false -> {runner_name}_runner.js + model.safetensors (fp16)")
    print(f"  forceFP32:true  -> {runner_name}_f32_runner.js + model_f32.safetensors (fp32)")


if __name__ == "__main__":
    main()
