#!/usr/bin/env python3
"""
Roofline profiling for brainchop models on iGPU.

Computes FLOPs, memory traffic, and arithmetic intensity per model/dtype.
Measures wall-clock inference time and plots results on a roofline chart.
Output filenames are derived from the detected hardware slug.

Usage:
    python examples/profile_roofline.py                          # all models, fp32+fp16
    python examples/profile_roofline.py --models tissue_fast     # one model
    python examples/profile_roofline.py --no-run                 # static analysis only
    python examples/profile_roofline.py --peak-gflops 500 --peak-bw 50
"""

import argparse
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# -- FLOPs / bandwidth helpers ------------------------------------------------

def conv3d_flops(in_c, out_c, k, spatial, has_bias=False):
    flops = 2 * (k ** 3) * in_c * out_c * spatial
    return flops + out_c * spatial if has_bias else flops

def conv3d_bytes(in_c, out_c, k, spatial, bpe, has_bias=False):
    b = (in_c * spatial + out_c * in_c * k**3 + out_c * spatial) * bpe
    return b + out_c * bpe if has_bias else b

def norm_flops(channels, spatial):
    # InstanceNorm (GroupNorm with num_groups=channels, affine=False):
    # mean (1 div) + variance (sub + sq + 1 div) + normalize (sub + div) = 6 ops/elem
    return 6 * channels * spatial

def silu_flops(n):
    # SiLU: x * sigmoid(x) = negate + exp + add + reciprocal + multiply = 5 ops/elem
    return 5 * n

def elem_flops(n):
    return n


# -- Dataclasses ---------------------------------------------------------------

@dataclass
class ModelProfile:
    model_name: str
    dtype: str
    total_flops: int
    total_bytes: int
    wall_time_s: float | None = None       # median of timed runs
    wall_time_std: float | None = None
    wall_time_min: float | None = None
    achieved_gflops: float | None = None
    achieved_bw_gbs: float | None = None

    @property
    def arithmetic_intensity(self):
        return self.total_flops / self.total_bytes if self.total_bytes else 0.0


# -- Static analysis -----------------------------------------------------------

def _bpe(dtype):
    return 2 if dtype == "fp16" else 4

def analyze_meshnet(config_path, dtype):
    with open(config_path) as f:
        config = json.load(f)

    bpe = _bpe(dtype)
    S = 256 ** 3
    total_f = total_b = 0
    has_bias = config.get("bias", False)
    has_bnorm = config.get("bnorm", True)

    for i, lc in enumerate(config["layers"]):
        ic, oc, k = lc["in_channels"], lc["out_channels"], lc["kernel_size"]
        is_last = (i == len(config["layers"]) - 1)

        total_f += conv3d_flops(ic, oc, k, S, has_bias)
        total_b += conv3d_bytes(ic, oc, k, S, bpe, has_bias)

        if not is_last:
            if has_bnorm:
                total_f += norm_flops(oc, S)
                total_b += 2 * oc * S * bpe
            total_f += elem_flops(oc * S)
            total_b += 2 * oc * S * bpe

    return ModelProfile("", dtype, total_f, total_b)


def analyze_sae(n_classes, dtype):
    from brainchop.sae_model import LAYER_INDICES

    bpe = _bpe(dtype)
    ch = 16
    S_full, S_half = 256**3, 128**3
    total_f = total_b = 0

    for i, idx in enumerate(LAYER_INDICES):
        is_last = (i == len(LAYER_INDICES) - 1)
        is_down, is_up = (idx == 10), (idx == 22)

        if is_last:      ic, oc, k, S = ch, n_classes, 1, S_full
        elif is_down:    ic, oc, k, S = ch, ch, 3, S_full
        elif is_up:      ic, oc, k, S = ch, ch, 2, S_half
        elif idx < 10:   ic, oc, k, S = (1 if idx == 0 else ch), ch, 3, S_full
        elif idx > 22:   ic, oc, k, S = ch, ch, 3, S_full
        else:            ic, oc, k, S = ch, ch, 3, S_half

        # SAE always has bias
        total_f += conv3d_flops(ic, oc, k, S, has_bias=True)
        total_b += conv3d_bytes(ic, oc, k, S, bpe, has_bias=True)

        # SiLU activation after every conv except ConvTranspose and final
        if not is_last and not is_up:
            out_S = S_half if is_down else S
            total_f += silu_flops(oc * out_S)
            total_b += 2 * oc * out_S * bpe

    return ModelProfile("", dtype, total_f, total_b)


# -- Runtime profiling ---------------------------------------------------------

N_WARMUP = 2
N_TIMED = 5

def profile_inference(model_name, vol, dtype, n_warmup=N_WARMUP, n_runs=N_TIMED):
    from brainchop.api import _load_model, _run_with_fuse_chunk
    from tinygrad import Tensor

    if dtype == "fp16":
        os.environ["FP16"] = "1"
    else:
        os.environ.pop("FP16", None)

    m = _load_model(model_name)
    x = Tensor(vol.data.numpy().astype(np.float32)).reshape(1, 1, 256, 256, 256)
    x = m.normalize(x)

    # Warm up (compile kernels, fill caches)
    for _ in range(n_warmup):
        out = _run_with_fuse_chunk(m, x, model_name)
        out.numpy()  # force GPU sync

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = _run_with_fuse_chunk(m, x, model_name)
        out.numpy()  # force GPU sync before stopping timer
        t1 = time.perf_counter()
        times.append(t1 - t0)

    os.environ.pop("FP16", None)
    return times


# -- Hardware detection --------------------------------------------------------

def _hw_slug():
    """Return a short, filesystem-safe hardware identifier."""
    name = _detect_gpu_name()
    if name:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug[:48]
    return platform.node().split(".")[0] or "unknown"


def _detect_gpu_name():
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if "Chipset Model" in line:
                    return line.split(":")[-1].strip()
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            r = subprocess.run(["lspci"], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if "VGA" in line or "Display" in line:
                    return line.split(":")[-1].strip()
        except Exception:
            pass
    return None


def _detect_hardware(user_gflops, user_bw):
    """Return (peak_gflops, peak_bw_gbs, gpu_name)."""
    name = _detect_gpu_name() or ""

    # Lookup table: substring → (GFLOP/s, GB/s)
    specs = {
        "M1":   (2600,  68), "M2":   (3600, 100),
        "M3":   (4100, 100), "M4":   (4600, 120),
        "780M": (8600,  51), "890M": (8600,  51),
        "680M": (4300,  51), "Vega": (2000,  38),
    }
    gflops = user_gflops
    bw = user_bw
    for key, (g, b) in specs.items():
        if key in name:
            gflops = gflops or g
            bw = bw or b
            break

    gflops = gflops or 500   # conservative fallback
    bw = bw or 50
    if name:
        print(f"  Detected: {name}")
    return gflops, bw, name


def _get_backend():
    try:
        from tinygrad import Device
        return Device.DEFAULT
    except Exception:
        return "unknown"


# -- Model config helpers ------------------------------------------------------

def _load_registry():
    p = Path(__file__).resolve().parent.parent / "models.json"
    with open(p) as f:
        return json.load(f)

def _resolve_model_dir(model_name):
    reg = _load_registry()
    folder = reg[model_name].get("folder", model_name)
    return Path.home() / ".cache" / "brainchop" / "models" / folder


# -- Plot ----------------------------------------------------------------------

def plot_roofline(profiles, peak_gflops, peak_bw, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ai_range = np.logspace(-2, 3, 500)
    roofline = np.minimum(peak_gflops, ai_range * peak_bw)
    ax.loglog(ai_range, roofline, "k-", lw=2, label="Roofline ceiling")

    ridge = peak_gflops / peak_bw
    ax.axvline(ridge, color="gray", ls=":", alpha=0.5)
    ax.text(ridge * 1.1, peak_gflops * 0.7, f"Ridge\n({ridge:.1f} F/B)",
            fontsize=8, color="gray")

    markers = {"fp32": "o", "fp16": "s"}
    cmap = plt.cm.tab10
    names = list({p.model_name for p in profiles})
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    for p in profiles:
        if p.achieved_gflops and p.achieved_gflops > 0:
            ax.plot(p.arithmetic_intensity, p.achieved_gflops,
                    marker=markers.get(p.dtype, "o"),
                    color=colors[p.model_name], ms=10, zorder=5)
            ax.annotate(f"{p.model_name}\n({p.dtype})",
                        (p.arithmetic_intensity, p.achieved_gflops),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color=colors[p.model_name])

    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title(f"Roofline — brainchop iGPU\n"
                 f"Peak: {peak_gflops} GFLOP/s | BW: {peak_bw} GB/s")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.1, peak_gflops * 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Roofline profiler for brainchop")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                        choices=["fp32", "fp16"])
    parser.add_argument("--peak-gflops", type=float, default=None)
    parser.add_argument("--peak-bw", type=float, default=None)
    parser.add_argument("--runs", type=int, default=N_TIMED,
                        help=f"Number of timed inference runs (default {N_TIMED})")
    parser.add_argument("--warmup", type=int, default=N_WARMUP,
                        help=f"Number of warm-up runs (default {N_WARMUP})")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-run", action="store_true",
                        help="Static analysis only — skip inference")
    parser.add_argument("--output-dir", default="examples")
    args = parser.parse_args()

    from tinygrad.helpers import fetch
    from brainchop import list_models, load

    registry = _load_registry()
    models = args.models or list(list_models().keys())
    peak_gflops, peak_bw, gpu_name = _detect_hardware(args.peak_gflops, args.peak_bw)
    slug = _hw_slug()
    backend = _get_backend()

    print("=" * 65)
    print(f"brainchop roofline profiler  [{slug}]")
    print(f"  models:  {', '.join(models)}")
    print(f"  dtypes:  {', '.join(args.dtypes)}")
    print(f"  peak:    {peak_gflops} GFLOP/s  |  {peak_bw} GB/s")
    print(f"  backend: {backend}")
    print("=" * 65)

    vol = None
    if not args.no_run:
        url = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
        vol = load(str(Path(fetch(url, "t1_crop.nii.gz"))))
        print(f"Loaded test volume: shape={vol.data.shape}\n")

    profiles: list[ModelProfile] = []

    for model_name in models:
        info = registry[model_name]
        model_type = info.get("type", "meshnet")
        n_classes = info.get("n_classes", 3)

        for dtype in args.dtypes:
            print(f"[{model_name} ({dtype})]")

            # Static analysis
            try:
                if model_type == "sae":
                    p = analyze_sae(n_classes, dtype)
                else:
                    d = _resolve_model_dir(model_name)
                    cfg = str(d / "model.json")
                    if not Path(cfg).exists():
                        from brainchop.utils import find_pth_files
                        find_pth_files(model_name)
                    p = analyze_meshnet(cfg, dtype)

                p.model_name = model_name
                p.dtype = dtype
                gf, gb = p.total_flops / 1e9, p.total_bytes / 1e9
                print(f"  Static:  {gf:.1f} GFLOP | {gb:.1f} GB | AI={p.arithmetic_intensity:.1f} F/B")
            except Exception as e:
                print(f"  Static analysis failed: {e}")
                p = ModelProfile(model_name, dtype, 0, 0)

            # Runtime
            if not args.no_run:
                try:
                    times = profile_inference(model_name, vol, dtype,
                                              n_warmup=args.warmup, n_runs=args.runs)
                    median_t = float(np.median(times))
                    p.wall_time_s = median_t
                    p.wall_time_std = float(np.std(times))
                    p.wall_time_min = float(np.min(times))
                    if p.total_flops > 0:
                        p.achieved_gflops = (p.total_flops / 1e9) / median_t
                        p.achieved_bw_gbs = (p.total_bytes / 1e9) / median_t
                    print(f"  Runtime: {median_t:.3f}s (std={p.wall_time_std:.3f}, n={len(times)}) "
                          f"| {p.achieved_gflops:.1f} GFLOP/s | {p.achieved_bw_gbs:.1f} GB/s")
                except Exception as e:
                    print(f"  Runtime failed: {e}")

            profiles.append(p)

    # Summary table
    print()
    w = max((len(p.model_name) for p in profiles), default=5)
    w = max(w, 5)
    hdr = f"{'Model':<{w}}  {'dtype':<5}  {'GFLOP':>7}  {'GB':>6}  {'AI':>6}  {'med(s)':>7}  {'std':>6}  {'GF/s':>7}  {'GB/s':>6}"
    print(hdr)
    print("-" * len(hdr))
    for p in profiles:
        gf = f"{p.total_flops/1e9:.1f}" if p.total_flops else "-"
        gb = f"{p.total_bytes/1e9:.1f}" if p.total_bytes else "-"
        ai = f"{p.arithmetic_intensity:.1f}" if p.total_bytes else "-"
        wt = f"{p.wall_time_s:.3f}" if p.wall_time_s else "-"
        sd = f"{p.wall_time_std:.3f}" if p.wall_time_std else "-"
        ag = f"{p.achieved_gflops:.1f}" if p.achieved_gflops else "-"
        ab = f"{p.achieved_bw_gbs:.1f}" if p.achieved_bw_gbs else "-"
        print(f"{p.model_name:<{w}}  {p.dtype:<5}  {gf:>7}  {gb:>6}  {ai:>6}  {wt:>7}  {sd:>6}  {ag:>7}  {ab:>6}")

    # Save artifacts with hardware slug
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / f"roofline_{slug}.json"
    results_data = {
        "hardware": {"name": gpu_name, "peak_gflops": peak_gflops,
                     "peak_bw_gbs": peak_bw, "backend": backend, "slug": slug},
        "profiles": [
            {"model": p.model_name, "dtype": p.dtype,
             "total_gflops": p.total_flops / 1e9, "total_gb": p.total_bytes / 1e9,
             "arithmetic_intensity": p.arithmetic_intensity,
             "wall_time_s": p.wall_time_s, "wall_time_std": p.wall_time_std,
             "wall_time_min": p.wall_time_min,
             "achieved_gflops": p.achieved_gflops, "achieved_bw_gbs": p.achieved_bw_gbs}
            for p in profiles
        ],
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults: {results_path}")

    if not args.no_plot:
        plot_path = out_dir / f"roofline_{slug}.png"
        plot_roofline(profiles, peak_gflops, peak_bw, str(plot_path))


if __name__ == "__main__":
    main()
