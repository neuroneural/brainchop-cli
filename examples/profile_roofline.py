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
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

BACKEND_ENV_VARS = ("DEV", "METAL", "AMD", "NV", "CUDA", "HIP", "HSA",
                    "GPU", "CL", "WEBGPU", "CPU")
ROCM_LIB_DIR = "/opt/rocm/lib"
ROCM_REEXEC_ENV = "BRAINCHOP_ROOFLINE_ROCM_REEXEC"


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

    if dtype == "fp16":
        x = x.half()
        # Validate that convolutions actually run in half precision.
        # We can't check the final output because fused argmax casts to float32.
        # Instead, probe the first conv layer directly.
        from tinygrad import nn
        first_conv = next(
            (l for l in (getattr(m, 'model', None) or [])
             if isinstance(l, nn.Conv2d)),
            None)
        if first_conv is not None:
            probe = x[:, :, :4, :4, :4].conv2d(first_conv.weight, first_conv.bias,
                                                 padding=first_conv.padding)
            probe_dtype = probe.realize().dtype.name
            del probe
            if probe_dtype != "half":
                raise RuntimeError(
                    f"fp16 requested but conv output dtype is {probe_dtype} — "
                    f"model does not run in true half precision")

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


def _detect_cpu_name():
    if platform.system() == "Linux":
        try:
            r = subprocess.run(["lscpu"], capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if line.startswith("Model name:"):
                    return line.split(":", 1)[-1].strip()
        except Exception:
            pass
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[-1].strip()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True)
            return r.stdout.strip() or None
        except Exception:
            pass
    return None


def _detect_hardware(user_gflops, user_gflops_fp16, user_bw):
    """Return (peak_gflops_fp32, peak_gflops_fp16, peak_bw_gbs, gpu_name).

    For matched Apple SKUs, fp16 peak is auto-derived as 2x fp32.
    For matched AMD Strix Halo SKUs, fp32/fp16/bandwidth are inferred from CPU model.
    For other unknown hardware, fp16 peak must be supplied via --peak-gflops-fp16.
    Raises SystemExit if peak specs cannot be determined.
    """
    name = _detect_gpu_name() or ""
    cpu_name = _detect_cpu_name() or ""

    # Exact-SKU lookup: full chip name → (fp32 GFLOP/s, mem BW GB/s)
    # Apple Silicon ALUs do 2x fp16 throughput; applied for matched SKUs only.
    apple_specs = {
        "Apple M1":          (2600,   68),
        "Apple M1 Pro":      (4100,  200),
        "Apple M1 Max":      (8200,  400),
        "Apple M1 Ultra":   (16400,  800),
        "Apple M2":          (3600,  100),
        "Apple M2 Pro":      (5700,  200),
        "Apple M2 Max":      (9800,  400),
        "Apple M2 Ultra":   (19600,  800),
        "Apple M3":          (4100,  100),
        "Apple M3 Pro":      (5700,  150),
        "Apple M3 Max":      (9800,  400),
        "Apple M4":          (4600,  120),
        "Apple M4 Pro":      (7400,  273),
        "Apple M4 Max":     (14200,  546),
    }
    # Linux lspci reports Strix Halo as an ambiguous 8050S/8060S string, so
    # distinguish the iGPU by CPU model instead.
    strix_halo_specs = [
        (r"Ryzen AI Max\+?(?: PRO)? 395\b", (14800, 29600, 256, "AMD Strix Halo Radeon 8060S")),
        (r"Ryzen AI Max(?: PRO)? 390\b",    (11500, 23000, 256, "AMD Strix Halo Radeon 8050S")),
        (r"Ryzen AI Max(?: PRO)? 385\b",    (11500, 23000, 256, "AMD Strix Halo Radeon 8050S")),
    ]

    gflops_fp32 = user_gflops
    gflops_fp16 = user_gflops_fp16
    bw = user_bw

    matched_sku = None
    if name in apple_specs:
        matched_sku = name
        g, b = apple_specs[name]
        gflops_fp32 = gflops_fp32 or g
        bw = bw or b
        # Apple Silicon: 2x fp16 throughput
        gflops_fp16 = gflops_fp16 or gflops_fp32 * 2
    elif "Strix Halo" in name or "Radeon 8050S" in name or "Radeon 8060S" in name:
        for pattern, (g32, g16, b, label) in strix_halo_specs:
            if re.search(pattern, cpu_name, re.IGNORECASE):
                matched_sku = label
                gflops_fp32 = gflops_fp32 or g32
                gflops_fp16 = gflops_fp16 or g16
                bw = bw or b
                break

    if name:
        print(f"  Detected GPU: {name}" + (" (matched SKU)" if matched_sku else " (unknown SKU)"))
    if cpu_name and matched_sku:
        print(f"  Detected CPU: {cpu_name}")

    if not gflops_fp32 or not bw:
        print(f"\n  ERROR: Could not determine peak specs for '{name or 'no GPU detected'}'.")
        if cpu_name:
            print(f"  CPU: {cpu_name}")
        print("  Supply --peak-gflops and --peak-bw explicitly.")
        raise SystemExit(1)

    # fp16 peak: require explicit value for non-Apple / unknown hardware
    if not gflops_fp16:
        gflops_fp16 = gflops_fp32  # conservative: assume no fp16 speedup

    return gflops_fp32, gflops_fp16, bw, name


def _add_rocm_library_path(env):
    if env.get("DIY") == "1" or not os.path.isdir(ROCM_LIB_DIR):
        return False

    ld = env.get("LD_LIBRARY_PATH", "")
    paths = [p for p in ld.split(":") if p]
    if ROCM_LIB_DIR in paths:
        return False

    env["LD_LIBRARY_PATH"] = f"{ROCM_LIB_DIR}:{ld}" if ld else ROCM_LIB_DIR
    return True


def _ensure_rocm_library_path_for_runtime():
    if platform.system() != "Linux":
        return

    added = _add_rocm_library_path(os.environ)
    if added and os.environ.get(ROCM_REEXEC_ENV) != "1":
        os.environ[ROCM_REEXEC_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ.copy())


def _has_explicit_backend():
    return any(os.environ.get(k) for k in BACKEND_ENV_VARS)


def _probe_backend(backend):
    env = os.environ.copy()
    for key in BACKEND_ENV_VARS:
        env.pop(key, None)
    _add_rocm_library_path(env)
    env[backend] = "1"
    code = """
from tinygrad import Device
dev = Device.DEFAULT
opened = Device[dev]
name = getattr(opened, "device_name", dev)
if dev == "CL" and "cpu" in name.lower():
    raise RuntimeError(f"OpenCL selected CPU device: {name}")
print(f"{dev}:{name}")
"""
    try:
        r = subprocess.run([sys.executable, "-c", code], env=env,
                           capture_output=True, text=True, timeout=15)
    except Exception as e:
        return False, str(e)
    msg = (r.stdout + r.stderr).strip()
    return r.returncode == 0, msg


def _configure_backend_for_gpu(gpu_name):
    """Set tinygrad backend env before tinygrad/brainchop import when it is unambiguous."""
    if _has_explicit_backend():
        return None, []

    candidates = []

    if platform.system() == "Darwin" and gpu_name.startswith("Apple "):
        candidates = ["METAL"]

    if platform.system() == "Linux" and ("AMD/ATI" in gpu_name or "Radeon" in gpu_name):
        candidates = ["AMD", "HIP", "CL", "WEBGPU"]

    errors = []
    for backend in candidates:
        ok, msg = _probe_backend(backend)
        if ok:
            os.environ[backend] = "1"
            return f"{backend}=1", errors
        errors.append(f"{backend}: {msg.splitlines()[-1] if msg else 'unavailable'}")

    return None, errors


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

def plot_roofline(profiles, peak_gflops_fp32, peak_gflops_fp16, peak_bw, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ai_range = np.logspace(-2, 3, 500)

    # Draw roofline per dtype if they differ
    has_fp16 = any(p.dtype == "fp16" for p in profiles if p.achieved_gflops)
    peak_max = peak_gflops_fp32

    roof_fp32 = np.minimum(peak_gflops_fp32, ai_range * peak_bw)
    ax.loglog(ai_range, roof_fp32, "k-", lw=2, label=f"fp32 ceiling ({peak_gflops_fp32} GF/s)")

    if has_fp16 and peak_gflops_fp16 != peak_gflops_fp32:
        roof_fp16 = np.minimum(peak_gflops_fp16, ai_range * peak_bw)
        ax.loglog(ai_range, roof_fp16, "k--", lw=1.5, label=f"fp16 ceiling ({peak_gflops_fp16} GF/s)")
        peak_max = peak_gflops_fp16

    ridge = peak_gflops_fp32 / peak_bw
    ax.axvline(ridge, color="gray", ls=":", alpha=0.5)
    ax.text(ridge * 1.1, peak_gflops_fp32 * 0.7, f"Ridge\n({ridge:.1f} F/B)",
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
                 f"Peak: {peak_gflops_fp32} GFLOP/s (fp32) | BW: {peak_bw} GB/s")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.1, peak_max * 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Roofline profiler for brainchop")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                        choices=["fp32", "fp16"])
    parser.add_argument("--peak-gflops", type=float, default=None,
                        help="Peak fp32 GFLOP/s for the target device")
    parser.add_argument("--peak-gflops-fp16", type=float, default=None,
                        help="Peak fp16 GFLOP/s (auto-derived as 2x fp32 for Apple Silicon)")
    parser.add_argument("--peak-bw", type=float, default=None,
                        help="Peak memory bandwidth in GB/s")
    parser.add_argument("--runs", type=int, default=N_TIMED,
                        help=f"Number of timed inference runs (default {N_TIMED})")
    parser.add_argument("--warmup", type=int, default=N_WARMUP,
                        help=f"Number of warm-up runs (default {N_WARMUP})")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-run", action="store_true",
                        help="Static analysis only — skip inference")
    parser.add_argument("--output-dir", default="examples")
    args = parser.parse_args()

    if not args.no_run:
        _ensure_rocm_library_path_for_runtime()

    registry = _load_registry()
    peak_fp32, peak_fp16, peak_bw, gpu_name = _detect_hardware(
        args.peak_gflops, args.peak_gflops_fp16, args.peak_bw)
    selected_backend, backend_errors = (None, [])
    if not args.no_run:
        selected_backend, backend_errors = _configure_backend_for_gpu(gpu_name)

    from tinygrad.helpers import fetch
    from brainchop import list_models, load

    models = args.models or list(list_models().keys())
    slug = _hw_slug()
    backend = _get_backend()

    # Refuse to profile if the runtime backend doesn't match the GPU whose
    # peak specs we're using — CPU timings against a GPU roofline are nonsense.
    gpu_backends = {"METAL", "AMD", "CUDA", "HIP", "HSA", "NV", "CL", "WEBGPU"}
    if not args.no_run and backend not in gpu_backends:
        print(f"\n  ERROR: tinygrad backend is '{backend}', but peak specs are for GPU '{gpu_name}'.")
        if backend_errors:
            print("  Auto backend probes failed:")
            for err in backend_errors:
                print(f"    {err}")
        print(f"  Set the backend (e.g. WEBGPU=1, CL=1, HIP=1, AMD=1, METAL=1) or use --no-run for static analysis only.")
        raise SystemExit(1)

    print("=" * 65)
    print(f"brainchop roofline profiler  [{slug}]")
    print(f"  models:  {', '.join(models)}")
    print(f"  dtypes:  {', '.join(args.dtypes)}")
    print(f"  peak:    {peak_fp32} GFLOP/s (fp32) | {peak_fp16} GFLOP/s (fp16) | {peak_bw} GB/s")
    print(f"  backend: {backend}")
    if selected_backend:
        print(f"  selected: {selected_backend}")
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
            # SAE models have no half() — fp16 weights aren't supported
            if dtype == "fp16" and model_type == "sae":
                print(f"[{model_name} ({dtype})]  skipped — SAE has no fp16 path")
                continue

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
                    gf_s = f"{p.achieved_gflops:.1f} GFLOP/s" if p.achieved_gflops else "- GFLOP/s"
                    bw_s = f"{p.achieved_bw_gbs:.1f} GB/s" if p.achieved_bw_gbs else "- GB/s"
                    print(f"  Runtime: {median_t:.3f}s (std={p.wall_time_std:.3f}, n={len(times)}) "
                          f"| {gf_s} | {bw_s}")
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
        "hardware": {"name": gpu_name, "peak_gflops_fp32": peak_fp32,
                     "peak_gflops_fp16": peak_fp16,
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
        plot_roofline(profiles, peak_fp32, peak_fp16, peak_bw, str(plot_path))


if __name__ == "__main__":
    main()
