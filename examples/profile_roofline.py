#!/usr/bin/env python3
"""
iGPU profiling with roofline-style analysis for brainchop models.

Measures FLOPs, memory bandwidth, and arithmetic intensity per model/dtype,
then produces a roofline plot. Optionally wraps AMD tools (AMDuProfCLI,
rocprof) when available.

Usage:
    # Manual roofline (works everywhere)
    python examples/profile_roofline.py

    # Specific models
    python examples/profile_roofline.py --models tissue_fast subcortical

    # FP16 only
    python examples/profile_roofline.py --dtypes fp16

    # With AMD UProf wrapper (Linux + AMD CPU)
    python examples/profile_roofline.py --uprof

    # With rocprof wrapper (Linux + ROCm GPU)
    python examples/profile_roofline.py --rocprof

    # Skip plot generation
    python examples/profile_roofline.py --no-plot

    # Custom hardware ceilings for roofline
    python examples/profile_roofline.py --peak-gflops 500 --peak-bw 50

Output:
    examples/roofline_results.json   — raw profiling data
    examples/roofline_plot.png       — roofline chart (requires matplotlib)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# FLOPs / bandwidth accounting
# ---------------------------------------------------------------------------

def conv3d_flops(in_c: int, out_c: int, k: int, spatial: int,
                 groups: int = 1, has_bias: bool = False) -> int:
    """FLOPs for a single 3-D convolution (multiply-accumulate × 2)."""
    # Each output element: k^3 * (in_c/groups) MACs
    macs_per_elem = (k ** 3) * (in_c // groups)
    total_elems = out_c * spatial
    flops = 2 * macs_per_elem * total_elems  # mul + add
    if has_bias:
        flops += total_elems
    return flops


def conv3d_bytes(in_c: int, out_c: int, k: int, spatial: int,
                 bytes_per_elem: int, has_bias: bool = False) -> int:
    """Bytes transferred: input + weights + output (lower bound)."""
    input_bytes = in_c * spatial * bytes_per_elem
    weight_bytes = out_c * in_c * (k ** 3) * bytes_per_elem
    if has_bias:
        weight_bytes += out_c * bytes_per_elem
    output_bytes = out_c * spatial * bytes_per_elem
    return input_bytes + weight_bytes + output_bytes


def groupnorm_flops(channels: int, spatial: int) -> int:
    """Approximate FLOPs for GroupNorm (mean + var + normalize)."""
    n = channels * spatial
    return 5 * n  # mean, var, sub, div, scale


def groupnorm_bytes(channels: int, spatial: int, bytes_per_elem: int) -> int:
    """Bytes for GroupNorm: read input + write output."""
    return 2 * channels * spatial * bytes_per_elem


def activation_flops(spatial: int) -> int:
    """FLOPs for element-wise activation (relu/gelu/silu/elu)."""
    return spatial  # 1 op per element (approximate)


def activation_bytes(spatial: int, bytes_per_elem: int) -> int:
    return 2 * spatial * bytes_per_elem  # read + write


# ---------------------------------------------------------------------------
# Model analysis: walk the architecture and tally FLOPs / bytes
# ---------------------------------------------------------------------------

@dataclass
class LayerProfile:
    name: str
    flops: int
    bytes_transferred: int

    @property
    def arithmetic_intensity(self) -> float:
        if self.bytes_transferred == 0:
            return 0.0
        return self.flops / self.bytes_transferred


@dataclass
class ModelProfile:
    model_name: str
    dtype: str
    total_flops: int
    total_bytes: int
    layers: list
    wall_time_s: float | None = None
    achieved_gflops: float | None = None
    achieved_bw_gbs: float | None = None

    @property
    def arithmetic_intensity(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return self.total_flops / self.total_bytes


def _bytes_per_elem(dtype: str) -> int:
    return 2 if dtype == "fp16" else 4


def analyze_meshnet(config_path: str, dtype: str) -> ModelProfile:
    """Static FLOPs/bandwidth analysis of a MeshNet model."""
    with open(config_path) as f:
        config = json.load(f)

    bpe = _bytes_per_elem(dtype)
    spatial = 256 ** 3  # 256×256×256
    layers_profile = []
    total_flops = 0
    total_bytes = 0
    has_bias = config.get("bias", False)
    has_bnorm = config.get("bnorm", True)

    for i, layer_cfg in enumerate(config["layers"]):
        in_c = layer_cfg["in_channels"]
        out_c = layer_cfg["out_channels"]
        k = layer_cfg["kernel_size"]

        # Conv
        f = conv3d_flops(in_c, out_c, k, spatial, has_bias=has_bias)
        b = conv3d_bytes(in_c, out_c, k, spatial, bpe, has_bias=has_bias)
        layers_profile.append(LayerProfile(f"conv_{i}", f, b))
        total_flops += f
        total_bytes += b

        # GroupNorm (all layers except last)
        if has_bnorm and i < len(config["layers"]) - 1:
            gf = groupnorm_flops(out_c, spatial)
            gb = groupnorm_bytes(out_c, spatial, bpe)
            layers_profile.append(LayerProfile(f"gnorm_{i}", gf, gb))
            total_flops += gf
            total_bytes += gb

        # Activation (all layers except last)
        if i < len(config["layers"]) - 1:
            af = activation_flops(out_c * spatial)
            ab = activation_bytes(out_c * spatial, bpe)
            layers_profile.append(LayerProfile(f"act_{i}", af, ab))
            total_flops += af
            total_bytes += ab

    return ModelProfile(
        model_name="",
        dtype=dtype,
        total_flops=total_flops,
        total_bytes=total_bytes,
        layers=[asdict(lp) for lp in layers_profile],
    )


def analyze_sae(n_classes: int, dtype: str) -> ModelProfile:
    """Static FLOPs/bandwidth analysis of an SAE model.

    Uses the known architecture: 5 encoder + downsample + 5 bottleneck +
    upsample + 5 decoder + final 1×1 conv.  All convs are 3×3 with dilations
    from DILATION_SCHEDULE, 16 channels throughout (except final).
    """
    from brainchop.sae_model import DILATION_SCHEDULE, LAYER_INDICES

    bpe = _bytes_per_elem(dtype)
    ch = 16  # SAE uses 16 channels throughout
    spatial_full = 256 ** 3
    spatial_half = 128 ** 3

    layers_profile = []
    total_flops = 0
    total_bytes = 0

    for i, idx in enumerate(LAYER_INDICES):
        is_last = (i == len(LAYER_INDICES) - 1)
        is_downsample = (idx == 10)
        is_upsample = (idx == 22)

        if is_last:
            # Final 1×1 conv → n_classes
            in_c, out_c, k = ch, n_classes, 1
            spatial = spatial_full
        elif is_downsample:
            in_c, out_c, k = ch, ch, 3
            spatial = spatial_full  # input spatial
        elif is_upsample:
            in_c, out_c, k = ch, ch, 2
            spatial = spatial_half  # input spatial (output is full)
        elif idx < 10:
            # Encoder layers (full resolution)
            in_c = 1 if idx == 0 else ch
            out_c, k = ch, 3
            spatial = spatial_full
        elif idx > 22:
            # Decoder layers (full resolution)
            in_c, out_c, k = ch, ch, 3
            spatial = spatial_full
        else:
            # Bottleneck layers (half resolution)
            in_c, out_c, k = ch, ch, 3
            spatial = spatial_half

        f = conv3d_flops(in_c, out_c, k, spatial)
        b = conv3d_bytes(in_c, out_c, k, spatial, bpe)
        layers_profile.append(LayerProfile(f"layer_{idx}", f, b))
        total_flops += f
        total_bytes += b

        # SiLU activation (all except last and upsample)
        if not is_last and not is_upsample:
            out_spatial = spatial_half if is_downsample else spatial
            af = activation_flops(out_c * out_spatial)
            ab = activation_bytes(out_c * out_spatial, bpe)
            layers_profile.append(LayerProfile(f"silu_{idx}", af, ab))
            total_flops += af
            total_bytes += ab

    return ModelProfile(
        model_name="",
        dtype=dtype,
        total_flops=total_flops,
        total_bytes=total_bytes,
        layers=[asdict(lp) for lp in layers_profile],
    )


# ---------------------------------------------------------------------------
# Runtime profiling: actually run inference and measure wall time
# ---------------------------------------------------------------------------

def profile_inference(model_name: str, vol, dtype: str) -> float:
    """Run one inference pass and return wall-clock seconds."""
    from brainchop import segment

    if dtype == "fp16":
        os.environ["FP16"] = "1"
    else:
        os.environ.pop("FP16", None)

    # Warm-up (JIT compile)
    _ = segment(vol, model_name)

    # Timed run
    t0 = time.perf_counter()
    _ = segment(vol, model_name)
    t1 = time.perf_counter()

    os.environ.pop("FP16", None)
    return t1 - t0


# ---------------------------------------------------------------------------
# AMD tool wrappers
# ---------------------------------------------------------------------------

def run_uprof(script_args: list[str], output_dir: str) -> str | None:
    """Wrap the profiling run with AMDuProfCLI for CPU-side roofline data.

    Requires AMDuProfCLI on PATH (Linux, AMD Zen CPU).
    Uses 'collect --config tbp' for time-based profiling with IPC/bandwidth
    counters.
    """
    uprof = shutil.which("AMDuProfCLI")
    if uprof is None:
        print("  AMDuProfCLI not found on PATH — skipping UProf collection")
        return None

    out = Path(output_dir) / "uprof"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        uprof, "collect",
        "--config", "tbp",
        "--output-dir", str(out),
        "--", sys.executable, *script_args,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Generate report
    # Find the most recent result directory
    result_dirs = sorted(out.iterdir(), key=lambda p: p.stat().st_mtime)
    if result_dirs:
        report_cmd = [uprof, "report", "--input-dir", str(result_dirs[-1])]
        result = subprocess.run(report_cmd, capture_output=True, text=True)
        report_path = str(out / "report.txt")
        with open(report_path, "w") as f:
            f.write(result.stdout)
        print(f"  UProf report saved to {report_path}")
        return report_path
    return None


def run_rocprof(script_args: list[str], output_dir: str) -> str | None:
    """Wrap the profiling run with rocprof for GPU kernel tracing.

    Requires rocprof on PATH (Linux, ROCm).
    """
    rocprof = shutil.which("rocprof")
    if rocprof is None:
        print("  rocprof not found on PATH — skipping rocprof collection")
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = str(out / "rocprof_trace.csv")
    cmd = [
        rocprof, "--stats",
        "-o", csv_path,
        sys.executable, *script_args,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    if Path(csv_path).exists():
        print(f"  rocprof trace saved to {csv_path}")
        return csv_path
    return None


# ---------------------------------------------------------------------------
# Roofline plotting
# ---------------------------------------------------------------------------

def plot_roofline(profiles: list[ModelProfile], peak_gflops: float,
                  peak_bw_gbs: float, output_path: str):
    """Generate a roofline chart.

    Args:
        profiles: List of model profiles with computed AI and achieved GFLOP/s.
        peak_gflops: Hardware peak compute (GFLOP/s).
        peak_bw_gbs: Hardware peak memory bandwidth (GB/s).
        output_path: Where to save the PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot generation")
        print("  pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline ceiling
    ridge_point = peak_gflops / peak_bw_gbs  # FLOPs/byte at ridge
    ai_range = np.logspace(-2, 3, 500)
    roofline = np.minimum(peak_gflops, ai_range * peak_bw_gbs)
    ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline ceiling")

    # Shade regions
    ax.fill_between(ai_range, roofline, alpha=0.05, color="gray")
    ax.axvline(ridge_point, color="gray", linestyle=":", alpha=0.5)
    ax.text(ridge_point * 1.1, peak_gflops * 0.7, f"Ridge point\n({ridge_point:.1f} F/B)",
            fontsize=8, color="gray")

    # Plot each model
    markers = {"fp32": "o", "fp16": "s", "int8": "^"}
    colors = {}
    cmap = plt.cm.tab10
    model_names = list({p.model_name for p in profiles})
    for i, name in enumerate(model_names):
        colors[name] = cmap(i % 10)

    for p in profiles:
        if p.achieved_gflops is not None and p.achieved_gflops > 0:
            ai = p.arithmetic_intensity
            ax.plot(ai, p.achieved_gflops,
                    marker=markers.get(p.dtype, "o"),
                    color=colors[p.model_name],
                    markersize=10, zorder=5)
            ax.annotate(f"{p.model_name}\n({p.dtype})",
                        (ai, p.achieved_gflops),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7, color=colors[p.model_name])

    # Labels
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title(f"Roofline Analysis — brainchop iGPU Inference\n"
                 f"Peak: {peak_gflops} GFLOP/s  |  BW: {peak_bw_gbs} GB/s",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.1, peak_gflops * 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nRoofline plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Model config resolution
# ---------------------------------------------------------------------------

def _resolve_model_dir(model_name: str) -> Path:
    """Resolve model cache directory."""
    models_json = Path(__file__).resolve().parent.parent / "models.json"
    with open(models_json) as f:
        registry = json.load(f)
    if model_name not in registry:
        raise ValueError(f"Unknown model: {model_name}")
    folder = registry[model_name].get("folder", model_name)
    return Path.home() / ".cache" / "brainchop" / "models" / folder


def _get_model_info(model_name: str) -> dict:
    """Get model metadata from registry."""
    models_json = Path(__file__).resolve().parent.parent / "models.json"
    with open(models_json) as f:
        registry = json.load(f)
    return registry[model_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile brainchop models with roofline analysis")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to profile (default: all)")
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"],
                        choices=["fp32", "fp16"],
                        help="Data types to test (default: fp32 fp16)")
    parser.add_argument("--peak-gflops", type=float, default=None,
                        help="Hardware peak GFLOP/s (auto-detected if omitted)")
    parser.add_argument("--peak-bw", type=float, default=None,
                        help="Hardware peak memory bandwidth in GB/s (auto-detected if omitted)")
    parser.add_argument("--uprof", action="store_true",
                        help="Run AMD UProf collection (requires AMDuProfCLI)")
    parser.add_argument("--rocprof", action="store_true",
                        help="Run rocprof GPU kernel tracing (requires rocprof)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip roofline plot generation")
    parser.add_argument("--no-run", action="store_true",
                        help="Static analysis only — skip actual inference")
    parser.add_argument("--output-dir", default="examples",
                        help="Output directory (default: examples)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per model in seconds (default: 600)")
    args = parser.parse_args()

    from tinygrad.helpers import fetch
    from brainchop import list_models, load

    # Resolve models
    all_models = list(list_models().keys())
    models = args.models or all_models

    # Auto-detect hardware ceilings
    peak_gflops, peak_bw = _detect_hardware(args.peak_gflops, args.peak_bw)

    print("=" * 65)
    print("brainchop roofline profiler")
    print(f"  models:      {', '.join(models)}")
    print(f"  dtypes:      {', '.join(args.dtypes)}")
    print(f"  peak GFLOP/s: {peak_gflops}")
    print(f"  peak BW GB/s: {peak_bw}")
    print(f"  backend:     {_get_backend()}")
    print("=" * 65)

    # Load test volume (only needed for runtime profiling)
    vol = None
    if not args.no_run:
        test_url = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
        nifti_path = str(Path(fetch(test_url, "t1_crop.nii.gz")))
        vol = load(nifti_path)
        print(f"Test volume loaded: shape={vol.data.shape}\n")

    # Profile each model × dtype
    profiles: list[ModelProfile] = []

    for model_name in models:
        info = _get_model_info(model_name)
        model_type = info.get("type", "meshnet")
        n_classes = info.get("n_classes", 3)

        for dtype in args.dtypes:
            label = f"{model_name} ({dtype})"
            print(f"[{label}]")

            # Static analysis
            try:
                if model_type == "sae":
                    profile = analyze_sae(n_classes, dtype)
                else:
                    model_dir = _resolve_model_dir(model_name)
                    config_path = str(model_dir / "model.json")
                    if not Path(config_path).exists():
                        # Download the model files
                        print(f"  Downloading model {model_name}…")
                        from brainchop.utils import find_pth_files
                        find_pth_files(model_name)
                    if not Path(config_path).exists():
                        print(f"  Config not found at {config_path} — skipping")
                        raise FileNotFoundError(config_path)
                    profile = analyze_meshnet(config_path, dtype)

                profile.model_name = model_name
                profile.dtype = dtype

                gflops = profile.total_flops / 1e9
                gbytes = profile.total_bytes / 1e9
                ai = profile.arithmetic_intensity
                print(f"  Static:  {gflops:.2f} GFLOP  |  {gbytes:.2f} GB transferred  |  AI = {ai:.2f} F/B")

            except Exception as e:
                print(f"  Static analysis failed: {e}")
                profile = ModelProfile(model_name, dtype, 0, 0, [])

            # Runtime profiling
            if not args.no_run:
                try:
                    print(f"  Running inference (warm-up + timed)…")
                    wall_time = profile_inference(model_name, vol, dtype)
                    profile.wall_time_s = wall_time

                    if profile.total_flops > 0:
                        profile.achieved_gflops = (profile.total_flops / 1e9) / wall_time
                        profile.achieved_bw_gbs = (profile.total_bytes / 1e9) / wall_time

                    print(f"  Runtime: {wall_time:.3f}s  |  "
                          f"{profile.achieved_gflops:.1f} GFLOP/s  |  "
                          f"{profile.achieved_bw_gbs:.1f} GB/s")
                except Exception as e:
                    print(f"  Runtime profiling failed: {e}")

            profiles.append(profile)
            print()

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = str(out_dir / "roofline_results.json")

    results_data = {
        "hardware": {
            "peak_gflops": peak_gflops,
            "peak_bw_gbs": peak_bw,
            "backend": _get_backend(),
        },
        "profiles": [],
    }
    for p in profiles:
        results_data["profiles"].append({
            "model": p.model_name,
            "dtype": p.dtype,
            "total_gflops": p.total_flops / 1e9,
            "total_gb": p.total_bytes / 1e9,
            "arithmetic_intensity": p.arithmetic_intensity,
            "wall_time_s": p.wall_time_s,
            "achieved_gflops": p.achieved_gflops,
            "achieved_bw_gbs": p.achieved_bw_gbs,
            "layers": p.layers,
        })

    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Results saved to {results_path}")

    # Print summary table
    _print_summary(profiles)

    # AMD tool wrappers
    if args.uprof:
        print("\n--- AMD UProf Collection ---")
        run_uprof(
            [__file__, "--models"] + models + ["--dtypes"] + args.dtypes + ["--no-plot"],
            str(out_dir / "amd_uprof"),
        )

    if args.rocprof:
        print("\n--- rocprof GPU Kernel Trace ---")
        run_rocprof(
            [__file__, "--models"] + models + ["--dtypes"] + args.dtypes + ["--no-plot"],
            str(out_dir / "rocprof"),
        )

    # Roofline plot
    if not args.no_plot:
        plot_path = str(out_dir / "roofline_plot.png")
        plot_roofline(profiles, peak_gflops, peak_bw, plot_path)


def _get_backend() -> str:
    try:
        from tinygrad import Device
        return Device.DEFAULT
    except Exception:
        return "unknown"


def _detect_hardware(user_gflops: float | None, user_bw: float | None) -> tuple[float, float]:
    """Auto-detect hardware ceilings or use user-supplied values.

    For common iGPUs, provides reasonable defaults. Falls back to
    conservative estimates.
    """
    import platform

    gflops = user_gflops
    bw = user_bw

    if gflops is None or bw is None:
        system = platform.system()
        machine = platform.machine()

        # Try to detect GPU info
        gpu_info = _detect_gpu_info()

        if gpu_info:
            # Use detected values
            gflops = gflops or gpu_info.get("peak_gflops", 200)
            bw = bw or gpu_info.get("peak_bw_gbs", 50)
        else:
            # Conservative defaults for typical iGPUs
            if system == "Darwin" and machine == "arm64":
                # Apple Silicon — M1 ~2.6 TFLOP/s FP32, ~200 GB/s
                gflops = gflops or 2600
                bw = bw or 200
            else:
                # Generic iGPU estimate
                gflops = gflops or 500
                bw = bw or 50

        if gpu_info:
            print(f"  Detected: {gpu_info.get('name', 'unknown GPU')}")

    return gflops, bw


def _detect_gpu_info() -> dict | None:
    """Try to detect GPU name and specs."""
    import platform

    # macOS: Apple Silicon detection
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True,
            )
            cpu = result.stdout.strip()
            # Also get GPU core count via system_profiler
            result2 = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True,
            )
            gpu_line = ""
            for line in result2.stdout.splitlines():
                if "Chipset Model" in line:
                    gpu_line = line.split(":")[-1].strip()
                    break

            name = gpu_line or cpu
            # Rough estimates for Apple Silicon
            if "M1" in name:
                return {"name": name, "peak_gflops": 2600, "peak_bw_gbs": 68}
            elif "M2" in name:
                return {"name": name, "peak_gflops": 3600, "peak_bw_gbs": 100}
            elif "M3" in name:
                return {"name": name, "peak_gflops": 4100, "peak_bw_gbs": 100}
            elif "M4" in name:
                return {"name": name, "peak_gflops": 4600, "peak_bw_gbs": 120}
            else:
                return {"name": name, "peak_gflops": 2600, "peak_bw_gbs": 68}
        except Exception:
            pass

    # Linux: try lspci for AMD/Intel iGPU
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True,
            )
            for line in result.stdout.splitlines():
                if "VGA" in line or "Display" in line:
                    name = line.split(":")[-1].strip()
                    # AMD Radeon iGPU estimates
                    if "Radeon" in name and any(x in name for x in ["Vega", "680M", "780M", "890M"]):
                        if "780M" in name or "890M" in name:
                            return {"name": name, "peak_gflops": 8600, "peak_bw_gbs": 51}
                        elif "680M" in name:
                            return {"name": name, "peak_gflops": 4300, "peak_bw_gbs": 51}
                        else:
                            return {"name": name, "peak_gflops": 2000, "peak_bw_gbs": 38}
                    # Intel iGPU
                    elif "Intel" in name:
                        return {"name": name, "peak_gflops": 500, "peak_bw_gbs": 50}
                    return {"name": name}
        except Exception:
            pass

    return None


def _print_summary(profiles: list[ModelProfile]):
    """Print a summary table of results."""
    print()
    print("=" * 90)
    name_w = max((len(p.model_name) for p in profiles), default=10)
    name_w = max(name_w, 5)
    print(f"{'Model':<{name_w}}  {'dtype':<5}  {'GFLOP':>8}  {'GB xfer':>8}  "
          f"{'AI (F/B)':>8}  {'Time (s)':>8}  {'GFLOP/s':>8}  {'GB/s':>8}")
    print("-" * 90)

    for p in profiles:
        gf = f"{p.total_flops/1e9:.1f}" if p.total_flops else "—"
        gb = f"{p.total_bytes/1e9:.1f}" if p.total_bytes else "—"
        ai = f"{p.arithmetic_intensity:.2f}" if p.total_bytes else "—"
        wt = f"{p.wall_time_s:.3f}" if p.wall_time_s else "—"
        ag = f"{p.achieved_gflops:.1f}" if p.achieved_gflops else "—"
        ab = f"{p.achieved_bw_gbs:.1f}" if p.achieved_bw_gbs else "—"
        print(f"{p.model_name:<{name_w}}  {p.dtype:<5}  {gf:>8}  {gb:>8}  "
              f"{ai:>8}  {wt:>8}  {ag:>8}  {ab:>8}")

    print("=" * 90)


if __name__ == "__main__":
    main()
