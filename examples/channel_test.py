#!/usr/bin/env python3
"""
Test custom MeshNet models with configurable channels on Metal vs WebGPU backends.

This script creates a random MeshNet model with configurable channel count and
runs inference on both Metal and WebGPU backends to compare outputs.

Usage:
    python examples/channel_test.py --channels 50 --size 64
    python examples/channel_test.py --channels 30 --size 32 --num-layers 7
    python examples/channel_test.py --channels 100 --size 64 --fp16

Single backend mode (used internally):
    DEVICE=METAL python examples/channel_test.py --channels 50 --size 32 --single-backend
"""

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

# Ensure we can import brainchop
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ComparisonResult:
    match: bool
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    metal_stats: Dict[str, float]
    webgpu_stats: Dict[str, float]


def compute_stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for an array."""
    flat = arr.flatten().astype(np.float64)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "nan_count": int(np.sum(np.isnan(flat))),
        "inf_count": int(np.sum(np.isinf(flat))),
    }


def compare_outputs(
    metal_out: np.ndarray,
    webgpu_out: np.ndarray,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> ComparisonResult:
    """Compare two arrays and return comparison statistics."""
    if metal_out.shape != webgpu_out.shape:
        raise ValueError(
            f"Shape mismatch: Metal {metal_out.shape} vs WebGPU {webgpu_out.shape}"
        )

    metal_flat = metal_out.flatten().astype(np.float64)
    webgpu_flat = webgpu_out.flatten().astype(np.float64)

    abs_diff = np.abs(metal_flat - webgpu_flat)
    denom = np.maximum(np.abs(metal_flat), np.abs(webgpu_flat))
    denom = np.where(denom == 0, 1.0, denom)
    rel_diff = abs_diff / denom

    is_close = np.allclose(metal_flat, webgpu_flat, atol=atol, rtol=rtol)

    return ComparisonResult(
        match=is_close,
        max_abs_diff=float(np.max(abs_diff)),
        mean_abs_diff=float(np.mean(abs_diff)),
        max_rel_diff=float(np.max(rel_diff)),
        metal_stats=compute_stats(metal_out),
        webgpu_stats=compute_stats(webgpu_out),
    )


class RandomMeshNet:
    """
    A MeshNet-style model with random weights and configurable channels.

    Architecture follows the dilated convolution pattern from:
    https://arxiv.org/pdf/1612.00940.pdf
    """

    def __init__(
        self,
        channels: int = 50,
        num_layers: int = 7,
        out_classes: int = 2,
        seed: int = 42,
    ):
        from tinygrad import nn
        from tinygrad.tensor import Tensor

        np.random.seed(seed)
        Tensor.manual_seed(seed)

        self.channels = channels
        self.num_layers = num_layers
        self.out_classes = out_classes
        self.layers: List = []

        # Dilations pattern: 1, 2, 4, 8, 16, 8, 4, 2, 1 (symmetric)
        dilations = self._compute_dilations(num_layers)

        # First layer: 1 input channel -> channels
        self.layers.append(
            nn.Conv2d(
                1,
                channels,
                kernel_size=(3, 3, 3),
                padding=(dilations[0], dilations[0], dilations[0]),
                stride=(1, 1, 1),
                dilation=(dilations[0], dilations[0], dilations[0]),
                bias=True,
            )
        )

        # Hidden layers: channels -> channels with varying dilation
        for i in range(1, num_layers):
            self.layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(3, 3, 3),
                    padding=(dilations[i], dilations[i], dilations[i]),
                    stride=(1, 1, 1),
                    dilation=(dilations[i], dilations[i], dilations[i]),
                    bias=True,
                )
            )

        # Output layer: channels -> out_classes (1x1x1 conv)
        self.output_conv = nn.Conv2d(
            channels,
            out_classes,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            bias=True,
        )

        # Initialize with random weights
        self._init_random_weights(seed)

    def _compute_dilations(self, num_layers: int) -> List[int]:
        """Compute symmetric dilation pattern for MeshNet."""
        if num_layers <= 1:
            return [1]

        # Build up: 1, 2, 4, 8, ... until midpoint
        half = (num_layers + 1) // 2
        up = [2**i for i in range(half)]

        # Build down symmetrically
        if num_layers % 2 == 0:
            # Even: mirror exactly, e.g., [1,2,4,4,2,1] for 6 layers
            down = up[::-1]
        else:
            # Odd: mirror without repeating peak, e.g., [1,2,4,2,1] for 5 layers
            down = up[-2::-1]

        result = up + down
        # Ensure we have exactly num_layers dilations
        return result[:num_layers]

    def _init_random_weights(self, seed: int):
        """Initialize all weights with random values (Xavier-like)."""
        from tinygrad.tensor import Tensor

        np.random.seed(seed)

        for layer in self.layers:
            in_ch = int(layer.weight.shape[1])
            out_ch = int(layer.weight.shape[0])
            k = int(layer.weight.shape[2])

            # Xavier initialization - use float32 explicitly
            std = float(np.sqrt(2.0 / (in_ch * k * k * k + out_ch * k * k * k)))
            w = (np.random.randn(*layer.weight.shape) * std).astype(np.float32)
            b = np.zeros(layer.bias.shape, dtype=np.float32)

            layer.weight = Tensor(w)
            layer.bias = Tensor(b)

        # Output layer
        in_ch = int(self.output_conv.weight.shape[1])
        out_ch = int(self.output_conv.weight.shape[0])
        std = float(np.sqrt(2.0 / (in_ch + out_ch)))
        w = (np.random.randn(*self.output_conv.weight.shape) * std).astype(np.float32)
        b = np.zeros(self.output_conv.bias.shape, dtype=np.float32)

        self.output_conv.weight = Tensor(w)
        self.output_conv.bias = Tensor(b)

    def __call__(self, x):
        """Forward pass with ELU activations."""
        for layer in self.layers:
            x = layer(x).elu()
        x = self.output_conv(x)
        return x

    def half(self):
        """Convert all weights to float16."""
        for layer in self.layers:
            layer.weight = layer.weight.half().realize()
            layer.bias = layer.bias.half().realize()
        self.output_conv.weight = self.output_conv.weight.half().realize()
        self.output_conv.bias = self.output_conv.bias.half().realize()
        return self


def run_single_backend(
    channels: int,
    num_layers: int,
    out_classes: int,
    input_path: str,
    output_path: str,
    use_fp16: bool = False,
    seed: int = 42,
):
    """Run model on current backend (set via BACKEND=1 env var) and save output."""
    from tinygrad.tensor import Tensor, Device

    print(f"Running on {Device.DEFAULT} backend")
    print(f"Channels: {channels}, Layers: {num_layers}, Classes: {out_classes}")
    print(f"FP16: {use_fp16}")

    # Load input
    input_data = np.load(input_path)
    print(f"Input shape: {input_data.shape}")

    # Create model
    model = RandomMeshNet(
        channels=channels,
        num_layers=num_layers,
        out_classes=out_classes,
        seed=seed,
    )

    # Create input tensor
    input_tensor = Tensor(input_data)
    if use_fp16:
        input_tensor = input_tensor.half()
        model = model.half()
    input_tensor = input_tensor.realize()

    # Run inference
    output = model(input_tensor).realize()
    output_np = output.numpy()

    print(f"Output shape: {output_np.shape}")
    stats = compute_stats(output_np)
    print(f"Output stats: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")

    # Save output
    np.save(output_path, output_np)
    print(f"Saved output to {output_path}")


def run_on_backend_subprocess(
    backend: str,
    channels: int,
    num_layers: int,
    out_classes: int,
    input_path: str,
    output_path: str,
    use_fp16: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Run model on specified backend using subprocess and return output."""
    env = os.environ.copy()
    # tinygrad uses BACKEND=1 pattern (e.g., METAL=1, WEBGPU=1)
    # Clear any existing backend env vars first
    for key in ["METAL", "WEBGPU", "CPU", "CUDA", "CL"]:
        env.pop(key, None)
    env[backend] = "1"

    cmd = [
        sys.executable,
        __file__,
        "--channels", str(channels),
        "--num-layers", str(num_layers),
        "--out-classes", str(out_classes),
        "--size", "0",  # Not used in single-backend mode
        "--seed", str(seed),
        "--single-backend",
        "--input-path", input_path,
        "--output-path", output_path,
    ]
    if use_fp16:
        cmd.append("--fp16")

    print(f"\n{'='*60}")
    print(f"Running on {backend} backend")
    print(f"{'='*60}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"{backend} backend failed with code {result.returncode}")

    return np.load(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Test custom MeshNet with configurable channels on Metal vs WebGPU"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=50,
        help="Number of hidden channels (default: 50)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=7,
        help="Number of conv layers (default: 7)",
    )
    parser.add_argument(
        "--out-classes",
        type=int,
        default=2,
        help="Number of output classes (default: 2)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Input volume size NxNxN (default: 64)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison (default: 1e-4)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison (default: 1e-3)",
    )
    # Internal args for single-backend mode
    parser.add_argument("--single-backend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-path", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Single backend mode - run inference and exit
    if args.single_backend:
        run_single_backend(
            args.channels,
            args.num_layers,
            args.out_classes,
            args.input_path,
            args.output_path,
            args.fp16,
            args.seed,
        )
        return 0

    # Main comparison mode
    print(f"\nChannel Test: Metal vs WebGPU Backend Comparison")
    print(f"=" * 60)
    print(f"Configuration:")
    print(f"  Channels: {args.channels}")
    print(f"  Layers: {args.num_layers}")
    print(f"  Output classes: {args.out_classes}")
    print(f"  Input size: {args.size}x{args.size}x{args.size}")
    print(f"  FP16: {args.fp16}")
    print(f"  Seed: {args.seed}")

    # Create deterministic input
    np.random.seed(args.seed)
    input_data = np.random.randn(1, 1, args.size, args.size, args.size).astype(
        np.float32
    )
    # Normalize to [0, 1] range
    input_data = (input_data - input_data.min()) / (
        input_data.max() - input_data.min() + 1e-8
    )

    print(f"\nInput shape: {input_data.shape}")
    print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")

    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.npy")
        np.save(input_path, input_data)

        # Run on Metal
        try:
            metal_output_path = os.path.join(tmpdir, "metal_output.npy")
            results["metal"] = run_on_backend_subprocess(
                "METAL",
                args.channels,
                args.num_layers,
                args.out_classes,
                input_path,
                metal_output_path,
                args.fp16,
                args.seed,
            )
        except Exception as e:
            print(f"Metal backend failed: {e}")
            import traceback
            traceback.print_exc()

        # Run on WebGPU
        try:
            webgpu_output_path = os.path.join(tmpdir, "webgpu_output.npy")
            results["webgpu"] = run_on_backend_subprocess(
                "WEBGPU",
                args.channels,
                args.num_layers,
                args.out_classes,
                input_path,
                webgpu_output_path,
                args.fp16,
                args.seed,
            )
        except Exception as e:
            print(f"WebGPU backend failed: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if "metal" in results and "webgpu" in results:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")

        comparison = compare_outputs(
            results["metal"], results["webgpu"], atol=args.atol, rtol=args.rtol
        )

        status = "PASS" if comparison.match else "FAIL"
        print(f"\nStatus: {status}")
        print(f"Max absolute difference: {comparison.max_abs_diff:.6e}")
        print(f"Mean absolute difference: {comparison.mean_abs_diff:.6e}")
        print(f"Max relative difference: {comparison.max_rel_diff:.6e}")

        if not comparison.match:
            print(f"\nMetal stats:")
            print(f"  min={comparison.metal_stats['min']:.6f}")
            print(f"  max={comparison.metal_stats['max']:.6f}")
            print(f"  mean={comparison.metal_stats['mean']:.6f}")
            print(f"  NaN count={comparison.metal_stats['nan_count']}")
            print(f"  Inf count={comparison.metal_stats['inf_count']}")

            print(f"\nWebGPU stats:")
            print(f"  min={comparison.webgpu_stats['min']:.6f}")
            print(f"  max={comparison.webgpu_stats['max']:.6f}")
            print(f"  mean={comparison.webgpu_stats['mean']:.6f}")
            print(f"  NaN count={comparison.webgpu_stats['nan_count']}")
            print(f"  Inf count={comparison.webgpu_stats['inf_count']}")

            # Find where differences are largest
            diff = np.abs(results["metal"] - results["webgpu"])
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"\nWorst mismatch at index {max_idx}:")
            print(f"  Metal: {results['metal'][max_idx]:.6f}")
            print(f"  WebGPU: {results['webgpu'][max_idx]:.6f}")
            print(f"  Diff: {diff[max_idx]:.6e}")

        return 0 if comparison.match else 1
    else:
        print("\nCould not compare - one or both backends failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
