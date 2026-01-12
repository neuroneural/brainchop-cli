#!/usr/bin/env python3
"""
Standalone forward pass for SAE-16 model.

Usage:
    python sae_forward.py input.nii.gz output.nii.gz
    python sae_forward.py input.nii.gz output.nii.gz --model path/to/model.pth
"""

import sys
import os

from tinygrad import Tensor
from tinygrad.nn.state import torch_load

from brainchop import load, save, Volume
from brainchop.tiny_meshnet import qnormalize, SequentialConvArgmax
from brainchop.niimath import bwlabel


# Layer indices in the fused model (odd indices are SiLU activations, no params)
LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27, 29, 31, 33]
DOWNSAMPLE_LAYER = 10  # stride=2, 256³ → 128³
UPSAMPLE_LAYER = 22    # ConvTranspose stride=2, 128³ → 256³

# Dilation schedule from modelAEgelu_dilated.json
# JSON has 16 layers: 5 encoder + 5 bottleneck + 5 decoder + 1 final
# Downsample/upsample are added by model class (not in JSON)
# padding = dilation for 3x3 kernel to maintain spatial size
DILATION_SCHEDULE = {
    # Encoder (JSON 0-4): d=1,2,4,6,8
    0: 1,
    2: 2,
    4: 4,
    6: 6,
    8: 8,
    # Downsample (not in JSON, strided conv)
    10: 1,
    # Bottleneck (JSON 5-9): d=10,12,16,12,10
    12: 10,
    14: 12,
    16: 16,  # peak
    18: 12,
    20: 10,
    # Upsample (ConvTranspose, no dilation)
    22: 1,
    # Decoder (JSON 10-14): d=8,6,4,2,1
    23: 8,
    25: 6,
    27: 4,
    29: 2,
    31: 1,
    # Final 1x1 (JSON 15)
    33: 1,
}


class SAENet:
    """
    Spatial AutoEncoder network for brain segmentation.

    Architecture:
        - Encoder: Conv3d layers with SiLU, one strided conv for downsampling
        - Bottleneck: Conv3d layers at reduced spatial resolution
        - Decoder: ConvTranspose3d for upsampling, Conv3d layers with SiLU
        - Output: 1x1x1 conv to 3 classes
    """

    def __init__(self, state_dict: dict, use_norm: bool = False):
        """Load model from state dict with numeric keys."""
        self.layers = []
        self.use_norm = use_norm

        for idx in LAYER_INDICES:
            # Move weights from DISK to compute device via numpy roundtrip
            weight = Tensor(state_dict[f"{idx}.weight"].numpy())
            bias = Tensor(state_dict[f"{idx}.bias"].numpy())

            if idx == UPSAMPLE_LAYER:
                layer_type = "convT"
            elif idx == DOWNSAMPLE_LAYER:
                layer_type = "conv_s2"
            else:
                layer_type = "conv"

            self.layers.append((layer_type, idx, weight, bias))

        # Output has 3 classes
        self.n_classes = 3
        self.seq_conv_argmax = SequentialConvArgmax(self.n_classes)

    def normalize(self, x: Tensor) -> Tensor:
        """Quantile normalization (same as MeshNet)."""
        return qnormalize(x)

    def __call__(self, x: Tensor, debug: bool = False) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, D, H, W)
            debug: If True, print norm of latents at each layer

        Returns:
            Segmentation (B, D, H, W) after argmax
        """
        if debug:
            xnp = x.numpy()
            print(f"  Input: shape={list(x.shape)}, norm={((xnp**2).sum()**0.5):.2f}, range=[{xnp.min():.3f}, {xnp.max():.3f}]")

        # Process all layers except the last (which has no activation)
        for layer_type, idx, weight, bias in self.layers[:-1]:
            w_shape = list(weight.shape)  # (out_ch, in_ch, D, H, W) or (in_ch, out_ch, D, H, W) for convT
            dilation = DILATION_SCHEDULE.get(idx, 1)
            padding = dilation  # padding = dilation for 3x3 kernel

            if layer_type == "conv":
                # Standard 3x3x3 conv with dilation
                out_ch = w_shape[0]
                op_str = f"Conv3d({w_shape[1]}→{out_ch}, k={w_shape[2]}, d={dilation}, p={padding})"
                x = x.conv2d(weight, bias, padding=padding, dilation=dilation)
            elif layer_type == "conv_s2":
                # Strided 3x3x3 conv for downsampling (with dilation)
                out_ch = w_shape[0]
                op_str = f"Conv3d({w_shape[1]}→{out_ch}, k={w_shape[2]}, s=2, d={dilation}, p={padding})"
                x = x.conv2d(weight, bias, padding=padding, stride=2, dilation=dilation)
            elif layer_type == "convT":
                # ConvTranspose for upsampling (2x2x2 kernel, stride 2, no dilation)
                # ConvTranspose weight shape is (in_ch, out_ch, D, H, W)
                out_ch = w_shape[1]
                op_str = f"ConvT3d({w_shape[0]}→{out_ch}, k={w_shape[2]}, s=2, p=0)"
                x = x.conv_transpose2d(weight, bias, stride=2, padding=0)

            # Add BatchNorm (compute stats from current batch)
            if self.use_norm:
                # x: (B, C, D, H, W) - normalize per channel
                mean = x.mean(axis=(0, 2, 3, 4), keepdim=True)
                var = ((x - mean) ** 2).mean(axis=(0, 2, 3, 4), keepdim=True)
                x = (x - mean) / (var + 1e-5).sqrt()
                op_str += " + BN"

            # SiLU activation after each conv (except output and ConvTranspose)
            # ConvTranspose (upsample) has no activation - goes directly to next conv
            if layer_type != "convT":
                x = x.silu()

            if debug:
                x = x.realize()
                xnp = x.numpy()
                act_str = "" if layer_type == "convT" else " + SiLU"
                print(f"  Layer {idx:2d}: {op_str}{act_str} → shape={list(x.shape)}, norm={((xnp**2).sum()**0.5):.2e}, range=[{xnp.min():.2f}, {xnp.max():.2f}]")

        # Final layer: 1x1x1 conv, no activation, no padding
        _, idx, weight, bias = self.layers[-1]
        w_shape = list(weight.shape)
        kernel_size = weight.shape[2]  # Get kernel size from weight shape
        padding = (kernel_size - 1) // 2  # 0 for 1x1, 1 for 3x3
        op_str = f"Conv3d({w_shape[1]}→{w_shape[0]}, k={w_shape[2]}, s=1, p={padding})"
        x = x.conv2d(weight, bias, padding=padding)

        if debug:
            x = x.realize()
            xnp = x.numpy()
            print(f"  Layer {idx:2d}: {op_str} → shape={list(x.shape)}, norm={((xnp**2).sum()**0.5):.2e}, range=[{xnp.min():.2f}, {xnp.max():.2f}]")
            # Per-class stats
            for c in range(xnp.shape[1]):
                print(f"    Class {c}: mean={xnp[0,c].mean():.2f}, std={xnp[0,c].std():.2f}")

        # Memory-efficient argmax
        return self.seq_conv_argmax(x)


def load_sae(model_path: str = "model_sae_16_fused.pth", use_norm: bool = False) -> SAENet:
    """Load SAENet model from .pth file."""
    state_dict = torch_load(model_path)
    return SAENet(state_dict, use_norm=use_norm)


def main():
    if len(sys.argv) < 3:
        print("Usage: python sae_forward.py input.nii.gz output.nii.gz [--model path/to/model.pth] [--debug] [--norm]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Optional model path
    model_path = "model_sae_16_fused.pth"
    if "--model" in sys.argv:
        model_idx = sys.argv.index("--model")
        model_path = sys.argv[model_idx + 1]

    # Debug mode
    debug = "--debug" in sys.argv

    # Add BatchNorm after each conv
    use_norm = "--norm" in sys.argv

    print(f"Loading model from {model_path}...")
    if use_norm:
        print("Using BatchNorm after each conv layer")
    model = load_sae(model_path, use_norm=use_norm)

    print(f"Loading input from {input_path}...")
    vol = load(input_path)

    # Preprocess: (X, Y, Z) -> (1, 1, D, H, W)
    x = vol.data.permute(2, 1, 0).cast("float32").rearrange("... -> 1 1 ...")
    x = model.normalize(x)

    print(f"Running inference (input shape: {x.shape})...")
    out = model(x, debug=debug)
    print(f"Output shape: {out.shape}")

    # Postprocess: (B, D, H, W) -> (X, Y, Z)
    out = out[0].permute(2, 1, 0).cast("uint8")
    out_np = out.numpy()

    # Connected components labeling (skip if single class)
    unique_classes = len(set(out_np.flatten()))
    if unique_classes > 1:
        out_np, _ = bwlabel(vol.header, out_np)
    else:
        print(f"Warning: Output has only {unique_classes} class(es), skipping bwlabel")

    print(f"Saving output to {output_path}...")
    save(Volume(Tensor(out_np), vol.header), output_path)

    print("Done!")


if __name__ == "__main__":
    main()
