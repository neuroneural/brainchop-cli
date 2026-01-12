"""
Flip ensemble segmentation script using brainchop native API.

Usage:
    python flip_ensemble.py <input> <output> <model> [--flip] [--debug]

Examples:
    python flip_ensemble.py brain.nii.gz out.nii.gz tissue_fast
    python flip_ensemble.py brain.nii.gz out.nii.gz subcortical --flip
    python flip_ensemble.py brain.nii.gz out.nii.gz tissue_fast --flip --debug
"""

import sys
from brainchop import load, save, Volume
from brainchop.api import _load_model
from brainchop.tiny_meshnet import chunked_conv, qnormalize
from brainchop.niimath import bwlabel
from tinygrad.tensor import Tensor
from tinygrad import nn


def forward_no_argmax(model, x: Tensor) -> Tensor:
    """Forward pass through MeshNet WITHOUT the final argmax."""
    for layer in model.model:
        if isinstance(layer, nn.Conv2d):
            x = chunked_conv(x, layer)
        else:
            x = layer(x)
    return x


def export_debug_volume(tensor: Tensor, header: bytes, path: str, name: str) -> None:
    """Export a debug tensor as a NIfTI volume."""
    # Convert from (D,H,W) or (C,D,H,W) to (X,Y,Z) format
    if len(tensor.shape) == 3:  # (D,H,W)
        output = tensor.permute(2, 1, 0).cast("uint8")
    elif len(tensor.shape) == 4:  # (C,D,H,W)
        # Take argmax across channels for visualization
        output = tensor.argmax(axis=0).permute(2, 1, 0).cast("uint8")
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    out_np, _ = bwlabel(header, output.numpy())
    result = Volume(Tensor(out_np), header)
    save(result, path)
    print(f"Debug: Saved {name} to {path}")


def main():
    args = sys.argv[1:]
    use_flip = "--flip" in args
    use_debug = "--debug" in args
    args = [a for a in args if a not in ["--flip", "--debug"]]

    input_path = args[0] if len(args) > 0 else "examples/brain.nii.gz"
    output_path = args[1] if len(args) > 1 else "output.nii.gz"
    model_name = args[2] if len(args) > 2 else "tissue_fast"

    print(f"Loading {input_path}...")
    volume = load(input_path)

    # Convert to tinygrad format following native API pattern
    data = volume.data.permute(2, 1, 0).cast("float32")  # (D, H, W)
    header = volume.header

    if use_flip:
        print("Using flip ensemble...")
        flipped = data.flip(2)
        data = data.reshape(1, 1, 256, 256, 256)
        flipped = flipped.reshape(1, 1, 256, 256, 256)
        batched = Tensor.cat(data, flipped, dim=0)  # (2, 1, 256, 256, 256)

        # Debug: Export post-flip inputs
        if use_debug:
            base_name = output_path.replace(".nii.gz", "").replace(".nii", "")
            export_debug_volume(
                data.squeeze().squeeze(),
                header,
                f"{base_name}_input_original.nii.gz",
                "original input",
            )
            export_debug_volume(
                flipped.squeeze().squeeze(),
                header,
                f"{base_name}_input_flipped.nii.gz",
                "flipped input",
            )
    else:
        print("Using vanilla (no flip)...")
        batched = data.reshape(1, 1, 256, 256, 256)  # (1, 1, 256, 256, 256)

        # Debug: Export single input
        if use_debug:
            base_name = output_path.replace(".nii.gz", "").replace(".nii", "")
            export_debug_volume(data, header, f"{base_name}_input.nii.gz", "input")

    print(f"Batched shape: {batched.shape}")

    print(f"Loading {model_name} model...")
    model = _load_model(model_name)

    batched = qnormalize(batched)

    print("Running forward pass...")
    raw_output = forward_no_argmax(model, batched).realize()
    print(f"Raw output shape: {raw_output.shape}")

    # Debug: Export pre-argmax individual outputs
    if use_debug:
        base_name = output_path.replace(".nii.gz", "").replace(".nii", "")
        if use_flip:
            # Export individual outputs before combining
            export_debug_volume(
                raw_output[0], header, f"{base_name}_output_original.nii.gz", "original output"
            )
            export_debug_volume(
                raw_output[1], header, f"{base_name}_output_flipped.nii.gz", "flipped output"
            )
        else:
            # Export single output
            export_debug_volume(raw_output[0], header, f"{base_name}_output.nii.gz", "output")

    if use_flip:
        # Split, unflip, sum
        original_out = raw_output[0:1]
        flipped_out = raw_output[1:2]
        unflipped_out = flipped_out.flip(4)
        combined = Tensor.cat(original_out, unflipped_out, dim=0)
        summed = combined.sum(axis=0)  # (C, D, H, W)
        segmentation = summed.argmax(axis=0)  # (D, H, W)
    else:
        segmentation = raw_output[0].argmax(axis=0)  # (D, H, W)

    print(f"Segmentation shape: {segmentation.shape}")

    # Convert back to Volume format using native API pattern
    output = segmentation.permute(2, 1, 0).cast("uint8")
    out_np, _ = bwlabel(header, output.numpy())

    result = Volume(Tensor(out_np), header)
    save(result, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
