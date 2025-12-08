"""
Convert tfjs .bin weights to PyTorch .pth format for brainchop models.

This script reads tfjs model.json + model.bin and outputs a model.pth file
that can be used with the tiny_meshnet backend.
"""
import os
import json
import numpy as np
from pathlib import Path
from functools import reduce
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import safe_save, get_state_dict


def load_tfjs_weights(json_path: str, bin_path: str):
    """Load tfjs model spec and weights."""
    with open(json_path, "r") as f:
        model_spec = json.load(f)
    with open(bin_path, "rb") as f:
        weights_data = np.frombuffer(f.read(), dtype=np.float32).copy()
    return model_spec, weights_data


def parse_tfjs_layers(model_spec):
    """Extract layer configs from tfjs model spec."""
    layers = model_spec["modelTopology"]["model_config"]["config"]["layers"][1:]  # Skip InputLayer
    conv_layers = []

    for layer in layers:
        if layer["class_name"] == "Conv3D":
            config = layer["config"]
            conv_layers.append({
                "name": config["name"],
                "filters": config["filters"],
                "kernel_size": config["kernel_size"][0],  # Assuming cubic kernels
                "strides": config["strides"][0],
                "dilation_rate": config["dilation_rate"][0],
                "use_bias": config["use_bias"],
            })

    return conv_layers


class MeshNetForConversion:
    """MeshNet model structure that mirrors tiny_meshnet for weight loading.

    Uses self.model as a list with ReLU activations interleaved to match
    the expected state_dict keys: model.0, model.2, model.4, etc.
    """

    def __init__(self, conv_configs):
        self.model = []
        in_channels = 1

        for i, cfg in enumerate(conv_configs):
            out_channels = cfg["filters"]
            kernel_size = cfg["kernel_size"]
            dilation = cfg["dilation_rate"]
            stride = cfg["strides"]

            # Calculate padding for 'same' behavior
            padding = (kernel_size - 1) * dilation // 2

            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=tuple([kernel_size] * 3),
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=cfg["use_bias"],
            )
            self.model.append(conv)
            # Add ReLU placeholder for all but the last layer
            # This ensures state_dict indices match: model.0, model.2, model.4...
            if i < len(conv_configs) - 1:
                self.model.append(lambda x: x.relu())
            in_channels = out_channels

    def __call__(self, x):
        for layer in self.model:
            x = layer(x)
        return x


def convert_tfjs_to_tinygrad(json_path: str, bin_path: str, output_path: str):
    """
    Convert tfjs model to tinygrad pth format.

    The tfjs weights are stored in NDHWC format (channels last) as [D, H, W, C_in, C_out]
    Tinygrad expects NCDHW format (channels first) as [C_out, C_in, D, H, W]
    """
    print(f"Loading tfjs model from {json_path} and {bin_path}")
    model_spec, weights_data = load_tfjs_weights(json_path, bin_path)
    conv_configs = parse_tfjs_layers(model_spec)

    print(f"Found {len(conv_configs)} conv layers:")
    for i, cfg in enumerate(conv_configs):
        print(f"  {i}: {cfg['name']} - filters={cfg['filters']}, kernel={cfg['kernel_size']}, dilation={cfg['dilation_rate']}")

    # Create model structure
    model = MeshNetForConversion(conv_configs)

    # Load weights into model
    # model.model contains interleaved conv and relu: [conv0, relu, conv1, relu, ..., convN]
    # So conv layers are at indices 0, 2, 4, 6, ...
    weight_index = 0
    in_channels = 1
    conv_idx = 0

    for i, cfg in enumerate(conv_configs):
        # Get the conv layer from model.model (at even indices)
        model_idx = i * 2 if i < len(conv_configs) - 1 else len(model.model) - 1
        conv = model.model[model_idx]

        out_channels = cfg["filters"]
        k = cfg["kernel_size"]

        # tfjs weight shape: [k, k, k, in_channels, out_channels]
        weight_shape_tfjs = [k, k, k, in_channels, out_channels]
        weight_size = reduce(lambda a, b: a * b, weight_shape_tfjs)

        # Extract weight
        weight_flat = weights_data[weight_index:weight_index + weight_size]
        weight_tfjs = weight_flat.reshape(weight_shape_tfjs)
        weight_index += weight_size

        # Transpose from [D, H, W, C_in, C_out] to [C_out, C_in, D, H, W]
        weight_tinygrad = np.transpose(weight_tfjs, (4, 3, 0, 1, 2))

        # Set conv weight
        conv.weight = Tensor(weight_tinygrad.astype(np.float32))

        # Extract and set bias if present
        if cfg["use_bias"]:
            bias_size = out_channels
            bias = weights_data[weight_index:weight_index + bias_size]
            weight_index += bias_size
            conv.bias = Tensor(bias.astype(np.float32))

        in_channels = out_channels
        print(f"  Loaded layer {i} (model.{model_idx}): weight shape {weight_tinygrad.shape}, bias shape {bias.shape if cfg['use_bias'] else 'None'}")

    print(f"Total weights loaded: {weight_index} / {len(weights_data)}")

    # Get state dict and save
    state_dict = get_state_dict(model)
    print(f"State dict keys: {list(state_dict.keys())}")

    # Save as safetensors (pth format)
    safe_save(state_dict, output_path)
    print(f"Saved converted model to {output_path}")

    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert tfjs model to pth format")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory containing model.json and model.bin")
    parser.add_argument("--output", type=str, default=None, help="Output path for model.pth")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    json_path = model_dir / "model.json"
    bin_path = model_dir / "model.bin"
    output_path = args.output or (model_dir / "model.pth")

    if not json_path.exists():
        raise FileNotFoundError(f"model.json not found in {model_dir}")
    if not bin_path.exists():
        raise FileNotFoundError(f"model.bin not found in {model_dir}")

    convert_tfjs_to_tinygrad(str(json_path), str(bin_path), str(output_path))


if __name__ == "__main__":
    main()
