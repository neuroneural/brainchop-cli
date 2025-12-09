import os
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.state import torch_load, safe_load, load_state_dict
import json
import numpy as np


def convert_keys(torch_state_dict, tiny_state_dict):
    torch_keys = torch_state_dict.keys()
    tiny_keys = tiny_state_dict.keys()
    new_dict = {}
    for f, t in zip(torch_keys, tiny_keys):
        new_dict[t] = torch_state_dict[f]
    return new_dict

def sequential_argmax(x: Tensor) -> Tensor:
    """
    Sequential argmax over channel dimension with eager evaluation.

    NOTE: This function receives an already-materialized tensor, so it does NOT
    save memory. For true memory savings, use SequentialConvArgmax which integrates
    the final conv layer with argmax to avoid materializing all channels at once.
    """
    print('using sequential argmax on new backend')
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    depth, height, width = x.shape[2], x.shape[3], x.shape[4]

    # outB tracks max values seen so far
    # outC tracks indices of the max channel
    outB = Tensor.full((batch_size, 1, depth, height, width), -1e9)
    outC = Tensor.zeros(batch_size, 1, depth, height, width)
    for i in range(int(num_channels)):
        # Extract i-th channel
        outA = x[:, i:i+1, :, :, :].realize()
        # Find where current channel > max so far
        greater = (outA > outB).float().realize()
        # Update max values
        outB = ((1 - greater) * outB + greater * outA).realize()
        # Update argmax indices
        outC = ((1 - greater) * outC + greater * float(i)).realize()
    return outC.squeeze(1)


class SequentialConvArgmax:
    """
    Replaces final conv + argmax with sequential per-channel convs.

    Instead of: Conv3d(in, 104, 1x1x1) -> argmax
    This does:  104 x Conv3d(in, 1, 1x1x1) with streaming argmax

    This avoids materializing the full (batch, 104, D, H, W) tensor,
    only keeping (batch, 1, D, H, W) for max values and indices.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.out_channels = out_channels
        self.convs = [
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1, 1), bias=False)
            for _ in range(out_channels)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        depth, height, width = x.shape[2], x.shape[3], x.shape[4]

        outB = Tensor.full((batch_size, 1, depth, height, width), -1e9)
        outC = Tensor.zeros(batch_size, 1, depth, height, width)

        for i, conv in enumerate(self.convs):
            outA = conv(x).realize()
            greater = (outA > outB).float().realize()
            outB = ((1 - greater) * outB + greater * outA).realize()
            outC = ((1 - greater) * outC + greater * float(i)).realize()

        return outC.squeeze(1)

    def load_from_conv(self, conv: nn.Conv2d):
        """Load weights from a standard Conv3d layer, splitting across per-channel convs."""
        # conv.weight shape: (out_channels, in_channels, 1, 1, 1)
        for i, c in enumerate(self.convs):
            c.weight = conv.weight[i:i+1].realize()

    def half(self):
        for conv in self.convs:
            conv.weight = conv.weight.half().realize()
        return self


def qnormalize(img: Tensor, qmin=0.02, qmax=0.98, eps=1e-3) -> Tensor:
    """Unit interval preprocessing with clipping and safe division for bf16"""
    img = img.numpy()
    qlow = np.quantile(img, qmin) #type:ignore . numpy api bad
    qhigh = np.quantile(img, qmax) #type:ignore
    img = (img - qlow) / (qhigh - qlow + eps)
    img = np.clip(img, 0, 1) #type:ignore
    return Tensor(img)


def set_channel_num(config, in_channels, n_classes, channels):
    # input layer
    config["layers"][0]["in_channels"] = in_channels
    config["layers"][0]["out_channels"] = channels
    # output layer
    config["layers"][-1]["in_channels"] = channels
    config["layers"][-1]["out_channels"] = n_classes
    # hidden layers
    for layer in config["layers"][1:-1]:
        layer["in_channels"] = layer["out_channels"] = channels
    return config


def construct_layer(dropout_p=0, bnorm=True, gelu=False, elu=False, *args, **kwargs):
    layers = []
    kwargs["kernel_size"] = [kwargs["kernel_size"]] * 3
    layers.append(nn.Conv2d(*args, **kwargs))
    if bnorm:
        layers.append(
            nn.GroupNorm(
                num_groups=kwargs["out_channels"],
                num_channels=kwargs["out_channels"],
                affine=False,
            )
        )

    relu_fn = lambda x: x.relu()
    gelu_fn = lambda x: x.gelu()
    elu_fn = lambda x: x.elu()
    dropout_fn = lambda x: x.dropout(dropout_p)

    if elu:
        layers.append(elu_fn)
    elif gelu:
        layers.append(gelu_fn)
    else:
        layers.append(relu_fn)

    if dropout_p > 0:
        layers.append(dropout_fn)
    return layers


class MeshNet:
    """Configurable MeshNet from https://arxiv.org/pdf/1612.00940.pdf"""

    def __init__(self, in_channels, n_classes, channels, config_file, fat=None):
        """Init"""
        with open(config_file, "r") as f:
            config = set_channel_num(json.load(f), in_channels, n_classes, channels)
        if fat is not None:
            chn = int(channels * 1.5)
            if fat in {"i", "io"}:
                config["layers"][0]["out_channels"] = chn
                config["layers"][1]["in_channels"] = chn
            if fat == "io":
                config["layers"][-1]["in_channels"] = chn
                config["layers"][-2]["out_channels"] = chn
            if fat == "b":
                config["layers"][3]["out_channels"] = chn
                config["layers"][4]["in_channels"] = chn

        self.model = []
        # Check if config specifies bias (default False for backward compat)
        use_bias = config.get("bias", False)

        for block_kwargs in config["layers"][:-1]:  # All but the last layer
            self.model.extend(
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config.get("gelu", False),
                    elu=config.get("elu", False),
                    **{**block_kwargs, "bias": use_bias},
                )
            )

        # Handle last layer specially - add it to model list
        last_config = config["layers"][-1]
        self.model.append(
            nn.Conv2d(
                last_config["in_channels"],
                last_config["out_channels"],
                kernel_size=tuple([last_config["kernel_size"]] * 3),
                padding=last_config["padding"],
                stride=last_config["stride"],
                dilation=last_config["dilation"],
                bias=use_bias,
            )
        )

        self.n_classes = last_config["out_channels"]
        self.seq_conv_argmax = None

    def init_seq_conv_argmax(self):
        """Initialize SequentialConvArgmax with identity weights for PREARGMAX path."""
        self.seq_conv_argmax = SequentialConvArgmax(self.n_classes, self.n_classes)
        for i, conv in enumerate(self.seq_conv_argmax.convs):
            w = np.zeros((1, self.n_classes, 1, 1, 1), dtype=np.float32)
            w[0, i, 0, 0, 0] = 1.0
            conv.weight = Tensor(w)

    def normalize(self, x):
        return qnormalize(x) # TODO: interpret normalization from config file

    def __call__(self, x):
        for layer in self.model:
            x = layer(x)
        if 'PREARGMAX' in os.environ:
            x = self.seq_conv_argmax(x)
        return x

    def half(self):
        """Convert all weights to float16/half precision"""
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                layer.weight = layer.weight.half().realize()
                if layer.bias is not None:
                    layer.bias = layer.bias.half().realize()
            elif isinstance(layer, nn.GroupNorm):
                if layer.weight is not None:
                    layer.weight = layer.weight.half().realize()
                if layer.bias is not None:
                    layer.bias = layer.bias.half().realize()
        return self

def load_meshnet(
    config_fn: str,
    model_fn: str,
    in_channels: int = 1,
    channels: int = 15,
    out_channels: int = 2,
):
    # Read config to check if it has explicit channel values
    with open(config_fn, "r") as f:
        config = json.load(f)

    # Check if config uses -1 placeholders (old style) or explicit values (new style)
    # Old style: in_channels=-1, out_channels=-1 for first/last, values get overridden
    # New style: all values are explicit, no -1 placeholders
    uses_placeholders = (
        config["layers"][0]["in_channels"] == -1 or
        config["layers"][-1]["out_channels"] == -1
    )

    if not uses_placeholders:
        # New style config with explicit values - read from config
        in_channels = config["layers"][0]["in_channels"]
        channels = config["layers"][0]["out_channels"]
        out_channels = config["layers"][-1]["out_channels"]

    model = MeshNet(
        in_channels=in_channels,
        n_classes=out_channels,
        channels=channels,
        config_file=config_fn,
    )
    # Try safetensors first, fall back to torch format
    try:
        state_dict = safe_load(model_fn)
    except Exception:
        state_dict = torch_load(model_fn)
        state_dict = convert_keys(state_dict, nn.state.get_state_dict(model))
    load_state_dict(model, state_dict, strict=True, verbose=False)
    # Initialize SequentialConvArgmax for PREARGMAX path (after loading weights)
    if 'PREARGMAX' in os.environ:
        model.init_seq_conv_argmax()
    # Convert to half precision if FP16 env var is set
    if os.environ.get("FP16"):
        model = model.half()
    return model


if __name__ == "__main__":
    # TODO @spikedoanz: load default meshnet in this snippet
    pass
