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


    def normalize(self, x):
        return qnormalize(x) # TODO: interpret normalization from config file

    def __call__(self, x):
        for layer in self.model:
            x = layer(x)
        if 'PREARGMAX' in os.environ: x = x.argmax(axis=1)
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
    # Convert to half precision if FP16 env var is set
    if os.environ.get("FP16"):
        model = model.half()
    return model


if __name__ == "__main__":
    # TODO @spikedoanz: load default meshnet in this snippet
    pass
