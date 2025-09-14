from tinygrad import Tensor, nn
from tinygrad.nn.state import torch_load, load_state_dict
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
    qlow = np.quantile(img, qmin)
    qhigh = np.quantile(img, qmax)
    img = (img - qlow) / (qhigh - qlow + eps)
    img = np.clip(img, 0, 1)
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


def construct_layer(dropout_p=0, bnorm=True, gelu=False, *args, **kwargs):
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

    relu = lambda x: x.relu()
    gelu = lambda x: x.gelu()
    dropout = lambda x: x.dropout(dropout_p)
    layers.append(gelu if gelu else relu)
    if dropout_p > 0:
        layers.append(dropout)
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
        for block_kwargs in config["layers"][:-1]:  # All but the last layer
            self.model.extend(
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config["gelu"],
                    **{**block_kwargs, "bias": False},  # middle layers have no bias
                )
            )

        # Handle last layer specially - add it to model list
        last_config = config["layers"][-1]
        self.model.append(
            nn.Conv2d(
                last_config["in_channels"],
                last_config["out_channels"],
                kernel_size=[last_config["kernel_size"]] * 3,
                padding=last_config["padding"],
                stride=last_config["stride"],
                dilation=last_config["dilation"],
                bias=False,  # Enable bias in the conv layer
            )
        )

    def __call__(self, x):
        x = qnormalize(x)  # TODO: interpret normalization from config file
        for layer in self.model:
            x = layer(x)
        return x


def load_meshnet(
    config_fn: str,
    model_fn: str,
    in_channels: int = 1,
    channels: int = 15,
    out_channels: int = 2,
):
    # TODO: Interpret channel info from config
    model = MeshNet(
        in_channels=in_channels,
        n_classes=out_channels,
        channels=channels,
        config_file=config_fn,
    )
    state_dict = torch_load(model_fn)
    state_dict = convert_keys(state_dict, nn.state.get_state_dict(model))
    load_state_dict(model, state_dict, strict=True, verbose=False)
    return model
