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
    return outC.squeeze(0)


class SequentialConvArgmax:
    """
    Fused conv + argmax with adaptive chunking.

    Tries the full conv first (fastest). On MemoryError, halves the chunk
    size and retries until it fits. The working chunk size is cached so
    subsequent runs skip the probing.

    Set FUSE_CHUNK envvar to override (0 = auto-probe).
    """

    def __init__(self, out_channels: int, final_conv: nn.Conv2d):
        self.out_channels = out_channels
        self.weight = final_conv.weight
        self.bias = final_conv.bias
        self.conv_kwargs = dict(
            stride=final_conv.stride,
            dilation=final_conv.dilation,
            padding=final_conv.padding,
        )

    def _conv_channels(self, x: Tensor, start: int, end: int) -> Tensor:
        """Compute a slice of output channels [start:end] of the final conv."""
        b = self.bias[start:end] if self.bias is not None else None
        return x.conv2d(self.weight[start:end], b, **self.conv_kwargs)

    def _chunked_argmax(self, x: Tensor, chunk_size: int) -> Tensor:
        """Conv + argmax in chunks of chunk_size channels."""
        # First chunk
        end = min(chunk_size, self.out_channels)
        logits = self._conv_channels(x, 0, end).realize()
        outB = logits.max(axis=1, keepdim=True)
        outC = logits.argmax(axis=1, keepdim=True).cast("float32")
        del logits
        outB, outC = outB.realize(), outC.realize()

        offset = end
        while offset < self.out_channels:
            end = min(offset + chunk_size, self.out_channels)
            logits = self._conv_channels(x, offset, end).realize()
            chunk_max = logits.max(axis=1, keepdim=True)
            chunk_idx = logits.argmax(axis=1, keepdim=True).cast("float32") + float(offset)
            del logits
            mask = chunk_max > outB
            outB = mask.where(chunk_max, outB).realize()
            outC = mask.where(chunk_idx, outC).realize()
            offset = end

        return outC.squeeze(1)

    def __call__(self, x: Tensor, chunk_size: int | None = None) -> Tensor:
        """Fused conv+argmax. x is the input to the final conv (pre-conv features)."""
        return self._chunked_argmax(x, chunk_size or self.out_channels)

    def argmax_only(self, x: Tensor) -> Tensor:
        """Argmax over pre-computed logits (B, C, D, H, W). No conv fusion."""
        outB = x[:, 0:1].realize()
        outC = Tensor.zeros_like(outB)

        for i in range(1, self.out_channels):
            mask = x[:, i:i+1] > outB
            outB = mask.where(x[:, i:i+1], outB).realize()
            outC = mask.where(float(i), outC).realize()

        return outC.squeeze(1)


def chunked_conv(x: Tensor, conv_layer: nn.Conv2d, chunk_limit: int | None = None) -> Tensor:
    """
    Workaround for WebGPU matmul bug: when output exceeds chunk_limit elements,
    process output channels in chunks to avoid 32-bit integer overflow.

    Set CHUNK_LIMIT env var to override the default of 2^30 (~1GB).
    """
    if chunk_limit is None:
        chunk_limit = int(os.environ.get("CHUNK_LIMIT", 2**30))
    from functools import reduce
    spatial = int(reduce(lambda a, b: int(a) * int(b), x.shape[2:], 1))
    out_channels = int(conv_layer.weight.shape[0])
    output_elements = out_channels * spatial

    if output_elements <= chunk_limit:
        return conv_layer(x)

    # Chunked processing
    max_channels = max(1, chunk_limit // spatial)
    chunks = []

    for start in range(0, out_channels, max_channels):
        end = min(start + max_channels, out_channels)
        chunk_weight = conv_layer.weight[start:end]
        chunk_bias = conv_layer.bias[start:end] if conv_layer.bias is not None else None

        chunk_out = x.conv2d(
            weight=chunk_weight,
            bias=chunk_bias,
            groups=conv_layer.groups,
            stride=conv_layer.stride,
            dilation=conv_layer.dilation,
            padding=conv_layer.padding,
        ).realize()
        chunks.append(chunk_out)

    return Tensor.cat(*chunks, dim=1)


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


def construct_layer(dropout_p=0, bnorm=True, gelu=False, elu=False, affine=False, *args, **kwargs):
    layers = []
    kwargs["kernel_size"] = [kwargs["kernel_size"]] * 3
    layers.append(nn.Conv2d(*args, **kwargs))
    if bnorm:
        layers.append(
            nn.GroupNorm(
                num_groups=kwargs["out_channels"],
                num_channels=kwargs["out_channels"],
                affine=affine,
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
        # Learnable per-channel scale/shift in GroupNorm (default False for
        # backward compat). Models trained with affine GroupNorm (e.g. the deep
        # gridding-free DK-atlas model) carry extra GN weight/bias params.
        use_affine = config.get("affine", False)

        for block_kwargs in config["layers"][:-1]:  # All but the last layer
            # Per-layer "bias" overrides the global default, so a stack can keep
            # bias-free hidden convs while still allowing a biased layer.
            layer_bias = block_kwargs.get("bias", use_bias)
            self.model.extend(
                construct_layer(
                    dropout_p=config["dropout_p"],
                    bnorm=config["bnorm"],
                    gelu=config.get("gelu", False),
                    elu=config.get("elu", False),
                    affine=use_affine,
                    **{**block_kwargs, "bias": layer_bias},
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
                bias=last_config.get("bias", use_bias),
            )
        )

        self.n_classes = last_config["out_channels"]
        self.seq_conv_argmax: SequentialConvArgmax | None = None

    def init_seq_conv_argmax(self):
        """Initialize SequentialConvArgmax with the final conv layer fused in."""
        final_conv = self.model[-1]  # last layer is always nn.Conv2d
        self.seq_conv_argmax = SequentialConvArgmax(self.n_classes, final_conv)

    def normalize(self, x):
        return qnormalize(x) # TODO: interpret normalization from config file

    def __call__(self, x, fuse_chunk=None):
        for layer in self.model[:-1]:  # all layers except final conv
            if isinstance(layer, nn.Conv2d):
                x = chunked_conv(x, layer)
            else:
                x = layer(x)
        assert self.seq_conv_argmax is not None
        return self.seq_conv_argmax(x, chunk_size=fuse_chunk)

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
    # Initialize SequentialConvArgmax for memory-efficient argmax
    model.init_seq_conv_argmax()
    # Convert to half precision if FP16 env var is set
    if os.environ.get("FP16"):
        model = model.half()
    return model


if __name__ == "__main__":
    # TODO @spikedoanz: load default meshnet in this snippet
    pass
