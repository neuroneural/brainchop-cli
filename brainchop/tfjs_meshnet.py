# DEPRECATED: please start using the native tiny_meshnet backend instead
import os
import json
import numpy as np
from tinygrad.tensor import Tensor
from typing import Tuple, Dict, Any, Callable
from functools import reduce


class SequentialArgmax:
    """
    Sequential argmax implementation for eager evaluation.

    Instead of stacking all channel outputs and doing a single argmax,
    this iterates through channels one at a time, tracking the max value
    and its index at each step. Forces eager evaluation with .realize().
    """

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def __call__(self, x: Tensor, get_channel_fn: Callable[[Tensor, int], Tensor]) -> Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, depth, height, width)
            get_channel_fn: Function that takes (x, channel_idx) and returns
                           the output for that channel (batch, 1, d, h, w)
        """
        batch_size = x.shape[0]
        depth, height, width = x.shape[2], x.shape[3], x.shape[4]

        # Initialize: track max values and their indices
        outB = Tensor.full((batch_size, 1, depth, height, width), -10000.0).realize()
        outC = Tensor.zeros(batch_size, 1, depth, height, width).realize()

        for i in range(self.out_channels):
            # Get output for channel i
            outA = get_channel_fn(x, i).realize()

            # Find where new channel gives greater response
            greater = (outA > outB).float().realize()

            # Update max values
            outB = ((1 - greater) * outB + greater * outA).realize()

            # Update indices
            outC = ((1 - greater) * outC + greater * i).realize()

        return outC

class MeshNetModel:
    def __init__(self):
        self.activation_map = {
            "relu": lambda x: x.relu(),
            "gelu": lambda x: x.gelu(),
            "elu": lambda x: x.elu(),
            "sigmoid": lambda x: x.sigmoid(),
            "tanh": lambda x: x.tanh(),
            "leaky_relu": lambda x: x.leakyrelu(),
        }
        self.normalization_map = {
            "minmax": self.min_max_normalize,
            "quantile": self.quantile_normalize
        }

    def load_model_spec(self, json_path: str, bin_path: str) -> Tuple[Dict[str, Any], Tensor]:
        with open(json_path, "r") as f:
            model_spec = json.load(f)
        with open(bin_path, "rb") as f:
            # .copy() is required because np.frombuffer returns a read-only array
            # (backed by the immutable bytes object), which causes issues when
            # tinygrad's CPU backend tries to copy data using ctypes.from_buffer()
            weights_data = Tensor(np.frombuffer(f.read(), dtype=np.float32).copy())
        return model_spec, weights_data

    def normalize(self, img: np.ndarray | Tensor, normalize_config: Dict[str, Any] | None = None) -> np.ndarray:
        """Normalize the input image based on the configuration."""
        if isinstance(img, Tensor):
            img = img.numpy()
            
        # Convert to float32 for normalization calculations
        img = img.astype(np.float32) #type:ignore
            
        if normalize_config is None:
            return self.min_max_normalize(img)
            
        norm_type = str(normalize_config.get("type", "minmax")).lower()
        if norm_type not in self.normalization_map:
            raise ValueError(f"Unsupported normalization type: {norm_type}")
            
        if norm_type == "quantile":
            qmin = float(normalize_config.get("min", 5))
            qmax = float(normalize_config.get("max", 95))
            return self.quantile_normalize(img, qmin, qmax)
        else:
            return self.min_max_normalize(img)

    def min_max_normalize(self, img: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0,1] range."""
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min == 0:
            return img - img_min
        return (img - img_min) / (img_max - img_min)
    
    def quantile_normalize(self, img: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
        """Normalize using quantile values."""
        img = img.astype(np.float32)
        qmin = float(qmin)
        qmax = float(qmax)
        
        # Calculate percentiles with float32 precision
        min_val = np.percentile(img, qmin).astype(np.float32)
        max_val = np.percentile(img, qmax).astype(np.float32)
        
        if max_val - min_val == 0:
            return (img - min_val).astype(np.float32)
            
        normalized = (img - min_val) / (max_val - min_val)
        return normalized.astype(np.float32)

    def calculate_padding(self, kernel_size: int | Tuple[int, ...], dilation: int | Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(dilation, int):
            dilation = (dilation,) * 3
        return tuple((k - 1) * d // 2 for k, d in zip(kernel_size, dilation))

    def process_conv_layer(self, x: Tensor, layer_config: Dict[str, Any],
                         weights_data: Tensor, weight_index: int,
                         in_channels: int,
                         out_channel_idx: int | None = None) -> Tuple[Tensor, int, int]:
        """
        Process a Conv3D layer.

        Args:
            x: Input tensor
            layer_config: Layer configuration dict
            weights_data: All model weights
            weight_index: Current position in weights_data
            in_channels: Number of input channels
            out_channel_idx: If specified, only compute this single output channel
                            (used for sequential argmax). If None, compute all channels.
        """
        padding = self.calculate_padding(
            layer_config["kernel_size"],
            layer_config["dilation_rate"]
        )

        out_channels = layer_config["filters"]
        k = layer_config["kernel_size"][0]

        weight_shape = [out_channels, in_channels, k, k, k]
        weight_shape = [weight_shape[i] for i in (2, 3, 4, 1, 0)]
        bias_shape = [out_channels]

        weight_size = reduce(lambda a, b: a*b, weight_shape)
        bias_size   = reduce(lambda a, b: a*b, bias_shape)

        # Extract and reshape weights
        weight = weights_data[weight_index:weight_index + weight_size].reshape(weight_shape)
        weight = weight.permute(4, 3, 0, 1, 2)
        weight_index += weight_size

        # Extract and reshape bias
        bias = weights_data[weight_index:weight_index + bias_size].reshape(bias_shape)
        weight_index += bias_size

        # If requesting a single output channel, slice weights/bias
        if out_channel_idx is not None:
            weight = weight[out_channel_idx:out_channel_idx+1]  # (1, in_channels, k, k, k)
            bias = bias[out_channel_idx:out_channel_idx+1]      # (1,)
            out_channels = 1

        # Perform convolution
        x = x.conv2d(
            weight=weight,
            bias=bias,
            groups=1,
            stride=layer_config["strides"][0],
            dilation=layer_config["dilation_rate"][0],
            padding=padding[0]
        )

        return x, weight_index, out_channels

class ModelContainer():
    def __init__(self, model, normalization_fn):
        self.model = model
        self.normalization_fn = normalization_fn

    def normalize(self, x):
        return self.normalization_fn(x)

    def __call__(self, x):
        return self.model(x)


def load_tfjs_meshnet(config_fn: str, binary_fn: str): # -> tinygrad "model"
    model = MeshNetModel()
    model_spec, weights_data = model.load_model_spec(config_fn, binary_fn)
    
    # Get normalization config from model spec if available
    normalize_config = model_spec.get("_normalize")
    
    def normalization_fn(x: Tensor, normalize_config=normalize_config) -> Tensor:
        # Convert to numpy for normalization if needed
        x_np = x.numpy() if isinstance(x, Tensor) else x
        x_norm = model.normalize(x_np, normalize_config)
        
        # Convert back to Tensor
        if not isinstance(x_norm, Tensor):
            x = Tensor(x_norm.astype(np.float32))
        else:
            x = x_norm
        return x

    def forward(x: Tensor, model=model, weights_data=weights_data) -> Tensor:
        weight_index = 0
        in_channels = 1

        spec = model_spec["modelTopology"]["model_config"]["config"]["layers"][1:]

        # Check if we should use sequential argmax (eager evaluation)
        use_sequential_argmax = 'PREARGMAX' in os.environ

        # Find the last Conv3D layer to handle specially if using sequential argmax
        last_conv_idx = None
        for i, layer in enumerate(spec):
            if layer["class_name"] == "Conv3D":
                last_conv_idx = i

        for i, layer in enumerate(spec):
            is_last_conv = (i == last_conv_idx)

            if layer["class_name"] == "Conv3D":
                if is_last_conv and use_sequential_argmax:
                    # For last conv with sequential argmax:
                    # Don't process the full conv, instead use SequentialArgmax
                    out_channels = layer["config"]["filters"]
                    last_layer_config = layer["config"]
                    last_weight_index = weight_index
                    last_in_channels = in_channels

                    # Create a function that computes a single output channel
                    def get_channel_fn(
                        input_tensor: Tensor,
                        channel_idx: int,
                        layer_config=last_layer_config,
                        w_idx=last_weight_index,
                        in_ch=last_in_channels
                    ) -> Tensor:
                        out, _, _ = model.process_conv_layer(
                            input_tensor, layer_config, weights_data,
                            w_idx, in_ch, out_channel_idx=channel_idx
                        )
                        return out

                    # Run sequential argmax
                    seq_argmax = SequentialArgmax(out_channels)
                    x = seq_argmax(x, get_channel_fn)

                    # Skip weight_index update since we're done
                    break
                else:
                    x, weight_index, in_channels = model.process_conv_layer(
                        x, layer["config"], weights_data, weight_index, in_channels
                    )
                    x = x.realize()
            elif layer["class_name"] == "Activation":
                activation = model.activation_map[layer["config"]["activation"]]
                x = activation(x).realize()

        return x

    model_container = ModelContainer(forward, normalization_fn)
    return model_container

if __name__ == "__main__":
    # TODO @spikedoanz: load default meshnet in this snippet
    pass
