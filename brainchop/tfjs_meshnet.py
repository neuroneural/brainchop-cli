# DEPRECATED: please start using the native tiny_meshnet backend instead
import json
import numpy as np
from tinygrad import Tensor
from typing import Tuple, Dict, Any

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

    def load_model_spec(self, json_path: str, bin_path: str) -> Tuple[Dict[str, Any], np.ndarray]:
        with open(json_path, "r") as f:
            model_spec = json.load(f)
        with open(bin_path, "rb") as f:
            weights_data = np.frombuffer(f.read(), dtype=np.float32)
        return model_spec, weights_data

    def normalize(self, img: np.ndarray | Tensor, normalize_config: Dict[str, Any] | None = None) -> np.ndarray:
        """Normalize the input image based on the configuration."""
        if isinstance(img, Tensor):
            img = img.numpy()
            
        # Convert to float32 for normalization calculations
        img = img.astype(np.float32)
            
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
                         weights_data: np.ndarray, weight_index: int, 
                         in_channels: int) -> Tuple[Tensor, int, int]:
        padding = self.calculate_padding(
            layer_config["kernel_size"],
            layer_config["dilation_rate"]
        )
        
        out_channels = layer_config["filters"]
        k = layer_config["kernel_size"][0]
        
        weight_shape = [out_channels, in_channels, k, k, k]
        weight_shape = [weight_shape[i] for i in (2, 3, 4, 1, 0)]
        bias_shape = [out_channels]
        
        weight_size = np.prod(weight_shape)
        bias_size = np.prod(bias_shape)
        
        # Extract and reshape weights
        weight = weights_data[weight_index:weight_index + weight_size].reshape(weight_shape)
        weight = np.transpose(weight, (4, 3, 0, 1, 2))
        weight_index += weight_size
        
        # Extract and reshape bias
        bias = weights_data[weight_index:weight_index + bias_size].reshape(bias_shape)
        weight_index += bias_size
        
        # Convert to Tensors
        weight_tensor = Tensor(weight.copy())
        bias_tensor = Tensor(bias.copy())
        
        # Perform convolution
        x = x.conv2d(
            weight=weight_tensor,
            bias=bias_tensor,
            groups=1,
            stride=layer_config["strides"][0],
            dilation=layer_config["dilation_rate"][0],
            padding=padding[0]
        )
        
        return x, weight_index, out_channels

def load_tfjs_meshnet(config_fn: str, binary_fn: str): # -> tinygrad "model"
    model = MeshNetModel()
    model_spec, weights_data = model.load_model_spec(config_fn, binary_fn)
    
    # Get normalization config from model spec if available
    normalize_config = model_spec.get("_normalize")
    
    def forward(x: Tensor) -> Tensor:
        # Convert to numpy for normalization if needed
        x_np = x.numpy() if isinstance(x, Tensor) else x
        x_norm = model.normalize(x_np, normalize_config)
        
        # Convert back to Tensor
        if not isinstance(x_norm, Tensor):
            x = Tensor(x_norm.astype(np.float32))
        else:
            x = x_norm
        
        weight_index = 0
        in_channels = 1
        
        spec = model_spec["modelTopology"]["model_config"]["config"]["layers"][1:]
        for layer in spec:
            if layer["class_name"] == "Conv3D":
                x, weight_index, in_channels = model.process_conv_layer(
                    x, layer["config"], weights_data, weight_index, in_channels
                )
            elif layer["class_name"] == "Activation":
                activation = model.activation_map[layer["config"]["activation"]]
                x = activation(x)
        
        # Return the raw tensor output
        return x
    
    return forward
