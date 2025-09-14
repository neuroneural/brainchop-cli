"""
BrainChop Model Specification
Read-only dataclass specification for model architecture loading
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal, Callable
from enum import Enum
import json


# ============================================================================
# Operation Enums - Map directly to torch.nn.functional / tinygrad operations
# ============================================================================

class PreprocessOp(Enum):
    """Preprocessing operations"""
    QNORMALIZE = "qnormalize"      # Custom quantile normalization
    MINMAX = "minmax"               # Min-max to [0,1]
    ZSCORE = "zscore"               # Standardization
    NONE = "none"                   # Identity


class Op(Enum):
    """All layer operations - matches torch.nn.functional names"""
    # Convolutions
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    
    # Normalization  
    BATCH_NORM = "batch_norm"
    BATCH_NORM1D = "batch_norm1d"
    BATCH_NORM2D = "batch_norm2d"
    BATCH_NORM3D = "batch_norm3d"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"
    INSTANCE_NORM1D = "instance_norm1d"
    INSTANCE_NORM2D = "instance_norm2d"
    INSTANCE_NORM3D = "instance_norm3d"
    LAYER_NORM = "layer_norm"
    
    # Activations (match F.* names)
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    
    # Dropout
    DROPOUT = "dropout"
    DROPOUT1D = "dropout1d"
    DROPOUT2D = "dropout2d"
    DROPOUT3D = "dropout3d"
    
    # Pooling
    MAX_POOL1D = "max_pool1d"
    MAX_POOL2D = "max_pool2d"
    MAX_POOL3D = "max_pool3d"
    AVG_POOL1D = "avg_pool1d"
    AVG_POOL2D = "avg_pool2d"
    AVG_POOL3D = "avg_pool3d"
    ADAPTIVE_AVG_POOL1D = "adaptive_avg_pool1d"
    ADAPTIVE_AVG_POOL2D = "adaptive_avg_pool2d"
    ADAPTIVE_AVG_POOL3D = "adaptive_avg_pool3d"
    
    # Linear
    LINEAR = "linear"
    
    # Upsampling
    UPSAMPLE = "upsample"
    INTERPOLATE = "interpolate"


class PostprocessOp(Enum):
    """Postprocessing operations"""
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    ARGMAX = "argmax"
    NONE = "none"


# ============================================================================
# Specification Dataclasses
# ============================================================================

@dataclass(frozen=True)
class Preprocess:
    op: PreprocessOp
    params: Dict[str, Any]


@dataclass(frozen=True)
class Layer:
    op: Op
    params: Dict[str, Any]
    training_only: bool = False  # For dropout layers


@dataclass(frozen=True)
class Postprocess:
    op: PostprocessOp
    params: Dict[str, Any]


@dataclass(frozen=True)
class Metadata:
    description: str
    framework: Literal["torch", "tinygrad"] = "tinygrad"
    input_shape: Optional[List[int]] = None  # [C, D, H, W] or [C, H, W]
    output_classes: Optional[int] = None


@dataclass(frozen=True)
class ModelSpec:
    """Complete model specification"""
    version: str
    metadata: Metadata
    preprocessing: Preprocess
    forward_pass: List[Layer]
    postprocessing: Postprocess


# ============================================================================
# Spec Loading
# ============================================================================

def load_spec(spec_path: str) -> ModelSpec:
    """Load model specification from JSON file"""
    with open(spec_path, 'r') as f:
        data = json.load(f)
    
    # Parse preprocessing
    preprocess_data = data.get("preprocessing", {"op": "none", "params": {}})
    preprocessing = Preprocess(
        op=PreprocessOp(preprocess_data["op"]),
        params=preprocess_data.get("params", {})
    )
    
    # Parse forward pass
    forward_pass = []
    for layer_data in data["forward_pass"]:
        forward_pass.append(Layer(
            op=Op(layer_data["op"]),
            params=layer_data.get("params", {}),
            training_only=layer_data.get("training_only", False)
        ))
    
    # Parse postprocessing
    postprocess_data = data.get("postprocessing", {"op": "none", "params": {}})
    postprocessing = Postprocess(
        op=PostprocessOp(postprocess_data["op"]),
        params=postprocess_data.get("params", {})
    )
    
    # Parse metadata
    meta_data = data.get("metadata", {})
    metadata = Metadata(
        description=meta_data.get("description", ""),
        framework=meta_data.get("framework", "tinygrad"),
        input_shape=meta_data.get("input_shape"),
        output_classes=meta_data.get("output_classes")
    )
    
    return ModelSpec(
        version=data.get("version", "2.0"),
        metadata=metadata,
        preprocessing=preprocessing,
        forward_pass=forward_pass,
        postprocessing=postprocessing
    )


# ============================================================================
# Model Building - Maps spec to actual tinygrad model
# ============================================================================

def build_model(spec_path: str, weights_path: str):
    """
    Build executable model from spec and weights.
    
    Args:
        spec_path: Path to JSON specification
        weights_path: Path to .pth or .safetensors weights
    
    Returns:
        Callable model with loaded weights
    """
    from tinygrad import nn
    from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
    from tinygrad.tensor import Tensor
    import numpy as np
    
    spec = load_spec(spec_path)
    
    # Load weights
    if weights_path.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(weights_path, framework="pt") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
    else:
        state_dict = torch_load(weights_path)
    
    # Build preprocessing
    preprocess_fn = _build_preprocess(spec.preprocessing)
    
    # Build layers
    layers = []
    weight_index = 0
    
    for layer_spec in spec.forward_pass:
        if layer_spec.op in [Op.CONV1D, Op.CONV2D, Op.CONV3D, Op.LINEAR]:
            # These ops consume weights
            layer = _build_weighted_layer(layer_spec, weight_index, state_dict)
            weight_index += 1
        else:
            # Non-weighted layers
            layer = _build_layer(layer_spec)
        
        layers.append((layer, layer_spec.training_only))
    
    # Build postprocessing
    postprocess_fn = _build_postprocess(spec.postprocessing)
    
    # Create model class
    class Model:
        def __init__(self):
            self.layers = [layer for layer, _ in layers]
            
        def __call__(self, x, training=False):
            x = preprocess_fn(x)
            
            for (layer, training_only) in layers:
                if not training_only or training:
                    x = layer(x)
            
            x = postprocess_fn(x)
            return x
    
    # Instantiate and load weights
    model = Model()
    
    # Map weights to model
    model_state = get_state_dict(model)
    mapped_weights = _map_weights(state_dict, model_state)
    load_state_dict(model, mapped_weights, strict=True)
    
    return model


def _build_preprocess(preprocess: Preprocess) -> Callable:
    """Build preprocessing function"""
    import numpy as np
    from tinygrad.tensor import Tensor
    
    if preprocess.op == PreprocessOp.QNORMALIZE:
        def qnormalize(x):
            x_np = x.numpy()
            qmin = preprocess.params.get("qmin", 0.02)
            qmax = preprocess.params.get("qmax", 0.98)
            eps = preprocess.params.get("eps", 1e-3)
            
            qlow = np.quantile(x_np, qmin)
            qhigh = np.quantile(x_np, qmax)
            x_np = (x_np - qlow) / (qhigh - qlow + eps)
            x_np = np.clip(x_np, 0, 1)
            return Tensor(x_np)
        return qnormalize
    
    elif preprocess.op == PreprocessOp.MINMAX:
        return lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    elif preprocess.op == PreprocessOp.ZSCORE:
        return lambda x: (x - x.mean()) / (x.std() + 1e-8)
    
    else:  # NONE
        return lambda x: x


def _build_weighted_layer(layer: Layer, weight_index: int, state_dict: dict):
    """Build layer that has weights (conv, linear)"""
    from tinygrad import nn
    
    # Map to tinygrad classes
    if layer.op == Op.CONV1D:
        return nn.Conv1d(**layer.params)
    elif layer.op == Op.CONV2D:
        return nn.Conv2d(**layer.params)
    elif layer.op == Op.CONV3D:
        # Tinygrad uses Conv2d for 3D
        return nn.Conv2d(**layer.params)
    elif layer.op == Op.LINEAR:
        return nn.Linear(**layer.params)
    else:
        raise ValueError(f"Unknown weighted layer type: {layer.op}")


def _build_layer(layer: Layer) -> Callable:
    """Build non-weighted layer (activation, norm, etc)"""
    from tinygrad import nn
    
    # Normalization layers
    if layer.op == Op.GROUP_NORM:
        num_groups = layer.params.get("num_groups")
        if num_groups == "auto":
            # Will be inferred from actual weight dimensions
            num_groups = layer.params.get("num_channels", 1)
        return nn.GroupNorm(
            num_groups=num_groups,
            num_channels=layer.params.get("num_channels", num_groups),
            affine=layer.params.get("affine", True)
        )
    
    elif layer.op == Op.BATCH_NORM3D:
        return nn.BatchNorm(layer.params.get("num_features"))
    
    elif layer.op == Op.LAYER_NORM:
        return nn.LayerNorm(layer.params.get("normalized_shape"))
    
    # Activation functions
    elif layer.op == Op.RELU:
        return lambda x: x.relu()
    
    elif layer.op == Op.GELU:
        return lambda x: x.gelu()
    
    elif layer.op == Op.SILU:
        return lambda x: x.silu()
    
    elif layer.op == Op.SIGMOID:
        return lambda x: x.sigmoid()
    
    elif layer.op == Op.TANH:
        return lambda x: x.tanh()
    
    elif layer.op == Op.LEAKY_RELU:
        alpha = layer.params.get("negative_slope", 0.01)
        return lambda x: x.leakyrelu(alpha)
    
    elif layer.op == Op.ELU:
        alpha = layer.params.get("alpha", 1.0)
        return lambda x: x.elu(alpha)
    
    elif layer.op == Op.SOFTMAX:
        dim = layer.params.get("dim", 1)
        return lambda x: x.softmax(axis=dim)
    
    # Dropout
    elif layer.op == Op.DROPOUT:
        p = layer.params.get("p", 0.5)
        return lambda x: x.dropout(p)
    
    elif layer.op in [Op.DROPOUT1D, Op.DROPOUT2D, Op.DROPOUT3D]:
        p = layer.params.get("p", 0.5)
        return lambda x: x.dropout(p)
    
    # Pooling
    elif layer.op == Op.MAX_POOL3D:
        kernel_size = layer.params["kernel_size"]
        stride = layer.params.get("stride", kernel_size)
        padding = layer.params.get("padding", 0)
        return lambda x: x.max_pool2d(kernel_size, stride, padding)
    
    elif layer.op == Op.AVG_POOL3D:
        kernel_size = layer.params["kernel_size"]
        stride = layer.params.get("stride", kernel_size)
        padding = layer.params.get("padding", 0)
        return lambda x: x.avg_pool2d(kernel_size, stride, padding)
    
    else:
        raise NotImplementedError(f"Operation {layer.op} not yet implemented")


def _build_postprocess(postprocess: Postprocess) -> Callable:
    """Build postprocessing function"""
    if postprocess.op == PostprocessOp.SOFTMAX:
        dim = postprocess.params.get("dim", 1)
        return lambda x: x.softmax(axis=dim)
    
    elif postprocess.op == PostprocessOp.SIGMOID:
        return lambda x: x.sigmoid()
    
    elif postprocess.op == PostprocessOp.ARGMAX:
        dim = postprocess.params.get("dim", 1)
        return lambda x: x.argmax(axis=dim)
    
    else:  # NONE
        return lambda x: x


def _map_weights(torch_dict: dict, tinygrad_dict: dict) -> dict:
    """Map torch weights to tinygrad model"""
    torch_keys = list(torch_dict.keys())
    tiny_keys = list(tinygrad_dict.keys())
    
    if set(torch_keys) == set(tiny_keys):
        # Direct mapping
        return torch_dict
    
    # Order-based mapping
    mapped = {}
    for torch_key, tiny_key in zip(torch_keys, tiny_keys):
        mapped[tiny_key] = torch_dict[torch_key]
    
    return mapped


# ============================================================================
# Example JSON spec for your MeshNet
# ============================================================================

EXAMPLE_MESHNET_SPEC = """
{
  "version": "2.0",
  "metadata": {
    "description": "MeshNet with 5 decoders, dilations 16->8->4->2->1",
    "framework": "tinygrad",
    "input_shape": [1, 256, 256, 256],
    "output_classes": 2
  },
  "preprocessing": {
    "op": "qnormalize",
    "params": {
      "qmin": 0.02,
      "qmax": 0.98,
      "eps": 1e-3
    }
  },
  "forward_pass": [
    {"op": "conv3d", "params": {"kernel_size": 3, "padding": 16, "dilation": 16, "bias": false}},
    {"op": "group_norm", "params": {"num_groups": "auto", "affine": false}},
    {"op": "gelu", "params": {}},
    
    {"op": "conv3d", "params": {"kernel_size": 3, "padding": 8, "dilation": 8, "bias": false}},
    {"op": "group_norm", "params": {"num_groups": "auto", "affine": false}},
    {"op": "gelu", "params": {}},
    
    {"op": "conv3d", "params": {"kernel_size": 3, "padding": 4, "dilation": 4, "bias": false}},
    {"op": "group_norm", "params": {"num_groups": "auto", "affine": false}},
    {"op": "gelu", "params": {}},
    
    {"op": "conv3d", "params": {"kernel_size": 3, "padding": 2, "dilation": 2, "bias": false}},
    {"op": "group_norm", "params": {"num_groups": "auto", "affine": false}},
    {"op": "gelu", "params": {}},
    
    {"op": "conv3d", "params": {"kernel_size": 3, "padding": 1, "dilation": 1, "bias": false}},
    {"op": "group_norm", "params": {"num_groups": "auto", "affine": false}},
    {"op": "gelu", "params": {}},
    
    {"op": "conv3d", "params": {"kernel_size": 1, "padding": 0, "dilation": 1, "bias": false}}
  ],
  "postprocessing": {
    "op": "none",
    "params": {}
  }
}
"""


# ============================================================================
# Usage
# ============================================================================

if __name__ == "__main__":
    # Example usage
    model = build_model(
        spec_path="meshnet.json",
        weights_path="meshnet.pth"
    )
    
    # Run inference
    import numpy as np
    from tinygrad import Tensor
    
    dummy_input = Tensor(np.random.randn(1, 1, 256, 256, 256).astype(np.float32))
    output = model(dummy_input, training=False)
    print(f"Output shape: {output.shape}")
