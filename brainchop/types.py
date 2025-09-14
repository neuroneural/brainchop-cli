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
    from tinygrad.nn.state import torch_load, load_state_dict, get_state_dict
    from tinygrad import nn
    
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
    
    # Build model with proper weight tracking
    model_layers = []
    
    # Track layer indices for weight mapping
    layer_idx = 0
    
    for layer_spec in spec.forward_pass:
        layer_info = _build_layer_with_weights(layer_spec, layer_idx, state_dict)
        if layer_info:
            model_layers.append(layer_info)
            # Only increment for layers that consume weights
            if layer_info['consumes_weights']:
                layer_idx += 1
    
    # Build postprocessing
    postprocess_fn = _build_postprocess(spec.postprocessing)
    
    # Create model class
    class Model:
        def __init__(self):
            self.layers = []
            for layer_info in model_layers:
                if layer_info['layer'] is not None:
                    self.layers.append((layer_info['layer'], layer_info['training_only']))
            
        def __call__(self, x, training=False):
            x = preprocess_fn(x)
            
            for layer, training_only in self.layers:
                if not training_only or training:
                    x = layer(x)
            
            x = postprocess_fn(x)
            return x
    
    # Instantiate model
    model = Model()
    
    # Load weights with proper mapping
    _load_weights_into_model(model, model_layers, state_dict)
    
    return model


def _build_layer_with_weights(layer_spec: Layer, layer_idx: int, state_dict: dict):
    """Build layer and track weight consumption"""
    from tinygrad import nn
    
    layer_info = {
        'layer': None,
        'training_only': layer_spec.training_only,
        'consumes_weights': False,
        'layer_idx': layer_idx,
        'op': layer_spec.op
    }
    
    # Weighted layers
    if layer_spec.op == Op.CONV1D:
        layer_info['layer'] = nn.Conv1d(**layer_spec.params)
        layer_info['consumes_weights'] = True
        
    elif layer_spec.op == Op.CONV2D:
        layer_info['layer'] = nn.Conv2d(**layer_spec.params)
        layer_info['consumes_weights'] = True
        
    elif layer_spec.op == Op.CONV3D:
        # For tinygrad, we need to handle 3D convs specially
        params = layer_spec.params.copy()
        layer_info['layer'] = nn.Conv2d(**params)  # Tinygrad handles 3D internally
        layer_info['consumes_weights'] = True
        
    elif layer_spec.op == Op.LINEAR:
        layer_info['layer'] = nn.Linear(**layer_spec.params)
        layer_info['consumes_weights'] = True
        
    # Batch normalization layers
    elif layer_spec.op == Op.BATCH_NORM3D:
        num_features = layer_spec.params.get("num_features")
        # Create batch norm but mark as inference-only
        layer_info['layer'] = _create_inference_batchnorm(num_features)
        layer_info['consumes_weights'] = True  # BatchNorm has weights/bias/running_mean/running_var
        
    elif layer_spec.op == Op.BATCH_NORM2D:
        num_features = layer_spec.params.get("num_features")
        layer_info['layer'] = _create_inference_batchnorm(num_features)
        layer_info['consumes_weights'] = True
        
    elif layer_spec.op == Op.BATCH_NORM1D:
        num_features = layer_spec.params.get("num_features")
        layer_info['layer'] = _create_inference_batchnorm(num_features)
        layer_info['consumes_weights'] = True
        
    # Non-weighted layers
    else:
        layer_info['layer'] = _build_functional_layer(layer_spec)
        layer_info['consumes_weights'] = False
    
    return layer_info


def _create_inference_batchnorm(num_features):
    """Create a batch norm layer for inference"""
    from tinygrad import Tensor
    
    class InferenceBatchNorm:
        def __init__(self, num_features):
            self.num_features = num_features
            # These will be loaded from state dict
            self.weight = None
            self.bias = None
            self.running_mean = None
            self.running_var = None
            self.eps = 1e-5
            
        def __call__(self, x):
            if self.running_mean is None or self.running_var is None:
                # Fallback to identity if stats not loaded
                return x
                
            # Apply batch normalization in inference mode
            # x_norm = (x - running_mean) / sqrt(running_var + eps)
            # y = weight * x_norm + bias
            
            # Reshape for broadcasting
            shape = [1] * len(x.shape)
            shape[1] = self.num_features  # Channel dimension
            
            mean = self.running_mean.reshape(shape)
            var = self.running_var.reshape(shape)
            
            x_norm = (x - mean) / (var + self.eps).sqrt()
            
            if self.weight is not None:
                weight = self.weight.reshape(shape)
                x_norm = x_norm * weight
                
            if self.bias is not None:
                bias = self.bias.reshape(shape)
                x_norm = x_norm + bias
                
            return x_norm
    
    return InferenceBatchNorm(num_features)


def _build_functional_layer(layer_spec: Layer):
    """Build non-weighted layer (activation, pooling, etc)"""
    
    # Activation functions
    if layer_spec.op == Op.RELU:
        return lambda x: x.relu()
    
    elif layer_spec.op == Op.GELU:
        return lambda x: x.gelu()
    
    elif layer_spec.op == Op.SILU:
        return lambda x: x.silu()
    
    elif layer_spec.op == Op.SIGMOID:
        return lambda x: x.sigmoid()
    
    elif layer_spec.op == Op.TANH:
        return lambda x: x.tanh()
    
    elif layer_spec.op == Op.LEAKY_RELU:
        alpha = layer_spec.params.get("negative_slope", 0.01)
        return lambda x: x.leakyrelu(alpha)
    
    elif layer_spec.op == Op.ELU:
        alpha = layer_spec.params.get("alpha", 1.0)
        return lambda x: x.elu(alpha)
    
    elif layer_spec.op == Op.SOFTMAX:
        dim = layer_spec.params.get("dim", 1)
        return lambda x: x.softmax(axis=dim)
    
    # Dropout
    elif layer_spec.op == Op.DROPOUT:
        p = layer_spec.params.get("p", 0.5)
        return lambda x: x.dropout(p)
    
    elif layer_spec.op in [Op.DROPOUT1D, Op.DROPOUT2D, Op.DROPOUT3D]:
        p = layer_spec.params.get("p", 0.5)
        return lambda x: x.dropout(p)
    
    # Pooling
    elif layer_spec.op == Op.MAX_POOL3D:
        kernel_size = layer_spec.params["kernel_size"]
        stride = layer_spec.params.get("stride", kernel_size)
        padding = layer_spec.params.get("padding", 0)
        return lambda x: x.max_pool2d(kernel_size, stride, padding)
    
    elif layer_spec.op == Op.AVG_POOL3D:
        kernel_size = layer_spec.params["kernel_size"]
        stride = layer_spec.params.get("stride", kernel_size)
        padding = layer_spec.params.get("padding", 0)
        return lambda x: x.avg_pool2d(kernel_size, stride, padding)
    
    else:
        #print(f"Warning: Operation {layer_spec.op} not implemented, skipping")
        return None


def _load_weights_into_model(model, model_layers, state_dict):
    """Load weights into model with proper mapping"""
    from tinygrad import Tensor
    
    # Create mapping from torch naming to our model structure
    torch_keys = sorted(state_dict.keys())
    
    # Group torch keys by layer
    layer_groups = {}
    for key in torch_keys:
        # Parse keys like "model.0.0.weight", "model.0.1.running_mean", or "model.9.weight"
        parts = key.split('.')
        if len(parts) >= 2 and parts[0] == 'model':
            layer_idx = int(parts[1])
            
            # Check if this is a nested layer (has sublayer index) or direct layer
            if len(parts) >= 4 and parts[2].isdigit():
                # Format: model.X.Y.param_name (nested layers)
                sublayer_idx = int(parts[2])
                param_name = '.'.join(parts[3:])
            else:
                # Format: model.X.param_name (direct layer, like final conv)
                sublayer_idx = 0  # Use 0 as default sublayer for non-nested
                param_name = '.'.join(parts[2:])
            
            if layer_idx not in layer_groups:
                layer_groups[layer_idx] = {}
            if sublayer_idx not in layer_groups[layer_idx]:
                layer_groups[layer_idx][sublayer_idx] = {}
                
            layer_groups[layer_idx][sublayer_idx][param_name] = key
    
    #print(f"Found {len(layer_groups)} layer groups in state dict")
    
    # Map to our model layers - need to handle the interleaved conv+batchnorm pattern
    weighted_layers = [info for info in model_layers if info['consumes_weights']]
    
    # From your weight structure, we can see the pattern:
    # model.0.0.weight/bias = first conv
    # model.0.1.weight/bias/running_mean/running_var = first batchnorm
    # model.1.0.weight/bias = second conv
    # model.1.1.weight/bias/running_mean/running_var = second batchnorm
    # ...
    # model.9.weight/bias = final conv (no batchnorm)
    
    block_idx = 0
    layer_in_block = 0
    
    for model_layer_idx, layer_info in enumerate(weighted_layers):
        layer = layer_info['layer']
        
        #print(f"Loading weights for model layer {model_layer_idx} (torch block {block_idx}, layer {layer_in_block})")
        
        if layer_info['op'] in [Op.CONV1D, Op.CONV2D, Op.CONV3D, Op.LINEAR]:
            # Load conv/linear weights from block_idx.layer_in_block
            if block_idx in layer_groups and layer_in_block in layer_groups[block_idx]:
                sublayer = layer_groups[block_idx][layer_in_block]
                
                if 'weight' in sublayer:
                    weight_key = sublayer['weight']
                    layer.weight = Tensor(state_dict[weight_key].numpy())
                    #print(f"  Loaded weight: {weight_key} -> {layer.weight.shape}")
                    
                if 'bias' in sublayer:
                    bias_key = sublayer['bias']
                    layer.bias = Tensor(state_dict[bias_key].numpy())
                    #print(f"  Loaded bias: {bias_key} -> {layer.bias.shape}")
            else:
                #print(f"  Warning: No weights found for conv layer at block {block_idx}, sublayer {layer_in_block}")
                pass
            
            # After conv, expect batchnorm next (except for final layer)
            layer_in_block += 1
            
        elif layer_info['op'] in [Op.BATCH_NORM1D, Op.BATCH_NORM2D, Op.BATCH_NORM3D]:
            # Load batch norm parameters from block_idx.layer_in_block
            if block_idx in layer_groups and layer_in_block in layer_groups[block_idx]:
                sublayer = layer_groups[block_idx][layer_in_block]
                
                if 'weight' in sublayer:
                    weight_key = sublayer['weight']
                    layer.weight = Tensor(state_dict[weight_key].numpy())
                    #print(f"  Loaded BN weight: {weight_key} -> {layer.weight.shape}")
                    
                if 'bias' in sublayer:
                    bias_key = sublayer['bias']
                    layer.bias = Tensor(state_dict[bias_key].numpy())
                    #print(f"  Loaded BN bias: {bias_key} -> {layer.bias.shape}")
                    
                if 'running_mean' in sublayer:
                    mean_key = sublayer['running_mean']
                    layer.running_mean = Tensor(state_dict[mean_key].numpy())
                    #print(f"  Loaded BN running_mean: {mean_key} -> {layer.running_mean.shape}")
                    
                if 'running_var' in sublayer:
                    var_key = sublayer['running_var']
                    layer.running_var = Tensor(state_dict[var_key].numpy())
                    #print(f"  Loaded BN running_var: {var_key} -> {layer.running_var.shape}")
            else:
                #print(f"  Warning: No weights found for batchnorm layer at block {block_idx}, sublayer {layer_in_block}")
                pass
            
            # After batchnorm, move to next block
            block_idx += 1
            layer_in_block = 0


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


# ============================================================================
# Example JSON spec for your MeshNet
# ============================================================================

EXAMPLE_MESHNET_SPEC = """
{
  "version": "2.0",
  "metadata": {
    "description": "MeshNet 3D CNN with dilations 1→2→4→8→16→8→4→2→1 for binary output",
    "framework": "tinygrad",
    "input_shape": [1, 1, 256, 256, 256],
    "output_classes": 2
  },
  "preprocessing": {
    "op": "none",
    "params": {}
  },
  "forward_pass": [
    {"op": "conv3d", "params": {"in_channels": 1, "out_channels": 26, "kernel_size": 3, "padding": 1, "stride": 1, "dilation": 1, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 2, "stride": 1, "dilation": 2, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 4, "stride": 1, "dilation": 4, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 8, "stride": 1, "dilation": 8, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 16, "stride": 1, "dilation": 16, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 8, "stride": 1, "dilation": 8, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 4, "stride": 1, "dilation": 4, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 2, "stride": 1, "dilation": 2, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 26, "kernel_size": 3, "padding": 1, "stride": 1, "dilation": 1, "bias": true}},
    {"op": "batch_norm3d", "params": {"num_features": 26}},
    {"op": "elu", "params": {"alpha": 1.0}},

    {"op": "conv3d", "params": {"in_channels": 26, "out_channels": 2, "kernel_size": 1, "padding": 0, "stride": 1, "dilation": 1, "bias": true}}
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
