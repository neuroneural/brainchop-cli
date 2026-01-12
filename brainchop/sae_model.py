"""
Spatial AutoEncoder (SAE) model for brain tissue segmentation.

This module provides SAENet, an encoder-decoder architecture with:
- Dilated convolutions for large receptive field
- Spatial bottleneck (256³ → 128³ → 256³)
- SiLU activation
"""

from tinygrad import Tensor
from tinygrad.nn.state import torch_load

from brainchop.tiny_meshnet import qnormalize, SequentialConvArgmax


# Layer indices in the fused model (odd indices are SiLU activations, no params)
LAYER_INDICES = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23, 25, 27, 29, 31, 33]
DOWNSAMPLE_LAYER = 10  # stride=2, 256³ → 128³
UPSAMPLE_LAYER = 22    # ConvTranspose stride=2, 128³ → 256³

# Dilation schedule from modelAEgelu_dilated.json
# JSON has 16 layers: 5 encoder + 5 bottleneck + 5 decoder + 1 final
# Downsample/upsample are added by model class (not in JSON)
# padding = dilation for 3x3 kernel to maintain spatial size
DILATION_SCHEDULE = {
    # Encoder (JSON 0-4): d=1,2,4,6,8
    0: 1,
    2: 2,
    4: 4,
    6: 6,
    8: 8,
    # Downsample (not in JSON, strided conv)
    10: 1,
    # Bottleneck (JSON 5-9): d=10,12,16,12,10
    12: 10,
    14: 12,
    16: 16,  # peak
    18: 12,
    20: 10,
    # Upsample (ConvTranspose, no dilation)
    22: 1,
    # Decoder (JSON 10-14): d=8,6,4,2,1
    23: 8,
    25: 6,
    27: 4,
    29: 2,
    31: 1,
    # Final 1x1 (JSON 15)
    33: 1,
}


def _permute_weights(weight, is_conv_transpose: bool = False):
    """
    Permute conv weights by swapping D and W spatial dimensions.

    This aligns PyTorch weights with the expected input orientation.
    PyTorch weight shape: (out_ch, in_ch, D, H, W) for Conv3d
                         (in_ch, out_ch, D, H, W) for ConvTranspose3d
    After permute: swap axes 2 and 4 (D <-> W)
    """
    import numpy as np
    w = weight.numpy()
    w = np.swapaxes(w, 2, 4)  # Swap D and W
    return Tensor(w)


class SAENet:
    """
    Spatial AutoEncoder network for brain segmentation.

    Architecture:
        - Encoder: Conv3d layers with SiLU, one strided conv for downsampling
        - Bottleneck: Conv3d layers at reduced spatial resolution
        - Decoder: ConvTranspose3d for upsampling, Conv3d layers with SiLU
        - Output: 1x1x1 conv to n_classes
    """

    def __init__(self, state_dict: dict, n_classes: int = 3, permute: bool = True):
        """Load model from state dict with numeric keys.

        Args:
            state_dict: Model weights with numeric keys (0.weight, 0.bias, etc.)
            n_classes: Number of output classes
            permute: If True, swap D and W dimensions in weights to match input orientation
        """
        self.layers = []
        self.n_classes = n_classes

        for idx in LAYER_INDICES:
            # Move weights from DISK to compute device via numpy roundtrip
            weight = Tensor(state_dict[f"{idx}.weight"].numpy())
            bias = Tensor(state_dict[f"{idx}.bias"].numpy())

            if idx == UPSAMPLE_LAYER:
                layer_type = "convT"
            elif idx == DOWNSAMPLE_LAYER:
                layer_type = "conv_s2"
            else:
                layer_type = "conv"

            # Permute weights if needed (swap D <-> W)
            if permute:
                weight = _permute_weights(weight, is_conv_transpose=(layer_type == "convT"))

            self.layers.append((layer_type, idx, weight, bias))

        self.seq_conv_argmax = SequentialConvArgmax(self.n_classes)

    def normalize(self, x: Tensor) -> Tensor:
        """Quantile normalization (same as MeshNet)."""
        return qnormalize(x)

    def forward_no_argmax(self, x: Tensor) -> Tensor:
        """
        Forward pass returning logits (no argmax).

        Args:
            x: Input tensor (B, 1, D, H, W)

        Returns:
            Logits tensor (B, n_classes, D, H, W)
        """
        # Process all layers except the last (which has no activation)
        for layer_type, idx, weight, bias in self.layers[:-1]:
            dilation = DILATION_SCHEDULE.get(idx, 1)
            padding = dilation  # padding = dilation for 3x3 kernel

            if layer_type == "conv":
                x = x.conv2d(weight, bias, padding=padding, dilation=dilation)
            elif layer_type == "conv_s2":
                x = x.conv2d(weight, bias, padding=padding, stride=2, dilation=dilation)
            elif layer_type == "convT":
                x = x.conv_transpose2d(weight, bias, stride=2, padding=0)

            # SiLU activation after each conv (except output and ConvTranspose)
            # ConvTranspose (upsample) has no activation - goes directly to next conv
            if layer_type != "convT":
                x = x.silu()

        # Final layer: 1x1x1 conv, no activation, no padding
        _, idx, weight, bias = self.layers[-1]
        kernel_size = weight.shape[2]
        padding = (kernel_size - 1) // 2  # 0 for 1x1, 1 for 3x3
        x = x.conv2d(weight, bias, padding=padding)

        return x

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, D, H, W)

        Returns:
            Segmentation (B, D, H, W) after argmax
        """
        x = self.forward_no_argmax(x)
        # Memory-efficient argmax
        return self.seq_conv_argmax(x)


def load_sae(model_path: str, n_classes: int = 3, permute: bool = True) -> SAENet:
    """Load SAENet model from .pth file.

    Args:
        model_path: Path to .pth file with fused model weights
        n_classes: Number of output classes
        permute: If True, swap D and W dimensions in weights (default True for correct orientation)
    """
    state_dict = torch_load(model_path)
    return SAENet(state_dict, n_classes=n_classes, permute=permute)
