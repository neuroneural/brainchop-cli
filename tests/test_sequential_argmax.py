"""Test sequential argmax implementation against standard argmax."""

from tinygrad.tensor import Tensor
from tinygrad import nn
import numpy as np


class SequentialConvLayer:
    """
    Sequential argmax implementation using Conv3d.

    This layer applies multiple 1x1x1 convolutions and tracks which filter
    gives the maximum response at each spatial location - effectively
    computing argmax over the channel dimension in a sequential manner.
    """

    def __init__(self, in_channels: int, out_channels: int):
        self.convs = [nn.Conv2d(in_channels, 1, kernel_size=(1, 1, 1)) for _ in range(out_channels)]

    def __call__(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        depth, height, width = x.shape[2], x.shape[3], x.shape[4]

        # Initialize output tensors
        # outB tracks max values seen so far (initialized to very negative)
        # outC tracks indices of the max filter
        outB = Tensor.full((batch_size, 1, depth, height, width), -10000.0)
        outC = Tensor.zeros(batch_size, 1, depth, height, width)

        for i, conv in enumerate(self.convs):
            # Apply the current filter
            outA = conv(x).realize()

            # Find where the new filter gives a greater response than the max so far
            greater = (outA > outB).float()
            greater = greater.realize()

            # Update outB with the max values so far
            outB = ((1 - greater) * outB + greater * outA).realize()

            # Update outC with the index of the filter with the max response so far
            outC = ((1 - greater) * outC + greater * i).realize()

        return outC


def test_sequential_argmax_matches_standard():
    """Test that SequentialConvLayer produces same result as argmax."""
    np.random.seed(42)

    in_channels = 4
    out_channels = 8
    batch_size = 1
    depth, height, width = 4, 4, 4

    # Create input tensor
    x_np = np.random.randn(batch_size, in_channels, depth, height, width).astype(np.float32)
    x = Tensor(x_np)

    # Create sequential conv layer
    seq_layer = SequentialConvLayer(in_channels, out_channels)

    # To compare against standard argmax, we need to:
    # 1. Apply all convolutions to get (batch, out_channels, d, h, w)
    # 2. Then do argmax over channel dimension

    # Stack all conv outputs
    conv_outputs = []
    for conv in seq_layer.convs:
        out = conv(x).realize()
        conv_outputs.append(out.numpy())

    # Stack along channel dimension: (batch, out_channels, d, h, w)
    stacked = np.concatenate(conv_outputs, axis=1)

    # Standard argmax over channel dimension
    standard_argmax = np.argmax(stacked, axis=1, keepdims=True).astype(np.float32)

    # Run sequential implementation
    seq_result = seq_layer(x).numpy()

    print(f"Input shape: {x.shape}")
    print(f"Stacked conv outputs shape: {stacked.shape}")
    print(f"Standard argmax shape: {standard_argmax.shape}")
    print(f"Sequential result shape: {seq_result.shape}")
    print()
    print(f"Standard argmax unique values: {np.unique(standard_argmax)}")
    print(f"Sequential result unique values: {np.unique(seq_result)}")
    print()

    # Compare
    match = np.allclose(standard_argmax, seq_result)
    print(f"Results match: {match}")

    if not match:
        diff_mask = standard_argmax != seq_result
        print(f"Number of differences: {np.sum(diff_mask)}")
        print(f"Total elements: {standard_argmax.size}")
        print(f"Difference locations (first 10):")
        diff_indices = np.where(diff_mask)
        for idx in zip(*[d[:10] for d in diff_indices]):
            print(f"  {idx}: standard={standard_argmax[idx]}, sequential={seq_result[idx]}")

    return match


def test_simple_case():
    """Test with a simple, predictable case."""
    print("=" * 50)
    print("Simple test case")
    print("=" * 50)

    in_channels = 2
    out_channels = 3

    # Create deterministic input
    x_np = np.ones((1, in_channels, 2, 2, 2), dtype=np.float32)
    x = Tensor(x_np)

    # Create layer with known weights
    seq_layer = SequentialConvLayer(in_channels, out_channels)

    # Set weights manually to make filter 1 always win
    # Filter 0: small response
    # Filter 1: large response
    # Filter 2: medium response
    for i, conv in enumerate(seq_layer.convs):
        weight_val = float(i - 1)  # -1, 0, 1
        new_weight = Tensor.full(conv.weight.shape, weight_val)
        conv.weight = new_weight
        if conv.bias is not None:
            conv.bias = Tensor.zeros(conv.bias.shape)

    # Run sequential
    result = seq_layer(x).numpy()
    print(f"Sequential result (expect all 2s):\n{result.squeeze()}")

    # Verify manually
    conv_outputs = []
    for conv in seq_layer.convs:
        out = conv(x).realize()
        conv_outputs.append(out.numpy())
        print(f"Conv output: {out.numpy().mean():.4f}")

    stacked = np.concatenate(conv_outputs, axis=1)
    standard_argmax = np.argmax(stacked, axis=1, keepdims=True)
    print(f"Standard argmax:\n{standard_argmax.squeeze()}")
    print()


if __name__ == "__main__":
    test_simple_case()
    print()
    print("=" * 50)
    print("Random test case")
    print("=" * 50)
    success = test_sequential_argmax_matches_standard()
    print()
    if success:
        print("SUCCESS: Sequential argmax matches standard argmax!")
    else:
        print("FAILURE: Results do not match!")
