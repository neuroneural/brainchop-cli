"""Test sequential argmax implementation against standard argmax."""

from tinygrad.tensor import Tensor
import numpy as np

from brainchop.tiny_meshnet import sequential_argmax


def test_sequential_argmax_matches_standard():
    """Test that sequential_argmax produces same result as vanilla argmax."""
    np.random.seed(42)

    batch_size = 1
    num_channels = 10
    depth, height, width = 4, 4, 4

    x_np = np.random.randn(batch_size, num_channels, depth, height, width).astype(np.float32)
    x = Tensor(x_np)

    standard_result = x.argmax(axis=1).numpy()
    sequential_result = sequential_argmax(x).numpy()

    print(f"Input shape: {x.shape}")
    print(f"Standard argmax shape: {standard_result.shape}")
    print(f"Sequential argmax shape: {sequential_result.shape}")

    assert standard_result.shape == sequential_result.shape, \
        f"Shape mismatch: standard={standard_result.shape}, sequential={sequential_result.shape}"

    match = np.allclose(standard_result, sequential_result)
    print(f"Results match: {match}")
    assert match, "Sequential argmax does not match standard argmax!"


def test_sequential_argmax_simple():
    """Test with a simple, predictable case."""
    x_np = np.zeros((1, 5, 2, 2, 2), dtype=np.float32)
    x_np[0, 2, :, :, :] = 1.0  # Channel 2 is max
    x = Tensor(x_np)

    result = sequential_argmax(x).numpy()
    expected = np.full((1, 2, 2, 2), 2.0, dtype=np.float32)

    assert np.allclose(result, expected), f"Expected all 2s, got {np.unique(result)}"


def test_sequential_argmax_varying():
    """Test with varying argmax positions."""
    x_np = np.zeros((1, 4, 2, 2, 2), dtype=np.float32)
    x_np[0, 0, 0, 0, 0] = 1.0
    x_np[0, 1, 0, 0, 1] = 1.0
    x_np[0, 2, 0, 1, 0] = 1.0
    x_np[0, 3, 0, 1, 1] = 1.0
    x = Tensor(x_np)

    standard = x.argmax(axis=1).numpy()
    sequential = sequential_argmax(x).numpy()

    assert np.allclose(standard, sequential)


if __name__ == "__main__":
    test_sequential_argmax_simple()
    print("test_sequential_argmax_simple passed")

    test_sequential_argmax_varying()
    print("test_sequential_argmax_varying passed")

    test_sequential_argmax_matches_standard()
    print("test_sequential_argmax_matches_standard passed")

    print("\nAll tests passed!")
