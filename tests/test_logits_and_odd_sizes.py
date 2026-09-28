"""Regression tests: per-class logits for --export-classes, and odd-sized
(e.g. --crop) inputs to the spatial autoencoder. Tiny random models, no downloads."""

import json

import numpy as np
import pytest
from tinygrad.tensor import Tensor

from brainchop.api import export_classes
from brainchop.sae_model import DOWNSAMPLE_LAYER, LAYER_INDICES, UPSAMPLE_LAYER, SAENet
from brainchop.tiny_meshnet import MeshNet

N_CLASSES = 3


def tiny_sae(channels=2):
    rng = np.random.default_rng(0)
    state = {}
    for idx in LAYER_INDICES:
        if idx == LAYER_INDICES[0]:
            shape = (channels, 1, 3, 3, 3)
        elif idx == LAYER_INDICES[-1]:
            shape = (N_CLASSES, channels, 1, 1, 1)
        elif idx == UPSAMPLE_LAYER:
            shape = (channels, channels, 2, 2, 2)  # conv_transpose: (in, out, k...)
        else:
            shape = (channels, channels, 3, 3, 3)
        out = shape[1] if idx == UPSAMPLE_LAYER else shape[0]
        state[f"{idx}.weight"] = Tensor(rng.standard_normal(shape).astype(np.float32) * 0.3)
        state[f"{idx}.bias"] = Tensor(np.zeros(out, dtype=np.float32))
    assert DOWNSAMPLE_LAYER in LAYER_INDICES
    return SAENet(state, n_classes=N_CLASSES)


def tiny_meshnet(tmp_path):
    layer = lambda d: {"kernel_size": 3, "padding": d, "stride": 1, "dilation": d}
    config = {
        "dropout_p": 0, "bnorm": True, "gelu": True,
        "layers": [layer(1), layer(2), {"kernel_size": 1, "padding": 0, "stride": 1, "dilation": 1}],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(config))
    m = MeshNet(in_channels=1, n_classes=N_CLASSES, channels=4, config_file=str(path))
    m.init_seq_conv_argmax()
    return m


def rand_input(size, seed):
    return Tensor(np.random.default_rng(seed).standard_normal((1, 1, *size)).astype(np.float32))


@pytest.mark.parametrize("size", [(8, 6, 4), (7, 6, 5), (9, 5, 3)])
def test_sae_keeps_input_grid(size):
    """Stride-2 down + up used to turn odd sizes n into n+1, breaking --crop."""
    assert tiny_sae()(rand_input(size, 1)).shape == (1, *size)


def test_sae_logits_match_fused_argmax():
    m = tiny_sae()
    x = rand_input((7, 6, 5), 1)
    logits = m(x, return_logits=True).numpy()
    assert logits.shape == (1, N_CLASSES, 7, 6, 5)
    np.testing.assert_array_equal(logits.argmax(axis=1), m(x).numpy())


def test_meshnet_logits_match_fused_argmax(tmp_path):
    """return_logits exposes what the fused conv+argmax computes internally."""
    m = tiny_meshnet(tmp_path)
    x = rand_input((6, 5, 4), 2)
    labels = m(x).numpy()
    logits = m(x, return_logits=True).numpy()
    assert logits.shape == (1, N_CLASSES, 6, 5, 4)
    np.testing.assert_array_equal(logits.argmax(axis=1), labels)


def test_export_classes_rejects_argmaxed_output(tmp_path):
    """Argmaxed labels used to crash deep inside with 'axes don't match array'."""
    with pytest.raises(ValueError, match="per-class output"):
        export_classes(Tensor.zeros(1, 6, 5, 4), bytes(352), str(tmp_path / "out.nii.gz"))
