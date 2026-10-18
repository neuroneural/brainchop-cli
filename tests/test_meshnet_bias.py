"""Legacy head-bias loading, without a PyTorch dependency or downloaded models."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tinygrad import Tensor
from brainchop.tiny_meshnet import convert_keys, load_meshnet


class MeshNetBiasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Path(self.tmp.name) / "model.json"
        self.architecture = {"bnorm": False, "dropout_p": 0, "layers": [
            {"in_channels": 1, "out_channels": 1, "kernel_size": 1,
             "padding": 0, "stride": 1, "dilation": 1},
            {"in_channels": 1, "out_channels": 2, "kernel_size": 1,
             "padding": 0, "stride": 1, "dilation": 1},
        ]}
        self.state = {
            "model.0.0.weight": Tensor.ones(1, 1, 1, 1, 1),
            "model.1.weight": Tensor([0., 1.]).reshape(2, 1, 1, 1, 1),
            "model.1.bias": Tensor([1.525439739227295, -1.5258055925369263]),
        }

    def load(self):
        self.config.write_text(json.dumps(self.architecture))
        with patch("brainchop.tiny_meshnet.safe_load", side_effect=ValueError), \
             patch("brainchop.tiny_meshnet.torch_load", return_value=self.state):
            return load_meshnet(str(self.config), "unused.pth")

    def test_missing_declaration_restores_trained_head_bias(self):
        model = self.load()
        np.testing.assert_array_equal(model.model[-1].bias.numpy(), self.state["model.1.bias"].numpy())
        self.assertIsNone(model.model[0].bias)
        x = Tensor.full((1, 1, 2, 2, 2), 2.)
        for chunk in (1, 2):
            np.testing.assert_array_equal(model(x, fuse_chunk=chunk).numpy(), np.zeros((1, 2, 2, 2)))

    def test_biasless_checkpoint_stays_biasless(self):
        del self.state["model.1.bias"]
        model = self.load()
        self.assertIsNone(model.model[-1].bias)
        np.testing.assert_array_equal(model(Tensor.full((1, 1, 2, 2, 2), 2.)).numpy(), np.ones((1, 2, 2, 2)))

    def test_explicit_head_bias_still_loads(self):
        self.architecture["layers"][-1]["bias"] = True
        np.testing.assert_array_equal(self.load().model[-1].bias.numpy(), self.state["model.1.bias"].numpy())

    def test_explicit_bias_false_does_not_silently_drop_checkpoint_bias(self):
        self.architecture["layers"][-1]["bias"] = False
        with self.assertRaisesRegex(ValueError, "parameters"):
            self.load()

    def test_convert_keys_rejects_extra_parameters_and_wrong_shapes(self):
        with self.assertRaisesRegex(ValueError, "parameters"):
            convert_keys({"a": Tensor.zeros(2), "b": Tensor.zeros(2)}, {"a": Tensor.zeros(2)})
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            convert_keys({"a": Tensor.zeros(2)}, {"a": Tensor.zeros(3)})


if __name__ == "__main__":
    unittest.main()
