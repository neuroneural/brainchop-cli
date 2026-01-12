"""
Export a brainchop model with test-time augmentation (TTA) to WebGPU format.

TTA uses flip ensemble: runs inference on both original and depth-flipped
inputs, then sums the logits for improved segmentation accuracy.

Usage:
    WEBGPU=1 python examples/export_taa.py
"""

import os
os.environ["WEBGPU"] = "1"

import brainchop as bc

js_path, weights_path = bc.export("tissue_fast", "/tmp/brainchop_export", taa=True)

print(f"Wrote model program to: {js_path}")
print(f"Wrote model weights to: {weights_path}")
