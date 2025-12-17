"""
Export a brainchop model to WebGPU format.

Usage:
    WEBGPU=1 python examples/export_webgpu.py
"""

import os
os.environ["WEBGPU"] = "1"

import brainchop as bc

js_path, weights_path = bc.export("tissue_fast", "/tmp/brainchop_export")

print(f"Exported to:")
print(f"  {js_path}")
print(f"  {weights_path}")
