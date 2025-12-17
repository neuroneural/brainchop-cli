"""
Export a brainchop model to WebGPU format.

Usage:
    WEBGPU=1 python examples/export_webgpu.py
"""

import os
os.environ["WEBGPU"] = "1"

from pathlib import Path
from tinygrad import Tensor
from tinygrad.nn.state import safe_save

from brainchop.utils import find_pth_files, unwrap_path
from brainchop.tiny_meshnet import load_meshnet
from brainchop.export_model import export_model

# Output directory
OUTPUT_DIR = Path("/tmp/brainchop_export")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
MODEL_NAME = "tissue_fast"
config_fn, model_fn = find_pth_files(MODEL_NAME)
model = load_meshnet(unwrap_path(config_fn), unwrap_path(model_fn))

# Create dummy input (1, 1, 256, 256, 256) float32
dummy_input = Tensor.zeros(1, 1, 256, 256, 256, dtype="float32")

# Export to WebGPU
print(f"Exporting {MODEL_NAME} to WebGPU...")
prg, input_sizes, output_sizes, state = export_model(model, "webgpu", dummy_input, model_name=MODEL_NAME)

# Save JavaScript code
js_path = OUTPUT_DIR / f"{MODEL_NAME}.js"
js_path.write_text(prg)
print(f"Saved: {js_path}")

# Save weights as safetensors
weights_path = OUTPUT_DIR / f"{MODEL_NAME}.safetensors"
safe_save(state, str(weights_path))
print(f"Saved: {weights_path}")

print(f"\nExport complete!")
print(f"  Input sizes: {input_sizes}")
print(f"  Output sizes: {output_sizes}")
print(f"  Output dir: {OUTPUT_DIR}")
