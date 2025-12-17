"""
Example: Using the brainchop Python API for brain segmentation.

This script demonstrates all the key API features:
1. List available models
2. Load NIfTI files (single or batch)
3. Load a model (with optional optimization)
4. Run inference
5. Save results
6. Advanced: export to WebGPU, direct tinygrad access
"""

from brainchop import NIfTI, Model, list_models

# -----------------------------------------------------------------------------
# 1. List available models
# -----------------------------------------------------------------------------
print("Available models:")
for model_info in list_models():
    print(f"  {model_info.name}: {model_info.description}")
    print(f"    - normalization: {model_info.normalization}")
print()

# -----------------------------------------------------------------------------
# 2. Load NIfTI files
# -----------------------------------------------------------------------------
# Single file
nifti = NIfTI.load("t1_crop.nii.gz")

# Multiple files (batch) - same API
# niftis = NIfTI.load(["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"])

# With options
# nifti = NIfTI.load("input.nii.gz", crop_percentile=2.0)  # crop for faster inference
# nifti = NIfTI.load("input.nii.gz", ct=True)  # CT scan conversion

# -----------------------------------------------------------------------------
# 3. Load a model
# -----------------------------------------------------------------------------
# Default (no optimization)
model = Model("subcortical")

# With BEAM optimization (explicit opt-in)
# model = Model("subcortical", optimize=True, beam_level=2)

# Custom model from local files
# model = Model(config_path="./my_model/model.json", weights_path="./my_model/model.pth")

# -----------------------------------------------------------------------------
# 4. Run inference
# -----------------------------------------------------------------------------
# Single file
segmentation = model.segment(nifti)

# Batch with shard_size for memory control
# segmentations = model.segment(niftis, shard_size=2)  # processes 2 at a time

# Raw output (no postprocessing)
# raw_output = model(nifti)  # returns numpy array (B, C, D, H, W)

# -----------------------------------------------------------------------------
# 5. Save results
# -----------------------------------------------------------------------------
segmentation.save("output.nii.gz")

# Uncompressed
# segmentation.save("output.nii", compress=False)

# Batch save
# for i, seg in enumerate(segmentations):
#     seg.save(f"output_{i}.nii.gz")

# -----------------------------------------------------------------------------
# 6. Advanced: WebGPU export
# -----------------------------------------------------------------------------
# Export model to WebGPU JavaScript
# NOTE: Must run script with WEBGPU=1 env var: WEBGPU=1 python examples/scripting.py
model.export(output_dir="/tmp")
# Creates: /tmp/subcortical.js + /tmp/subcortical.safetensors

# -----------------------------------------------------------------------------
# 7. Advanced: Direct tinygrad access
# -----------------------------------------------------------------------------
raw_model = model.tinygrad_model
print(raw_model)

tensor_input = nifti.to_tensor()  # returns Tensor (1, 1, 256, 256, 256)

raw_output = raw_model(tensor_input) # Run inference directly
