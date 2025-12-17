"""
Example: brainchop Python API
"""

from brainchop import load, segment, save, list_models

# List models
print("Available models:")
for name, desc in list_models().items():
    print(f"  {name}: {desc}")

# Load, segment, save
volume, header = load("t1_crop.nii.gz")
result = segment(volume, "tissue_fast", header)
save(result, header, "output.nii.gz")
