"""
Example: brainchop Python API
"""

from brainchop import load, segment, save, list_models, Volume

# List models
print("Available models:")
for name, desc in list_models().items():
    print(f"  {name}: {desc}")

# Load, segment, save
vol = load("t1_crop.nii.gz")
result = segment(vol, "tissue_fast")
assert isinstance(result, Volume)  # single input -> single output
save(result, "output.nii.gz")
