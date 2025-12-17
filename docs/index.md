# brainchop

GPU-accelerated brain MRI segmentation using tinygrad.

## Installation

```bash
pip install brainchop
```

## CLI Usage

```bash
# Segment a brain MRI
brainchop input.nii.gz -o output.nii.gz

# List available models
brainchop --list

# Use a specific model
brainchop input.nii.gz -m subcortical -o output.nii.gz

# Skull stripping
brainchop input.nii.gz --skull-strip -o brain.nii.gz
```

## Python API

```python
from brainchop import load, segment, save, list_models

# Load and segment
vol = load("input.nii.gz")
result = segment(vol, "subcortical")
save(result, "output.nii.gz")

# List models
for name, desc in list_models().items():
    print(f"{name}: {desc}")
```

See [Examples](examples.md) for more use cases and [API Reference](api.md) for details.

## Available Models

| Model | Description |
|-------|-------------|
| `subcortical` | Subcortical structures (default) |
| `tissue_fast` | Fast tissue segmentation (GM/WM/CSF) |
| `tissue_full` | Full tissue segmentation |
| `mindgrab` | Skull stripping |
| `dkatlas` | DK atlas parcellation |
