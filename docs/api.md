# API Reference

## Volume

::: brainchop.Volume
    options:
      show_source: false

## Functions

::: brainchop.load
    options:
      show_source: false

::: brainchop.segment
    options:
      show_source: false

::: brainchop.save
    options:
      show_source: false

::: brainchop.list_models
    options:
      show_source: false

## Examples

### Batch Processing

```python
from brainchop import load, segment, save

# Process multiple files
volumes = [load(f"scan_{i}.nii.gz") for i in range(4)]
results = segment(volumes, "subcortical", shard_size=2)

for i, result in enumerate(results):
    save(result, f"output_{i}.nii.gz")
```

### Custom Model

```python
from brainchop import load, segment, save

vol = load("input.nii.gz")
result = segment(vol, "/path/to/custom/model")
save(result, "output.nii.gz")
```

Custom model directory must contain:

- `model.json` - MeshNet configuration
- `model.pth` or `model.bin` - Weights
