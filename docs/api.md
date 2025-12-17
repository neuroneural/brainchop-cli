# API Reference

## Volume

A brain volume with its NIfTI header.

```python
@dataclass
class Volume:
    data: Tensor    # (256, 256, 256) uint8
    header: bytes   # 352-byte NIfTI header
```

## Functions

### load

```python
def load(path: str, *, crop: float | None = None, ct: bool = False) -> Volume
```

Load NIfTI file and conform to 256x256x256.

- `path`: Path to NIfTI file (.nii or .nii.gz)
- `crop`: Crop to percentile (e.g., 0.01 removes bottom 1%)
- `ct`: Use CT windowing instead of MRI normalization

### segment

```python
def segment(
    volume: Volume | list[Volume],
    model: str,
    shard_size: int = 1,
) -> Volume | list[Volume]
```

Segment brain volume(s).

- `volume`: Single Volume or list of Volumes
- `model`: Model name (e.g., "subcortical") or path to model directory
- `shard_size`: Batch size for processing multiple volumes

Returns segmented Volume(s) matching input type.

### save

```python
def save(volume: Volume, path: str) -> None
```

Save volume to NIfTI file.

### list_models

```python
def list_models() -> dict[str, str]
```

Return available models as `{name: description}`.

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
