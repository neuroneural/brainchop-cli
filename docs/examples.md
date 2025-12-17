# Examples

## Basic Usage

```python
import brainchop as bc

# Load a NIfTI file
vol = bc.load("brain.nii.gz")

# Segment with a model
result = bc.segment(vol, "subcortical")

# Save the result
bc.save(result, "segmented.nii.gz")
```

## List Available Models

```python
import brainchop as bc

for name, description in bc.list_models().items():
    print(f"{name}: {description}")
```

## Batch Processing

```python
import brainchop as bc

# Load multiple volumes
volumes = [bc.load(f"scan_{i}.nii.gz") for i in range(4)]

# Segment all at once with shard_size for memory control
results = bc.segment(volumes, "tissue_fast", shard_size=2)

# Save results
for i, result in enumerate(results):
    bc.save(result, f"output_{i}.nii.gz")
```

## BEAM Optimization

BEAM optimization compiles optimized GPU kernels for faster inference.

### Pre-optimize a Model

```python
import brainchop as bc

# Optimize once (cached for future runs)
bc.optimize("tissue_fast", beam=2)

# Now segment() uses cached optimization
vol = bc.load("brain.nii.gz")
result = bc.segment(vol, "tissue_fast")  # uses cached BEAM=2
```

### Explicit BEAM Level

```python
import brainchop as bc

vol = bc.load("brain.nii.gz")

# Specify BEAM directly (no caching)
result = bc.segment(vol, "tissue_fast", beam=2)
```

### Optimize for Batch Size

```python
import brainchop as bc

# Optimize for batch_size=4
bc.optimize("tissue_fast", beam=2, batch_size=4)

# Now batched inference is optimized
volumes = [bc.load(f"scan_{i}.nii.gz") for i in range(4)]
results = bc.segment(volumes, "tissue_fast", shard_size=4)
```

## WebGPU Export

Export models for browser-based inference.

```python
import brainchop as bc

# Basic export
js_path, weights_path = bc.export("tissue_fast", "/tmp/export")
print(f"Model: {js_path}")
print(f"Weights: {weights_path}")

# With BEAM optimization
js_path, weights_path = bc.export("tissue_fast", "/tmp/export", beam=2)
```

Note: WebGPU export requires `WEBGPU=1` environment variable:

```bash
WEBGPU=1 python export_script.py
```

## Custom Models

Load models from a local directory:

```python
import brainchop as bc

vol = bc.load("brain.nii.gz")

# Custom model directory must contain:
#   - model.json (MeshNet config)
#   - model.pth or model.bin (weights)
result = bc.segment(vol, "/path/to/custom/model")

bc.save(result, "output.nii.gz")
```

## CLI Examples

```bash
# Basic segmentation
brainchop input.nii.gz -o output.nii.gz

# Specific model
brainchop input.nii.gz -m tissue_fast -o tissue.nii.gz

# Skull stripping
brainchop input.nii.gz --skull-strip -o brain.nii.gz

# List models
brainchop --list

# With explicit BEAM optimization
brainchop input.nii.gz -m tissue_fast --beam 2 -o output.nii.gz

# Skip optimization prompt
brainchop input.nii.gz -m tissue_fast --no-optimize -o output.nii.gz

# Batch processing
brainchop scan1.nii.gz scan2.nii.gz scan3.nii.gz -m subcortical

# CT scan
brainchop ct_scan.nii.gz --ct -m tissue_fast -o output.nii.gz
```
