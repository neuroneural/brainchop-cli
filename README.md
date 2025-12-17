# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## Installation

```bash
pip install brainchop
```

For development (includes docs, testing):

```bash
pip install -e ".[all]"
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

# With BEAM optimization
brainchop input.nii.gz -m tissue_fast --beam 2 -o output.nii.gz
```

## Python API

```python
import brainchop as bc

# List available models
print(bc.list_models())

# Load, segment, save
vol = bc.load("input.nii.gz")
result = bc.segment(vol, "subcortical")
bc.save(result, "output.nii.gz")

# With BEAM optimization
bc.optimize("tissue_fast", beam=2)
result = bc.segment(vol, "tissue_fast")

# Export to WebGPU
bc.export("tissue_fast", "/tmp/export")
```

## Documentation

Serve docs locally:

```bash
mkdocs serve -w brainchop/
```

## Docker

```bash
git clone git@github.com:neuroneural/brainchop-cli.git
cd brainchop-cli
docker build -t brainchop .
```

Then to run:

```bash
docker run --rm -it --device=nvidia.com/gpu=all -v [[output directory]]:/app brainchop [[input nifti file]] -o [[output nifti file]]
```

## Requirements

- Python 3.10+
- tinygrad
- numpy
- requests

## License

MIT License
