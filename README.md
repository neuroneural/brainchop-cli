# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## Installation

You can install BrainChop using pip (Python > 3.10)

```
pip install brainchop
```

## CLI Usage

To use BrainChop from the command line:

```bash
brainchop input.nii.gz -o output.nii.gz
```

List available models:

```bash
brainchop --list
```

Use a specific model:

```bash
brainchop input.nii.gz -m subcortical -o output.nii.gz
```

## Python API

BrainChop provides a clean Python API for scripting:

```python
from brainchop import NIfTI, Model, list_models

# List available models
for m in list_models():
    print(f"{m.name}: {m.description}")

# Load and segment a brain scan
nifti = NIfTI.load("input.nii.gz")
model = Model("subcortical")
result = model.segment(nifti)
result.save("output.nii.gz")
```

### Batch Processing

```python
# Load multiple scans
niftis = NIfTI.load(["scan1.nii.gz", "scan2.nii.gz", "scan3.nii.gz"])

# Segment all (with memory-efficient sharding)
model = Model("tissue_fast")
results = model.segment(niftis, shard_size=2)

# Save outputs
for i, result in enumerate(results):
    result.save(f"output_{i}.nii.gz")
```

### Custom Models

```python
# Load a custom model with MeshNet format
model = Model(config_path="custom/model.json", weights_path="custom/model.pth")

# Load a custom model with Spec format (supports funky layers)
model = Model(config_path="spec/model.json", weights_path="spec/model.pth")
```

### WebGPU Export

Export models for browser deployment:

```bash
# Must set WEBGPU=1 before importing brainchop
WEBGPU=1 python export_script.py
```

```python
# export_script.py
from brainchop import Model

model = Model("tissue_fast")
code_path, weights_path = model.export(output_dir="./web_model")
```

## Docker

You can also install BrainChop using docker:

```bash
git clone git@github.com:neuroneural/brainchop-cli.git
cd brainchop-cli
docker build -t brainchop .
```

Then to run:

```bash
docker run --rm -it --device=nvidia.com/gpu=all -v [[output directory]]:/app brainchop [[input nifti file]] -o [[output nifti file]]
```

On some systems (like recent 25.05 nixos), the docker run command will need to be prepended with:

```bash
docker run --rm -it --device=nvidia.com/gpu=all
```

## Requirements

- Python 3.10+
- tinygrad : our tiny and portable (but powerful) ML inference engine
- numpy : basic tensor operations
- requests : to download models

Sometimes it may be necessary to install tinygrad from master branch:

```bash
uv pip install git+ssh://git@github.com/tinygrad/tinygrad.git
```

## License

This project is licensed under the MIT License.
