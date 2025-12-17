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

```python
from brainchop import load, segment, save, list_models

# List available models
print(list_models())

# Load, segment, save
volume, header = load("input.nii.gz")
result = segment(volume, "subcortical", header)
save(result, header, "output.nii.gz")
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
