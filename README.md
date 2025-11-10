# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## Installation

You can install BrainChop using pip (Python > 3.10)


```
pip install brainchop
```

## Usage

To use BrainChop, run the following command:

```
brainchop input.nii.gz -o output.nii.gz
```




## Docker

You can also install BrainChop using docker
```
git clone git@github.com:neuroneural/brainchop-cli.git
cd brainchop-cli
docker build -t brainchop .
```

Then to run, use
```
docker run --rm -it --device=nvidia.com/gpu=all -v [[output directory]]:/app brainchop [[input nifti file]] -o [[output nifti file]]
```

On some systems (like recent 25.05 nixos), the docker run command will need to be prepended with
```
docker run --rm -it --device=nvidia.com/gpu=all
```

Where:
- `input.nii.gz` is your input NIfTI file
- `output.nii.gz` is the desired output file name


## Requirements

- Python 3.10+
- tinygrad : our tiny and portable (but powerful) ML inference engine
- numpy : basic tensor operations
- requests : to download models

sometimes it may be necessary to install tinygrad from master branch. in that case:
```
uv pip install git+ssh://git@github.com/tinygrad/tinygrad.git
```

## (Experimental) Webgpu Model export

prepend these flags to the brainchop call (only works off of github install)
```
PREARGMAX=1 WEBGPU=1 PYTHONPATH=. EXPORT=1
```


## License

This project is licensed under the MIT License.
