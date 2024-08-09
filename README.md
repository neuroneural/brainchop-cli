# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## Installation

You can install BrainChop using pip:

```
pip install brainchop
```

## Usage

To use BrainChop, run the following command:

```
brainchop input.nii.gz -o output.nii.gz
```

Where:
- `input.nii.gz` is your input NIfTI file
- `output.nii.gz` is the desired output file name


## Requirements

- Python 3.6+
- tinygrad : our tiny and portable (but powerful) ML inference engine
- numpy : basic tensor operations
- nibabel : to read nifti files
- requests : to download models

## License

This project is licensed under the MIT License.
