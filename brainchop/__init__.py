"""
brainchop - Portable brain segmentation tool.

API-first design: use the Python API directly or via CLI.

Example:
    from brainchop import NIfTI, Model, list_models

    # List models
    for m in list_models():
        print(f"{m.name}: {m.description}")

    # Segment
    nifti = NIfTI.load("input.nii.gz")
    model = Model("subcortical")
    result = model.segment(nifti)
    result.save("output.nii.gz")
"""

from brainchop.api import (
    NIfTI,
    Model,
    ModelInfo,
    list_models,
    skull_strip,
    argmax,
    largest_component,
    export_channels,
)

__all__ = [
    "NIfTI",
    "Model",
    "ModelInfo",
    "list_models",
    "skull_strip",
    "argmax",
    "largest_component",
    "export_channels",
]
