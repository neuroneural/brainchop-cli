"""
brainchop - Portable brain segmentation tool.

API-first design: use the Python API directly or via CLI.

Example:
    from brainchop import load_nifti, save_nifti, Model, list_models

    # List models
    for m in list_models():
        print(f"{m.name}: {m.description}")

    # Segment
    nifti = load_nifti("input.nii.gz")
    model = Model("subcortical")
    result = model.segment(nifti)
    save_nifti(result, "output.nii.gz")
"""

from brainchop.api import (
    NIfTI,
    Model,
    ModelInfo,
    list_models,
    load_nifti,
    load_niftis,
    save_nifti,
    nifti_to_tensor,
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
    "load_nifti",
    "load_niftis",
    "save_nifti",
    "nifti_to_tensor",
    "skull_strip",
    "argmax",
    "largest_component",
    "export_channels",
]
