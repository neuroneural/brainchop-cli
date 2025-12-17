"""
brainchop - Portable brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    vol = load("input.nii.gz")
    result = segment(vol, "subcortical")
    save(result, "output.nii.gz")
"""

from brainchop.api import Volume, load, save, segment, list_models, export, optimize, export_classes

__all__ = ["Volume", "load", "save", "segment", "list_models", "export", "optimize", "export_classes"]
