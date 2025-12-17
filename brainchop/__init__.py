"""
brainchop - Portable brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    volume, header = load("input.nii.gz")
    result = segment(volume, "subcortical", header)
    save(result, header, "output.nii.gz")
"""

from brainchop.api import load, save, segment, segment_batch, list_models

__all__ = ["load", "save", "segment", "segment_batch", "list_models"]
