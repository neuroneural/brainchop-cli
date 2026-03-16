"""
brainchop - Portable brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    vol = load("input.nii.gz")
    result = segment(vol, "subcortical")
    save(result, "output.nii.gz")
"""

import os

# Auto-configure ROCm LD_LIBRARY_PATH if on a ROCm node. Set DIY=1 to skip.
if os.environ.get("DIY") != "1" and os.path.isdir("/opt/rocm/lib"):
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if "/opt/rocm/lib" not in ld:
        os.environ["LD_LIBRARY_PATH"] = f"/opt/rocm/lib:{ld}" if ld else "/opt/rocm/lib"

from brainchop.api import Volume, load, save, segment, list_models, export, optimize, export_classes

__all__ = ["Volume", "load", "save", "segment", "list_models", "export", "optimize", "export_classes"]
