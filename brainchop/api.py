"""
brainchop API - Core scripting interface for brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    volume, header = load("input.nii.gz")
    result = segment(volume, "subcortical")
    save(result, header, "output.nii.gz")
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
from tinygrad.tensor import Tensor

from brainchop.niimath import (
    conform,
    bwlabel,
    truncate_header_bytes,
)
from brainchop.tiny_meshnet import load_meshnet


def list_models() -> dict:
    """Return available models as {name: description}."""
    from brainchop.utils import AVAILABLE_MODELS
    return {name: details["description"] for name, details in AVAILABLE_MODELS.items()}


def load(path: str, *, crop: float | None = None, ct: bool = False) -> tuple[np.ndarray, bytes]:
    """
    Load NIfTI file, conform to 256^3.

    Returns:
        (volume, header) - volume is uint8 (256,256,256), header is bytes
    """
    from brainchop.utils import crop_to_cutoff

    volume, header = conform(os.path.abspath(path), ct=ct)
    if crop is not None:
        volume, _ = crop_to_cutoff(volume, crop)
    return volume, header


def save(volume: np.ndarray, header: bytes, path: str) -> None:
    """Save volume with header to NIfTI file."""
    header = truncate_header_bytes(header)
    gz = "1" if path.endswith(".gz") else "0"
    subprocess.run(
        ["niimath", "-", "-gz", gz, path, "-odt", "char"],
        input=header + volume.astype(np.uint8).tobytes(),
        check=True,
    )


def _load_model(name: str):
    """Load model by name, return tinygrad model."""
    from brainchop.utils import find_pth_files, AVAILABLE_MODELS, unwrap_path

    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(AVAILABLE_MODELS.keys())}")

    config_fn, model_fn = find_pth_files(name)
    return load_meshnet(unwrap_path(config_fn), unwrap_path(model_fn))


def segment(volume: np.ndarray, model: str, header: bytes | None = None) -> np.ndarray:
    """
    Segment brain volume.

    Args:
        volume: Input volume (256,256,256) uint8
        model: Model name (e.g., "subcortical", "tissue_fast")
        header: Optional header for bwlabel postprocessing

    Returns:
        Segmented volume (256,256,256) uint8
    """
    m = _load_model(model)

    # Prepare tensor (Z,Y,X) -> (1,1,D,H,W)
    tensor = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange("... -> 1 1 ...")

    if hasattr(m, "normalize"):
        tensor = m.normalize(tensor)

    # Run inference
    output = m(tensor).numpy()

    # Output is (1,D,H,W), transpose to (X,Y,Z)
    result = output[0].transpose((2, 1, 0)).astype(np.uint8)

    # Postprocess with bwlabel if header provided
    if header is not None:
        result, _ = bwlabel(header, result)

    return result


def segment_batch(
    volumes: list[np.ndarray],
    model: str,
    headers: list[bytes] | None = None,
    shard_size: int = 1,
) -> list[np.ndarray]:
    """Segment multiple volumes."""
    m = _load_model(model)
    results = []

    for i in range(0, len(volumes), shard_size):
        shard = volumes[i : i + shard_size]

        # Stack tensors
        tensors = [Tensor(v.transpose((2, 1, 0)).astype(np.float32)).rearrange("... -> 1 ...") for v in shard]
        batched = Tensor.stack(*tensors, dim=0) if len(tensors) > 1 else tensors[0].rearrange("... -> 1 ...")

        if hasattr(m, "normalize"):
            batched = m.normalize(batched)

        output = m(batched).numpy()

        for j, out in enumerate(output):
            result = out.transpose((2, 1, 0)).astype(np.uint8)
            if headers is not None:
                result, _ = bwlabel(headers[i + j], result)
            results.append(result)

    return results
