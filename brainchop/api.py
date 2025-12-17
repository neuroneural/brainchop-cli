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


def _load_model(model: str):
    """
    Load model by name or path.

    Args:
        model: Model name (e.g., "subcortical") or path to model directory
               containing model.json and model.pth/model.bin
    """
    from pathlib import Path
    from brainchop.utils import find_pth_files, AVAILABLE_MODELS, unwrap_path

    # Check if it's a path (absolute, relative, or file://)
    if model.startswith("file://"):
        model = model.replace("file://", "")
        if model.startswith("~"):
            model = os.path.expanduser(model)

    model_path = Path(model)
    if model_path.exists() and model_path.is_dir():
        # Custom model directory
        config_fn = model_path / "model.json"
        if not config_fn.exists():
            raise FileNotFoundError(f"No model.json found in {model_path}")

        # Find weights file
        pth_fn = model_path / "model.pth"
        bin_fn = model_path / "model.bin"
        if pth_fn.exists():
            weights_fn = pth_fn
        elif bin_fn.exists():
            weights_fn = bin_fn
        else:
            raise FileNotFoundError(f"No model.pth or model.bin found in {model_path}")

        return load_meshnet(str(config_fn), str(weights_fn))

    # Otherwise treat as model name
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model}. Available: {list(AVAILABLE_MODELS.keys())}")

    config_fn, model_fn = find_pth_files(model)
    return load_meshnet(unwrap_path(config_fn), unwrap_path(model_fn))


def segment(
    volume: np.ndarray | list[np.ndarray],
    model: str,
    header: bytes | list[bytes] | None = None,
    shard_size: int = 1,
) -> np.ndarray | list[np.ndarray]:
    """
    Segment brain volume(s).

    Args:
        volume: Single volume (256,256,256) or list of volumes
        model: Model name (e.g., "subcortical", "tissue_fast")
        header: Optional header(s) for bwlabel postprocessing
        shard_size: Batch size for processing multiple volumes

    Returns:
        Segmented volume(s) - single array if input was single, list if input was list
    """
    # Handle single volume case
    single_input = not isinstance(volume, list)
    volumes = [volume] if single_input else volume
    headers = [header] if single_input and header is not None else header

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

    return results[0] if single_input else results
