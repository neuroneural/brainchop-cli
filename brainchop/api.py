"""
brainchop API - Core scripting interface for brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    vol = load("input.nii.gz")
    result = segment(vol, "subcortical")
    save(result, "output.nii.gz")
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from tinygrad import Tensor

from brainchop.niimath import (
    conform,
    bwlabel,
    truncate_header_bytes,
)
from brainchop.tiny_meshnet import load_meshnet


@dataclass
class Volume:
    """A brain volume with its NIfTI header."""
    data: Tensor  # (256, 256, 256) uint8
    header: bytes  # 352-byte NIfTI header


def list_models() -> dict[str, str]:
    """Return available models as {name: description}."""
    from brainchop.utils import AVAILABLE_MODELS
    return {name: details["description"] for name, details in AVAILABLE_MODELS.items()}


def load(path: str, *, crop: float | None = None, ct: bool = False) -> Volume:
    """
    Load NIfTI file, conform to 256^3.

    Returns:
        Volume with data Tensor (256,256,256) and header bytes
    """
    from brainchop.utils import crop_to_cutoff

    data, header = conform(os.path.abspath(path), ct=ct)
    if crop is not None:
        data, _ = crop_to_cutoff(data, crop)
    return Volume(Tensor(data), header)


def save(volume: Volume, path: str) -> None:
    """Save volume to NIfTI file."""
    header = truncate_header_bytes(volume.header)
    gz = "1" if path.endswith(".gz") else "0"
    data = volume.data.cast("uint8").numpy().tobytes()
    subprocess.run(
        ["niimath", "-", "-gz", gz, path, "-odt", "char"],
        input=header + data,
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
    volume: Volume | list[Volume],
    model: str,
    shard_size: int = 1,
) -> Volume | list[Volume]:
    """
    Segment brain volume(s).

    Args:
        volume: Single Volume or list of Volumes
        model: Model name (e.g., "subcortical", "tissue_fast") or path to model dir
        shard_size: Batch size for processing multiple volumes

    Returns:
        Segmented Volume(s) - single if input was single, list if input was list
    """
    # Handle single volume case
    single_input = not isinstance(volume, list)
    volumes: list[Volume] = [volume] if single_input else volume  # type: ignore[assignment]

    m = _load_model(model)
    results: list[Volume] = []

    for i in range(0, len(volumes), shard_size):
        shard = volumes[i : i + shard_size]

        # Stack tensors: (X,Y,Z) -> (1,1,D,H,W) then batch
        tensors = [v.data.permute(2, 1, 0).cast("float32").rearrange("... -> 1 1 ...") for v in shard]
        batched = Tensor.stack(*tensors, dim=0).rearrange("b 1 ... -> b ...") if len(tensors) > 1 else tensors[0]

        if hasattr(m, "normalize"):
            batched = m.normalize(batched)

        output = m(batched)  # (B, D, H, W)

        # Split batch and convert back to (X,Y,Z)
        for j in range(output.shape[0]):
            out = output[j].permute(2, 1, 0).cast("uint8")  # (D,H,W) -> (X,Y,Z)
            header = shard[j].header
            out_np, _ = bwlabel(header, out.numpy())
            results.append(Volume(Tensor(out_np), header))

    return results[0] if single_input else results
