"""
brainchop API - Core scripting interface for brain segmentation.

Example:
    from brainchop import load, segment, save, list_models

    vol = load("input.nii.gz")
    result = segment(vol, "subcortical")
    save(result, "output.nii.gz")
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tinygrad import Tensor

from brainchop.niimath import (
    conform,
    bwlabel,
    truncate_header_bytes,
)
from brainchop.tiny_meshnet import load_meshnet, chunked_conv
from tinygrad import nn


# =============================================================================
# BEAM Optimization Cache
# =============================================================================

def _get_cache_dir(model_name: str) -> Path:
    """Get cache directory for a model."""
    return Path.home() / ".cache" / "brainchop" / "models" / model_name


def _load_optimization_cache(model_name: str) -> dict:
    """Load optimization cache for a model."""
    cache_file = _get_cache_dir(model_name) / "optimizations.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"beams": []}


def _save_optimization_cache(model_name: str, batch_size: int, beam_value: int) -> None:
    """Save optimization cache entry."""
    cache_dir = _get_cache_dir(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "optimizations.json"

    cache_data = _load_optimization_cache(model_name)

    # Check if already cached
    for entry in cache_data["beams"]:
        if entry["BS"] == batch_size and entry["BEAM"] == beam_value:
            return

    # Add new entry
    cache_data["beams"].append({"BS": batch_size, "BEAM": beam_value})
    cache_data["beams"].sort(key=lambda x: (x["BS"], x["BEAM"]))

    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def _get_best_beam(model_name: str, batch_size: int) -> int | None:
    """Get best cached BEAM value for a batch size."""
    cache_data = _load_optimization_cache(model_name)
    beam_values = [e["BEAM"] for e in cache_data["beams"] if e["BS"] == batch_size]
    return max(beam_values) if beam_values else None


def _is_first_run(model_name: str, batch_size: int) -> bool:
    """Check if this is first run for model/batch_size combo."""
    cache_data = _load_optimization_cache(model_name)
    return not any(e["BS"] == batch_size for e in cache_data["beams"])


@dataclass
class Volume:
    """
    A brain volume with its NIfTI header.

    Attributes:
        data: Tensor of shape (256, 256, 256), dtype uint8
        header: 352-byte NIfTI header

    Example:
        ```python
        >>> vol = load("brain.nii.gz")
        >>> vol.data.shape
        (256, 256, 256)
        >>> len(vol.header)
        352
        ```
    """
    data: Tensor  # (256, 256, 256) uint8
    header: bytes  # 352-byte NIfTI header


def list_models() -> dict[str, str]:
    """
    Return available models as {name: description}.

    Example:
        ```python
        >>> list_models()
        {'subcortical': 'Subcortical structures', 'tissue_fast': 'Fast 3-class tissue', ...}
        ```
    """
    from brainchop.utils import AVAILABLE_MODELS
    return {name: details["description"] for name, details in AVAILABLE_MODELS.items()}


def load(path: str, *, crop: float | None = None, ct: bool = False, comply: bool = False) -> Volume:
    """
    Load NIfTI file and conform to 256x256x256.

    Args:
        path: Path to NIfTI file (.nii or .nii.gz)
        crop: Crop intensity percentile (e.g., 0.01 removes bottom 1%)
        ct: Use CT windowing instead of MRI normalization
        comply: Insert niimath compliance arguments before conform

    Returns:
        Volume with data Tensor (256,256,256) and header bytes

    Example:
        ```python
        >>> vol = load("brain.nii.gz")
        >>> vol.data.shape
        (256, 256, 256)

        >>> vol = load("brain.nii.gz", crop=0.01)  # crop dark voxels
        ```
    """
    from brainchop.utils import crop_to_cutoff

    data, header = conform(os.path.abspath(path), ct=ct, comply=comply)
    if crop is not None:
        data, _ = crop_to_cutoff(data, crop)
    return Volume(Tensor(data), header)


def save(volume: Volume, path: str) -> None:
    """
    Save volume to NIfTI file.

    Args:
        volume: Volume to save
        path: Output path (.nii or .nii.gz)

    Example:
        ```python
        >>> vol = load("brain.nii.gz")
        >>> result = segment(vol, "subcortical")
        >>> save(result, "segmented.nii.gz")
        ```
    """
    header = truncate_header_bytes(volume.header)
    gz = "1" if path.endswith(".gz") else "0"
    data = volume.data.cast("uint8").numpy().tobytes()
    subprocess.run(
        ["niimath", "-", "-gz", gz, path, "-odt", "char"],
        input=header + data,
        check=True,
    )


def _create_flip_matrix(size: int) -> Tensor:
    """Create a permutation matrix that reverses order along an axis."""
    import numpy as np
    # Anti-diagonal identity matrix: P[i, j] = 1 if j = size-1-i
    P = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        P[i, size - 1 - i] = 1.0
    return Tensor(P)


class TTAModel:
    """Test-Time Augmentation wrapper using flip ensemble."""

    def __init__(self, model):
        self._model = model
        self.n_classes = model.n_classes
        # Pre-create flip matrix for depth dimension (256)
        self._flip_matrix = _create_flip_matrix(256)

    def _flip_depth(self, x: Tensor) -> Tensor:
        """Flip tensor along depth axis using matrix multiplication."""
        # x: (B, C, D, H, W) - flip along D (axis 2)
        # Reshape to (B*C, D, H*W), apply permutation, reshape back
        b, c, d, h, w = x.shape
        x_reshaped = x.reshape(b * c, d, h * w)  # (B*C, D, H*W)
        # Permute depth: (B*C, D, H*W) @ (D, D) -> need to transpose for matmul
        # x_reshaped is (B*C, D, H*W), we want to permute the D dimension
        # So we do: P @ x_reshaped along the D axis
        # Reshape to (B*C, H*W, D) for matmul, then back
        x_t = x_reshaped.permute(0, 2, 1)  # (B*C, H*W, D)
        x_flipped_t = x_t @ self._flip_matrix  # (B*C, H*W, D)
        x_flipped = x_flipped_t.permute(0, 2, 1)  # (B*C, D, H*W)
        return x_flipped.reshape(b, c, d, h, w)

    def _forward_no_argmax(self, x: Tensor) -> Tensor:
        """Forward pass through conv layers without final argmax."""
        for layer in self._model.model:
            if isinstance(layer, nn.Conv2d):
                x = chunked_conv(x, layer)
            else:
                x = layer(x)
        return x

    def normalize(self, x: Tensor) -> Tensor:
        """Delegate normalization to underlying model."""
        return self._model.normalize(x)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass with flip ensemble (test-time augmentation).

        Runs inference on both original and depth-flipped inputs,
        sums the logits, then applies argmax.

        Args:
            x: Input tensor (B, 1, D, H, W)

        Returns:
            Segmentation (B, D, H, W) after argmax
        """
        # Force input into computation graph - add zero to create explicit dependency
        x = x + Tensor.zeros(1)

        # Forward on original
        logits_orig = self._forward_no_argmax(x)

        # Flip input on depth axis using permutation matrix
        x_flipped = self._flip_depth(x)
        logits_flipped = self._forward_no_argmax(x_flipped)

        # Unflip the flipped output on depth axis
        logits_unflipped = self._flip_depth(logits_flipped)

        # Sum logits and argmax (use model's seq_conv_argmax for WebGPU compatibility)
        summed = logits_orig + logits_unflipped
        return self._model.seq_conv_argmax(summed)


def _load_model(model: str, *, tta: bool = False):
    """
    Load model by name or path.

    Args:
        model: Model name (e.g., "subcortical") or path to model directory
               containing model.json and model.pth/model.bin
        tta: If True, wrap model with test-time augmentation (flip ensemble)
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

        m = load_meshnet(str(config_fn), str(weights_fn))
        return TTAModel(m) if tta else m

    # Otherwise treat as model name
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model}. Available: {list(AVAILABLE_MODELS.keys())}")

    config_fn, model_fn = find_pth_files(model)
    m = load_meshnet(unwrap_path(config_fn), unwrap_path(model_fn))
    return TTAModel(m) if tta else m


def optimize(model: str, *, beam: int = 2, batch_size: int = 1) -> None:
    """
    Pre-optimize a model with BEAM search for faster inference.

    Args:
        model: Model name (e.g., "subcortical") or path to model dir
        beam: BEAM optimization level (default: 2)
        batch_size: Batch size to optimize for (default: 1)

    Example:
        ```python
        >>> optimize("tissue_fast", beam=2)
        >>> # Now segment() will be faster
        >>> result = segment(vol, "tissue_fast")
        ```
    """
    import numpy as np

    model_name = Path(model).name if "/" in model else model

    print(f"brainchop :: Optimizing '{model_name}' with BEAM={beam}, BS={batch_size}...")

    original_beam = os.environ.get("BEAM")
    try:
        os.environ["BEAM"] = str(beam)
        m = _load_model(model)
        dummy = Tensor(np.random.randn(batch_size, 1, 256, 256, 256).astype(np.float32))
        _ = m(dummy).realize()
        _save_optimization_cache(model_name, batch_size, beam)
        print("brainchop :: Optimization complete!")
    finally:
        if original_beam is not None:
            os.environ["BEAM"] = original_beam
        elif "BEAM" in os.environ:
            del os.environ["BEAM"]


def segment(
    volume: Volume | list[Volume],
    model: str,
    *,
    shard_size: int = 1,
    beam: int = 0,
    return_raw: bool = False,
    tta: bool = False,
):  # type: ignore[return]
    """
    Segment brain volume(s).

    Args:
        volume: Single Volume or list of Volumes
        model: Model name (e.g., "subcortical", "tissue_fast") or path to model dir
        shard_size: Batch size for processing multiple volumes
        beam: BEAM optimization level (0 = use cached or none)
        return_raw: If True, also return raw model output (pre-argmax) for export_classes
        tta: If True, use test-time augmentation (flip ensemble)

    Returns:
        Segmented Volume(s) - single if input was single, list if input was list
        If return_raw=True, returns tuple of (result, raw_output)

    Example:
        ```python
        >>> vol = load("brain.nii.gz")
        >>> result = segment(vol, "subcortical")
        >>> result.data.shape
        (256, 256, 256)

        # With BEAM optimization
        >>> result = segment(vol, "subcortical", beam=2)

        # Batch processing
        >>> vols = [load(f"brain_{i}.nii.gz") for i in range(4)]
        >>> results = segment(vols, "tissue_fast", shard_size=2)
        >>> len(results)
        4

        # Get raw output for export_classes
        >>> result, raw = segment(vol, "subcortical", return_raw=True)
        >>> export_classes(raw, vol.header, "output.nii.gz")
        ```
    """
    # Handle single volume case
    single_input = not isinstance(volume, list)
    volumes: list[Volume] = [volume] if single_input else volume  # type: ignore[assignment]

    # Determine BEAM value
    model_name = Path(model).name if "/" in model else model
    if beam == 0:
        # Use cached value if available
        cached_beam = _get_best_beam(model_name, shard_size)
        beam = cached_beam if cached_beam else 0

    original_beam = os.environ.get("BEAM")
    try:
        if beam > 0:
            os.environ["BEAM"] = str(beam)

        m = _load_model(model, tta=tta)
        results: list[Volume] = []
        raw_outputs: list[Tensor] = []

        for i in range(0, len(volumes), shard_size):
            shard = volumes[i : i + shard_size]

            # Stack tensors: (X,Y,Z) -> (1,1,D,H,W) then batch
            tensors = [v.data.permute(2, 1, 0).cast("float32").rearrange("... -> 1 1 ...") for v in shard]
            batched = Tensor.stack(*tensors, dim=0).rearrange("b 1 ... -> b ...") if len(tensors) > 1 else tensors[0]

            if hasattr(m, "normalize"):
                batched = m.normalize(batched)

            raw_output = m(batched)
            # TTAModel returns raw logits (B, C, D, H, W), MeshNet returns argmaxed (B, D, H, W)
            if len(raw_output.shape) == 5:
                output = raw_output.argmax(axis=1)  # (B, C, D, H, W) -> (B, D, H, W)
            else:
                output = raw_output  # Already (B, D, H, W)

            # Split batch and convert back to (X,Y,Z)
            for j in range(output.shape[0]):
                out = output[j].permute(2, 1, 0).cast("uint8")  # (D,H,W) -> (X,Y,Z)
                header = shard[j].header
                out_np, _ = bwlabel(header, out.numpy())
                results.append(Volume(Tensor(out_np), header))
                if return_raw:
                    raw_outputs.append(raw_output[j:j+1])

        result = results[0] if single_input else results
        if return_raw:
            raw = raw_outputs[0] if single_input else raw_outputs
            return result, raw
        return result
    finally:
        if original_beam is not None:
            os.environ["BEAM"] = original_beam
        elif "BEAM" in os.environ:
            del os.environ["BEAM"]


def export(
    model: str,
    output_dir: str,
    *,
    target: str = "webgpu",
    beam: int = 0,
    tta: bool = False,
) -> tuple[str, str]:
    """
    Export model to WebGPU or other target format.

    Args:
        model: Model name (e.g., "tissue_fast") or path to model dir
        output_dir: Directory to save exported files
        target: Export target ("webgpu", "clang", "wasm")
        beam: BEAM optimization level (0 = no optimization)
        tta: If True, export with test-time augmentation (flip ensemble)

    Returns:
        Tuple of (js_path, weights_path)

    Example:
        ```python
        >>> js_path, weights_path = export("tissue_fast", "/tmp/export")
        >>> print(js_path)
        /tmp/export/tissue_fast.js

        # With BEAM optimization
        >>> js_path, weights_path = export("tissue_fast", "/tmp/export", beam=2)

        # With test-time augmentation
        >>> js_path, weights_path = export("tissue_fast", "/tmp/export", tta=True)
        ```
    """
    from tinygrad.nn.state import safe_save
    from brainchop.export_model import export_model

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get model name for output files
    model_name = Path(model).name if "/" in model else model
    if tta:
        model_name = f"{model_name}_tta"

    original_beam = os.environ.get("BEAM")
    try:
        if beam > 0:
            os.environ["BEAM"] = str(beam)

        # Load model
        m = _load_model(model, tta=tta)

        # Create dummy input - use randn to prevent JIT from treating as constant
        dummy_input = Tensor.randn(1, 1, 256, 256, 256, dtype="float32")

        # Export
        prg, _input_sizes, _output_sizes, state = export_model(
            m, target, dummy_input, model_name=model_name
        )

        # Save JS code
        js_path = output_path / f"{model_name}.js"
        js_path.write_text(prg)

        # Save weights
        weights_path = output_path / f"{model_name}.safetensors"
        safe_save(state, str(weights_path))

        return str(js_path), str(weights_path)
    finally:
        if original_beam is not None:
            os.environ["BEAM"] = original_beam
        elif "BEAM" in os.environ:
            del os.environ["BEAM"]


def export_classes(raw_output: Tensor, header: bytes, output_path: str) -> list[str]:
    """
    Export per-class probability maps as separate NIfTI files.

    Args:
        raw_output: Raw model output tensor, shape (1, C, D, H, W) before argmax
        header: 352-byte NIfTI header from input volume
        output_path: Base output path (e.g., "output.nii.gz")

    Returns:
        List of saved file paths

    Example:
        ```python
        >>> vol = load("brain.nii.gz")
        >>> result, raw = segment(vol, "subcortical", return_raw=True)
        >>> paths = export_classes(raw, vol.header, "output.nii.gz")
        >>> print(paths)
        ['output_c0.nii', 'output_c1.nii', ...]
        ```
    """
    from brainchop.niimath import _write_nifti

    # Strip extensions to append "_c{i}.nii"
    base = output_path
    for ext in [".nii.gz", ".nii"]:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    # Get numpy array and drop batch dim: (1, C, D, H, W) -> (C, D, H, W)
    ch_np = raw_output.numpy().squeeze(0)

    # Modify header for float data
    header_ba = bytearray(header)
    header_ba[70:74] = b"\x10\x00\x20\x00"  # Set datatype to float32
    header_bytes = bytes(header_ba)

    saved_paths = []
    for i in range(ch_np.shape[0]):
        # Transpose from (D, H, W) to (W, H, D) for NIfTI
        chan = ch_np[i].transpose((2, 1, 0))
        out_fname = f"{base}_c{i}.nii"
        _write_nifti(out_fname, chan, header_bytes)
        saved_paths.append(out_fname)
        print(f"brainchop :: Saved class {i} to {out_fname}")

    return saved_paths
