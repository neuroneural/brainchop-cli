"""
brainchop API - Core scripting interface for brain segmentation.

This module provides a clean, API-first interface for brain MRI segmentation.
The CLI is a thin wrapper around this API.

Example usage:
    from brainchop import NIfTI, Model, list_models

    # List available models
    for m in list_models():
        print(f"{m.name}: {m.description}")

    # Load and segment
    nifti = NIfTI.load("input.nii.gz")
    model = Model("subcortical")
    result = model.segment(nifti)
    result.save("output.nii.gz")
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
from tinygrad.tensor import Tensor

from brainchop.niimath import (
    conform,
    bwlabel,
    set_header_intent_label,
    truncate_header_bytes,
)
from brainchop.tiny_meshnet import load_meshnet


@dataclass
class ModelInfo:
    """Metadata about an available model."""

    name: str
    description: str
    folder: str
    normalization: str  # "minmax" | "quantile"


def list_models() -> list[ModelInfo]:
    """Return rich metadata for all available models."""
    from brainchop.utils import AVAILABLE_MODELS

    result = []
    for name, details in AVAILABLE_MODELS.items():
        # Parse normalization from considerations field
        considerations = details.get("considerations", "")
        if "quantile" in considerations.lower():
            normalization = "quantile"
        else:
            normalization = "minmax"

        result.append(
            ModelInfo(
                name=name,
                description=details.get("description", ""),
                folder=details.get("folder", ""),
                normalization=normalization,
            )
        )
    return result


@dataclass
class NIfTI:
    """
    Wrapper for NIfTI brain volumes.

    Holds the conformed 256x256x256 volume, header bytes, and optional
    crop coordinates for restoring original size.
    """

    volume: np.ndarray  # 256x256x256 uint8 (or cropped shape)
    header: bytes  # NIfTI header bytes
    crop_coords: tuple[int, ...] | None = None  # (x_min, x_max, y_min, y_max, z_min, z_max)
    source_path: str | None = None  # Original file path

    @overload
    @classmethod
    def load(
        cls,
        path: str,
        *,
        crop_percentile: float | None = None,
        ct: bool = False,
        comply: bool = False,
    ) -> NIfTI: ...

    @overload
    @classmethod
    def load(
        cls,
        path: list[str],
        *,
        crop_percentile: float | None = None,
        ct: bool = False,
        comply: bool = False,
    ) -> list[NIfTI]: ...

    @classmethod
    def load(
        cls,
        path: str | list[str],
        *,
        crop_percentile: float | None = None,
        ct: bool = False,
        comply: bool = False,
    ) -> NIfTI | list[NIfTI]:
        """
        Load and conform NIfTI file(s) to 256x256x256 uint8.

        Args:
            path: Path to NIfTI file, or list of paths
            crop_percentile: If set, crop volume by this percentile cutoff (faster inference)
            ct: Convert CT scans from Hounsfield to Cormack units
            comply: Insert compliance arguments to niimath

        Returns:
            Single NIfTI if path is string, list of NIfTI if path is list
        """
        if isinstance(path, list):
            return [
                cls._load_single(p, crop_percentile=crop_percentile, ct=ct, comply=comply)
                for p in path
            ]
        return cls._load_single(path, crop_percentile=crop_percentile, ct=ct, comply=comply)

    @classmethod
    def _load_single(
        cls,
        path: str,
        *,
        crop_percentile: float | None = None,
        ct: bool = False,
        comply: bool = False,
    ) -> NIfTI:
        """Load a single NIfTI file."""
        from brainchop.utils import crop_to_cutoff

        abs_path = os.path.abspath(path)
        volume, header = conform(abs_path, comply=comply, ct=ct)

        crop_coords = None
        if crop_percentile is not None:
            volume, crop_coords = crop_to_cutoff(volume, crop_percentile)

        return cls(
            volume=volume,
            header=header,
            crop_coords=crop_coords,
            source_path=abs_path,
        )

    def to_tensor(self) -> Tensor:
        """Convert to tinygrad Tensor in model-ready format (1, 1, D, H, W)."""
        # Transpose from (X, Y, Z) to (Z, Y, X) for model input
        arr = self.volume.transpose((2, 1, 0)).astype(np.float32)
        return Tensor(arr).rearrange("... -> 1 1 ...")

    def save(self, path: str, *, compress: bool = True) -> None:
        """
        Save NIfTI volume to path.

        Args:
            path: Output path (.nii or .nii.gz)
            compress: Whether to gzip compress (default True, inferred from extension)
        """
        gzip_flag = "1" if compress and not path.endswith(".nii") else "0"
        header = truncate_header_bytes(self.header)

        cmd = ["niimath", "-", "-gz", gzip_flag, path, "-odt", "char"]
        subprocess.run(
            cmd,
            input=header + self.volume.tobytes(),
            check=True,
        )


class Model:
    """
    Brain segmentation model wrapper.

    Provides high-level inference API while exposing the underlying
    tinygrad model for advanced use cases like export.
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        config_path: str | None = None,
        weights_path: str | None = None,
        optimize: bool = False,
        beam_level: int = 2,
    ):
        """
        Load a model by name or from custom paths.

        Args:
            name: Model name (e.g., "subcortical", "mindgrab")
            config_path: Path to custom model.json (alternative to name)
            weights_path: Path to custom model.pth (required if config_path set)
            optimize: Enable BEAM optimization (explicit opt-in)
            beam_level: BEAM level when optimize=True (default 2)
        """
        self._name = name
        self._optimize = optimize
        self._beam_level = beam_level

        # Handle optimization
        original_beam = os.environ.get("BEAM")
        if optimize:
            os.environ["BEAM"] = str(beam_level)

        try:
            if config_path and weights_path:
                # Custom model from paths
                self._model = load_meshnet(config_path, weights_path)
                self._info = ModelInfo(
                    name="custom",
                    description="Custom model",
                    folder="",
                    normalization="minmax",  # Default assumption
                )
            elif name:
                # Load by name
                self._model, self._info = self._load_by_name(name)
            else:
                raise ValueError("Must provide either 'name' or both 'config_path' and 'weights_path'")
        finally:
            # Restore original BEAM
            if original_beam is not None:
                os.environ["BEAM"] = original_beam
            elif "BEAM" in os.environ and optimize:
                del os.environ["BEAM"]

    def _load_by_name(self, name: str) -> tuple:
        """Load model by name from registry."""
        from brainchop.utils import find_pth_files, AVAILABLE_MODELS, unwrap_path

        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {name}. Use list_models() to see available models.")

        config_fn, model_fn = find_pth_files(name)
        config_fn = unwrap_path(config_fn)
        model_fn = unwrap_path(model_fn)

        model = load_meshnet(config_fn, model_fn)

        # Build ModelInfo
        details = AVAILABLE_MODELS[name]
        considerations = details.get("considerations", "")
        normalization = "quantile" if "quantile" in considerations.lower() else "minmax"

        info = ModelInfo(
            name=name,
            description=details.get("description", ""),
            folder=details.get("folder", ""),
            normalization=normalization,
        )

        return model, info

    @property
    def info(self) -> ModelInfo:
        """Model metadata."""
        return self._info

    @property
    def tinygrad_model(self):
        """Access the underlying tinygrad model directly for export/advanced use."""
        return self._model

    @overload
    def __call__(self, nifti: NIfTI, *, shard_size: int = 1) -> np.ndarray: ...

    @overload
    def __call__(self, nifti: list[NIfTI], *, shard_size: int = 1) -> np.ndarray: ...

    def __call__(self, nifti: NIfTI | list[NIfTI], *, shard_size: int = 1) -> np.ndarray:
        """
        Run inference, return raw output (B, C, D, H, W).

        Args:
            nifti: Single NIfTI or list of NIfTIs
            shard_size: Process inputs in chunks of this size (for memory management)

        Returns:
            Raw model output as numpy array
        """
        niftis = [nifti] if isinstance(nifti, NIfTI) else nifti

        all_outputs = []
        for i in range(0, len(niftis), shard_size):
            shard = niftis[i : i + shard_size]

            # Stack tensors
            tensors = [n.to_tensor() for n in shard]
            if len(tensors) == 1:
                batched = tensors[0]
            else:
                # Stack along batch dimension
                batched = Tensor.stack(*[t.squeeze(0) for t in tensors], dim=0)

            # Normalize
            if hasattr(self._model, "normalize"):
                batched = self._model.normalize(batched)

            # Run inference
            output = self._model(batched)
            all_outputs.append(output.numpy())

        # Concatenate all shard outputs
        return np.concatenate(all_outputs, axis=0)

    @overload
    def segment(
        self,
        nifti: NIfTI,
        *,
        postprocess: bool = True,
        shard_size: int = 1,
    ) -> NIfTI: ...

    @overload
    def segment(
        self,
        nifti: list[NIfTI],
        *,
        postprocess: bool = True,
        shard_size: int = 1,
    ) -> list[NIfTI]: ...

    def segment(
        self,
        nifti: NIfTI | list[NIfTI],
        *,
        postprocess: bool = True,
        shard_size: int = 1,
    ) -> NIfTI | list[NIfTI]:
        """
        Run inference + postprocessing, return segmented NIfTI(s).

        Args:
            nifti: Single NIfTI or list of NIfTIs
            postprocess: Apply argmax + bwlabel (default True)
            shard_size: Process inputs in chunks of this size

        Returns:
            Segmented NIfTI(s) - single if input was single, list if input was list
        """
        from brainchop.utils import pad_to_original_size

        single_input = isinstance(nifti, NIfTI)
        niftis_list: list[NIfTI] = [nifti] if single_input else nifti  # type: ignore[assignment]

        # Run inference
        raw_output = self.__call__(niftis_list, shard_size=shard_size)  # type: ignore[arg-type]

        # Post-process each result
        results: list[NIfTI] = []
        for i, n in enumerate(niftis_list):
            output_channels = raw_output[i : i + 1]  # Keep batch dim

            if postprocess:
                # Model always outputs argmaxed labels (1, D, H, W)
                # Rearrange from (1, D, H, W) to (Z, Y, X)
                output = output_channels[0].transpose((2, 1, 0)).astype(np.uint8)

                # Pad back if cropped
                if n.crop_coords is not None:
                    output = pad_to_original_size(output, n.crop_coords)

                # Connected component labeling
                labels, new_header = bwlabel(n.header, output)
                new_header = set_header_intent_label(new_header)
            else:
                # No postprocessing - just return raw
                output = output_channels[0].transpose((0, 3, 2, 1)).astype(np.float32)
                new_header = n.header
                labels = output

            results.append(
                NIfTI(
                    volume=labels,
                    header=new_header,
                    crop_coords=None,  # Already restored
                    source_path=n.source_path,
                )
            )

        return results[0] if single_input else results

    def export(
        self,
        output_dir: str = ".",
        *,
        stream_weights: bool = False,
    ) -> tuple[str, str]:
        """
        Export model to WebGPU format.

        NOTE: Must run with WEBGPU=1 environment variable set before importing brainchop.
        Example: WEBGPU=1 python my_export_script.py

        Args:
            output_dir: Directory to write output files
            stream_weights: Whether to stream weights

        Returns:
            Tuple of (code_path, weights_path)
        """
        from brainchop.export_model import export_model
        from tinygrad.nn.state import safe_save
        from tinygrad.tensor import Device

        if Device.DEFAULT != "WEBGPU":
            raise RuntimeError(
                "Export requires WEBGPU device. Run with WEBGPU=1 env var set before importing brainchop.\n"
                "Example: WEBGPU=1 python my_script.py"
            )

        # Create a dummy input tensor for tracing
        dummy_input = Tensor(np.random.randn(1, 1, 256, 256, 256).astype(np.float32))

        model_name = self._name or "model"
        prg, _, _, state = export_model(
            self._model,
            "webgpu",
            dummy_input,
            model_name=model_name,
            stream_weights=stream_weights,
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save code
        code_path = out_path / f"{model_name}.js"
        with open(code_path, "w") as f:
            f.write(prg)

        # Save weights
        weights_path = out_path / f"{model_name}.safetensors"
        safe_save(state, str(weights_path))

        return str(code_path), str(weights_path)


def skull_strip(
    nifti: NIfTI | list[NIfTI],
    *,
    border_mm: int = 0,
    shard_size: int = 1,
) -> NIfTI | list[NIfTI]:
    """
    Skull strip using mindgrab model.

    Convenience function, equivalent to Model('mindgrab').segment(...)

    Args:
        nifti: Single NIfTI or list of NIfTIs
        border_mm: Morphological border in mm (default 0)
        shard_size: Process inputs in chunks of this size

    Returns:
        Skull-stripped NIfTI(s)
    """
    model = Model("mindgrab")
    return model.segment(nifti, shard_size=shard_size)


def argmax(output: np.ndarray) -> np.ndarray:
    """Convert multi-channel output to single-channel labels."""
    return output.argmax(axis=1)


def largest_component(volume: np.ndarray, neighbors: int = 26) -> np.ndarray:
    """Keep only the largest connected component."""
    counts = np.bincount(volume.ravel().astype(np.int32))
    if len(counts) > 1:
        largest_label = counts[1:].argmax() + 1
        volume[volume != largest_label] = 0
    return volume


def export_channels(
    output: np.ndarray,
    header: bytes,
    output_dir: str,
) -> list[str]:
    """
    Save each channel as separate NIfTI file.

    Args:
        output: Model output array (C, D, H, W) or (B, C, D, H, W)
        header: NIfTI header bytes
        output_dir: Directory to write files

    Returns:
        List of output file paths
    """
    from brainchop.utils import export_classes as _export_classes

    # Handle batch dimension
    if output.ndim == 5:
        output = output[0]  # Take first batch

    output_tensor = Tensor(output).rearrange("c d h w -> 1 c d h w")
    output_path = str(Path(output_dir) / "output.nii.gz")
    _export_classes(output_tensor, header, output_path)

    # Return list of created files
    base = str(Path(output_dir) / "output")
    return [f"{base}_c{i}.nii" for i in range(output.shape[0])]
