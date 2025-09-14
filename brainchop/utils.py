import requests
import os
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Any, Tuple
from urllib.parse import urlparse
from .niimath import _write_nifti


from .tfjs_meshnet import load_tfjs_meshnet
from .tiny_meshnet import load_meshnet
from .types import build_model  # Import the new model builder

# ! : is of type termination (meaning runtime is interrupted)

def download_model_listing():  # -> Json | !
    response = requests.get(MODELS_JSON_URL)
    response.raise_for_status()
    models = response.json()

    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    local_models_file.parent.mkdir(parents=True, exist_ok=True)

    with open(local_models_file, "w") as f:
        json.dump(models, f, indent=2)

    print(f"Downloaded models.json file to {local_models_file}")
    return models


def load_models():  # -> Json
    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    if local_models_file.exists():
        with open(local_models_file, "r") as f:
            return json.load(f)
    else:
        return download_model_listing()


def update_models() -> None:
    AVAILABLE_MODELS = download_model_listing()
    print("Model listing updated successfully.")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")


BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/"
MESHNET_BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"
MODELS_JSON_URL = (
    "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"
)
AVAILABLE_MODELS = load_models()
NEW_BACKEND = {"mindgrab", "."}
NEW_ARCHITECTURE_MODELS = set()  # Models using the new architecture format


def list_models() -> None:
    print("Available models:")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")


def download(url, local_path) -> None:  # -> None | !
    print(f"Downloading from {url} to {local_path}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def unwrap_path(path):  # -> String | !
    assert os.path.isfile(path), f"Error: {path} is not a file"
    return str(path)


def unwrap_model_name(s: str):  # -> String | !
    assert s in AVAILABLE_MODELS.keys(), f"Error: {s} is not an available model"
    return s


def detect_architecture_version(json_path: Path) -> str:
    """
    Detect whether a model JSON uses the new or old architecture format.
    
    Args:
        json_path: Path to the model JSON file
        
    Returns:
        str: "new" for new architecture format, "old" for deprecated format
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check for new architecture format indicators
        if "version" in data and "forward_pass" in data:
            return "new"
        # Check for old format indicators
        elif "layers" in data and isinstance(data.get("layers"), list):
            return "old"
        else:
            # Default to old if unclear
            return "old"
    except:
        return "old"


def find_pth_files(model_name) -> Tuple[Path | Any, Path | Any]:
    """New native backend for models"""
    if model_name == ".":
        return Path("model.json"), Path("model.pth")  # local model support model_name = unwrap_model_name(model_name)
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    json_fn = cache_dir / "model.json"
    pth_fn = cache_dir / "model.pth"

    base_url = MESHNET_BASE_URL
    for file in ["model.json", "model.pth"]:
        url = f"{base_url}{model_dir}/{file}"
        local_path = cache_dir / file
        if not local_path.exists():
            download(url, local_path)
    return json_fn, pth_fn


def find_tfjs_files(model_name) -> Tuple[Path | Any, Path | Any]:
    """Deprecated tfjs weight backend"""
    model_name = unwrap_model_name(model_name)
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    json_fn = cache_dir / "model.json"
    bin_fn = cache_dir / "model.bin"
    base_url = MESHNET_BASE_URL
    for file in ["model.json", "model.bin"]:
        url = f"{base_url}{model_dir}/{file}"
        local_path = cache_dir / file
        if not local_path.exists():
            download(url, local_path)
    return json_fn, bin_fn


def _load_model_from_uri(uri: str):
    """Loads a model from a file URI, correctly handling all path formats."""
    parsed_uri = urlparse(uri)
    if parsed_uri.scheme != 'file':
        raise ValueError(f"Unsupported URI scheme: {parsed_uri.scheme}")

    # Reconstruct path for non-standard cases like file://Users/spike/...
    path_str = parsed_uri.path
    if parsed_uri.netloc and parsed_uri.netloc != 'localhost':
        path_str = "/" + parsed_uri.netloc + path_str

    # **THE FIX**: If urlparse gives a path like '/~', strip the leading '/'
    # so that .expanduser() can correctly interpret the tilde.
    if path_str.startswith('/~'):
        path_str = path_str[1:]

    # Now, expand the tilde and resolve to an absolute path
    model_dir = Path(path_str).expanduser().resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_fn = model_dir / "model.json"
    pth_fn = model_dir / "model.pth"
    bin_fn = model_dir / "model.bin"

    if not config_fn.exists():
        raise FileNotFoundError(f"model.json not found in {model_dir}")

    if pth_fn.exists():
        weights_fn = pth_fn
    elif bin_fn.exists():
        weights_fn = bin_fn
    else:
        raise FileNotFoundError(f"No model weights (.pth or .bin) found in {model_dir}")

    return get_model_from_custom_path(str(config_fn), str(weights_fn))


# tinygrad model :: (pre-preprocessed) Tensor(1, ic,256,256,256) -> (pre-argmaxed) Tensor(1, oc, 256, 256, 256)
def get_model(model_name):  # -> tinygrad model
    if model_name.startswith("file://"):
        return _load_model_from_uri(model_name)
        
    if model_name in NEW_BACKEND:
        config_fn, model_fn = find_pth_files(model_name)
        config_fn = unwrap_path(config_fn)
        model_fn = unwrap_path(model_fn)
        
        # Detect architecture version
        arch_version = detect_architecture_version(Path(config_fn))
        
        if arch_version == "new":
            #print("brainchop :: Loading model with new architecture format")
            return build_model(config_fn, model_fn)
        else:
            #print("brainchop :: Loading model with legacy architecture format")
            return load_meshnet(config_fn, model_fn)
    else:  # oldbackend
        config_fn, binary_fn = find_tfjs_files(model_name)
        config_fn = unwrap_path(config_fn)
        binary_fn = unwrap_path(binary_fn)
        return load_tfjs_meshnet(config_fn, binary_fn)


def get_model_from_custom_path(config_path: str, weights_path: str):
    """
    Load a model from custom paths, auto-detecting the architecture format.
    
    Args:
        config_path: Path to the model JSON configuration. Can be "." to load the local model.
        weights_path: Path to the model weights (.pth or .bin). Can be "." to load the local model.
        
    Returns:
        Loaded model callable
    """
    # If the special "." local model name is passed, defer to the standard get_model logic.
    # This creates a dedicated code branch for explicitly specified local models.
    if config_path == "." or weights_path == ".":
        return get_model(".")

    config_p = Path(config_path)
    weights_p = Path(weights_path)
    
    if not config_p.exists():
        raise FileNotFoundError(f"Config file not found: {config_p}")
    if not weights_p.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_p}")
    
    # Detect architecture version
    arch_version = detect_architecture_version(config_p)
    
    if arch_version == "new":
        print("brainchop :: Loading custom model with new architecture format")
        if weights_p.suffix == ".bin":
            raise ValueError("New architecture format requires .pth weights file, got .bin")
        return build_model(str(config_p), str(weights_p))
    else:
        print("brainchop :: Loading custom model with legacy architecture format")
        if weights_p.suffix == ".bin":
            return load_tfjs_meshnet(str(config_p), str(weights_p))
        else:
            return load_meshnet(str(config_p), str(weights_p))


def cleanup() -> None:
    if os.path.exists("conformed.nii"):
        subprocess.run(["rm", "conformed.nii"])


def export_classes(output_channels, header: bytes, output_path: str):
    """
    Split the model's output channels and write each as a separate NIfTI
    using a pre‐built 352 B header (with vox_offset reset, ext_flag zeroed).

    Args:
        output_channels: tinygrad Tensor of shape (1, C, Z, Y, X)
        header:          352‐byte NIfTI header (bytes), no extensions
        output_path:     filename for first channel (e.g. "out.nii.gz")
    """
    # strip extensions so we can append "_c{i}.nii.gz"
    base, _ = os.path.splitext(output_path)
    if base.endswith(".nii"):
        base, _ = os.path.splitext(base)

    # pull into NumPy and drop the batch dim
    ch_np = output_channels.numpy().squeeze(0)  # shape (C, Z, Y, X)

    header = bytearray(header)
    header[70:74] = b"\x10\x00\x20\x00"
    header = bytes(header)

    # write each channel with our _write_nifti
    for i in range(ch_np.shape[0]):
        chan = ch_np[i].transpose((2, 1, 0))
        out_fname = f"{base}_c{i}.nii"
        _write_nifti(out_fname, chan, header)
        print(f"Saved channel {i} to {out_fname}")


def crop_to_cutoff(arr: np.ndarray, cutoff_percent: float = 2.0):
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        raise ValueError("Input must be a 3D numpy array.")

    # Compute cutoff using percentile without creating full flattened copy
    cutoff_value = np.percentile(arr, cutoff_percent)

    # Compute bounding axes projections faster than manual looping
    def axis_indices_max(arr, axis):
        axis_opt = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        projected_mask = np.any(arr > cutoff_value, axis=axis_opt[axis])
        indices = np.where(projected_mask)[0]
        return (indices[0], indices[-1]) if indices.size > 0 else (0, -1)

    x_min, x_max = 0, 255  # axis_indices_max(arr, 0)
    y_min, y_max = axis_indices_max(arr, 1)
    z_min, z_max = axis_indices_max(arr, 2)

    # Handle complete elimination
    if x_min > x_max or y_min > y_max or z_min > z_max:
        return np.empty((0, 0, 0), dtype=arr.dtype), (0, 0, 0, 0, 0, 0)

    cropped_arr = arr[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    return cropped_arr, (x_min, x_max, y_min, y_max, z_min, z_max)


def pad_to_original_size(
    cropped_arr: np.ndarray, coords: tuple, original_shape: tuple = (256, 256, 256)
):
    x_min, x_max, y_min, y_max, z_min, z_max = coords

    # Fast padding using zero padding with offset slicing
    padded_arr = np.zeros(original_shape, dtype=cropped_arr.dtype)

    # Check if crop is empty
    if (slice_size := cropped_arr.size) == 0:
        return padded_arr

    # Calculate crop dimensions dynamically instead copying shape
    x_size = x_max - x_min + 1
    y_size = y_max - y_min + 1
    z_size = z_max - z_min + 1

    # Coordinate-aware slicing that adjusts automatically to empty cases
    if x_size > 0 and y_size > 0 and z_size > 0:
        padded_arr[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1] = (
            cropped_arr
        )

    return padded_arr
