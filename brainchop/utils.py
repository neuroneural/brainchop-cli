import requests
import os
import json
import numpy as np
from pathlib import Path
from typing import Any, Tuple


def download_model_listing():
    """Download models.json from GitHub."""
    response = requests.get(MODELS_JSON_URL)
    response.raise_for_status()
    models = response.json()

    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    local_models_file.parent.mkdir(parents=True, exist_ok=True)

    with open(local_models_file, "w") as f:
        json.dump(models, f, indent=2)

    print(f"Downloaded models.json to {local_models_file}")
    return models


def load_models():
    """Load models.json from cache or download."""
    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    if local_models_file.exists():
        with open(local_models_file, "r") as f:
            return json.load(f)
    else:
        return download_model_listing()


def update_models() -> None:
    """Update model listing from GitHub."""
    models = download_model_listing()
    print("Model listing updated:")
    for model, details in models.items():
        print(f"  {model}: {details['description']}")


BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/"
MESHNET_BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"
MODELS_JSON_URL = (
    "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"
)
AVAILABLE_MODELS = load_models()


def download(url, local_path) -> None:
    """Download file from URL."""
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def unwrap_path(path) -> str:
    """Assert path exists and return as string."""
    assert os.path.isfile(path), f"Error: {path} is not a file"
    return str(path)


def find_pth_files(model_name) -> Tuple[Path | Any, Path | Any]:
    """Find or download model config and weights."""
    if model_name == ".":
        return Path("model.json"), Path("model.pth")

    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    json_fn = cache_dir / "model.json"
    pth_fn = cache_dir / "model.pth"
    bin_fn = cache_dir / "model.bin"

    if not json_fn.exists():
        download(f"{MESHNET_BASE_URL}{model_dir}/model.json", json_fn)

    if pth_fn.exists():
        return json_fn, pth_fn
    elif bin_fn.exists():
        return json_fn, bin_fn
    else:
        try:
            download(f"{MESHNET_BASE_URL}{model_dir}/model.pth", pth_fn)
            return json_fn, pth_fn
        except Exception:
            download(f"{MESHNET_BASE_URL}{model_dir}/model.bin", bin_fn)
            return json_fn, bin_fn


def crop_to_cutoff(arr: np.ndarray, cutoff_percent: float = 2.0):
    """Crop volume to bounding box above percentile cutoff."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        raise ValueError("Input must be a 3D numpy array.")

    cutoff_value = np.percentile(arr, cutoff_percent)

    def axis_indices_max(arr, axis):
        axis_opt = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        projected_mask = np.any(arr > cutoff_value, axis=axis_opt[axis])
        indices = np.where(projected_mask)[0]
        return (indices[0], indices[-1]) if indices.size > 0 else (0, -1)

    x_min, x_max = 0, 255
    y_min, y_max = axis_indices_max(arr, 1)
    z_min, z_max = axis_indices_max(arr, 2)

    if x_min > x_max or y_min > y_max or z_min > z_max:
        return np.empty((0, 0, 0), dtype=arr.dtype), (0, 0, 0, 0, 0, 0)

    cropped_arr = arr[x_min : x_max + 1, y_min : y_max + 1, z_min : z_max + 1]
    return cropped_arr, (x_min, x_max, y_min, y_max, z_min, z_max)
