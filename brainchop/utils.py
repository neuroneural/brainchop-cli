import requests
import os
import subprocess
import json
from pathlib import Path
from typing import Any, Tuple

from .tfjs_meshnet import load_tfjs_meshnet
from .tiny_meshnet import load_meshnet

# ! : is of type termination (meaning runtime is interrupted)

def download_model_listing(): # -> Json | !
    response = requests.get(MODELS_JSON_URL)
    response.raise_for_status()
    models = response.json()
    
    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    local_models_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(local_models_file, "w") as f:
        json.dump(models, f, indent=2)
    
    print(f"Downloaded models.json file to {local_models_file}")
    return models

def load_models(): # -> Json
    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    if local_models_file.exists():
        with open(local_models_file, "r") as f:
            return json.load(f)
    else:
        return download_model_listing()

global BASE_URL
global MESHNET_BASE_URL
global MULTIAXIAL_BASE_URL
global MODELS_JSON_URL
global AVAILABLE_MODELS
global NEW_BACKEND

BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/"
MESHNET_BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"
MODELS_JSON_URL = "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"
AVAILABLE_MODELS = load_models()
NEW_BACKEND = {"mindgrab"}

def update_models() -> None:
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = download_model_listing()
    print("Model listing updated successfully.")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def list_models() -> None:
    print("Available models:")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def download(url, local_path) -> None: # -> None | !
    print(f"Downloading from {url} to {local_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def find_pth_files(model_name) -> Tuple[Path|Any, Path|Any]:
    """ New native backend for models """
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

def find_tfjs_files(model_name)-> Tuple[Path|Any, Path|Any]:
    """ Deprecated tfjs weight backend """
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


def unwrap_path(path):
    assert os.path.isfile(path), f"Error: {path} is not a file"
    return str(path)

# tinygrad model :: (unpreprocessed) Tensor(1, ic,256,256,256) -> Tensor(1, oc, 256, 256, 256)
def get_model(model_name): # -> tinygrad model
    if model_name in NEW_BACKEND:
        config_fn, model_fn = find_pth_files(model_name)
        config_fn = unwrap_path(config_fn)
        model_fn = unwrap_path(model_fn)
        return load_meshnet(config_fn, model_fn) # TODO: other configs should be loaded from json
    else: # oldbackend
        config_fn, binary_fn = find_tfjs_files(model_name)
        config_fn = unwrap_path(config_fn)
        binary_fn = unwrap_path(binary_fn)
        return load_tfjs_meshnet(config_fn, binary_fn)
    # even elser: load multiaxial and other models (this should be a standalone file)

def cleanup() -> None:
    if os.path.exists("conformed.nii.gz"):
        subprocess.run(["rm", "conformed.nii.gz"])
