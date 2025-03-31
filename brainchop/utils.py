import requests
import json
from pathlib import Path
import sys

def download_models_json():
    try:
        response = requests.get(MODELS_JSON_URL)
        response.raise_for_status()
        models = response.json()
        
        local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
        local_models_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_models_file, "w") as f:
            json.dump(models, f, indent=2)
        
        print(f"Downloaded models.json file to {local_models_file}")
        return models
    except Exception as e:
        print(f"Error downloading models.json: {str(e)}")
        sys.exit(1)

def load_models():
    local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
    if local_models_file.exists():
        with open(local_models_file, "r") as f:
            return json.load(f)
    else:
        return download_models_json()

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

def update_models():
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = download_models_json()
    print("Model listing updated successfully.")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def list_models():
    print("Available models:")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def download_file(url, local_path):
    """Helper function to download a file from URL to local path."""
    try:
        print(f"Downloading from {url} to {local_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded to {local_path}")
        return True
    except requests.RequestException as e:
        print(f"Failed to download: {str(e)}")
        return False

def download_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available.")
        return None
    
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = {}
    
    if model_name == "mindgrab": # new backend goes in here
        base_url = MESHNET_BASE_URL
        files_to_download = ["model.json", "model.pth"]
    else:
        base_url = MESHNET_BASE_URL
        files_to_download = ["model.json", "model.bin"]

    
    for file in files_to_download:
        url = f"{base_url}{model_dir}/{file}"
        local_path = cache_dir / file
        
        if not local_path.exists():
            if not download_file(url, str(local_path)):
                return None
        else:
            print(f"{file} already exists at {local_path}")
        downloaded_paths[file] = str(local_path)
    return downloaded_paths

def find_model_files(model_name):
    if model_name == ".":
        current_dir = Path.cwd()
        json_file = current_dir / "model.json"
        bin_file = current_dir / "model.bin"
        pth_file = current_dir / "model.pth"

        if json_file.is_file() and bin_file.is_file() and model_name != "mindgrab":
            return str(json_file), str(bin_file)
        if model_name == "mindgrab":
            return str(json_file), str(pth_file)
        
    if not model_name:
        model_name = next(iter(AVAILABLE_MODELS))
        print(f"No model specified, defaulting to: {model_name}")
    
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available.")
        return None, None
    
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    print(f"Checking cache directory: {cache_dir}")
    
    json_file = cache_dir / "model.json"
    bin_file = cache_dir / "model.bin"
    
    if not json_file.is_file() or not bin_file.is_file():
        print(f"MeshNet model files not found locally. Downloading...")
        downloaded_files = download_model(model_name)
        if downloaded_files:
            json_file = Path(downloaded_files["model.json"])
            bin_file = Path(downloaded_files["model.bin"])
            print(f"Successfully downloaded MeshNet model files to {cache_dir}")
        else:
            return None, None
    else:
        print(f"Using cached MeshNet model files from {cache_dir}")
    
    return str(json_file), str(bin_file)

def find_custom_model(model_path: str) -> tuple[str | None, str | None]:
    import os
    try:
        model_path = os.path.abspath(model_path)
        if os.path.isfile(model_path):
            dirname = os.path.dirname(model_path)
            basename = os.path.basename(model_path)
            if basename.endswith('.json'):
                bin_path = os.path.join(dirname, 'model.bin')
                return model_path, bin_path if os.path.exists(bin_path) else None
            elif basename.endswith('.bin'):
                json_path = os.path.join(dirname, 'model.json')
                return json_path if os.path.exists(json_path) else None, model_path
        json_path = os.path.join(model_path, 'model.json')
        bin_path = os.path.join(model_path, 'model.bin')
        if os.path.isfile(json_path) and os.path.isfile(bin_path):
            return json_path, bin_path
        return None, None
    except Exception as e:
        print(f"Error finding custom model files: {str(e)}")
        return None, None
