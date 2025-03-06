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
global MODELS_JSON_URL
global AVAILABLE_MODELS

BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/"
MODELS_JSON_URL = "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"
AVAILABLE_MODELS = load_models()

def update_models():
    global AVAILABLE_MODELS
    AVAILABLE_MODELS = download_models_json()
    print("Model listing updated successfully.")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def list_available_models():
    print("Available models:")
    for model, details in AVAILABLE_MODELS.items():
        print(f"- {model}: {details['description']}")

def download_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available.")
        return None
    
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target cache directory: {cache_dir}")
    
    downloaded_paths = {}
    is_multiaxial = AVAILABLE_MODELS[model_name].get("model_type") == "multiaxial"
    
    files_to_download = (
        [
            "axial_model.onnx",
            "coronal_model.onnx",
            "sagittal_model.onnx",
            "consensus_layer.onnx"
        ] if is_multiaxial else
        ["model.json", "model.bin"]
    )
    
    for file in files_to_download:
        url = f"{BASE_URL}{model_dir}/{file}"
        local_path = cache_dir / file
        
        if not local_path.exists():
            print(f"Downloading {file} from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {file} to {local_path}")
            except requests.RequestException as e:
                print(f"Failed to download {file}: {str(e)}")
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
        if json_file.is_file() and bin_file.is_file():
            return str(json_file), str(bin_file)
        
        # Also check for multiaxial model files in current directory
        required_multiaxial_files = [
            "axial_model.onnx",
            "coronal_model.onnx",
            "sagittal_model.onnx",
            "consensus_layer.onnx"
        ]
        if all((current_dir / f).is_file() for f in required_multiaxial_files):
            return str(current_dir), None
        
        print("Model files not found in the current directory.")
        return None, None
    
    if not model_name:
        model_name = next(iter(AVAILABLE_MODELS))
        print(f"No model specified, defaulting to: {model_name}")
    
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' is not available.")
        return None, None
    
    model_dir = AVAILABLE_MODELS[model_name]["folder"]
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
    print(f"Checking cache directory: {cache_dir}")
    
    is_multiaxial = AVAILABLE_MODELS[model_name].get("model_type") == "multiaxial"
    
    if is_multiaxial:
        required_files = [
            "axial_model.onnx",
            "coronal_model.onnx",
            "sagittal_model.onnx",
            "consensus_layer.onnx"
        ]
        all_files = {f: cache_dir / f for f in required_files}
        
        missing_files = [f for f, path in all_files.items() if not path.is_file()]
        if missing_files:
            print(f"Missing multiaxial files: {missing_files}. Initiating download...")
            downloaded_files = download_model(model_name)
            if not downloaded_files or len(downloaded_files) != len(required_files):
                print(f"Failed to download all required multiaxial model files: {required_files}")
                return None, None
            print(f"Successfully downloaded multiaxial model files to {cache_dir}")
        else:
            print(f"Using cached multiaxial model files from {cache_dir}")
        return str(cache_dir), None
    else:
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
