import requests
import json
from pathlib import Path
import nibabel as nib
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

BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/"
MESHNET_BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"
MULTIAXIAL_BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/multiaxial/"
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

def download_multiaxial_model(target_dir):
    """Download multiaxial model files to the specified directory."""
    required_files = [
        "axial_model.onnx",
        "coronal_model.onnx", 
        "sagittal_model.onnx",
        "consensus_layer.onnx"
    ]
    
    # Ensure target directory exists
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading multiaxial model files to {target_dir}")
    
    success = True
    for file in required_files:
        url = f"{MULTIAXIAL_BASE_URL}{file}"
        local_path = target_dir / file
        
        # Skip download if file already exists
        if local_path.exists():
            print(f"File {file} already exists at {local_path}")
            continue
        
        if not download_file(url, local_path):
            success = False
            print(f"Failed to download {file}")
    
    if success:
        print("Successfully downloaded all multiaxial model files")
    else:
        print("Failed to download some or all multiaxial model files")
    
    return success

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
    
    if is_multiaxial:
        files_to_download = [
            "axial_model.onnx",
            "coronal_model.onnx",
            "sagittal_model.onnx",
            "consensus_layer.onnx"
        ]
        base_url = MULTIAXIAL_BASE_URL
    else:
        files_to_download = ["model.json", "model.bin"]
        base_url = MESHNET_BASE_URL
    
    for file in files_to_download:
        url = f"{base_url}{model_dir}/{file}" if not is_multiaxial else f"{base_url}{file}"
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

def check_multiaxial_cache():
    """Check if multiaxial model files exist in the default cache location."""
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / "multiaxial"
    
    required_files = [
        "axial_model.onnx",
        "coronal_model.onnx", 
        "sagittal_model.onnx",
        "consensus_layer.onnx"
    ]
    
    if not cache_dir.exists():
        return False, cache_dir
    
    missing_files = [f for f in required_files if not (cache_dir / f).is_file()]
    if missing_files:
        return False, cache_dir
    
    return True, cache_dir

def reorient_to_lia(nii_img):
    """
    Reorient a NIfTI image from any orientation to LIA (Left-Inferior-Anterior).
    
    Args:
        nii_img (nibabel.Nifti1Image): The input NIfTI image.
        
    Returns:
        nibabel.Nifti1Image: The reoriented NIfTI image.
    """
    # Get the current orientation
    current_orientation = nib.aff2axcodes(nii_img.affine)
    print(f"Current orientation: {''.join(current_orientation)}")
    
    # Define target orientation as LIA
    target_orientation = "LIA"
    
    # Transform from current to target orientation
    orig_ornt = nib.io_orientation(nii_img.affine)
    targ_ornt = axcodes2ornt(target_orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    
    # Apply the transformation
    reoriented_img = nii_img.as_reoriented(transform)
    
    # Verify the new orientation
    new_orientation = nib.aff2axcodes(reoriented_img.affine)
    print(f"New orientation: {''.join(new_orientation)}")
    
    return reoriented_img
