import requests
import json
from pathlib import Path
import os
import sys
import argparse
from nibabel import save, load, Nifti1Image
from tinygrad import Tensor
import numpy as np
from brainchop.model import tinygrad_model

BASE_URL = "https://github.com/neuroneural/brainchop-models/raw/main/meshnet/"
MODELS_JSON_URL = "https://raw.githubusercontent.com/neuroneural/brainchop-cli/main/models.json"

def load_models():
  local_models_file = Path.home() / ".cache" / "brainchop" / "models.json"
  if local_models_file.exists():
    with open(local_models_file, "r") as f:
      return json.load(f)
  else:
    return download_models_json()

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

AVAILABLE_MODELS = load_models()

def update_models():
  global AVAILABLE_MODELS
  AVAILABLE_MODELS = download_models_json()
  print("Model listing updated successfully.")

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
  
  files_to_download = ["model.json", "model.bin"]
  downloaded_paths = {}
  
  for file in files_to_download:
    url = f"{BASE_URL}{model_dir}/{file}"
    local_path = cache_dir / file
    
    if not local_path.exists():
      response = requests.get(url)
      if response.status_code == 200:
        with open(local_path, "wb") as f:
          f.write(response.content)
        print(f"Downloaded {file} to {local_path}")
      else:
        print(f"Failed to download {file}. Status code: {response.status_code}")
        return None
    else:
      print(f"{file} already exists at {local_path}")
    
    downloaded_paths[file] = str(local_path)
  
  return downloaded_paths

def find_model_files(model_name):
  if model_name == ".":
    # Look for model files in the current directory
    current_dir = Path.cwd()
    json_file = current_dir / "model.json"
    bin_file = current_dir / "model.bin"
    if json_file.is_file() and bin_file.is_file():
      return str(json_file), str(bin_file)
    else:
      print("Model files not found in the current directory.")
      return None, None
  
  if not model_name:
    # Default to the first model in AVAILABLE_MODELS
    model_name = next(iter(AVAILABLE_MODELS))
  
  if model_name not in AVAILABLE_MODELS:
    print(f"Error: Model '{model_name}' is not available.")
    return None, None
  
  # Check in ~/.cache/brainchop/models/
  model_dir = AVAILABLE_MODELS[model_name]["folder"]
  cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
  json_file = cache_dir / "model.json"
  bin_file = cache_dir / "model.bin"
  
  if not json_file.is_file() or not bin_file.is_file():
    print(f"Model files for '{model_name}' not found locally. Downloading...")
    downloaded_files = download_model(model_name)
    if downloaded_files:
      json_file = Path(downloaded_files["model.json"])
      bin_file = Path(downloaded_files["model.bin"])
    else:
      return None, None
  
  return str(json_file), str(bin_file)

def main():
  default_model = next(iter(AVAILABLE_MODELS.keys()))
  parser = argparse.ArgumentParser(description="BrainChop: portable brain segmentation tool")
  parser.add_argument("input", nargs="?", help="Input NIfTI file path")
  parser.add_argument("-l", "--list", action="store_true", help="List available models")
  parser.add_argument("-u", "--update", action="store_true", help="Update the model listing")
  parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
  parser.add_argument("-m", "--model", default="",
                      help=f"Name of segmentation model, default: {default_model}")
  args = parser.parse_args()

  if args.update:
    update_models()
    return

  if args.list:
    list_available_models()
    return

  if not args.input:
    parser.print_help()
    return

  # Convert input and output paths to absolute paths
  args.input = os.path.abspath(args.input)
  args.output = os.path.abspath(args.output)

  # Find model files
  json_file, bin_file = find_model_files(args.model)

  if not json_file or not bin_file:
    print("Error: Unable to locate or download the required model files.")
    sys.exit(1)

  # Verify all required files exist
  for file_path in [args.input, json_file, bin_file]:
    if not os.path.isfile(file_path):
      print(f"Error: File not found: {file_path}")
      sys.exit(1)

  try:
    img = load(args.input)
    tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
    t = Tensor(tensor.astype(np.float16))
    out_tensor = tinygrad_model(json_file, bin_file, t)
    save(Nifti1Image(out_tensor, img.affine, img.header), args.output)
    print(f"Output saved as {args.output}")
  except Exception as e:
    print(f"An error occurred: {str(e)}")
    sys.exit(1)

if __name__ == "__main__":
  main()
