import os
import sys
import argparse
from pathlib import Path
from nibabel import save, load, Nifti1Image
from tinygrad import Tensor
import numpy as np
from brainchop.model import tinygrad_model
from brainchop.download import download_model, list_available_models, AVAILABLE_MODELS

# Add the directory containing the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

def find_model_files(model_name):
  if not model_name:
    # Default to the first model in AVAILABLE_MODELS
    model_name = next(iter(AVAILABLE_MODELS))
  
  # Check in ~/.cache/brainchop/models/
  cache_dir = Path.home() / ".cache" / "brainchop" / "models" / AVAILABLE_MODELS.get(model_name, model_name)
  json_file = cache_dir / "model.json"
  bin_file = cache_dir / "model.bin"
  if json_file.is_file() and bin_file.is_file():
    return str(json_file), str(bin_file)
  
  return None, None  # Always return a tuple, even if files are not found

def main():
  parser = argparse.ArgumentParser(description="BrainChop: portable brain segmentation tool")
  parser.add_argument("input", nargs="?", help="Input NIfTI file path")
  parser.add_argument("-l", "--list", action="store_true", help="List available models")
  parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
  parser.add_argument("-m", "--model", default="", help="Name of model")
  args = parser.parse_args()

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

  # If model files are not found, attempt to download them
  if not json_file or not bin_file:
    model_to_download = args.model or next(iter(AVAILABLE_MODELS))
    print(f"Model files for {model_to_download} not found locally. Downloading...")
    try:
      downloaded_files = download_model(model_to_download)
      if downloaded_files:
        json_file = downloaded_files["model.json"]
        bin_file = downloaded_files["model.bin"]
      else:
        raise Exception("Download failed")
    except Exception as e:
      print(f"Failed to download model files: {str(e)}")
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
