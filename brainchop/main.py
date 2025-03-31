import os
import sys
import argparse
import subprocess
from nibabel.nifti1 import save, Nifti1Image, load
from tinygrad.tensor import Tensor
import numpy as np

from brainchop.model import meshnet
from brainchop.tiny_meshnet import load_meshnet, qnormalize
from brainchop.niimath import conform, inverse_conform, bwlabel

from tinygrad.device import Device
from tinygrad.helpers import getenv
from pathlib import Path
from .utils import update_models, list_models, find_model_files, AVAILABLE_MODELS

def validate_file(file_path: str, description: str) -> None:
    if not os.path.isfile(file_path):
        print(f"Error: {description} not found: {file_path}")
        sys.exit(1)


def inference(args, model):
    # preprocess (should be factored out, since this is model dependent)
    conform_result = conform(args.input)
    img = conform_result[0]

    # forward pass
    tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
    t = Tensor(tensor.astype(np.float16))

    # post process (TODO: add export classes back in)
    out_tensor = model(t)
    save(Nifti1Image(out_tensor, img.affine, img.header), args.output)

    if args.inverse_conform:
        inverse_conform(args.input, args.output)
    print(f"Output saved as {args.output}")



def process_meshnet_model(args, config_fn: str, binary_fn: str) -> None:
    validate_file(args.input, "Input file")
    validate_file(config_fn, "Model JSON file")
    validate_file(binary_fn, "Model binary file")
    model = load_meshnet_tfjs(config_fn, binary_fn)
    inference(args, model)
    


def process_tiny_meshnet_model(args, config_fn: str, model_fn: str) -> None: # new backend
    # only supports mindgrab for now
    validate_file_exists(args.input, "Input file")
    validate_file_exists(config_fn, "Model JSON file")
    validate_file_exists(model_fn, "Model state dict file")
    model = load_meshnet(config_fn, model_fn)
    inference(args, model)


def process_multiaxial_model(args, model_dir: str) -> None:
    validate_file_exists(args.input, "Input file")

    try:
        from .multiaxial import multiaxial_segmentation
        
        # Ensure model_dir is an absolute path
        model_dir = os.path.abspath(model_dir)
        
        # Check if all required ONNX files exist, download if needed
        if not ensure_multiaxial_model_files(model_dir):
            print("Error: Failed to ensure all required multiaxial model files are available")
            sys.exit(1)
        
        print(f"Using multiaxial model from: {model_dir}")
        
        conform_result = conform(args.input)
        img = conform_result[0]
        out_image = multiaxial_segmentation(img, model_dir)
        save(out_image, args.output)

        
        bwlabel(args.output)
        if args.inverse_conform:
            print("Warning: Inverse conformation is not yet supported for multiaxial models")

        print(f"Output saved as {args.output}")

    except Exception as e:
        print(f"Error processing Multiaxial model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    default_model = next(iter(AVAILABLE_MODELS.keys()))
    parser = argparse.ArgumentParser(description="BrainChop: portable brain segmentation tool")
    parser.add_argument("input", nargs="?", help="Input NIfTI file path")
    parser.add_argument("-l", "--list", action="store_true", help="List available models")
    parser.add_argument("-i", "--inverse_conform", action="store_true", help="Perform inverse conformation into original image space")
    parser.add_argument("-u", "--update", action="store_true", help="Update the model listing")
    parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
    parser.add_argument("-m", "--model", default="", help=f"Name of segmentation model, default: {default_model}")
    parser.add_argument("-c", "--custom", type=str, help="Path to custom model directory or file (MeshNet or Multiaxial model)")
    parser.add_argument("-ec", "--export-classes", action="store_true", help="Export class probability maps (MeshNet only)")
    parser.add_argument("--cache-dir", type=str, default=str(Path.home() / ".cache" / "brainchop" / "models" / "multiaxial"),
                        help="Directory to cache downloaded multiaxial models")

    if getenv("PRINT_DEVICE", 0):
        print(Device.default)
    args = parser.parse_args()

    if args.update:
        update_models()
        return
    if args.list:
        list_models()
        return
    if not args.input:
        parser.print_help()
        return

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Handle custom model path provided with -c
    if args.custom:
        # Check if this directory has multiaxial model files
        custom_path = os.path.abspath(args.custom)
        required_onnx_files = ["sagittal_model.onnx", "coronal_model.onnx", "axial_model.onnx", "consensus_layer.onnx"]
        has_multiaxial_files = os.path.isdir(custom_path) and all(
            os.path.isfile(os.path.join(custom_path, f)) for f in required_onnx_files
        )
        
        # Check if this directory/file has MeshNet model files
        json_file, bin_file = find_custom_model(args.custom)
        has_meshnet_files = json_file is not None and bin_file is not None
        
        if has_multiaxial_files:
            # Process as multiaxial model using the custom directory
            process_multiaxial_model(args, custom_path)
        elif has_meshnet_files:
            # Process as MeshNet model using found files
            process_meshnet_model(args, json_file, bin_file)
        else:
            # No valid model files found at the custom path
            print(f"Error: No valid model files found at {args.custom}")
            print("For multiaxial models, directory must contain: sagittal_model.onnx, coronal_model.onnx, axial_model.onnx, consensus_layer.onnx")
            print("For MeshNet models, directory must contain: model.json and model.bin")
            sys.exit(1)
    # No custom path provided, use standard model from repository
    else:
        is_multiaxial = args.model in AVAILABLE_MODELS and AVAILABLE_MODELS[args.model].get("model_type") == "multiaxial"
        
        if is_multiaxial:
            # For multiaxial models, use cache directory directly
            cache_dir = Path(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            process_multiaxial_model(args, str(cache_dir))
        else:
            # For MeshNet models, use standard approach

            if args.model == "mindgrab":
                config_fn, model_fn = find_model_files(args.model)
                print(config_fn, model_fn)
                process_tiny_meshnet_model(args, config_fn, model_fn)

            else:
                model_dir_or_json, bin_file = find_model_files(args.model)
                
                if not model_dir_or_json:
                    print("Error: Unable to locate or download the required model files.")
                    sys.exit(1)
                if not bin_file:
                    print("Error: MeshNet model requires both JSON and binary files.")
                    sys.exit(1)
                process_meshnet_model(args, model_dir_or_json, bin_file)

    if os.path.exists("conformed.nii.gz"):
        subprocess.run(["rm", "conformed.nii.gz"])

if __name__ == "__main__":
    main()
