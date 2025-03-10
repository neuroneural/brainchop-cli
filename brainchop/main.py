import os
import sys
import argparse
import subprocess
from nibabel.nifti1 import save, Nifti1Image, load
from tinygrad.tensor import Tensor
import numpy as np
from brainchop.model import meshnet
from brainchop.niimath import conform, inverse_conform, bwlabel
from tinygrad.device import Device
from tinygrad.helpers import getenv
from pathlib import Path
from .utils import update_models, list_available_models, find_model_files, download_multiaxial_model, AVAILABLE_MODELS

def find_custom_model(model_path: str) -> tuple[str | None, str | None]:
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

def validate_file_exists(file_path: str, description: str) -> None:
    if not os.path.isfile(file_path):
        print(f"Error: {description} not found: {file_path}")
        sys.exit(1)

def ensure_multiaxial_model_files(model_dir: str) -> bool:
    """Ensure all required ONNX files for multiaxial model exist, download if needed."""
    model_dir_path = Path(model_dir)
    required_files = ["sagittal_model.onnx", "coronal_model.onnx", "axial_model.onnx", "consensus_layer.onnx"]
    
    # Check if all files exist
    all_exist = all(os.path.isfile(model_dir_path / file) for file in required_files)
    
    if not all_exist:
        print(f"Some multiaxial model files are missing in {model_dir}")
        # Try to download from GitHub
        return download_multiaxial_model(model_dir_path)
    
    return True

def process_meshnet_model(args, json_file: str, bin_file: str) -> None:
    validate_file_exists(args.input, "Input file")
    validate_file_exists(json_file, "Model JSON file")
    validate_file_exists(bin_file, "Model binary file")

    try:
        conform_result = conform(args.input)
        img = conform_result[0]
        tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
        t = Tensor(tensor.astype(np.float16))

        out_tensor, raw_classes = meshnet(json_file, bin_file, t, args.export_classes)
        save(Nifti1Image(out_tensor, img.affine, img.header), args.output)

        if args.export_classes and raw_classes is not None:
            output_base = os.path.splitext(args.output)[0]
            if output_base.endswith('.nii'):
                output_base = os.path.splitext(output_base)[0]
            num_classes = raw_classes.shape[1]
            for class_idx in range(num_classes):
                class_output_path = f"{output_base}_class{class_idx}.nii.gz"
                class_data = raw_classes[0, class_idx]
                save(Nifti1Image(class_data, img.affine, img.header), class_output_path)
                print(f"Class {class_idx} probability map saved as {class_output_path}")

        bwlabel(args.output)
        if args.inverse_conform:
            inverse_conform(args.input, args.output)

        print(f"Output saved as {args.output}")

    except Exception as e:
        print(f"Error processing MeshNet model: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

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
        # img = load(args.input)
        out_tensor = multiaxial_segmentation(img, model_dir)
        save(Nifti1Image(out_tensor, img.affine, img.header), args.output)
        
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
        list_available_models()
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
