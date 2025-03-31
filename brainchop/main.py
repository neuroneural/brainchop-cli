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

from pathlib import Path
from .utils import update_models, list_models, find_model_files, AVAILABLE_MODELS, NEW_BACKEND

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

def validate_fn(file_path: str, description: str) -> None:
    if not os.path.isfile(file_path):
        print(f"Error: {description} not found: {file_path}")
        sys.exit(1)

def cleanup():
    if os.path.exists("conformed.nii.gz"):
        subprocess.run(["rm", "conformed.nii.gz"])

def get_parser():
    parser = argparse.ArgumentParser(description="BrainChop: portable brain segmentation tool")
    parser.add_argument("input", nargs="?", 
                        help="Input NIfTI file path")
    parser.add_argument("-l", "--list", action="store_true", 
                        help="List available models")
    parser.add_argument("-i", "--inverse_conform", action="store_true", 
                        help="Perform inverse conformation into original image space")
    parser.add_argument("-u", "--update", action="store_true", 
                        help="Update the model listing")
    parser.add_argument("-o", "--output", default="output.nii.gz", 
                        help="Output NIfTI file path")
    parser.add_argument("-m", "--model", default="", 
                        help=f"Name of segmentation model, default: {next(iter(AVAILABLE_MODELS.keys()))}")
    parser.add_argument("-c", "--custom", type=str, 
                        help="Path to custom model directory or file (MeshNet or Multiaxial model)")
    parser.add_argument("-ec", "--export-classes", action="store_true", 
                        help="Export class probability maps (MeshNet only)")
    parser.add_argument("--cache-dir", type=str, 
                        default=str(Path.home() / ".cache" / "brainchop" / "models" / "multiaxial"),
                        help="Directory to cache downloaded multiaxial models")
    return parser.parse_args()

def get_model(model_name):
    if model_name in NEW_BACKEND:
        config_fn, model_fn = find_model_files(model_name)
        return load_meshnet(config_fn, model_fn) # other configs should be loaded from json
    else: # oldbackend
        config_fn, bin_fn = find_model_files(model_name)
        





def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.update: update_models(); return
    if args.list: list_models() ; return
    if not args.input: parser.print_help(); return

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)



    # handle custom model
    # handle new backend
    # handle old backend
    # (optional) handle multiaxial

    if args.model == "mindgrab":
        config_fn, model_fn = find_model_files(args.model)
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


    cleanup()

if __name__ == "__main__":
    main()
