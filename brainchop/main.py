import os
import argparse
from pathlib import Path

from tinygrad import Tensor

from brainchop.utils import (
        update_models, 
        list_models, 
        get_model,
        AVAILABLE_MODELS, 
        cleanup)



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
    parser.add_argument("-m", "--model", default=next(iter(AVAILABLE_MODELS.keys())), 
                        help=f"Name of segmentation model, default: {next(iter(AVAILABLE_MODELS.keys()))}")
    parser.add_argument("-c", "--custom", type=str, 
                        help="Path to custom model directory (model.json and model.bin)")
    parser.add_argument("-ec", "--export-classes", action="store_true", 
                        help="Export class probability maps (MeshNet only)")
    parser.add_argument("--cache-dir", type=str, 
                        default=str(Path.home() / ".cache" / "brainchop" / "models" / "multiaxial"),
                        help="Directory to cache downloaded multiaxial models")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.update:     update_models();        return
    if args.list:       list_models() ;         return
    if not args.input:  parser.print_help();    return

    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)


    model = get_model(args.model)
    print(f"    brainchop :: Loaded model {args.model}")


    # handle custom model
    # handle new backend
    # handle old backend
    # (optional) handle multiaxial



    cleanup()

if __name__ == "__main__":
    main()
