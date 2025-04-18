import os
import argparse
from pathlib import Path

import nibabel as nib

import numpy as np
from tinygrad import Tensor, dtypes
from brainchop.niimath import conform, inverse_conform, bwlabel

from brainchop.utils import (
        update_models, 
        list_models, 
        get_model,
        export_classes,
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
                        help="Export class probability maps")
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


    # load input
    nifti = conform(args.input)[0]
    image = Tensor(nifti.get_fdata().astype(np.float32)).rearrange("... -> 1 1 ...")

    output_channels = model(image)
    output = output_channels.argmax(axis=1).reshape(256,256,256).numpy()
    
    output_nifti = nib.Nifti1Image(output,nifti.affine)
    nib.save(output_nifti, args.output)
    bwlabel(args.output)
    if args.inverse_conform: inverse_conform(args.input, args.output)
    

    if args.export_classes: 
        export_classes(output_channels, nifti.affine, args.output)
        print(f"    brainchop :: Exported classes to c[channel_number]_{args.output}")
    cleanup()

if __name__ == "__main__":
    main()
