import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
from tinygrad import Tensor
from brainchop.niimath import (
    conform,
    set_header_intent_label,
    bwlabel,
    grow_border,
    niimath_dtype,
)

from brainchop.utils import (
    update_models,
    list_models,
    get_model,
    export_classes,
    AVAILABLE_MODELS,
    cleanup,
    crop_to_cutoff,
    pad_to_original_size,
)


def get_parser():
    parser = argparse.ArgumentParser(
        description="BrainChop: portable brain segmentation tool"
    )
    parser.add_argument("input", nargs="?", help="Input NIfTI file path")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available models"
    )
    parser.add_argument(
        "-i",
        "--inverse-conform",
        action="store_true",
        help="Perform inverse conformation into original image space",
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="Update the model listing"
    )
    parser.add_argument(
        "-o", "--output", default="output.nii.gz", help="Output NIfTI file path"
    )
    parser.add_argument(
        "-a",
        "--mask",
        nargs="?",  # 0 or 1 arguments
        const="mask.nii.gz",  # if they just say `--mask` with no value
        default=None,  # if they don’t mention `--mask` at all
        help="If provided and using mindgrab, write out the mask (defaults to mask.nii.gz when used without a value)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=next(iter(AVAILABLE_MODELS.keys())),
        help=f"Name of segmentation model, default: {next(iter(AVAILABLE_MODELS.keys()))}",
    )
    parser.add_argument(
        "-c",
        "--custom",
        type=str,
        help="Path to custom model directory (model.json and model.bin)",
    )
    parser.add_argument(
        "--comply",
        action="store_true",
        default=False,
        help="Insert compliance arguments to `niimath` before '-conform'",
    )
    parser.add_argument(
        "--ct",
        action="store_true",
        default=False,
        help="Convert CT scans from 'Hounsfield' to 'Cormack' units to emphasize soft tissue contrast",
    )
    parser.add_argument(
        "--crop",
        nargs="?",  # 0 or 1 arguments
        type=float,
        const=2,  # if they just say `--crop` with no value
        default=False,  # if they don’t mention `--crop` at all
        help="Crop the input for faster execution. May reduce accuracy.(defaults to percentile 2 cutoff)",
    )
    parser.add_argument(
        "-ss",
        "--skull-strip",
        action="store_true",
        help="Return just the brain compartment. An alias for -m mindgrab, that overrides -m parameter",
    )
    parser.add_argument(
        "-ec",
        "--export-classes",
        action="store_true",
        help="Export class probability maps",
    )
    parser.add_argument(
        "-b",
        "--border",
        type=int,
        default=0,
        help="Mask border threshold in mm. Default is 0. Makes a difference only if the model is `mindgrab`",
    )
    return parser


def preprocess_input(args):
    """
    Handle input preprocessing: loading, conforming, and cropping.
    
    Returns:
        tuple: (image_tensor, volume, header, crop_coords)
    """
    # Load and conform input volume
    volume, header = conform(args.input, comply=args.comply, ct=args.ct)
    crop_coords = None
    
    # Apply cropping if requested
    if args.crop:
        volume, crop_coords = crop_to_cutoff(volume, args.crop)
        print(f"    brainchop :: cropped to {volume.shape}")
    
    # Convert to tensor format expected by model
    image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )
    
    return image, volume, header, crop_coords


def run_inference(model, image):
    """
    Execute model inference on the preprocessed image.
    
    Args:
        model: The loaded segmentation model
        image: Preprocessed image tensor
        
    Returns:
        Tensor: Raw model output channels
    """
    return model(image)


def postprocess_output(output_channels, header, crop_coords=None):
    """
    Handle output postprocessing: argmax, padding, and labeling.
    
    Args:
        output_channels: Raw model output tensor
        header: Original NIfTI header
        crop_coords: Coordinates for uncropping (if cropping was applied)
        
    Returns:
        tuple: (processed_labels_data, new_header)
    """
    # Convert model output to segmentation labels
    output = (
        output_channels.argmax(axis=1)
        .rearrange("1 x y z -> z y x")
        .numpy()
        .astype(np.uint8)
    )
    
    # Restore original size if cropping was applied
    if crop_coords is not None:
        output = pad_to_original_size(output, crop_coords)
    
    # Generate labeled output with proper header
    labels, new_header = bwlabel(header, output)
    processed_data = set_header_intent_label(new_header) + labels.tobytes()
    
    return processed_data, new_header


def write_output(processed_data, args):
    """
    Handle file output operations including niimath commands and subprocess calls.
    
    Args:
        processed_data: Processed segmentation data ready for output
        args: Command line arguments containing output settings
    """
    output_dtype = "char"
    
    # Handle class probability export if requested
    if args.export_classes:
        # Note: This requires access to output_channels, will need to be called separately
        print(f"    brainchop :: Exported classes to c[channel_number]_{args.output}")
    
    # Determine gzip compression based on file extension
    gzip_flag = "0" if str(args.output).endswith(".nii") else "1"
    
    # Build base niimath command
    cmd = ["niimath", "-"]
    if args.inverse_conform and args.model != "mindgrab":
        cmd += ["-reslice_nn", args.input]
    
    # Handle mindgrab-specific processing
    data_to_write = processed_data
    if args.model == "mindgrab":
        cmd = ["niimath", str(args.input)]
        
        # Apply border growth if specified
        if args.border > 0:
            data_to_write = grow_border(processed_data, args.border)
        
        # Write mask file if requested
        if args.mask is not None:
            cmdm = ["niimath", "-"]
            cmdm += ["-reslice_nn", args.input]
            subprocess.run(
                cmdm + ["-gz", "1", args.mask, "-odt", "char"],
                input=data_to_write,
                check=True,
            )
        
        cmd += ["-reslice_mask", "-"]
        output_dtype = "input_force"
    
    # Finalize command and execute
    cmd += ["-gz", gzip_flag, str(args.output), "-odt", output_dtype]
    subprocess.run(cmd, input=data_to_write, check=True)


def run_cli():
    """Main CLI function that orchestrates brainchop command-line operations."""
    parser = get_parser()
    args = parser.parse_args()

    # Handle simple commands that don't require processing
    if args.update:
        update_models()
        return
    if args.list:
        list_models()
        return
    if not args.input:
        parser.print_help()
        return

    # Prepare file paths
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Load model
    modelname = args.model
    if args.skull_strip:
        modelname = "mindgrab"
        args.model = modelname
    model = get_model(modelname)
    print(f"    brainchop :: Loaded model {modelname}")

    # Execute processing pipeline
    image, volume, header, crop_coords = preprocess_input(args)
    output_channels = run_inference(model, image)
    processed_data, new_header = postprocess_output(output_channels, header, crop_coords)
    
    # Handle class export before writing main output
    if args.export_classes:
        export_classes(output_channels, header, args.output)
        print(f"    brainchop :: Exported classes to c[channel_number]_{args.output}")
    
    write_output(processed_data, args)
    cleanup()


if __name__ == "__main__":
    run_cli()
