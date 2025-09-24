import os
import argparse
import subprocess
import json
from pathlib import Path

import numpy as np
from tinygrad.tensor import Tensor
from brainchop.niimath import (
    conform,
    set_header_intent_label,
    bwlabel,
    grow_border,
)

from brainchop.utils import (
    update_models,
    list_models,
    get_model,
    get_model_from_custom_path,
    export_classes,
    AVAILABLE_MODELS,
    cleanup,
    crop_to_cutoff,
    pad_to_original_size,
)


def load_optimization_cache(model_name):
    """
    Load optimization cache for a given model.

    Args:
        model_name: Name of the model

    Returns:
        dict: Optimization cache data with 'beams' list, or empty structure if not found
    """
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_name
    cache_file = cache_dir / "optimizations.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted, return empty structure
            pass

    return {"beams": []}


def save_optimization_cache(model_name, batch_size, beam_value):
    """
    Save optimization data to cache.

    Args:
        model_name: Name of the model
        batch_size: Batch size used
        beam_value: BEAM value that was successful
    """
    cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "optimizations.json"

    # Load existing cache
    cache_data = load_optimization_cache(model_name)

    # Check if this BS/BEAM combination already exists
    for entry in cache_data["beams"]:
        if entry["BS"] == batch_size and entry["BEAM"] == beam_value:
            return  # Already cached

    # Add new entry
    cache_data["beams"].append({"BS": batch_size, "BEAM": beam_value})

    # Sort by batch size for easier reading
    cache_data["beams"].sort(key=lambda x: (x["BS"], x["BEAM"]))

    # Save to file
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)


def get_best_beam_for_batch_size(model_name, batch_size):
    """
    Get the best (largest) BEAM value for a given batch size.

    Args:
        model_name: Name of the model
        batch_size: Current batch size

    Returns:
        int: Best BEAM value for this batch size, or None if not found
    """
    cache_data = load_optimization_cache(model_name)

    # Find all BEAM values for this batch size
    beam_values = [
        entry["BEAM"] for entry in cache_data["beams"] if entry["BS"] == batch_size
    ]

    if beam_values:
        return max(beam_values)

    return None


def is_first_run(model_name, batch_size):
    """
    Check if this is the first run for a given model and batch size.

    Args:
        model_name: Name of the model
        batch_size: Batch size to check

    Returns:
        bool: True if no optimization exists for this model/batch_size combo
    """
    cache_data = load_optimization_cache(model_name)

    # Check if any optimization exists for this batch size
    for entry in cache_data["beams"]:
        if entry["BS"] == batch_size:
            return False

    return True


def preoptimize(
    model_name, beam, batch_size=1, custom_config=None, custom_weights=None
):
    """
    Pre-optimize a model by running it with a random input tensor and specified BEAM value.

    Args:
        model_name: Name of the model to optimize
        beam: BEAM optimization value to use
        batch_size: Batch size for the input tensor (default: 1)
        custom_config: Path to custom model config (optional)
        custom_weights: Path to custom model weights (optional)
    """
    print(
        f"brainchop :: Pre-optimizing model '{model_name}' with BEAM={beam}, BS={batch_size}..."
    )
    print("brainchop :: This may take a few moments for the initial compilation...")

    # Store original BEAM value
    original_beam = os.environ.get("BEAM")

    try:
        # Set BEAM environment variable for optimization
        os.environ["BEAM"] = str(beam)

        # Load the model with the specified BEAM value
        if custom_config and custom_weights:
            model = get_model_from_custom_path(custom_config, custom_weights)
        else:
            model = get_model(model_name)

        # Generate random input tensor with shape (BS, 1, 256, 256, 256)
        random_input = np.random.randn(batch_size, 1, 256, 256, 256).astype(np.float32)
        input_tensor = Tensor(random_input)

        print("brainchop :: Running optimization pass...")

        # Run inference to trigger compilation/optimization
        output = model(input_tensor)

        # Force computation to complete (realize the tensor)
        output.realize()

        print(
            f"brainchop :: Pre-optimization complete! Model is now optimized for BS={batch_size}"
        )

        # Save this optimization to cache
        save_optimization_cache(model_name, batch_size, beam)

        return True

    except Exception as e:
        print(f"brainchop :: Pre-optimization failed: {e}")
        return False

    finally:
        # Restore original BEAM environment variable
        if original_beam is not None:
            os.environ["BEAM"] = original_beam
        elif "BEAM" in os.environ:
            del os.environ["BEAM"]


def prompt_for_optimization(
    model_name, batch_size, custom_config=None, custom_weights=None
):
    """
    Prompt user to optimize the model on first run.

    Args:
        model_name: Name of the model
        batch_size: Batch size to optimize for
        custom_config: Path to custom model config (optional)
        custom_weights: Path to custom model weights (optional)

    Returns:
        bool: True if optimization was performed successfully, False otherwise
    """
    print(
        f"\nbrainchop :: First run detected for model '{model_name}' with batch size {batch_size}"
    )
    print(
        "brainchop :: Would you like to pre-optimize the model for faster subsequent runs?"
    )
    print(
        "brainchop :: This will compile the model with BEAM=2 optimization (recommended)"
    )

    while True:
        response = input("brainchop :: Optimize now? [y/n]: ").strip().lower()

        if response == "y":
            return preoptimize(
                model_name,
                beam=2,
                batch_size=batch_size,
                custom_config=custom_config,
                custom_weights=custom_weights,
            )
        elif response == "n":
            print(
                "brainchop :: Skipping optimization. Proceeding with unoptimized model..."
            )
            return False
        else:
            print("brainchop :: Please enter 'y' for yes or 'n' for no")


def generate_output_filename(input_path, modelname, index, output_dir=None):
    """
    Generate output filename based on input filename, model name, and index.

    Args:
        input_path: Path to input file
        modelname: Name of the segmentation model used
        index: Processing index/order
        output_dir: Optional output directory (uses current dir if None)

    Returns:
        str: Generated output filename in format {input_name}_{modelname}_output_{index}.nii.gz
    """
    input_file = Path(input_path)
    import hashlib, base64

    hash_string = lambda s: base64.urlsafe_b64encode(
        hashlib.sha1(s.encode()).digest()
    ).decode()[:8]
    if modelname not in AVAILABLE_MODELS:
        modelname = hash_string(modelname)

    # Extract base name without extensions (.nii.gz or .nii)
    base_name = input_file.name
    if base_name.endswith(".nii.gz"):
        base_name = base_name[:-7]  # Remove .nii.gz
    elif base_name.endswith(".nii"):
        base_name = base_name[:-4]  # Remove .nii
    else:
        # Remove any extension for non-nii files
        base_name = input_file.stem

    # Generate output filename with model name and index
    output_filename = f"{base_name}_{modelname}_output_{index}.nii.gz"

    # Use output directory if specified, otherwise use current directory
    if output_dir:
        output_path = Path(output_dir) / output_filename
    else:
        output_path = Path(output_filename)

    return str(output_path.absolute())


def get_parser():
    parser = argparse.ArgumentParser(
        description="BrainChop: portable brain segmentation tool"
    )
    parser.add_argument("input", nargs="*", help="Input NIfTI file path(s)")
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
        "-o",
        "--output",
        default="output.nii.gz",
        help="Output NIfTI file path (for single input) or output directory (for multiple inputs)",
    )
    parser.add_argument(
        "-a",
        "--mask",
        nargs="?",  # 0 or 1 arguments
        const="mask.nii.gz",  # if they just say `--mask` with no value
        default=None,  # if they don't mention `--mask` at all
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
        help="Path to custom model directory (containing model.json and model.pth or model.bin)",
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
        default=False,  # if they don't mention `--crop` at all
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
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing multiple inputs (default: 1)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip the optimization prompt on first run",
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
        print(f"brainchop :: cropped to {volume.shape}")

    # Convert to tensor format expected by model
    image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )

    return image, volume, header, crop_coords


def preprocess_batch(input_files, args):
    """
    Handle batch preprocessing: loading, conforming, and cropping multiple inputs.

    Args:
        input_files: List of input file paths
        args: Command line arguments

    Returns:
        tuple: (batched_tensor, list_of_volumes, list_of_headers, list_of_crop_coords)
    """
    batch_tensors = []
    volumes = []
    headers = []
    crop_coords_list = []

    for input_file in input_files:
        # Create temporary args for this input
        temp_args = argparse.Namespace(**vars(args))
        temp_args.input = input_file

        # Preprocess individual file
        image_tensor, volume, header, crop_coords = preprocess_input(temp_args)

        # Remove the batch dimension (1) from individual tensor to prepare for batching
        image_tensor = image_tensor.squeeze(0)  # Shape: (1, H, W, D)

        batch_tensors.append(image_tensor)
        volumes.append(volume)
        headers.append(header)
        crop_coords_list.append(crop_coords)

    # Stack tensors along batch dimension
    batched_tensor = Tensor.stack(*batch_tensors, dim=0)  # Shape: (BS, 1, H, W, D)

    return batched_tensor, volumes, headers, crop_coords_list


def run_inference(model, image):
    """
    Execute model inference on the preprocessed image.

    Args:
        model: The loaded segmentation model
        image: Preprocessed image tensor (single or batched)

    Returns:
        Tensor: Raw model output channels
    """
    return model(image)


def run_batch_inference(model, batched_image):
    """
    Execute model inference on batched preprocessed images.

    Args:
        model: The loaded segmentation model
        batched_image: Batched preprocessed image tensor (BS, 1, H, W, D)

    Returns:
        Tensor: Raw batched model output channels
    """
    return model(batched_image)


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


def postprocess_batch_output(batched_output_channels, headers, crop_coords_list):
    """
    Handle batch output postprocessing: argmax, padding, and labeling for multiple outputs.

    Args:
        batched_output_channels: Raw batched model output tensor (BS, C, H, W, D)
        headers: List of original NIfTI headers
        crop_coords_list: List of coordinates for uncropping (if cropping was applied)

    Returns:
        list: List of (processed_labels_data, new_header) tuples
    """
    results = []
    batch_size = batched_output_channels.shape[0]

    for i in range(batch_size):
        # Extract individual output from batch
        output_channels = batched_output_channels[
            i : i + 1
        ]  # Keep batch dimension for consistency
        header = headers[i]
        crop_coords = crop_coords_list[i]

        # Process individual output
        processed_data, new_header = postprocess_output(
            output_channels, header, crop_coords
        )
        results.append((processed_data, new_header))

    return results


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
        print(f"brainchop :: Exported classes to c[channel_number]_{args.output}")

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

    # Prepare file paths - convert input list to absolute paths
    input_files = [os.path.abspath(input_file) for input_file in args.input]

    # Store original output value before converting to absolute path
    original_output = args.output
    args.output = os.path.abspath(args.output)

    print(f"brainchop :: Processing {len(input_files)} input file(s)")

    # Determine model name and handle custom models
    modelname = args.model
    custom_config = None
    custom_weights = None

    if args.skull_strip:
        modelname = "mindgrab"
        args.model = modelname

    # Handle custom model path
    if args.custom:
        custom_dir = Path(args.custom)
        if not custom_dir.exists():
            print(f"Error: Custom model directory not found: {custom_dir}")
            return

        # Look for model files in custom directory
        json_files = list(custom_dir.glob("model.json"))
        pth_files = list(custom_dir.glob("model.pth"))
        bin_files = list(custom_dir.glob("model.bin"))

        if not json_files:
            print(f"Error: No model.json found in {custom_dir}")
            return

        custom_config = str(json_files[0])

        # Determine weights file based on what's available
        if pth_files:
            custom_weights = str(pth_files[0])
        elif bin_files:
            custom_weights = str(bin_files[0])
        else:
            print(f"Error: No model.pth or model.bin found in {custom_dir}")
            return

        modelname = "custom"
        print(f"brainchop :: Using custom model from {custom_dir}")

    # Check if this is the first run for this model/batch_size combination
    batch_size = args.batch_size
    original_beam = os.environ.get("BEAM")
    if (
        not args.no_optimize
        and is_first_run(modelname, batch_size)
        and not original_beam
    ):
        # Prompt for optimization on first run
        optimization_success = prompt_for_optimization(
            modelname, batch_size, custom_config, custom_weights
        )
        if optimization_success:
            print(
                "brainchop :: Model optimized successfully. Continuing with processing..."
            )
        print()  # Add blank line for clarity

    # Check for cached optimization and set BEAM environment variable
    best_beam = get_best_beam_for_batch_size(modelname, batch_size)

    if (
        best_beam is not None
        and original_beam is not None
        and best_beam > original_beam
    ):
        os.environ["BEAM"] = str(best_beam)
        print(
            f"brainchop :: Using cached optimization BEAM={best_beam} for batch size {batch_size}"
        )

    # Load model (will use the BEAM environment variable if set)
    if custom_config and custom_weights:
        model = get_model_from_custom_path(custom_config, custom_weights)
    else:
        model = get_model(modelname)

    print(f"brainchop :: Loaded model {modelname}")

    # Process input files in batches
    print(f"brainchop :: Using batch size: {batch_size}")

    for batch_start in range(0, len(input_files), batch_size):
        batch_end = min(batch_start + batch_size, len(input_files))
        batch_files = input_files[batch_start:batch_end]

        # Process batch using proper batching
        batched_tensor, volumes, headers, crop_coords_list = preprocess_batch(
            batch_files, args
        )

        batched_output_channels = run_batch_inference(model, batched_tensor)

        batch_results = postprocess_batch_output(
            batched_output_channels, headers, crop_coords_list
        )

        # Process each file's results
        for i, input_file in enumerate(batch_files):
            global_index = batch_start + i + 1
            processed_data, new_header = batch_results[i]

            # Generate output filename based on input filename, model, and index
            # Always use the new naming format unless user explicitly specified a custom output
            if original_output == "output.nii.gz":
                # Default output - always use the new dynamic naming format
                output_file = generate_output_filename(
                    input_file, modelname, global_index, None
                )
            elif len(input_files) == 1:
                # Single file with custom output specified - use the custom output
                output_file = args.output
            else:
                # Multiple files with custom output directory - generate dynamic name in that directory
                output_dir = str(Path(args.output).parent)
                output_file = generate_output_filename(
                    input_file, modelname, global_index, output_dir
                )

            print(
                f"Processing file {global_index}/{len(input_files)}: {input_file} -> {output_file}"
            )

            # Create a temporary args object for this specific input
            current_args = argparse.Namespace(**vars(args))
            current_args.input = input_file
            current_args.output = output_file
            current_args.model = (
                args.model
            )  # Preserve the original model name for mindgrab check

            # Handle class export before writing main output
            if args.export_classes:
                # For batched processing, we need to extract the individual output channels
                individual_output_channels = batched_output_channels[i : i + 1]
                export_classes(
                    individual_output_channels, headers[i], current_args.output
                )
                print(
                    f"brainchop :: Exported classes to c[channel_number]_{current_args.output}"
                )

            write_output(processed_data, current_args)

    # Save optimization data to cache if BEAM was used (and not already saved during pre-optimization)
    current_beam = os.environ.get("BEAM")
    if current_beam is not None and not is_first_run(modelname, batch_size):
        try:
            beam_value = int(current_beam)
            save_optimization_cache(modelname, batch_size, beam_value)
        except ValueError:
            pass  # Invalid BEAM value, skip caching

    # Restore original BEAM environment variable
    if original_beam is not None:
        os.environ["BEAM"] = original_beam
    elif "BEAM" in os.environ:
        del os.environ["BEAM"]

    cleanup()


if __name__ == "__main__":
    run_cli()
