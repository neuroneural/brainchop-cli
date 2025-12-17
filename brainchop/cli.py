"""
brainchop CLI - Command-line interface for brain segmentation.

This is a thin wrapper around the brainchop API.
"""

import argparse
import os
import hashlib
import base64
from pathlib import Path

from brainchop.api import NIfTI, Model, list_models as api_list_models


def _hash_string(s: str) -> str:
    """Generate short hash for custom model names."""
    return base64.urlsafe_b64encode(hashlib.sha1(s.encode()).digest()).decode()[:8]


def _generate_output_path(
    input_path: str,
    model_name: str,
    index: int,
    output_arg: str,
    total_inputs: int,
) -> str:
    """Generate output filename based on input, model, and index."""
    input_file = Path(input_path)

    # Extract base name without extensions
    base_name = input_file.name
    if base_name.endswith(".nii.gz"):
        base_name = base_name[:-7]
    elif base_name.endswith(".nii"):
        base_name = base_name[:-4]
    else:
        base_name = input_file.stem

    # Hash custom model names
    available = {m.name for m in api_list_models()}
    if model_name not in available:
        model_name = _hash_string(model_name)

    # Single file with explicit output
    if total_inputs == 1 and output_arg != "output.nii.gz":
        return os.path.abspath(output_arg)

    # Generate dynamic name
    output_filename = f"{base_name}_{model_name}_output_{index}.nii.gz"

    # Use output directory if specified and not default
    if output_arg != "output.nii.gz":
        output_dir = Path(output_arg).parent
        return str((output_dir / output_filename).absolute())

    return str(Path(output_filename).absolute())


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    available_models = api_list_models()
    default_model = available_models[0].name if available_models else "tissue_fast"

    parser = argparse.ArgumentParser(
        description="BrainChop: portable brain segmentation tool"
    )

    # Input/output
    parser.add_argument("input", nargs="*", help="Input NIfTI file path(s)")
    parser.add_argument(
        "-o",
        "--output",
        default="output.nii.gz",
        help="Output NIfTI file path (single input) or directory (multiple inputs)",
    )

    # Model selection
    parser.add_argument(
        "-m",
        "--model",
        default=default_model,
        help=f"Name of segmentation model (default: {default_model})",
    )
    parser.add_argument(
        "-c",
        "--custom",
        type=str,
        help="Path to custom model directory (containing model.json and model.pth)",
    )
    parser.add_argument(
        "-ss",
        "--skull-strip",
        action="store_true",
        help="Skull strip only (alias for -m mindgrab)",
    )

    # Model management
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="Update the model listing",
    )

    # Preprocessing
    parser.add_argument(
        "--comply",
        action="store_true",
        default=False,
        help="Insert compliance arguments to niimath before '-conform'",
    )
    parser.add_argument(
        "--ct",
        action="store_true",
        default=False,
        help="Convert CT scans from Hounsfield to Cormack units",
    )
    parser.add_argument(
        "--crop",
        nargs="?",
        type=float,
        const=2.0,
        default=None,
        help="Crop input by percentile cutoff for faster execution (default: 2)",
    )

    # Postprocessing
    parser.add_argument(
        "-i",
        "--inverse-conform",
        action="store_true",
        help="Inverse conform output to original image space",
    )
    parser.add_argument(
        "-a",
        "--mask",
        nargs="?",
        const="mask.nii.gz",
        default=None,
        help="Save mask file (mindgrab only)",
    )
    parser.add_argument(
        "-b",
        "--border",
        type=int,
        default=0,
        help="Mask border threshold in mm (mindgrab only)",
    )
    parser.add_argument(
        "-ec",
        "--export-classes",
        action="store_true",
        help="Export class probability maps",
    )

    # Performance
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=1,
        help="Shard size for processing multiple inputs (default: 1)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip BEAM optimization",
    )

    return parser


def print_models() -> None:
    """Print available models."""
    print("Available models:")
    for m in api_list_models():
        print(f"  {m.name}: {m.description}")


def find_weights(custom_dir: str) -> str:
    """Find weights file in custom model directory."""
    custom_path = Path(custom_dir)
    pth_files = list(custom_path.glob("model.pth"))
    bin_files = list(custom_path.glob("model.bin"))

    if pth_files:
        return str(pth_files[0])
    elif bin_files:
        return str(bin_files[0])
    else:
        raise FileNotFoundError(f"No model.pth or model.bin found in {custom_dir}")


def _as_list(x):
    """Ensure x is a list."""
    return x if isinstance(x, list) else [x]


def main():
    """Main CLI entry point."""
    parser = get_parser()
    args = parser.parse_args()

    # Handle model management commands
    if args.update:
        from brainchop.utils import update_models
        update_models()
        return

    if args.list:
        print_models()
        return

    if not args.input:
        parser.print_help()
        return

    # Prepare input paths
    input_files = [os.path.abspath(f) for f in args.input]
    print(f"brainchop :: Processing {len(input_files)} input file(s)")

    # Load inputs
    niftis = NIfTI.load(
        input_files,
        crop_percentile=args.crop,
        ct=args.ct,
        comply=args.comply,
    )

    # Determine model
    model_name = "mindgrab" if args.skull_strip else args.model

    # Load model
    if args.custom:
        custom_path = Path(args.custom)
        if not custom_path.exists():
            print(f"Error: Custom model directory not found: {custom_path}")
            return

        config_path = str(custom_path / "model.json")
        weights_path = find_weights(args.custom)
        print(f"brainchop :: Using custom model from {custom_path}")

        model = Model(
            config_path=config_path,
            weights_path=weights_path,
            optimize=not args.no_optimize,
        )
        model_name = "custom"
    else:
        model = Model(model_name, optimize=not args.no_optimize)

    print(f"brainchop :: Loaded model {model_name}")
    print(f"brainchop :: Using shard size: {args.batch_size}")

    # Run inference
    results = model.segment(niftis, shard_size=args.batch_size)
    results = _as_list(results)
    niftis = _as_list(niftis)

    # Save outputs
    for i, (result, input_nifti) in enumerate(zip(results, niftis)):
        output_path = _generate_output_path(
            input_nifti.source_path or args.input[i],
            model_name,
            i + 1,
            args.output,
            len(input_files),
        )

        print(f"brainchop :: Saving {i + 1}/{len(results)}: {output_path}")

        # Handle mindgrab special cases
        if model_name == "mindgrab":
            _save_mindgrab_output(result, input_nifti, args, output_path)
        else:
            # Handle inverse conform
            if args.inverse_conform:
                _save_with_inverse_conform(result, input_nifti, output_path)
            else:
                result.save(output_path)

        # Export classes if requested
        if args.export_classes:
            from brainchop.api import export_channels
            raw_output = model(input_nifti)
            export_channels(raw_output, result.header, str(Path(output_path).parent))
            print(f"brainchop :: Exported classes to c[channel]_{output_path}")

    print("brainchop :: Done!")


def _save_mindgrab_output(result: NIfTI, input_nifti: NIfTI, args, output_path: str):
    """Handle mindgrab-specific output saving."""
    import subprocess
    from brainchop.niimath import grow_border, truncate_header_bytes

    if input_nifti.source_path is None:
        raise ValueError("Cannot save mindgrab output without source_path")

    header = truncate_header_bytes(result.header)
    data = header + result.volume.tobytes()

    # Apply border growth if specified
    if args.border > 0:
        data = grow_border(data, args.border)

    # Save mask if requested
    if args.mask is not None:
        cmd: list[str] = ["niimath", "-", "-reslice_nn", input_nifti.source_path, "-gz", "1", args.mask, "-odt", "char"]
        subprocess.run(cmd, input=data, check=True)

    # Apply to original image
    gzip_flag = "0" if output_path.endswith(".nii") else "1"
    cmd = [
        "niimath",
        input_nifti.source_path,
        "-reslice_mask",
        "-",
        "-gz",
        gzip_flag,
        output_path,
        "-odt",
        "input_force",
    ]
    subprocess.run(cmd, input=data, check=True)


def _save_with_inverse_conform(result: NIfTI, input_nifti: NIfTI, output_path: str):
    """Save with inverse conformation to original space."""
    import subprocess
    from brainchop.niimath import truncate_header_bytes

    if input_nifti.source_path is None:
        raise ValueError("Cannot inverse conform without source_path")

    header = truncate_header_bytes(result.header)
    gzip_flag = "0" if output_path.endswith(".nii") else "1"

    cmd: list[str] = [
        "niimath",
        "-",
        "-reslice_nn",
        input_nifti.source_path,
        "-gz",
        gzip_flag,
        output_path,
        "-odt",
        "char",
    ]
    subprocess.run(cmd, input=header + result.volume.tobytes(), check=True)


if __name__ == "__main__":
    main()
