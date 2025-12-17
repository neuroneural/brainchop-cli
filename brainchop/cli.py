"""
brainchop CLI - Command-line interface for brain segmentation.
"""

import argparse
import os
import subprocess
from pathlib import Path

from brainchop.api import (
    Volume, load, save, segment, list_models, optimize,
    _is_first_run, _get_best_beam,
)
from brainchop.niimath import grow_border, truncate_header_bytes


def get_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    models = list_models()
    default_model = list(models.keys())[0] if models else "tissue_fast"

    parser = argparse.ArgumentParser(
        description="BrainChop: portable brain segmentation tool"
    )

    parser.add_argument("input", nargs="*", help="Input NIfTI file(s)")
    parser.add_argument("-o", "--output", default="output.nii.gz", help="Output path")
    parser.add_argument("-m", "--model", default=default_model, help=f"Model (default: {default_model})")
    parser.add_argument("-c", "--custom", help="Custom model directory")
    parser.add_argument("-ss", "--skull-strip", action="store_true", help="Skull strip (alias for -m mindgrab)")
    parser.add_argument("-l", "--list", action="store_true", help="List models")
    parser.add_argument("-u", "--update", action="store_true", help="Update model listing")
    parser.add_argument("--ct", action="store_true", help="CT scan conversion")
    parser.add_argument("--crop", nargs="?", type=float, const=2.0, help="Crop percentile")
    parser.add_argument("-i", "--inverse-conform", action="store_true", help="Inverse conform output")
    parser.add_argument("-a", "--mask", nargs="?", const="mask.nii.gz", help="Save mask (mindgrab)")
    parser.add_argument("-b", "--border", type=int, default=0, help="Mask border mm (mindgrab)")
    parser.add_argument("-bs", "--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--no-optimize", action="store_true", help="Skip optimization prompt")
    parser.add_argument("--beam", type=int, default=None, help="BEAM optimization level")

    return parser


def _prompt_for_optimization(model_name: str, batch_size: int) -> bool:
    """Prompt user for first-run optimization."""
    print(f"\nbrainchop :: First run detected for '{model_name}' with batch size {batch_size}")
    print("brainchop :: Would you like to pre-optimize for faster subsequent runs?")
    print("brainchop :: This compiles the model with BEAM=2 optimization (recommended)")
    while True:
        try:
            response = input("brainchop :: Optimize now? [y/n]: ").strip().lower()
        except EOFError:
            return False
        if response == "y":
            optimize(model_name, beam=2, batch_size=batch_size)
            return True
        elif response == "n":
            print("brainchop :: Skipping optimization.")
            return False
        else:
            print("brainchop :: Please enter 'y' or 'n'")


def main():
    """Main CLI entry point."""
    args = get_parser().parse_args()

    if args.update:
        from brainchop.utils import update_models
        update_models()
        return

    if args.list:
        for name, desc in list_models().items():
            print(f"  {name}: {desc}")
        return

    if not args.input:
        get_parser().print_help()
        return

    # Determine model: --custom overrides -m, --skull-strip overrides both
    if args.skull_strip:
        model_name = "mindgrab"
    elif args.custom:
        model_name = args.custom
    else:
        model_name = args.model

    # Handle BEAM optimization
    beam = args.beam if args.beam is not None else 0
    if beam == 0 and not args.no_optimize:
        # Check if first run, prompt for optimization
        if _is_first_run(model_name, args.batch_size):
            _prompt_for_optimization(model_name, args.batch_size)
        # Use cached beam value
        cached = _get_best_beam(model_name, args.batch_size)
        if cached:
            beam = cached

    # Process each input
    for i, input_path in enumerate(args.input):
        abs_path = os.path.abspath(input_path)
        print(f"brainchop :: [{i+1}/{len(args.input)}] {input_path}")

        # Load
        vol = load(abs_path, crop=args.crop, ct=args.ct)

        # Segment (single volume, so result is always Volume)
        result = segment(vol, model_name, beam=beam)
        assert isinstance(result, Volume)

        # Output path
        if len(args.input) == 1 and args.output != "output.nii.gz":
            output_path = args.output
        else:
            base = Path(input_path).stem.replace(".nii", "")
            # Use model name for output, hash if it's a path
            if Path(model_name).is_dir():
                import hashlib
                model_label = hashlib.sha1(model_name.encode()).hexdigest()[:8]
            else:
                model_label = model_name
            output_path = f"{base}_{model_label}_{i+1}.nii.gz"

        # Save
        if model_name == "mindgrab":
            _save_mindgrab(result, abs_path, args, output_path)
        elif args.inverse_conform:
            _save_inverse_conform(result, abs_path, output_path)
        else:
            save(result, output_path)

        print(f"brainchop :: Saved {output_path}")

    print("brainchop :: Done!")


def _save_mindgrab(result: Volume, source_path: str, args, output_path: str):
    """Handle mindgrab output."""
    header = truncate_header_bytes(result.header)
    data = header + result.data.cast("uint8").numpy().tobytes()

    if args.border > 0:
        data = grow_border(data, args.border)

    if args.mask:
        subprocess.run(
            ["niimath", "-", "-reslice_nn", source_path, "-gz", "1", args.mask, "-odt", "char"],
            input=data, check=True
        )

    gz = "0" if output_path.endswith(".nii") else "1"
    subprocess.run(
        ["niimath", source_path, "-reslice_mask", "-", "-gz", gz, output_path, "-odt", "input_force"],
        input=data, check=True
    )


def _save_inverse_conform(result: Volume, source_path: str, output_path: str):
    """Save with inverse conform."""
    header = truncate_header_bytes(result.header)
    gz = "0" if output_path.endswith(".nii") else "1"
    subprocess.run(
        ["niimath", "-", "-reslice_nn", source_path, "-gz", gz, output_path, "-odt", "char"],
        input=header + result.data.cast("uint8").numpy().tobytes(), check=True
    )


if __name__ == "__main__":
    main()
