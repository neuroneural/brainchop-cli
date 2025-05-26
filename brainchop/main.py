import os
import argparse
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes
from brainchop.niimath import (
    conform,
    bwlabel,
    _write_nifti,
    _run_niimath,
    _get_temp_dir,
)

from brainchop.utils import (
    update_models,
    list_models,
    get_model,
    export_classes,
    AVAILABLE_MODELS,
    cleanup,
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
        "--inverse_conform",
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


def main():
    parser = get_parser()
    args = parser.parse_args()

    temp_dir = _get_temp_dir()
    model_output = temp_dir / "model_output.nii"
    model_output_path = Path(model_output).absolute()

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

    model = get_model(args.model)
    print(f"    brainchop :: Loaded model {args.model}")

    # load input
    volume, header = conform(args.input, comply=args.comply)

    image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )

    output_channels = model(image / image.max())
    output = (
        output_channels.argmax(axis=1)
        .rearrange("1 x y z -> z y x")
        .numpy()
        .astype(np.uint8)
    )

    _write_nifti(str(model_output_path), output, header)

    bwlabel(str(model_output_path), image=output)

    if args.export_classes:
        export_classes(output_channels, header, args.output)
        print(f"    brainchop :: Exported classes to c[channel_number]_{args.output}")

    cmd = [str(model_output_path)]
    if args.inverse_conform or args.model == "mindgrab":
        cmd += ["-reslice_nn", args.input]

    if args.model == "mindgrab":
        if args.border > 0:
            cmd += ["-sedt", "-add", str(args.border), "-bin"]
        if args.mask is not None:
            _run_niimath(cmd + ["-gz", "1", args.mask])
        cmd += ["-mul", args.input]
    cmd += ["-gz", "1", str(args.output)]

    _run_niimath(cmd)

    cleanup()

    try:
        model_output_path.unlink()  # Use pathlib's unlink
    except OSError:
        pass  # Handle case where file doesn't exist or can't be removed


if __name__ == "__main__":
    main()
