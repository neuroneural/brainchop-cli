import os
import argparse
import subprocess
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes
from brainchop.niimath import (
    conform,
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
        "--ct",
        action="store_true",
        default=False,
        help="Convert CT scans from 'Hounsfield' to 'Cormack' units to emphasize soft tissue contrast",
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

    output_dtype = "char"
    # load input
    volume, header = conform(args.input, comply=args.comply, ct=args.ct)

    image = Tensor(volume.transpose((2, 1, 0)).astype(np.float32)).rearrange(
        "... -> 1 1 ..."
    )

    output_channels = model(image)
    output = (
        output_channels.argmax(axis=1)
        .rearrange("1 x y z -> z y x")
        .numpy()
        .astype(np.uint8)
    )

    labels, new_header = bwlabel(header, output)
    full_input = new_header + labels.tobytes()

    if args.export_classes:
        export_classes(output_channels, header, args.output)
        print(f"    brainchop :: Exported classes to c[channel_number]_{args.output}")

    cmd = ["niimath", "-"]
    if args.inverse_conform and not args.model == "mindgrab":
        cmd += ["-reslice_nn", args.input]

    if args.model == "mindgrab":
        cmd = ["niimath", str(args.input)]
        if args.border > 0:
            full_input = grow_border(full_input, args.border)
        if args.mask is not None:
            cmdm = ["niimath", "-"]
            cmdm += ["-reslice_nn", args.input]
            subprocess.run(
                cmdm + ["-gz", "1", args.mask, "-odt", "char"],
                input=full_input,
                check=True,
            )
        cmd += ["-reslice_mask", "-"]
        output_dtype = "input_force"
    cmd += ["-gz", "1", str(args.output), "-odt", output_dtype]

    subprocess.run(cmd, input=full_input, check=True)

    cleanup()


if __name__ == "__main__":
    main()
