import os, sys, shutil
import struct
import subprocess
from pathlib import Path
import numpy as np
from tinygrad import Tensor


def _get_executable():
    """
    Locate the niimath binary, either via NIIMATH_PATH or on your PATH.
    Raises FileNotFoundError if not found, RuntimeError on unknown platform.
    """
    # pick the binary name for this platform
    exe_name = "niimath.exe" if sys.platform.startswith("win") else "niimath"

    # 1) if NIIMATH_PATH is set, look there first
    niimath_dir = os.getenv("NIIMATH_PATH")
    if niimath_dir:
        candidate = Path(niimath_dir) / exe_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        raise FileNotFoundError(
            f"NIIMATH_PATH={niimath_dir}, but {exe_name} not found/executable"
        )

    # 2) else search the PATH
    fullpath = shutil.which(exe_name)
    if fullpath:
        return fullpath

    # not found anywhere
    raise FileNotFoundError(
        f"Could not find `{exe_name}` on your PATH. "
        "Install niimath or set NIIMATH_PATH to its folder."
    )


def _get_temp_dir():
    """
    Gets the temporary directory path from environment or system default.

    Returns:
        Path: Path to temporary directory
    """
    temp_dir = os.getenv("NIIMATH_TEMP", "/tmp")
    return Path(temp_dir)


def _run_niimath(args):
    """
    Executes the niimath command with specified arguments.

    Parameters:
        args (list): List of command-line arguments to pass to niimath.

    Returns:
        int: Return code from niimath.

    Raises:
        subprocess.CalledProcessError: If the niimath command fails.
    """
    exe = _get_executable()
    cmd = [exe] + args

    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"niimath failed with error:\n{e.stderr}", file=sys.stderr)
        raise RuntimeError(f"niimath failed with error:\n{e.stderr}") from e


def _read_nifti(filename, voxel_size=1):
    EXPECTED_DIM = (256, 256, 256)
    VOXEL_COUNT = np.prod(EXPECTED_DIM)
    HEADER_SIZE = 352
    VOXEL_SIZE = voxel_size  # 1 for uint8
    EXPECTED_SIZE = HEADER_SIZE + VOXEL_COUNT * VOXEL_SIZE

    dtypes = {1: np.uint8, 2: np.uint16, 4: np.uint32}

    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        header = bytearray(f.read(HEADER_SIZE))  # skip header

        # —————— skip NIfTI‐1 extensions if present ——————
        ext_flag = struct.unpack("<i", header[348:352])[0]
        if ext_flag:
            ext_size = struct.unpack("<i", f.read(4))[0]
            f.seek(ext_size - 4, os.SEEK_CUR)

        # unpack datatype code (unused here) and bits per voxel
        _, bitpix = struct.unpack("<hh", header[70:74])
        VOXEL_SIZE = bitpix // 8  # bytes per voxel

        # now at start of voxel data
        data_start = f.tell()
        remaining = file_size - data_start
        expected = VOXEL_COUNT * VOXEL_SIZE
        if remaining != expected:
            raise ValueError(f"Data block is {remaining} bytes, expected {expected}")

        # ————————————————————————————————————————————————

        data = np.frombuffer(f.read(), dtype=dtypes[voxel_size])

    # Zero out the history offset and location
    header[348:352] = b"\x00\x00\x00\x00"
    header[108:112] = b"\x00\x00\xb0\x43"

    header = bytes(header)

    if data.size != VOXEL_COUNT:
        raise ValueError(f"Read {data.size} voxels, expected {VOXEL_COUNT}")

    return data.reshape(EXPECTED_DIM), header


def _write_nifti(path, data, header):
    # write header + raw voxel data
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


def conform(input_image_path, output_image_path="conformed.nii", comply=False):
    """
    Conform a NIfTI image to the specified shape using niimath.

    Parameters:
        input_image_path (str): Path to the input NIfTI file.
        output_image_path (str): Path to save the conformated NIfTI file.

    Returns:
        data, header: The conform numpy image, and binary header of 352 bytes.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If the conform operation fails.
    """
    input_path = Path(input_image_path).absolute()
    if not input_path.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {input_path}")

    # Convert output path to absolute path
    output_path = Path(output_image_path).absolute()

    comply_args = [
        "-comply",
        "256",
        "256",
        "256",
        "1",
        "1",
        "1",
        "0.98",
        "1",
    ]
    # Construct niimath arguments
    args = [
        str(input_path),
        "-conform",
        "-gz",
        "0",
        str(output_path),
        "-odt",
        "char",
    ]
    if comply:
        args[1:1] = comply_args

    # Run niimath
    _run_niimath(args)

    # Load and return the conformated image
    conform_img, header = _read_nifti(
        output_path, voxel_size=1
    )  # todo: do this all in mem

    try:
        output_path.unlink()  # Use pathlib's unlink instead of subprocess rm
    except OSError:
        pass  # Handle case where file doesn't exist or can't be removed

    return conform_img, header


def largest_cluster(data):
    counts = np.bincount(data.ravel().astype(np.int32))
    largest_label = counts[1:].argmax() + 1
    return largest_label


def bwlabel(image_path, neighbors=26, image=None):
    """
    Performs in place connected component labelling for non-zero voxels
    (conn sets neighbors: 6, 18, 26)
    """
    temp_dir = _get_temp_dir()
    mask_path = temp_dir / "bwlabel_mask.nii"
    image_path = Path(image_path).absolute()

    args = [
        str(image_path),
        "-bwlabel",
        str(neighbors),
        "-gz",
        "0",
        str(mask_path),
        "-odt",
        "char",
    ]
    _run_niimath(args)

    if image is None:
        image = _read_nifti(image_path)[0].astype(np.uint8)

    clusters, header = _read_nifti(mask_path)
    cluster_label = largest_cluster(clusters)
    image[clusters != cluster_label] = 0
    _write_nifti(image_path, image, header)

    try:
        mask_path.unlink()  # Use pathlib's unlink instead of subprocess rm
    except OSError:
        pass  # Handle case where file doesn't exist or can't be removed
