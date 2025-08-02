import os, sys, shutil
import struct
import gzip
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


def read_header_bytes(path, size=352):
    if path.endswith((".nii.gz", ".gz")):
        opener = gzip.open
    else:
        opener = open
    with opener(path, "rb") as f:
        return f.read(size)


def niimath_dtype(path: str):
    header = read_header_bytes(path)
    # 1) detect endianness via sizeof_hdr (should be 348)
    endian = header2endian(header)
    # 2) unpack using the detected endianness
    datatype, bitpix = struct.unpack(f"{endian}hh", header[70:74])

    dtype_map = {
        2: "char",  # uint8
        4: "short",  # int16
        8: "int",  # int32
        16: "float",  # float32
        64: "double",  # float64
        512: "ushort",  # uint16
        768: "long",  # int64
        1024: "uint",  # uint32
        1280: "ulong",  # uint64
    }
    return dtype_map.get(datatype, f"unknown({datatype})")


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


def conform(input_image_path, comply=False, ct=False):
    """
    Conform a NIfTI image to 256³ uint8 using niimath, returning
    the volume (256×256×256) and the 352‑byte header.
    """
    inp = Path(input_image_path).absolute()
    if not inp.exists():
        raise FileNotFoundError(f"Input NIfTI file not found: {inp}")

    comply_args = ["-comply", "256", "256", "256", "1", "1", "1", "1", "1"]

    cmd = ["niimath", str(inp)]
    if ct:
        cmd += ["-h2c"]
    if comply:
        cmd += comply_args
    cmd += ["-conform", "-gz", "0", "-", "-odt", "char"]

    # run niimath, capture stdout (header+raw voxels)
    res = subprocess.run(cmd, capture_output=True, check=True)
    out = res.stdout

    # split off header and data
    header = out[:352]
    data = out[352:]

    # reshape into (256,256,256) uint8 volume
    volume = np.frombuffer(data, dtype=np.uint8).reshape((256, 256, 256))
    return volume, header


def _conform(
    input_image_path, output_image_path="conformed.nii", comply=False, ct=False
):
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
        "1",
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
    if ct:
        args[1:1] = ["-h2c"]
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


def header2dimensions(header_bytes):
    """
    Extracts the meaningful dimensions (nx, ny, nz, nt, nu, nv, nw)
    from a NIfTI header ignoring trailing dimensions of size 0.

    Args:
        header_bytes: A bytes object containing the NIfTI header (at least 352 bytes).

    Returns:
        A tuple containing the meaningful dimensions (nx, ny, nz, nt, nu, nv, nw).
        Returns an empty tuple if the header_bytes are not valid or too short.
    """
    if not isinstance(header_bytes, bytes) or len(header_bytes) < 352:
        print("Error: Invalid or incomplete header_bytes provided.")
        return ()

    try:
        # Unpack the dimensions from bytes 40-56 of the header
        dimensions = struct.unpack_from("<hhhh hhhh", header_bytes, offset=40)

        # The first element of the dimensions array is ndim.
        # The subsequent elements correspond to
        # nx, ny, nz, nt, nu, nv, nw.
        # We should extract dimensions based on ndim.
        ndim = dimensions[0]

        # Meaningful dimensions start from index 1 (nx) up to ndim.
        # If ndim is less than 1 (though spec says > 0), or greater than 7,
        # we still want to return the first `ndim` dimensions.
        # If ndim is 1, we return only nx.
        # If ndim is 3, we return nx, ny, nz.
        # We take the slice from index 1 up to min(ndim + 1, 8)
        # because dimensions is 0-indexed, and we want ndim elements
        # starting from index 1. min(ndim + 1, 8) ensures we don't go
        # out of bounds of the dimensions tuple which has 8 elements.

        meaningful_dims = dimensions[1 : min(ndim + 1, 8)]

        return meaningful_dims

    except struct.error as e:
        print(f"Error unpacking dimensions from header: {e}")
        return ()


def header2endian(header: bytes):
    # 1) detect endianness via sizeof_hdr (should be 348)
    le_size = struct.unpack("<i", header[0:4])[0]
    if le_size == 348:
        endian = "<"
    else:
        # try big‑endian
        be_size = struct.unpack(">i", header[0:4])[0]
        if be_size == 348:
            endian = ">"
        else:
            raise ValueError(f"Unrecognized sizeof_hdr: {le_size!r}/{be_size!r}")
    return endian


def set_header_intent(header: bytes, intent_code: int) -> bytes:
    """Sets the NIFTI intent code to `intent_code` in the header."""
    # 1) detect endianness
    endian = header2endian(header)

    # Convert to mutable bytearray to allow modification
    header_array = bytearray(header)

    # NIFTI intent code for labels is 1007. It is a short at offset 68.
    intent_offset = 68

    # Pack the new intent_code into the header_array at the correct offset
    struct.pack_into(f"{endian}h", header_array, intent_offset, intent_code)

    # Return the modified header as an immutable bytes object
    return bytes(header_array)


def set_header_intent_label(header: bytes) -> bytes:
    """Sets the NIFTI intent code to 'label' (1002) in the header."""
    return set_header_intent(header, 1002)


def header2datatype(header: bytes):
    # 1) detect endianness
    endian = header2endian(header)
    # 2) unpack using the detected endianness
    datatype, bitpix = struct.unpack(f"{endian}hh", header[70:74])
    return datatype, bitpix


def header2dtype(header: bytes):
    datatype, _ = header2datatype(header)

    dtype_map = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        512: np.uint16,
        768: np.int64,
        1024: np.uint32,
        1280: np.uint64,
    }
    return dtype_map.get(datatype, np.uint8)  # fallback to uint8


def niimath_pipe_process(cmd: list, full_input: bytes):
    res = subprocess.run(cmd, input=full_input, capture_output=True, check=True)
    out = res.stdout
    # split off the 352‑byte header
    out_header, out_data = out[:352], out[352:]
    shape = header2dimensions(out_header)
    # reinterpret and reshape into (Z,Y,X)
    numpyarray = np.frombuffer(out_data, dtype=header2dtype(out_header)).reshape(shape)
    return numpyarray, out_header


def grow_border(full_input: bytes, border: int):
    cmd = [
        "niimath",
        "-",
        "-close",
        "1",
        str(border),
        "0",
        "-gz",
        "0",
        "-",
        "-odt",
        "char",
    ]
    res = subprocess.run(cmd, input=full_input, capture_output=True, check=True)
    return res.stdout


def largest_cluster(data):
    counts = np.bincount(data.ravel().astype(np.int32))
    largest_label = counts[1:].argmax() + 1
    return largest_label


def bwlabel(header: bytes, vol_data: np.ndarray, neighbors=26):
    # fire niimath, pipe in header+data, capture its stdout
    res = subprocess.run(
        ["niimath", "-", "-bwlabel", str(neighbors), "-gz", "0", "-", "-odt", "char"],
        input=header + vol_data.tobytes(),
        capture_output=True,
        check=True,
    )
    out = res.stdout
    # split off the 352‑byte header
    out_header, out_data = out[:352], out[352:]
    # reinterpret and reshape into (Z,Y,X)
    clusters = np.frombuffer(out_data, dtype=np.uint8).reshape(vol_data.shape)
    cluster_label = largest_cluster(clusters)
    vol_data[clusters != cluster_label] = 0
    return vol_data, out_header


def _bwlabel(image_path, neighbors=26, image=None):
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
