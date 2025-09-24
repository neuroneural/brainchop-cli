import os
import sys
import shutil
import struct
import gzip
import subprocess
from pathlib import Path
import numpy as np
from tinygrad import Tensor
from typing import Tuple


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


def header_data_split(nifti_bytes: bytes) -> Tuple[bytes, bytes]:
    """
    Splits a NIfTI file's byte content into header and image data.

    This function correctly finds the start of the image data by reading
    the 'vox_offset' field from the header, providing a robust alternative
    to assuming a fixed header size.

    Args:
        nifti_bytes: The full content of a NIfTI file as a bytes object.

    Returns:
        A tuple containing two bytes objects: (header, image_data).
        The 'header' includes the standard 348-byte header plus any extensions.

    Raises:
        ValueError: If the input is too short to contain the vox_offset field
                    or if the offset points beyond the data's boundaries.
    """
    # The vox_offset field is a 4-byte float at byte offset 108.
    # We must have at least 108 + 4 = 112 bytes to read it.
    if len(nifti_bytes) < 112:
        raise ValueError(
            f"Input is too short ({len(nifti_bytes)} bytes) to be a valid NIfTI file."
        )

    # Unpack the little-endian float ('<f') from bytes 108 to 112.
    vox_offset = int(struct.unpack("<f", nifti_bytes[108:112])[0])

    # Sanity check: the offset cannot be larger than the total size of the input.
    if vox_offset > len(nifti_bytes):
        raise ValueError(
            f"Invalid vox_offset: {vox_offset}. It points beyond the end of the input data."
        )

    # Split the byte string at the dynamically found offset.
    out_header = nifti_bytes[:vox_offset]
    out_data = nifti_bytes[vox_offset:]

    return out_header, out_data


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


def _read_nifti(filename):

    INITIAL_HEADER_READ_SIZE = 352

    # NIfTI datatype codes to numpy dtypes mapping
    nifti_dtypes = {
        1: np.uint8,  # DT_UNSIGNED_CHAR
        2: np.int16,  # DT_SIGNED_SHORT
        4: np.int32,  # DT_SIGNED_INT
        8: np.float32,  # DT_FLOAT
        16: np.complex64,  # DT_COMPLEX
        32: np.float64,  # DT_DOUBLE
        64: np.int8,  # DT_INT8 (NIfTI-1 extension)
        128: np.uint16,  # DT_UINT16 (NIfTI-1 extension)
        256: np.uint32,  # DT_UINT32 (NIfTI-1 extension)
        512: np.int64,  # DT_INT64 (NIfTI-1 extension)
        768: np.uint64,  # DT_UINT64 (NIfTI-1 extension)
        1024: np.float128,  # DT_FLOAT128 (NIfTI-1 extension)
        1792: np.complex128,  # DT_COMPLEX128 (NIfTI-1 extension)
    }

    file_size = os.path.getsize(filename)
    with open(filename, "rb") as f:
        # Read the initial header block. This block contains all necessary info
        # including vox_offset, dim, datatype, bitpix.
        header = bytearray(f.read(INITIAL_HEADER_READ_SIZE))
        if len(header) < INITIAL_HEADER_READ_SIZE:
            raise ValueError(
                f"File is too small to contain a {INITIAL_HEADER_READ_SIZE}-byte NIfTI header."
            )

        # --- Parse critical fields from the header ---

        # 1. Image Dimensions (dim[0] is number of dimensions, dim[1..7] are sizes)
        # NIfTI dim array is at byte offset 40, consists of 8 short integers (16 bytes total)
        all_dims_raw = struct.unpack("<hhhhhhhh", header[40:56])

        ndim = all_dims_raw[0]  # Number of dimensions actually used (e.g., 3 for 3D)
        if not (1 <= ndim <= 7):
            raise ValueError(
                f"Invalid number of dimensions: {ndim}. NIfTI files usually have 3 or 4 dimensions."
            )

        # Extract actual image dimensions (e.g., dim[1], dim[2], dim[3] for a 3D image)
        # We take positive dimensions as NIfTI can pad with zeros.
        # NIfTI's dim[1]=x, dim[2]=y, dim[3]=z.
        img_dims_nifti_order = tuple(d for d in all_dims_raw[1 : ndim + 1] if d > 0)
        if not img_dims_nifti_order:
            raise ValueError(
                "Could not determine valid image dimensions from NIfTI header."
            )

        # Calculate total voxel count
        VOXEL_COUNT = np.prod(img_dims_nifti_order)

        # 2. Datatype and Bits per voxel
        # datatype_code at byte offset 70 (short), bitpix at byte offset 72 (short)
        datatype_code, bitpix = struct.unpack("<hh", header[70:74])

        image_dtype = nifti_dtypes.get(datatype_code)
        if image_dtype is None:
            raise ValueError(f"Unsupported NIfTI datatype code: {datatype_code}")

        VOXEL_SIZE_BYTES = bitpix // 8  # Bytes per voxel
        if VOXEL_SIZE_BYTES == 0:
            raise ValueError("Bitpix is 0, cannot determine voxel size.")

        # 3. Image Data Offset (vox_offset)
        # vox_offset is at byte offset 108, a float value representing byte offset.
        # It accounts for header, extensions, etc.
        vox_offset_float = struct.unpack("<f", header[108:112])[0]
        image_data_start_offset = int(vox_offset_float)

        # --- Now, seek to the actual image data start using vox_offset ---
        f.seek(
            image_data_start_offset, os.SEEK_SET
        )  # os.SEEK_SET means from the beginning of the file

        # --- Validate remaining file size and read data ---
        current_pos = f.tell()
        if current_pos != image_data_start_offset:
            raise IOError(
                f"Failed to seek to {image_data_start_offset} bytes. Current position is {current_pos}."
            )

        remaining_file_bytes = file_size - current_pos
        expected_data_bytes = VOXEL_COUNT * VOXEL_SIZE_BYTES

        if remaining_file_bytes < expected_data_bytes:
            raise ValueError(
                f"Data block is {remaining_file_bytes} bytes, "
                f"expected at least {expected_data_bytes} for {VOXEL_COUNT} voxels of {VOXEL_SIZE_BYTES} bytes each. "
                "File might be truncated or header dimensions/datatype incorrect."
            )

        # Read exactly the expected amount of image data
        data_bytes = f.read(expected_data_bytes)
        data = np.frombuffer(data_bytes, dtype=image_dtype)

    # --- Header modifications (preserving original user logic) ---
    # Zero out the extension field 'extension' at byte 348.
    header[348:352] = b"\x00\x00\x00\x00"

    # Modify vox_offset field at byte 108.
    header[108:112] = b"\x00\x00\xb0\x43"

    header_bytes = bytes(header)

    if data.size != VOXEL_COUNT:
        raise ValueError(f"Read {data.size} voxels, expected {VOXEL_COUNT}")

    # --- Reshape the data ---
    # NIfTI stores data in 'x-fastest' order (dim[1] changes fastest, then dim[2], then dim[3]).
    # For a typical NumPy array where indexing is (z, y, x) (e.g., slice, row, column),
    # we need to reverse the dimensions from the NIfTI header.
    # If img_dims_nifti_order is (dx, dy, dz), we reshape to (dz, dy, dx).
    reshaped_dims = img_dims_nifti_order[::-1]

    return data.reshape(reshaped_dims), header_bytes


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
    header, data = header_data_split(out)

    # reshape into (256,256,256) uint8 volume
    volume = np.frombuffer(data, dtype=np.uint8).reshape((256, 256, 256))
    return volume, header


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
    out_header, out_data = header_data_split(out)
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


def truncate_header_bytes(header: bytes):
    # Make a mutable copy of the header to modify it safely.
    new_header = bytearray(header)

    # 1. Set vox_offset (at byte 108) to 352.0
    # '<f' specifies a little-endian float.
    new_header[108:112] = struct.pack("<f", 352.0)

    # 2. Zero out the extension flag (at byte 348)
    # to declare that no extensions exist in this new in-memory file.
    new_header[348:352] = b"\x00\x00\x00\x00"

    # 3. Truncate the header to exactly 352 bytes, in case the original
    #    header was longer (due to extensions).
    return bytes(new_header[:352])


def bwlabel(header: bytes, vol_data: np.ndarray, neighbors=26):
    header = truncate_header_bytes(header)
    # fire niimath, pipe in header+data, capture its stdout
    res = subprocess.run(
        ["niimath", "-", "-bwlabel", str(neighbors), "-gz", "0", "-", "-odt", "char"],
        input=header + vol_data.tobytes(),
        capture_output=True,
        check=True,
    )
    out = res.stdout
    # split off the 352‑byte header
    out_header, out_data = header_data_split(out)
    # reinterpret and reshape into (Z,Y,X)
    clusters = np.frombuffer(out_data, dtype=np.uint8).reshape(vol_data.shape)
    cluster_label = largest_cluster(clusters)
    vol_data[clusters != cluster_label] = 0
    return vol_data, out_header
