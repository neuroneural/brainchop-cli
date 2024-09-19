# brainchop/niimath/__init__.py

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from tinygrad import Tensor

import nibabel as nib # todo: remove nibabel


def _get_executable():
    """
    Determines the path to the niimath executable based on the operating system.
    Ensures the executable exists and has the correct permissions.

    Returns:
        str: Path to the niimath executable.

    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If the platform is unsupported.
    """
    base_path = Path(__file__).parent
    if sys.platform.startswith('linux'):
        exe = base_path / 'linux' / 'niimath'
    elif sys.platform.startswith('darwin'):
        exe = base_path / 'macos' / 'niimath'
    elif sys.platform.startswith('win'):
        exe = base_path / 'windows' / 'niimath.exe'
    else:
        raise RuntimeError('Unsupported platform')

    if not exe.exists():
        raise FileNotFoundError(f'niimath executable not found: {exe}')

    # Ensure the executable has execute permissions (for Unix-like systems)
    """
    if not sys.platform.startswith('win'):
        st = os.stat(exe)
        os.chmod(exe, st.st_mode | stat.S_IEXEC)
    """

    return str(exe)

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

    # Initialize the command with the executable
    cmd = [exe] + args

    try:
        # Execute the command
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Optionally, process result.stdout if needed
        print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        # Print the error message from niimath
        print(f'niimath failed with error:\n{e.stderr}', file=sys.stderr)
        raise RuntimeError(f'niimath failed with error:\n{e.stderr}') from e

def conform(input_image_path, output_image_path="conformed.nii.gz"):
    """
    Conform a NIfTI image to the specified shape using niimath.

    Parameters:
        input_image_path (str): Path to the input NIfTI file.
        output_image_path (str): Path to save the conformated NIfTI file.
        dt (str, optional): Internal datatype (e.g., 'float', 'double'). Defaults to 'float'.
        odt (str, optional): Output datatype (e.g., 'char', 'short', 'int', 'float', 'double', 'input'). Defaults to 'float'.

    Returns:
        nibabel.Nifti1Image: The conformated NIfTI image.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If the conform operation fails.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f'Input NIfTI file not found: {input_image_path}')

    # Load the input image
    img = nib.load(input_image_path)
    affine = img.affine
    header = img.header

    # Construct niimath arguments
    args = [input_image_path] + ['-conform'] + [output_image_path] + ['-odt', 'char']

    # Run niimath
    _run_niimath(args)

    # Load and return the conformated image
    conform_img = nib.load(output_image_path) # todo: do this all in mem

    return conform_img, affine, header

def inverse_conform(input_image_path, output_image_path):
    """
    Performs an inverse conform in place of the image at output_image_path into
    the shape of the input_image_path.
    """
    img = nib.load(input_image_path)
    shape = [str(i) for i in img.header.get_data_shape()]
    voxel_size = ['1']*3
    f_high = ['0.98'] # top 2%
    isLinear = ['1'] # replace with 0 for nearest neighbor
    comply_args = ['-comply'] + shape + voxel_size + f_high + isLinear
    args = [output_image_path] + comply_args  + [output_image_path]
    _run_niimath(args)

def bwlabel(image_path, neighbors=26):
    """
    Performs in place connected component labelling for non-zero voxels 
    (conn sets neighbors: 6, 18, 26)
    """
    mask_path = "bwlabel_mask.nii.gz" # TODO: do this in memory
    args = [image_path] + ['-bwlabel', str(neighbors)] + [mask_path]
    _run_niimath(args)

    img = nib.load(image_path)

    image, affine, header = Tensor(np.array(img.dataobj)), img.affine, img.header # obtain image data

    mask = Tensor(np.array(nib.load(mask_path).dataobj)) # obtain mask tensor

    ret = (mask * image).numpy() # apply mask

    subprocess.run(['rm', mask_path])
    nib.save(nib.Nifti1Image(ret, affine, header), image_path)
