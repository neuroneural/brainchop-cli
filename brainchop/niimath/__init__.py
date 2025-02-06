import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from tinygrad import Tensor
import nibabel as nib  # todo: remove nibabel

def _get_executable():
    """
    Determines the path to the niimath executable based on the operating system and environment.
    Uses NIIMATH_PATH environment variable if set.

    Returns:
        str: Path to the niimath executable.

    Raises:
        FileNotFoundError: If the executable is not found.
        RuntimeError: If the platform is unsupported.
    """
    # First check for environment variable
    niimath_path = os.getenv('NIIMATH_PATH')
    if niimath_path:
        if sys.platform.startswith('linux'):
            exe = Path(niimath_path) / 'linux' / 'niimath'
        elif sys.platform.startswith('darwin'):
            exe = Path(niimath_path) / 'macos' / 'niimath'
        elif sys.platform.startswith('win'):
            exe = Path(niimath_path) / 'windows' / 'niimath.exe'
        else:
            raise RuntimeError('Unsupported platform')
    else:
        # Fallback to package directory if environment variable not set
        base_path = Path(__file__).parent.absolute()
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

    return str(exe)

def _get_temp_dir():
    """
    Gets the temporary directory path from environment or system default.
    
    Returns:
        Path: Path to temporary directory
    """
    temp_dir = os.getenv('NIIMATH_TEMP', '/tmp')
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
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(result.stdout)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f'niimath failed with error:\n{e.stderr}', file=sys.stderr)
        raise RuntimeError(f'niimath failed with error:\n{e.stderr}') from e

def conform(input_image_path, output_image_path="conformed.nii.gz"):
    """
    Conform a NIfTI image to the specified shape using niimath.

    Parameters:
        input_image_path (str): Path to the input NIfTI file.
        output_image_path (str): Path to save the conformated NIfTI file.

    Returns:
        nibabel.Nifti1Image: The conformated NIfTI image.

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If the conform operation fails.
    """
    input_path = Path(input_image_path).absolute()
    if not input_path.exists():
        raise FileNotFoundError(f'Input NIfTI file not found: {input_path}')

    # Convert output path to absolute path
    output_path = Path(output_image_path).absolute()

    # Load the input image
    img = nib.load(input_path)
    affine = img.affine
    header = img.header

    # Construct niimath arguments
    args = [str(input_path), '-conform', str(output_path), '-odt', 'char']

    # Run niimath
    _run_niimath(args)

    # Load and return the conformated image
    conform_img = nib.load(output_path)  # todo: do this all in mem

    return conform_img, affine, header

def inverse_conform(input_image_path, output_image_path):
    """
    Performs an inverse conform in place of the image at output_image_path into
    the shape of the input_image_path.
    """
    input_path = Path(input_image_path).absolute()
    output_path = Path(output_image_path).absolute()
    
    img = nib.load(input_path)
    shape = [str(i) for i in img.header.get_data_shape()]
    voxel_size = ['1']*3
    f_high = ['0.98']  # top 2%
    isLinear = ['1']  # replace with 0 for nearest neighbor
    comply_args = ['-comply'] + shape + voxel_size + f_high + isLinear
    args = [str(output_path)] + comply_args + [str(output_path)]
    _run_niimath(args)

def bwlabel(image_path, neighbors=26):
    """
    Performs in place connected component labelling for non-zero voxels 
    (conn sets neighbors: 6, 18, 26)
    """
    temp_dir = _get_temp_dir()
    mask_path = temp_dir / "bwlabel_mask.nii.gz"
    image_path = Path(image_path).absolute()
    
    args = [str(image_path), '-bwlabel', str(neighbors), str(mask_path)]
    _run_niimath(args)

    img = nib.load(image_path)
    image, affine, header = Tensor(np.array(img.dataobj)), img.affine, img.header

    mask = Tensor(np.array(nib.load(mask_path).dataobj))
    ret = (mask * image).numpy()

    try:
        mask_path.unlink()  # Use pathlib's unlink instead of subprocess rm
    except OSError:
        pass  # Handle case where file doesn't exist or can't be removed
        
    nib.save(nib.Nifti1Image(ret, affine, header), image_path)
