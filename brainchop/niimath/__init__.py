# brainchop/niimath/__init__.py

import os
import sys
import subprocess
from pathlib import Path
import shlex

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

    # For debugging: print the command being executed
    cmd_str = ' '.join(shlex.quote(arg) for arg in cmd)
    print(f"Executing command: {cmd_str}")

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
