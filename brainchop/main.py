import os
import sys
import argparse
from nibabel import save, load, Nifti1Image
from tinygrad import Tensor
import numpy as np

# Add the directory containing the script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from brainchop.model import tinygrad_model

def find_file(filename, search_paths):
    for path in search_paths:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            return file_path
    return None

def main():
    parser = argparse.ArgumentParser(description="BrainChop: Brain segmentation tool")
    parser.add_argument("input", help="Input NIfTI file path")
    parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
    parser.add_argument("--json", default="model.json", help="Path to model JSON file")
    parser.add_argument("--bin", default="model.bin", help="Path to model binary file")
    args = parser.parse_args()

    # Convert input and output paths to absolute paths
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Define search paths
    search_paths = [
        os.getcwd(),  # Current working directory
        os.path.dirname(args.input),  # Directory of the input file
        script_dir,  # Directory of the main.py script
        project_root,  # Root directory of the project
        os.path.join(project_root, 'example'),  # Example directory
    ]

    # Find JSON and bin files
    args.json = find_file(args.json, search_paths) or args.json
    args.bin = find_file(args.bin, search_paths) or args.bin

    # Verify all required files exist
    for file_path in [args.input, args.json, args.bin]:
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    try:
        img = load(args.input)
        tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
        t = Tensor(tensor.astype(np.float16))
        out_tensor = tinygrad_model(args.json, args.bin, t)
        save(Nifti1Image(out_tensor, img.affine, img.header), args.output)
        print(f"Output saved as {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
