import os
import subprocess
import glob
from pathlib import Path

def process_nifti_files(directory_path):
    """
    Find all NIFTI files in a directory and process them with BrainChop.
    Outputs are saved with 'output_' prepended to the original filename.
    
    Args:
        directory_path (str): Path to the directory containing NIFTI files
    """
    # Make sure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist")
        return
    
    # Find all NIFTI files (both .nii and .nii.gz extensions)
    nifti_files = []
    nifti_files.extend(glob.glob(os.path.join(directory_path, "*.nii")))
    nifti_files.extend(glob.glob(os.path.join(directory_path, "*.nii.gz")))
    
    if not nifti_files:
        print(f"No NIFTI files found in '{directory_path}'")
        return
    
    print(f"Found {len(nifti_files)} NIFTI files")
    
    # Process each file with BrainChop
    for nifti_file in nifti_files:
        file_path = Path(nifti_file)
        output_name = f"output_{file_path.name}"
        output_path = os.path.join(directory_path, output_name)
        
        print(f"Processing: {file_path.name}")
        try:
            subprocess.run(
                ["brainchop", nifti_file, "-o", output_path],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  Output will be saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process NIFTI files with BrainChop")
    parser.add_argument("directory", help="Directory containing NIFTI files")
    args = parser.parse_args()
    
    process_nifti_files(args.directory)
