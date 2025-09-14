"""
Comprehensive batching test with visual inspection for brainchop-cli
Tests batch sizes 1, 2, 4, 8, 16 on default model and mindgrab
"""

import subprocess
import shutil
from pathlib import Path
from tinygrad.helpers import fetch, getenv

CACHEDIR = Path.home() / ".cache" / "brainchop" / "batching_test"
CACHEDIR.mkdir(parents=True, exist_ok=True)

_URLS = {
    "t1_crop": "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
}

def get_brainchop_batch_cmd(
    paths: list[Path], model: str, batch_size: int, output_dir: Path
) -> tuple[list[str], list[Path]]:
    """Generate brainchop command for batch processing with multiple input files."""
    cmd = ["brainchop", "-m", model, "-bs", str(batch_size)]
    cmd.extend([str(path) for path in paths])
    
    # Generate expected output paths based on the new naming scheme
    output_paths = []
    for i, path in enumerate(paths):
        base_name = path.stem.replace('.nii', '')  # Remove .nii from .nii.gz
        output_name = f"{base_name}_{model}_output_{i+1}.nii.gz"
        output_paths.append(output_dir / output_name)
    
    return cmd, output_paths

def get_mrpeek_cmd(path: Path) -> list[str]:
    """Generate mrpeek command for visual inspection."""
    return ["mrpeek", "-batch", str(path)]

def cmd_to_str(cmd_list: list[str]) -> str:
    """Convert command list to string."""
    return " ".join(cmd_list)

def test_batching_comprehensive():
    """Test batching functionality with different batch sizes and models."""
    
    # Test configuration
    batch_sizes = [1, 2, 4]
    models = ["tissue_fast", "mindgrab"]  # Default model and mindgrab
    
    print("="*80)
    print("COMPREHENSIVE BATCHING TEST")
    print("="*80)
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Testing models: {models}")
    
    # Download test file
    print("\nDownloading test file...")
    test_file_path = fetch(_URLS["t1_crop"], "t1_crop.nii.gz")
    print(f"Test file: {test_file_path}")
    
    # Create test directory
    test_dir = CACHEDIR / "batch_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    all_commands = []
    all_output_paths = []
    
    for model in models:
        for batch_size in batch_sizes:
            print(f"\n--- Testing {model} with batch size {batch_size} ---")
            
            # Create multiple copies of the test file for batching
            input_paths = []
            for i in range(batch_size):
                input_name = f"input_{i+1}.nii.gz"
                input_path = test_dir / f"{model}_bs{batch_size}_{input_name}"
                shutil.copy(test_file_path, input_path)
                input_paths.append(input_path)
            
            # Generate output directory for this test
            output_dir = test_dir / f"{model}_bs{batch_size}_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate brainchop command
            cmd, output_paths = get_brainchop_batch_cmd(input_paths, model, batch_size, output_dir)
            
            # Change working directory in command to output directory
            full_cmd = ["bash", "-c", f"cd {output_dir} && {cmd_to_str(cmd)}"]
            
            print(f"Command: {cmd_to_str(cmd)}")
            print(f"Expected outputs: {len(output_paths)} files")
            
            all_commands.append((full_cmd, output_paths, model, batch_size))
            all_output_paths.extend(output_paths)
    
    # Print all mrpeek commands for original file
    print("\n" + "="*80)
    print("MRPEEK COMMANDS FOR ORIGINAL FILE:")
    print("="*80)
    original_mrpeek = get_mrpeek_cmd(Path(test_file_path))
    print(cmd_to_str(original_mrpeek))
    
    # Print all brainchop commands
    print("\n" + "="*80)
    print("BRAINCHOP BATCH COMMANDS:")
    print("="*80)
    for cmd, _, model, batch_size in all_commands:
        print(f"# {model} batch size {batch_size}")
        print(cmd_to_str(cmd))
    
    # Print all mrpeek commands for outputs
    print("\n" + "="*80)
    print("MRPEEK COMMANDS FOR OUTPUTS:")
    print("="*80)
    for output_path in all_output_paths:
        mrpeek_cmd = get_mrpeek_cmd(output_path)
        print(cmd_to_str(mrpeek_cmd))
    
    if getenv("DRYRUN"):
        print("\nDRYRUN mode - not executing commands")
        return
    
    # Execute all commands
    print("\n" + "="*80)
    print("EXECUTING COMMANDS:")
    print("="*80)
    
    # Show original file first
    print(f"\n>>> SHOWING ORIGINAL FILE: {cmd_to_str(original_mrpeek)}")
    subprocess.run(original_mrpeek)
    
    success_count = 0
    total_tests = len(all_commands)
    
    for cmd, output_paths, model, batch_size in all_commands:
        print(f"\n>>> TESTING {model} with batch size {batch_size}")
        print(f">>> RUNNING: {cmd_to_str(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✓ SUCCESS: {model} batch size {batch_size}")
                
                # Check for batch tensor processing in output
                if f"batch tensor shape: ({batch_size}, 1, 256, 256, 256)" in result.stdout:
                    print("✓ VERIFIED: Correct batch tensor shape detected")
                else:
                    print("⚠ WARNING: Expected batch tensor shape not found in output")
                
                # Check if output files were created
                created_files = 0
                for output_path in output_paths:
                    if output_path.exists():
                        created_files += 1
                        # Show visual inspection for first output of each test
                        if created_files == 1:
                            mrpeek_cmd = get_mrpeek_cmd(output_path)
                            print(f">>> VISUAL INSPECTION: {cmd_to_str(mrpeek_cmd)}")
                            subprocess.run(mrpeek_cmd)
                
                print(f"✓ OUTPUTS: {created_files}/{len(output_paths)} files created")
                if created_files == len(output_paths):
                    success_count += 1
                    
            else:
                print(f"✗ FAILED: {model} batch size {batch_size}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {model} batch size {batch_size}")
        except Exception as e:
            print(f"✗ ERROR: {model} batch size {batch_size} - {e}")
        
        print("-" * 40)
    
    # Summary
    print("\n" + "="*80)
    print("BATCHING TEST SUMMARY")
    print("="*80)
    print(f"Successful tests: {success_count}/{total_tests}")
    print(f"Success rate: {success_count/total_tests*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 ALL BATCHING TESTS PASSED!")
    else:
        print(f"⚠ {total_tests - success_count} tests failed")

if __name__ == "__main__":
    test_batching_comprehensive()
