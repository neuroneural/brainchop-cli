"""
Test comparing METAL vs WEBGPU backend outputs.

Usage:
    # Run METAL backend first, save output
    python tests/test_backend_comparison.py --backend metal --save

    # Run WEBGPU backend and compare
    WEBGPU=1 python tests/test_backend_comparison.py --backend webgpu --compare

    # Or run both sequentially (spawns subprocess for WEBGPU)
    python tests/test_backend_comparison.py --both
"""

import argparse
import subprocess
import sys
import os
import numpy as np
from pathlib import Path

# Cache directory for test artifacts
CACHE_DIR = Path.home() / ".cache" / "brainchop" / "backend_test"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

METAL_OUTPUT = CACHE_DIR / "metal_output.npy"
WEBGPU_OUTPUT = CACHE_DIR / "webgpu_output.npy"


def get_test_nifti_path() -> Path:
    """Download and cache test NIfTI file."""
    from tinygrad.helpers import fetch
    TEST_URL = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
    return Path(fetch(TEST_URL, "t1_crop.nii.gz"))


def run_inference(model_name: str = "tissue_fast") -> np.ndarray:
    """Run inference and return raw output array."""
    from brainchop import Volume, load, segment

    nifti_path = get_test_nifti_path()
    vol = load(str(nifti_path))
    result = segment(vol, model_name)
    assert isinstance(result, Volume)
    return result.data.numpy()


def get_current_backend() -> str:
    """Get current tinygrad backend."""
    from tinygrad.tensor import Device
    return Device.DEFAULT


def save_output(output: np.ndarray, path: Path) -> None:
    """Save output array to file."""
    np.save(path, output)
    print(f"Saved output to {path} (shape: {output.shape}, dtype: {output.dtype})")


def load_output(path: Path) -> np.ndarray:
    """Load output array from file."""
    return np.load(path)


def compare_outputs(metal_output: np.ndarray, webgpu_output: np.ndarray) -> dict:
    """Compare two output arrays and return statistics."""
    # Basic shape check
    if metal_output.shape != webgpu_output.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: METAL {metal_output.shape} vs WEBGPU {webgpu_output.shape}"
        }

    # Compute differences
    diff = np.abs(metal_output.astype(np.float64) - webgpu_output.astype(np.float64))

    # For integer outputs (argmax), check exact match
    if metal_output.dtype in (np.uint8, np.int32, np.int64):
        exact_match = np.array_equal(metal_output, webgpu_output)
        mismatch_count = np.sum(metal_output != webgpu_output)
        mismatch_pct = 100 * mismatch_count / metal_output.size

        return {
            "match": exact_match,
            "dtype": str(metal_output.dtype),
            "shape": metal_output.shape,
            "mismatch_count": int(mismatch_count),
            "mismatch_percent": float(mismatch_pct),
            "max_diff": int(diff.max()),
        }

    # For float outputs, use tolerance-based comparison
    rtol = 1e-4
    atol = 1e-5
    close = np.allclose(metal_output, webgpu_output, rtol=rtol, atol=atol)

    return {
        "match": close,
        "dtype": str(metal_output.dtype),
        "shape": metal_output.shape,
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "std_abs_diff": float(diff.std()),
        "percentile_99": float(np.percentile(diff, 99)),
        "percentile_999": float(np.percentile(diff, 99.9)),
    }


def run_metal():
    """Run inference on METAL backend."""
    backend = get_current_backend()
    print(f"Running on {backend} backend...")

    if backend != "METAL":
        print(f"Warning: Expected METAL, got {backend}")

    output = run_inference()
    save_output(output, METAL_OUTPUT)
    return output


def run_webgpu():
    """Run inference on WEBGPU backend."""
    backend = get_current_backend()
    print(f"Running on {backend} backend...")

    if backend != "WEBGPU":
        print(f"Warning: Expected WEBGPU, got {backend}")
        print("Make sure to run with WEBGPU=1 environment variable")

    output = run_inference()
    save_output(output, WEBGPU_OUTPUT)
    return output


def compare():
    """Compare saved METAL and WEBGPU outputs."""
    if not METAL_OUTPUT.exists():
        print(f"Error: METAL output not found at {METAL_OUTPUT}")
        print("Run with --backend metal --save first")
        return False

    if not WEBGPU_OUTPUT.exists():
        print(f"Error: WEBGPU output not found at {WEBGPU_OUTPUT}")
        print("Run with WEBGPU=1 --backend webgpu --save first")
        return False

    metal_output = load_output(METAL_OUTPUT)
    webgpu_output = load_output(WEBGPU_OUTPUT)

    print(f"\nMETAL output: shape={metal_output.shape}, dtype={metal_output.dtype}")
    print(f"WEBGPU output: shape={webgpu_output.shape}, dtype={webgpu_output.dtype}")

    results = compare_outputs(metal_output, webgpu_output)

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    for key, value in results.items():
        print(f"  {key}: {value}")

    if results["match"]:
        print("\n✓ METAL and WEBGPU outputs MATCH")
    else:
        print("\n✗ METAL and WEBGPU outputs DIFFER")

    return results["match"]


def run_both():
    """Run both backends sequentially using subprocesses."""
    print("=" * 60)
    print("STEP 1: Running METAL backend (subprocess)")
    print("=" * 60)

    # Run METAL in subprocess with clean env
    env = os.environ.copy()
    env["METAL"] = "1"
    # Remove any conflicting device settings
    for key in ["WEBGPU", "CPU", "CUDA", "CLANG"]:
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, __file__, "--backend", "metal", "--save"],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Error running METAL backend")
        return False

    print("\n" + "=" * 60)
    print("STEP 2: Running WEBGPU backend (subprocess)")
    print("=" * 60)

    # Run WEBGPU in subprocess with clean env
    env = os.environ.copy()
    env["WEBGPU"] = "1"
    # Remove any conflicting device settings
    for key in ["METAL", "CPU", "CUDA", "CLANG"]:
        env.pop(key, None)

    result = subprocess.run(
        [sys.executable, __file__, "--backend", "webgpu", "--save"],
        env=env,
        capture_output=False,
    )

    if result.returncode != 0:
        print("Error running WEBGPU backend")
        return False

    print("\n" + "=" * 60)
    print("STEP 3: Comparing outputs")
    print("=" * 60)

    return compare()


def main():
    parser = argparse.ArgumentParser(description="Compare METAL vs WEBGPU backend outputs")
    parser.add_argument("--backend", choices=["metal", "webgpu"], help="Backend to run")
    parser.add_argument("--save", action="store_true", help="Save output after running")
    parser.add_argument("--compare", action="store_true", help="Compare saved outputs")
    parser.add_argument("--both", action="store_true", help="Run both backends and compare")
    parser.add_argument("--model", default="tissue_fast", help="Model to test (default: tissue_fast)")

    args = parser.parse_args()

    if args.both:
        success = run_both()
        sys.exit(0 if success else 1)

    if args.compare:
        success = compare()
        sys.exit(0 if success else 1)

    if args.backend == "metal":
        run_metal()
    elif args.backend == "webgpu":
        run_webgpu()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
