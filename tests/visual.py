"""
Convenience test for manual visual inspection of brainchop output.
Runs all models through the API and generates mrpeek commands for inspection.
"""

import hashlib
from pathlib import Path
from tinygrad.helpers import fetch, getenv

import brainchop as bc

CACHEDIR = Path.home() / ".cache" / "brainchop" / "output"
CACHEDIR.mkdir(parents=True, exist_ok=True)

_URLS = {
    "t1_crop": "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
}

_MODELS = sorted(bc.list_models().keys())


def get_output_path(input_path: Path, model: str) -> Path:
    """Generate deterministic output path based on input and model."""
    key = f"{input_path}_{model}"
    hash_str = hashlib.md5(key.encode()).hexdigest()[:8]
    return CACHEDIR / f"{hash_str}.nii.gz"


def get_mrpeek_cmd(path: Path) -> str:
    return f"mrpeek -batch {path}"


# List available models
print("available models:", _MODELS)

# Download test files
test_files = ["t1_crop"]
paths = [Path(fetch(_URLS[name], name + ".nii.gz")) for name in test_files]

# Print mrpeek commands for original files
print("=" * 80)
print("mrpeek commands for original files:")
for path in paths:
    print(get_mrpeek_cmd(path))

# Generate output paths
output_paths = [(path, model, get_output_path(path, model)) for path in paths for model in _MODELS]

# Print expected output paths
print("=" * 80)
print("output files:")
for input_path, model, output_path in output_paths:
    print(f"  {model}: {output_path}")

# Print mrpeek commands for output files
print("=" * 80)
print("mrpeek commands for outputs:")
for _, _, output_path in output_paths:
    print(get_mrpeek_cmd(output_path))

if getenv("DRYRUN"):
    print("\nDRYRUN mode - not running inference")
    exit(0)

# Run inference using API (no interactive prompts)
print("=" * 80)
print("running inference...")
for input_path, model, output_path in output_paths:
    print(f"\n>>> {model}")

    try:
        # Load
        vol = bc.load(str(input_path))

        # Segment (beam=0 means no optimization, no prompts)
        result = bc.segment(vol, model, beam=0)

        # Save
        bc.save(result, str(output_path))
        print(f"    saved: {output_path}")
    except Exception as e:
        print(f"    FAILED: {e}")

print("=" * 80)
print("done! run mrpeek commands above to inspect outputs")
