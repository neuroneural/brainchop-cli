#!/usr/bin/env python3
"""Re-export all axial TTA models after bugfix."""

import shutil
from pathlib import Path

import brainchop as bc

OUTPUT_DIR = "../niivue-tinygrad"
PUBLIC_DIR = "../niivue-tinygrad/public"

MODELS = [
    "tissue_fast",
    "robust_tissue",
    "big_robust_tissue",
    "subcortical",
    "DKatlas",
    "mindgrab",
    "aparc50",
]


def main():
    for model in MODELS:
        print(f"\nExporting {model}_tta_axial...")
        try:
            js_path, weights_path = bc.export(
                model, OUTPUT_DIR, tta=True, flip_axis="axial"
            )
            print(f"  -> {js_path}")
            print(f"  -> {weights_path}")

            # Move safetensors to public/
            weights_file = Path(weights_path)
            dest = Path(PUBLIC_DIR) / weights_file.name
            shutil.move(weights_path, dest)
            print(f"  -> moved to {dest}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
