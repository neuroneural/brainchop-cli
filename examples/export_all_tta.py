#!/usr/bin/env python3
"""Export all models with all three TTA flip variants to niivue-tinygrad."""

import brainchop as bc

OUTPUT_DIR = "../niivue-tinygrad"

# Models that still need coronal and axial TTA exports
# (sagittal TTA already done for all, tissue_fast coronal/axial already done)
MODELS = [
    "robust_tissue",
    "big_robust_tissue",
    "subcortical",
    "DKatlas",
    "mindgrab",
    "aparc50",
]

# Only need coronal and axial (sagittal already done)
FLIP_AXES = ["coronal", "axial"]


def main():
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Exporting {model}")
        print('='*60)

        # Export TTA variants (coronal, axial)
        for flip_axis in FLIP_AXES:
            print(f"\n  TTA {flip_axis}...")
            try:
                js_path, weights_path = bc.export(
                    model, OUTPUT_DIR, tta=True, flip_axis=flip_axis
                )
                print(f"    -> {js_path}")
                print(f"    -> {weights_path}")
            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\n{'='*60}")
    print("Export complete!")
    print('='*60)


if __name__ == "__main__":
    main()
