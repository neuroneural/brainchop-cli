#!/usr/bin/env python3
"""Re-export every active brainchop-test menu model that has a WebGPU runner.

For each entry in MODELS below this drives examples/export_meshnet_webgpu.py
with `--which both`, producing the fp16 runner+weights and the fp32
runner+weights and dropping them straight into a brainchop-test checkout:

    webgpu_runners/<runner>_runner.js       + public/models/<web>/model.safetensors      (fp16)
    webgpu_runners/<runner>_f32_runner.js   + public/models/<web>/model_f32.safetensors  (fp32)

`runner` and `web` are taken from the brainchop-test model entry's
`webgpu_runner` and `webgpu_safetensor` fields, so the emitted names line up
with what inference-webgpu.js loads. Edit MODELS if you add/rename menu models.

Usage
-----
    source ~/venv/torch/bin/activate         # or: cd brainchop-cli && uv run ...
    cd brainchop-cli
    python examples/reexport_all_webgpu.py --bct ../brainchop-test --beam 3
    # subset / dry-run / faster first pass:
    python examples/reexport_all_webgpu.py --bct ../brainchop-test --only dkatlas24,mindgrab
    python examples/reexport_all_webgpu.py --bct ../brainchop-test --beam 0 --dry-run

Notes
-----
* The FIRST BEAM run at 256^3 is very slow (tens of minutes/model) then caches
  in ~/.cache/tinygrad. Use --beam 0 for a quick correctness pass, --beam 3 for
  the tuned production runners.
* SAE models (robust_tissue / robust_subcortical) are NOT MeshNets; this script
  skips them (see SKIP_SAE). They need the SAE export path, not this one.
* Models are exported in ascending memory cost so cheap ones validate the
  toolchain before the expensive 104-class run.
"""
import argparse
import subprocess
import sys
from pathlib import Path

# (runner-name, web-model-dir, source checkpoint dir relative to --models-root, chunk)
#   runner-name   == brainchop-test entry's `webgpu_runner`
#   web-model-dir == public/models/<dir> from its `webgpu_safetensor`
#   chunk         == FUSE_CHUNK for the final classifier conv (<= hidden channels
#                    keeps peak at one hidden activation; 24 is safe for all here)
MODELS = [
    # runner            web-model-dir       source (under --models-root)        chunk
    ("model5",           "model5_gw_ae",     "meshnet/model5_gw_ae",             24),
    ("mindgrab",         "mindgrab",         "meshnet/mindgrab",                 24),
    ("model21chan18cls", "model18cls",       "meshnet/model18cls",               24),  # subcortical-mini 21ch/18cls
    ("model30chan18cls", "model30chan18cls", "meshnet/subcortical",              24),  # subcortical 30ch/18cls
    ("model30chan50cls", "model30chan50cls", "meshnet/model30chan50cls",         24),  # aparc50 30ch/50cls
    ("model21",          "model21_104class", "meshnet/model21_104class",         24),  # legacy 21ch/104cls
    ("dkatlas24",        "model24chan104cls","meshnet/model24chan104cls",        24),  # deep 24ch/104cls (the heavy one)
]

# Not MeshNets -- spatial-autoencoder architecture; export_meshnet_webgpu.py
# can't trace them. Listed only so it's obvious they were intentionally skipped.
SKIP_SAE = {
    "robust_tissue":      "model_sae16ch3_tfjs",
    "robust_subcortical": "model_sae32ch18_tfjs",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bct", required=True, help="path to brainchop-test checkout")
    ap.add_argument("--models-root", default="../brainchop-models",
                    help="root holding the source checkpoints (default ../brainchop-models)")
    ap.add_argument("--beam", type=int, default=3, help="BEAM kernel-search level (0 = quick/untuned)")
    ap.add_argument("--which", choices=["fp16", "fp32", "both"], default="both")
    ap.add_argument("--only", default=None,
                    help="comma-separated runner names to limit to (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    exporter = here / "export_meshnet_webgpu.py"
    if not exporter.exists():
        sys.exit(f"missing exporter: {exporter}")

    models_root = Path(args.models_root)
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    jobs = [m for m in MODELS if (only is None or m[0] in only)]
    if not jobs:
        sys.exit(f"no models matched --only={args.only}; known: {[m[0] for m in MODELS]}")

    print(f"Re-exporting {len(jobs)} model(s) -> {args.bct}  (beam={args.beam}, which={args.which})")
    if SKIP_SAE:
        print(f"Skipping SAE (non-MeshNet): {', '.join(SKIP_SAE)}\n")

    failures = []
    for runner, web, src, chunk in jobs:
        model_dir = models_root / src
        cmd = [
            sys.executable, str(exporter),
            "--model-dir", str(model_dir),
            "--bct", args.bct,
            "--runner-name", runner,
            "--web-model-dir", web,
            "--chunk", str(chunk),
            "--beam", str(args.beam),
            "--which", args.which,
        ]
        print("=" * 78)
        print(f"[{runner}]  src={model_dir}  web={web}  chunk={chunk}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        if not model_dir.exists():
            print(f"  !! source checkpoint not found: {model_dir} -- skipping")
            failures.append(runner)
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  !! export failed for {runner} (exit {rc})")
            failures.append(runner)

    print("=" * 78)
    if failures:
        sys.exit(f"Done with errors. Failed/skipped: {', '.join(failures)}")
    print("All exports completed." if not args.dry_run else "Dry run complete.")


if __name__ == "__main__":
    main()
