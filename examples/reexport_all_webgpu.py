#!/usr/bin/env python3
"""Stage tuned low-memory WebGPU exports for every active non-TTA model.

The production path is deliberately two-stage:

1. Trace the contiguous-fp16 graph on native WebGPU with BEAM and record the
   selected tinygrad optimization list for every unique kernel AST.
2. Trace the same graph on Metal, lower those ASTs to WGSL using the recorded
   WebGPU choices, and export lifetime-reused ``arena_*`` buffers.

Nothing is copied over a working brainchop-test runner. Candidates are written
under ``--stage-root`` and must pass browser parity/timing before installation.
TTA runners are intentionally out of scope.

Examples:

    python examples/reexport_all_webgpu.py \
      --bct ../brainchop-test --only model5 --beam 2

    python examples/reexport_all_webgpu.py \
      --bct ../brainchop-test --beam 2 --reuse-catalog
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


# runner, web model directory, source relative to --models-root, classifier
# chunk, architecture, class count. This mirrors the active base entries in
# brainchop-test/brainchop-parameters.js as of 2026-08-10. TTA and legacy
# runners are deliberately absent.
MODELS = [
    ("model5",             "model5_gw_ae",              "meshnet/model5_gw_ae",              3,  "meshnet", 3),
    ("model6chan3cls",     "model6chan3cls",            "meshnet/model6chan3cls",            3,  "meshnet", 3),
    ("mindgrab",           "mindgrab",                  "meshnet/mindgrab",                  1,  "meshnet", 1),
    ("model16chan18cls",   "model16chan18cls",          "meshnet/model16chan18cls",         16,  "meshnet", 18),
    ("robust_tissue",      "model_sae16ch3_tfjs",       "sae/robust_tissue",                 3,  "sae",     3),
    ("model30chan50cls",   "model30chan50cls",          "meshnet/model30chan50cls",         24,  "meshnet", 50),
    ("model32chan18cls",   "model32chan18cls",          "meshnet/model32chan18cls",         18,  "meshnet", 18),
    ("dkatlas24_synth",    "model24chan104cls_synth",   "meshnet/model24chan104cls_synth", 24,  "meshnet", 104),
]


def runner_memory(path: Path):
    if not path.is_file():
        return None
    source = path.read_text()
    sizes = [int(x) for x in re.findall(r"createEmptyBuf\(device,\s*(\d+)\)", source)]
    return {
        "count": len(sizes),
        "total_mib": sum(sizes) / 2**20,
        "largest_mib": max(sizes, default=0) / 2**20,
        "arenas": len(re.findall(r"const arena_\d+\s*=\s*createEmptyBuf", source)),
    }


def format_memory(value):
    if value is None:
        return "missing"
    return (f"{value['total_mib']:.1f} MiB total / {value['largest_mib']:.1f} MiB largest "
            f"/ {value['count']} buffers / {value['arenas']} arenas")


def run(command, dry_run=False):
    print("  " + " ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bct", required=True, help="brainchop-test checkout used for current-runner comparison")
    parser.add_argument("--models-root", default="../brainchop-models")
    parser.add_argument("--stage-root", default="/tmp/brainchop-webgpu-lowmem",
                        help="candidate tree; never overwrites --bct (default: %(default)s)")
    parser.add_argument("--beam", type=int, default=2)
    parser.add_argument("--fp16-conv-store", choices=("scheduled", "contiguous"),
                        default="contiguous",
                        help="activation boundary to export (default: %(default)s)")
    parser.add_argument("--only", default=None, help="comma-separated runner names")
    parser.add_argument("--reuse-catalog", action="store_true",
                        help="reuse an existing per-model schedule catalog instead of rerunning BEAM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.beam < 0:
        parser.error("--beam must be nonnegative")

    here = Path(__file__).resolve().parent
    exporter = here / "export_meshnet_webgpu.py"
    models_root = Path(args.models_root).resolve()
    bct = Path(args.bct).resolve()
    stage_root = Path(args.stage_root).resolve()
    catalogs = stage_root / "catalogs"
    catalogs.mkdir(parents=True, exist_ok=True)

    selected = {name.strip() for name in args.only.split(",")} if args.only else None
    jobs = [model for model in MODELS if selected is None or model[0] in selected]
    known = {model[0] for model in MODELS}
    unknown = set() if selected is None else selected - known
    if unknown:
        parser.error(f"unknown runner(s): {', '.join(sorted(unknown))}")

    print(f"Staging {len(jobs)} low-memory base runner(s) in {stage_root}")
    print("TTA runners are unchanged. Working brainchop-test files are never overwritten.\n")

    failures = []
    for runner, web_dir, source_rel, chunk, kind, n_classes in jobs:
        source = models_root / source_rel
        catalog = catalogs / f"{runner}.pkl"
        current_js = bct / "webgpu_runners" / f"{runner}_runner.js"
        candidate_js = stage_root / "webgpu_runners" / f"{runner}_runner.js"
        print("=" * 78)
        print(f"[{runner}] {kind}, source={source}, chunk={chunk}, BEAM={args.beam}, "
              f"conv-store={args.fp16_conv_store}")
        print(f"  current:   {format_memory(runner_memory(current_js))}")

        if not source.exists():
            print(f"  ERROR: source model is missing: {source}")
            failures.append(runner)
            continue

        common = [
            sys.executable, str(exporter),
            "--model-dir", str(source),
            "--runner-name", runner,
            "--web-model-dir", web_dir,
            "--chunk", str(chunk),
            "--which", "fp16",
            "--fp16-norm", "rescale",
            "--fp16-conv-store", args.fp16_conv_store,
            "--model-kind", kind,
            "--n-classes", str(n_classes),
        ]

        try:
            if not args.reuse_catalog or not catalog.is_file():
                with tempfile.TemporaryDirectory(prefix=f"bc-{runner}-tune-") as tune_root:
                    run(common + [
                        "--bct", tune_root,
                        "--beam", str(args.beam),
                        "--capture-device", "webgpu",
                        "--opt-dump", str(catalog),
                    ], args.dry_run)
            else:
                print(f"  schedule: reusing {catalog}")

            run(common + [
                "--bct", str(stage_root),
                "--beam", "0",
                "--capture-device", "metal",
                "--opt-catalog", str(catalog),
            ], args.dry_run)
        except subprocess.CalledProcessError as error:
            print(f"  ERROR: export exited {error.returncode}")
            failures.append(runner)
            continue

        if not args.dry_run:
            candidate = runner_memory(candidate_js)
            print(f"  candidate: {format_memory(candidate)}")
            if candidate is None or candidate["arenas"] == 0:
                print("  ERROR: candidate does not contain lifetime-reused arena buffers")
                failures.append(runner)

    print("=" * 78)
    if failures:
        sys.exit(f"Completed with failures: {', '.join(failures)}")
    print("All requested candidates staged. No working brainchop-test runner was replaced.")


if __name__ == "__main__":
    main()
