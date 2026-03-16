#!/usr/bin/env python3
"""
Benchmark script for brainchop models.

Sweeps through models, measuring e2e inference time and memory.
BEAM is read from the envvar — set it before running.

Usage:
    python examples/benchmark.py                # all models, BEAM from env (0 if unset)
    python examples/benchmark.py 3              # first 3 models
    BEAM=2 python examples/benchmark.py         # all models with BEAM=2
    python examples/benchmark.py --timeout 300  # 5 min timeout per model
"""

import argparse
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from tinygrad.helpers import fetch

TEST_URL = "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
DEFAULT_TIMEOUT = 600  # 10 minutes


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Timed out")


def get_memory_stats():
    """Try to get memory stats from tinygrad if available."""
    stats = {}
    try:
        from tinygrad import Device
        dev = Device.DEFAULT
        if hasattr(Device[dev], "mem_used"):
            stats["mem_used_mb"] = Device[dev].mem_used / (1024 * 1024)
        if hasattr(Device[dev], "mem_total"):
            stats["mem_total_mb"] = Device[dev].mem_total / (1024 * 1024)
    except Exception:
        pass

    # Fallback: process RSS via resource module
    try:
        import resource
        import platform
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes, Linux in kilobytes
        divisor = (1024 * 1024) if platform.system() == "Darwin" else 1024
        stats["rss_mb"] = rusage.ru_maxrss / divisor
    except Exception:
        pass

    return stats


def run_single_benchmark(model_name, vol, timeout):
    """Benchmark a single model. Returns dict with results."""
    from brainchop import segment

    result = {
        "model": model_name,
        "status": "ok",
        "time_s": None,
        "error": None,
        "memory": {},
    }

    # Set up timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        mem_before = get_memory_stats()
        t0 = time.perf_counter()
        _ = segment(vol, model_name)
        t1 = time.perf_counter()
        result["time_s"] = round(t1 - t0, 2)
        mem_after = get_memory_stats()
        result["memory"] = {
            "before": mem_before,
            "after": mem_after,
        }
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Exceeded {timeout}s timeout"
    except Exception as e:
        result["status"] = "fail"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return result


def print_results(results):
    """Print a summary table."""
    # Header
    name_w = max(len(r["model"]) for r in results)
    name_w = max(name_w, 5)
    print()
    print("=" * (name_w + 55))
    print(f"{'Model':<{name_w}}  {'Status':<9} {'Time':>8}  {'RSS (MB)':>10}  Error")
    print("-" * (name_w + 55))

    for r in results:
        time_str = f"{r['time_s']:.2f}s" if r["time_s"] is not None else "—"
        rss = ""
        if r["memory"].get("after", {}).get("rss_mb"):
            rss = f"{r['memory']['after']['rss_mb']:.0f}"

        err = r.get("error") or ""
        if len(err) > 60:
            err = err[:60] + "…"

        print(f"{r['model']:<{name_w}}  {r['status']:<9} {time_str:>8}  {rss:>10}  {err}")

    print("=" * (name_w + 55))

    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "fail")
    to = sum(1 for r in results if r["status"] == "timeout")
    total_time = sum(r["time_s"] for r in results if r["time_s"] is not None)
    print(f"\n{ok} passed, {fail} failed, {to} timed out  |  total: {total_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark brainchop models")
    parser.add_argument("n", nargs="?", type=int, default=None,
                        help="Number of models to benchmark (1-N). Default: all")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Timeout per model in seconds (default: {DEFAULT_TIMEOUT})")
    args = parser.parse_args()

    beam = os.environ.get("BEAM", "0")

    # Get models
    from brainchop import list_models, load
    models = list(list_models().keys())

    if args.n is not None:
        models = models[:args.n]

    print(f"brainchop benchmark")
    print(f"  models:  {len(models)} — {', '.join(models)}")
    print(f"  BEAM:    {beam} (from {'envvar' if 'BEAM' in os.environ else 'default'})")
    print(f"  timeout: {args.timeout}s per model")
    print()

    # Download test input once
    print("Fetching test volume…")
    nifti_path = str(Path(fetch(TEST_URL, "t1_crop.nii.gz")))
    vol = load(nifti_path)
    print(f"Loaded {nifti_path}  shape={vol.data.shape}\n")

    # Run benchmarks
    results = []
    for i, model_name in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {model_name} …", end=" ", flush=True)
        r = run_single_benchmark(model_name, vol, args.timeout)
        if r["status"] == "ok":
            print(f"{r['time_s']}s")
        else:
            print(f"{r['status'].upper()}: {r.get('error', '')}")
        results.append(r)

    print_results(results)
    # Exit with error code if any failures
    if any(r["status"] != "ok" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
