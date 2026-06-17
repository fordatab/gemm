#!/usr/bin/env python3
"""Timing / profiling wrapper for the `modern` CNN binary.

Everything after ``--args`` is treated as the command to run, e.g.:

    python3 utils/profile.py --args ./modern cuda --batch 1000
    python3 utils/profile.py --runs 5 --warmup 1 --args ./modern cpu --batch 1000

It runs the command, measures wall-clock time (and peak RSS on POSIX), parses
the binary's own "Layer Time" / "Op Time" / "Test Accuracy" lines, and prints a
summary. With ``--runs > 1`` it reports mean/min/max wall time across runs.
"""
import argparse
import re
import statistics
import subprocess
import sys
import time

try:
    import resource  # POSIX only; used for peak RSS
except ImportError:  # pragma: no cover - Windows
    resource = None

LAYER_RE = re.compile(r"Layer Time:\s*([\d.]+)\s*ms")
OP_RE = re.compile(r"Op Time:\s*([\d.]+)\s*ms")
ACC_RE = re.compile(r"Test Accuracy:\s*([\d.]+)")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Profile the `modern` binary (or any command).",
        usage="%(prog)s [--runs N] [--warmup N] [-q] --args CMD [CMD_ARGS...]",
    )
    p.add_argument("--runs", type=int, default=1,
                   help="number of timed runs (default: 1)")
    p.add_argument("--warmup", type=int, default=0,
                   help="untimed warmup runs before timing (default: 0)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the child's stdout/stderr")
    p.add_argument("--args", nargs=argparse.REMAINDER, default=[],
                   help="command to profile; must be the LAST flag")
    a = p.parse_args(argv)
    if not a.args:
        p.error("--args requires a command, e.g. --args ./modern cuda --batch 1000")
    if a.runs < 1:
        p.error("--runs must be >= 1")
    return a


def peak_rss_kb():
    """Max RSS (KiB) of terminated children so far, or None if unavailable.

    ru_maxrss is KiB on Linux/WSL and bytes on macOS; we normalize to KiB.
    """
    if resource is None:
        return None
    val = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        val //= 1024
    return val


def run_once(cmd, quiet):
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if not quiet:
        sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)
    return elapsed, proc.stdout + proc.stderr, proc.returncode


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    cmd = a.args
    print(f"Profiling: {' '.join(cmd)}", file=sys.stderr)

    for i in range(a.warmup):
        print(f"  warmup {i + 1}/{a.warmup} ...", file=sys.stderr)
        _, _, rc = run_once(cmd, quiet=True)
        if rc != 0:
            print(f"warmup run failed (exit {rc})", file=sys.stderr)
            return rc

    walls = []
    last_output = ""
    for i in range(a.runs):
        if a.runs > 1:
            print(f"  run {i + 1}/{a.runs} ...", file=sys.stderr)
        elapsed, output, rc = run_once(cmd, a.quiet)
        if rc != 0:
            print(f"command failed (exit {rc})", file=sys.stderr)
            return rc
        walls.append(elapsed)
        last_output = output

    layers = [float(x) for x in LAYER_RE.findall(last_output)]
    ops = [float(x) for x in OP_RE.findall(last_output)]
    acc = ACC_RE.search(last_output)

    print("\n=== profile summary ===")
    print(f"command      : {' '.join(cmd)}")
    print(f"runs         : {a.runs}" + (f" (+{a.warmup} warmup)" if a.warmup else ""))
    if a.runs == 1:
        print(f"wall time    : {walls[0] * 1000:.2f} ms")
    else:
        print(f"wall time    : mean {statistics.mean(walls) * 1000:.2f} ms | "
              f"min {min(walls) * 1000:.2f} | max {max(walls) * 1000:.2f} | "
              f"stdev {statistics.pstdev(walls) * 1000:.2f}")

    peak = peak_rss_kb()
    if peak is not None:
        print(f"peak rss     : {peak / 1024:.1f} MiB")

    if layers:
        print(f"\nper-conv-layer timing (last run, {len(layers)} layers):")
        for i, (lt, ot) in enumerate(zip(layers, ops or [float('nan')] * len(layers))):
            print(f"  layer {i}: layer {lt:8.2f} ms | op {ot:8.2f} ms")
        print(f"  total  : layer {sum(layers):8.2f} ms | op {sum(ops):8.2f} ms")
        if walls:
            overhead = walls[-1] * 1000 - sum(layers)
            print(f"  (non-conv / overhead vs wall: {overhead:.2f} ms)")

    if acc:
        print(f"\ntest accuracy: {acc.group(1)}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
