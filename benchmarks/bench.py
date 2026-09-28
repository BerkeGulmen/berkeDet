"""Reproducible benchmarks for the berkeDet package.

    python benchmarks/bench.py

Compares meryemPer with itertools.permutations and with a chain generator
(Narayana Pandita, i.e. std::next_permutation) written in the same C as
berkeDet, so that the language is the same on both sides.  Median of 5 runs.
"""

import gc
import itertools
import math
import os
import platform
import sys
import time

import numpy as np

import berkeDet
from berkeDet import _core, berkeDet as det, meryemPer, meryemSign


def median_time(fn, repeat=5):
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
        del result
    return sorted(times)[repeat // 2]


def chain(n):
    out = np.empty((math.factorial(n), n), dtype=np.uint8)
    _core.chain_rows(n, out)
    return out


def itertools_array(n):
    return np.fromiter(
        itertools.chain.from_iterable(itertools.permutations(range(n))),
        dtype=np.uint8,
        count=n * math.factorial(n),
    ).reshape(-1, n)


def main():
    cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    print(f"berkeDet {berkeDet.__version__} | Python {platform.python_version()} | "
          f"NumPy {np.__version__} | {platform.machine()} | {cores} cores")
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    print(f"CPU: {cpu}\n")

    print("All permutations of n items, lexicographic order, as a NumPy array")
    print(f"{'n':>3}  {'method':<46} {'seconds':>9} {'vs C chain':>11}")
    for n in (10, 11):
        base = median_time(lambda: chain(n))
        rows = []
        if n == 10:
            rows.append(("itertools.permutations -> list of tuples",
                         median_time(lambda: list(itertools.permutations(range(n))))))
        rows.append(("itertools.permutations -> NumPy array",
                     median_time(lambda: itertools_array(n), repeat=3)))
        rows.append(("chain generator in C (Narayana)", base))
        for order, label in (("C", "row layout"), ("F", "column layout")):
            for threads in sorted({1, cores}):
                t = median_time(lambda: meryemPer(n, order=order, threads=threads))
                rows.append((f"meryemPer, {label}, {threads} thread(s)", t))
        for label, t in rows:
            print(f"{n:>3}  {label:<46} {t:>9.4f} {base / t:>10.2f}x")
        print()

    print("All signs, lexicographic order")
    for n in (11, 12):
        t = median_time(lambda: meryemSign(n))
        print(f"{n:>3}  meryemSign: {math.factorial(n):>11,} signs in {t:.4f} s")
    print()

    print("Exact integer determinant, entries in [-4, 4]")
    rng = np.random.default_rng(42)
    for n in (9, 10, 11, 12):
        M = rng.integers(-4, 5, size=(n, n))
        t = median_time(lambda: det(M), repeat=3)
        value = det(M)
        lu = float(np.linalg.det(M.astype(float)))
        rel = abs(value - lu) / max(1.0, abs(value))
        print(f"{n:>3}  berkeDet = {value:>14}  in {t:8.4f} s   "
              f"(LU float: {lu:.6e}, relative difference {rel:.1e})")


if __name__ == "__main__":
    sys.exit(main())
