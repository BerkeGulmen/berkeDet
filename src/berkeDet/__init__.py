"""berkeDet: lexicographic permutations, their signs, and exact determinants,
built by block replication and index mapping.

Three independent tools; use whichever you need:

    >>> from berkeDet import berkeDet, meryemPer, meryemSign
    >>> meryemPer("ABC").tolist()
    [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B'], ['C', 'B', 'A']]
    >>> meryemSign(3).tolist()
    [1, -1, -1, 1, 1, -1]
    >>> berkeDet([[2, 0, 1], [1, 3, 2], [1, 1, 4]])
    18

The output order is exactly the order of ``itertools.permutations``.

Authors: Berke Gülmen, Meryem Gülmen, Ömer Gülmen.
Free for noncommercial use under the PolyForm Noncommercial License 1.0.0;
commercial licenses: gulmenberkemeryem@gmail.com
Preprint: https://doi.org/10.5281/zenodo.21709734
"""

from __future__ import annotations

import math
import operator
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from . import _core

__all__ = ["berkeDet", "meryemPer", "meryemSign"]
__version__ = "1.0.0"

_MAX_N = 20  # 20! < 2**64, so every rank fits in 64 bits
_DEFAULT_MAX_BYTES = 8 * 2**30  # largest result a call allocates unless told otherwise
_TABLE_M = 10  # suffix tables of up to 10! lines (36 MB) are built once and cached
_LIMIT = 2**62  # block-size law: m! * prod_i max_j |a_ij| < 2**62 keeps sums exact
# Below these sizes, starting threads costs more than it saves (measured).
_PAR_MIN_COLS = 1 << 26  # column-layout construction: 64 MB of output
_PAR_MIN_ROWS = 1 << 24  # row-layout windows: 16 MB of output
_PAR_MIN_DET = 1 << 22  # determinant: 4 million Leibniz terms


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _n_threads(threads):
    if threads is None:
        count = getattr(os, "process_cpu_count", os.cpu_count)()
        return count or 1
    t = operator.index(threads)
    if t < 1:
        raise ValueError(f"threads must be a positive integer, got {t}")
    return t


def _run(fn, tasks, threads):
    if threads <= 1 or len(tasks) <= 1:
        for task in tasks:
            fn(*task)
        return
    with ThreadPoolExecutor(max_workers=min(threads, len(tasks))) as pool:
        for future in [pool.submit(fn, *task) for task in tasks]:
            future.result()


def _split(lo, hi, parts):
    parts = max(1, min(parts, hi - lo))
    edges = [lo + (hi - lo) * k // parts for k in range(parts + 1)]
    return [(edges[k], edges[k + 1]) for k in range(parts) if edges[k] < edges[k + 1]]


def _build_tables(m):
    lines = math.factorial(m)
    cols = np.empty(m * lines, dtype=np.uint8)
    _core.cols_full(m, cols)
    signs = np.empty(lines, dtype=np.int8)
    _core.signs_full(m, signs)
    cols.flags.writeable = False
    signs.flags.writeable = False
    return cols, signs


@lru_cache(maxsize=None)
def _tables(m):
    """Column-major table of S_m and its signs (m <= _TABLE_M, cached)."""
    return _build_tables(m)


def _as_labels(seq):
    """Items as a 1-D array.  A typed NumPy array is used only when converting
    cannot change any item (exact built-in types; NumPy strips trailing NUL
    characters from strings, so those stay Python objects)."""
    kinds = {type(x) for x in seq}
    if len(kinds) == 1:
        kind = kinds.pop()
        expected = {bool: "b", int: "iu", float: "f", complex: "c", str: "U", bytes: "S"}
        if kind in (str, bytes):
            nul = "\0" if kind is str else b"\0"
            safe = not any(x.endswith(nul) for x in seq)
        else:
            safe = kind in expected
        if safe:
            arr = np.asarray(seq)
            # NumPy may widen types (e.g. ints beyond 64 bits mixed with small
            # ones become floats); keep the typed array only if it did not
            if arr.ndim == 1 and arr.dtype.kind in expected[kind]:
                return arr
    arr = np.empty(len(seq), dtype=object)
    for i, x in enumerate(seq):
        arr[i] = x
    return arr


def _parse_items(items):
    """Return (n, labels); labels is None when items is an integer n."""
    if isinstance(items, (int, np.integer)) and not isinstance(items, (bool, np.bool_)):
        n = int(items)
        if not 0 <= n <= _MAX_N:
            raise ValueError(f"n must be between 0 and {_MAX_N}, got {n}")
        return n, None
    if isinstance(items, np.ndarray):
        if items.ndim == 0:
            raise TypeError("expected an integer n or a sequence of items")
        labels = items
    else:
        try:
            seq = list(items)
        except TypeError:
            raise TypeError("expected an integer n or a sequence of items") from None
        labels = _as_labels(seq)
    if len(labels) > _MAX_N:
        raise ValueError(f"at most {_MAX_N} items are supported, got {len(labels)}")
    return len(labels), labels


def _window(start, stop, total):
    a = 0 if start is None else operator.index(start)
    b = total if stop is None else operator.index(stop)
    if not 0 <= a <= b <= total:
        raise ValueError(f"need 0 <= start <= stop <= {total}, got start={a}, stop={b}")
    return a, b


def _check_size(nbytes, max_bytes):
    if nbytes > max_bytes:
        raise MemoryError(
            f"this call would allocate {nbytes / 2**30:.1f} GiB, above the limit of "
            f"{max_bytes / 2**30:.1f} GiB. Request a window with start/stop, or pass "
            "a larger max_bytes if your machine has the memory."
        )


def _pick_m(n, count):
    """Window size m! for a request of `count` lines: large enough to cover
    the request, never larger than the cached tables."""
    if n == 1:
        return 1
    m = min(n - 1, _TABLE_M)
    while m > 1 and math.factorial(m - 1) >= count:
        m -= 1
    return m


def _lines(n, start, count, threads, rows, signs):
    """Rows and/or signs of lines start .. start+count-1 (n >= 1)."""
    R = np.empty((count, n), dtype=np.uint8) if rows else None
    S = np.empty(count, dtype=np.int8) if signs else None
    if count == 0:
        return R, S
    m = _pick_m(n, count)
    cols, sm = _tables(m)
    work = count * (n if rows else 1)
    parts = 4 * threads if (threads > 1 and work >= _PAR_MIN_ROWS) else 1
    tasks = [
        (n, m, cols, sm, start + lo, hi - lo,
         R[lo:hi] if rows else None, S[lo:hi] if signs else None)
        for lo, hi in _split(0, count, parts)
    ]
    _run(_core.rows_range, tasks, threads)
    return R, S


def _lines_columns(n, start, count, threads):
    """Lines start .. start+count-1 in column layout: shape (count, n),
    Fortran-ordered, each position stored contiguously (n >= 1)."""
    out = np.empty((n, count), dtype=np.uint8)  # out[j] is position j
    if count:
        m = _pick_m(n, count)
        cols, _ = _tables(m)
        parallel = threads > 1 and count * n >= _PAR_MIN_COLS
        parts = 4 * threads if parallel else 1
        tasks = [
            (n, m, cols, start + lo, hi - lo, out, count, lo)
            for lo, hi in _split(0, count, parts)
        ]
        _run(_core.cols_range, tasks, threads if parallel else 1)
    return out.T


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def meryemPer(items, start=None, stop=None, *, order="F", threads=None,
              max_bytes=_DEFAULT_MAX_BYTES):
    """All permutations in lexicographic order, as one NumPy array.

    Parameters
    ----------
    items : int or sequence
        An integer ``n`` gives the permutations of ``0, 1, ..., n-1`` as a
        ``uint8`` array of shape ``(n!, n)``.  A sequence (string, list,
        tuple, NumPy array) gives the permutations of its items, in the same
        order as ``itertools.permutations(items)``.  At most 20 items.
    start, stop : int, optional
        Return only the permutations with lexicographic ranks
        ``start <= rank < stop``.  They are computed directly, without
        generating the ones before them, so you can reach anywhere in the
        listing of up to 20 items.
    order : {"F", "C"}
        Memory layout of the result; the values are identical either way.
        "F" (default) stores each position contiguously: this is how the
        algorithm builds the listing, as long contiguous block operations,
        and it is the fastest.  "C" stores each permutation contiguously, for
        code that needs raw row-major memory.
    threads : int, optional
        Number of threads.  Default: all available CPU cores.
    max_bytes : int
        Refuse (with ``MemoryError``) to allocate a result larger than this;
        default 8 GiB.

    Returns
    -------
    numpy.ndarray of shape ``(stop - start, n)``.

    Examples
    --------
    >>> meryemPer(3).tolist()
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    >>> meryemPer(4, start=6, stop=8).tolist()
    [[1, 0, 2, 3], [1, 0, 3, 2]]
    >>> meryemPer(20, start=10**18, stop=10**18 + 1)[0].tolist()
    [8, 4, 3, 10, 16, 7, 13, 6, 17, 9, 18, 12, 2, 5, 19, 1, 14, 15, 0, 11]
    """
    n, labels = _parse_items(items)
    if order not in ("C", "F"):
        raise ValueError(f"order must be 'C' or 'F', got {order!r}")
    nthreads = _n_threads(threads)
    total = math.factorial(n)
    a, b = _window(start, stop, total)
    count = b - a
    extra = 0 if labels is None else labels.dtype.itemsize * max(1, labels[0:1].size)
    _check_size(count * n * (1 + extra), max_bytes)

    if n == 0:
        idx = np.empty((count, 0), dtype=np.uint8, order=order)
    elif order == "F":
        idx = _lines_columns(n, a, count, nthreads)
    else:
        idx = _lines(n, a, count, nthreads, rows=True, signs=False)[0]

    if labels is None:
        return idx
    if order == "F":  # gather position by position, keep column layout, no copy
        return np.moveaxis(labels[idx.T], 0, 1)
    return labels[idx]


def meryemSign(items, start=None, stop=None, *, threads=None,
               max_bytes=_DEFAULT_MAX_BYTES):
    """Signs (+1 or -1) of all permutations, in lexicographic order.

    Entry ``k`` is the sign of the ``k``-th permutation listed by
    ``meryemPer`` (and by ``itertools.permutations``).  The complete list is
    built by block replication with alternating flips, with no inversion
    counting; a window only computes the sign of its short shared prefix, once
    per block of up to 10! consecutive signs.

    Parameters
    ----------
    items : int or sequence
        ``n``, or a sequence whose length is ``n``.  At most 20.
    start, stop : int, optional
        Return only the signs with ranks ``start <= rank < stop``.
    threads : int, optional
        Number of threads for windows.  Default: all available CPU cores.
    max_bytes : int
        Refuse (with ``MemoryError``) to allocate a result larger than this;
        default 8 GiB.

    Returns
    -------
    numpy.ndarray of ``int8``, length ``stop - start``.

    Examples
    --------
    >>> meryemSign(3).tolist()
    [1, -1, -1, 1, 1, -1]
    >>> int(meryemSign(4).sum())
    0
    """
    n, _ = _parse_items(items)
    nthreads = _n_threads(threads)
    total = math.factorial(n)
    a, b = _window(start, stop, total)
    count = b - a
    _check_size(count, max_bytes)
    if n == 0:
        return np.ones(count, dtype=np.int8)
    if count == total:
        out = np.empty(total, dtype=np.int8)
        _core.signs_full(n, out)
        return out
    return _lines(n, a, count, nthreads, rows=False, signs=True)[1]


def berkeDet(matrix, *, threads=None):
    """Determinant by the Leibniz formula, streamed through meryemPer and
    meryemSign.

    * Integer (and boolean) matrices give the exact integer determinant.
      While the product of the row maxima, ``prod_i max_j |a_ij|``, stays
      below ``2**62``, the sum runs in C with an exact 128-bit accumulator;
      beyond that it runs in Python's unbounded integers (exact, slower).
    * Floating-point matrices are computed in ``float64`` and give a
      ``float``.
    * Anything else is summed in its own Python arithmetic, on a slower path:
      exact for ``fractions.Fraction``, symbolic for SymPy expressions, and as
      precise as the type for ``decimal.Decimal`` (context precision) and
      complex numbers (floating point).

    The Leibniz formula has ``n!`` terms, so the cost grows as ``n * n!``.
    For numerical work on larger matrices use an ``O(n^3)`` method such as
    ``numpy.linalg.det``.  At most 20 x 20.

    Parameters
    ----------
    matrix : array_like, square
    threads : int, optional
        Number of threads.  Default: all available CPU cores.

    Examples
    --------
    >>> berkeDet([[2, 0, 1], [1, 3, 2], [1, 1, 4]])
    18
    >>> from fractions import Fraction
    >>> berkeDet([[Fraction(1, 2), 1], [1, Fraction(1, 3)]])
    Fraction(-5, 6)
    """
    A = np.asarray(matrix)
    if A.ndim == 1 and A.size == 0:
        A = A.reshape(0, 0)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"berkeDet needs a square matrix, got shape {A.shape}")
    n = A.shape[0]
    if n > _MAX_N:
        raise ValueError(f"at most {_MAX_N} x {_MAX_N} is supported, got {n} x {n}")
    nthreads = _n_threads(threads)
    kind = A.dtype.kind
    if n == 0:
        return 1.0 if kind == "f" else 1
    if kind in "biu":
        return _det_int(A.tolist(), n, nthreads)
    if kind == "f":
        return _det_float(A, n, nthreads)
    return _det_generic(A, n)


def _det_int(rows, n, threads):
    if n == 1:
        return int(rows[0][0])
    pmax = 1
    for row in rows:
        pmax *= max(1, max(abs(int(v)) for v in row))
    if pmax >= _LIMIT:  # a single term could leave 64 bits: stay in Python
        return _det_generic(np.array([[int(v) for v in row] for row in rows], dtype=object), n)
    run = min(512, (_LIMIT - 1) // pmax)  # terms summed in 64 bits before carrying
    m = min(n - 1, _TABLE_M)
    windows = math.factorial(n) // math.factorial(m)
    cols, sm = _tables(m)
    A = np.asarray(rows, dtype=np.int64)
    parallel = threads > 1 and math.factorial(n) >= _PAR_MIN_DET
    ranges = _split(0, windows, 4 * threads if parallel else 1)
    totals = np.zeros((len(ranges), 2), dtype=np.int64)
    tasks = [(n, m, A, cols, sm, lo, hi, run, totals[k]) for k, (lo, hi) in enumerate(ranges)]
    _run(_core.det_i64, tasks, threads if parallel else 1)
    return sum((int(hi) << 64) + (int(lo) & 0xFFFFFFFFFFFFFFFF) for lo, hi in totals.tolist())


def _det_float(A, n, threads):
    A = np.require(A, dtype=np.float64, requirements=["C", "A"])
    if n == 1:
        return float(A[0, 0])
    m = min(n - 1, _TABLE_M)
    windows = math.factorial(n) // math.factorial(m)
    cols, sm = _tables(m)
    partials = np.empty(windows, dtype=np.float64)
    parallel = threads > 1 and math.factorial(n) >= _PAR_MIN_DET
    finite = bool(np.isfinite(A).all())
    tasks = [
        (n, m, A, cols, sm, lo, hi, finite, partials[lo:hi])
        for lo, hi in _split(0, windows, 4 * threads if parallel else 1)
    ]
    _run(_core.det_f64, tasks, threads if parallel else 1)
    return math.fsum(partials.tolist())


def _det_generic(M, n):
    """Leibniz sum in the entries' own Python arithmetic, vectorized over
    blocks of permutations with NumPy object arrays."""
    M = np.asarray(M, dtype=object)
    if n == 1:
        return M[0, 0]
    total = math.factorial(n)
    det = 0
    step = 1 << 15
    for a in range(0, total, step):
        b = min(total, a + step)
        P = meryemPer(n, a, b, threads=1)
        S = meryemSign(n, a, b, threads=1)
        terms = M[0][P[:, 0]]
        for i in range(1, n):
            terms = terms * M[i][P[:, i]]
        plus, minus = terms[S > 0], terms[S < 0]
        if plus.size:
            det = det + plus.sum()
        if minus.size:
            det = det - minus.sum()
    return det
