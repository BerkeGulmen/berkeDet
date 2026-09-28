"""Tests for berkeDet.  Every result is checked against an independent
reference: itertools, inversion counting, factorial-number-system decoding,
and fraction-free (Bareiss) elimination."""

import cmath
import doctest
import enum
import itertools
import math
import random
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import berkeDet as pkg
from berkeDet import _core, berkeDet, meryemPer, meryemSign

# ------------------------------------------------------------------ references


def kth_perm(n, k):
    """k-th permutation of 0..n-1 in lexicographic order (factorial base)."""
    elems, out = list(range(n)), []
    for i in range(n, 0, -1):
        d, k = divmod(k, math.factorial(i - 1))
        out.append(elems.pop(d))
    return out


def inv_sign(p):
    inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return -1 if inv % 2 else 1


def bareiss(M):
    """Exact integer determinant, O(n^3), shares no code with berkeDet."""
    A = [[int(v) for v in row] for row in M]
    n, sign, prev = len(A), 1, 1
    if n == 0:
        return 1
    for k in range(n - 1):
        if A[k][k] == 0:
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    A[k], A[r] = A[r], A[k]
                    sign = -sign
                    break
            else:
                return 0
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def leibniz(M):
    """Textbook Leibniz formula in plain Python arithmetic."""
    n, det = len(M), 0
    for p in itertools.permutations(range(n)):
        term = 1
        for i in range(n):
            term = term * M[i][p[i]]
        det = det + term if inv_sign(p) > 0 else det - term
    return det


def ref_rows(n):
    return np.fromiter(
        itertools.chain.from_iterable(itertools.permutations(range(n))),
        dtype=np.uint8,
        count=n * math.factorial(n),
    ).reshape(-1, n)


# ------------------------------------------------------------------ docs


def test_docstring_examples():
    failures, tried = doctest.testmod(pkg, verbose=False)
    assert tried >= 8 and failures == 0


# ------------------------------------------------------------------ meryemPer


@pytest.mark.parametrize("n", range(0, 10))
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("threads", [1, 3])
def test_perm_exhaustive(n, order, threads):
    ref = np.array(list(itertools.permutations(range(n))), dtype=np.uint8)
    ref = ref.reshape(math.factorial(n), n)
    got = meryemPer(n, order=order, threads=threads)
    assert got.dtype == np.uint8 and got.shape == ref.shape
    assert np.array_equal(got, ref)
    if n >= 1:
        assert got.flags.c_contiguous if order == "C" else got.flags.f_contiguous
    if order == "F":
        assert np.array_equal(meryemPer(n, threads=threads), ref)  # the default


@pytest.mark.parametrize("order", ["C", "F"])
def test_perm_exhaustive_n10_and_n11_prefixes(order):
    ref = ref_rows(10)
    assert np.array_equal(meryemPer(10, order=order, threads=1), ref)
    assert np.array_equal(meryemPer(10, order=order, threads=4), ref)
    # n = 12 uses two-element prefixes; check whole windows of it against rank decoding
    rng = random.Random(21)
    for a in (0, 479_001_600 - 5000, rng.randrange(479_001_600 - 5000)):
        got = meryemPer(12, a, a + 5000, order=order, threads=4)
        assert got.tolist() == [kth_perm(12, r) for r in range(a, a + 5000)]


def test_items_like_itertools():
    for items in ["ABC", "AAB", [1, "x", None], [(1, 2), (3, 4), (5, 6)], [2.5, -1.0, 7.25, 0.0]]:
        ref = [list(p) for p in itertools.permutations(items)]
        assert meryemPer(items).tolist() == ref
        assert meryemPer(items, order="C").tolist() == ref
        assert meryemPer(items, order="F").flags.f_contiguous
    assert meryemPer("ABCDE", 10, 20).tolist() == [
        list(p) for p in itertools.islice(itertools.permutations("ABCDE"), 10, 20)
    ]
    assert meryemPer([]).shape == (1, 0)


def test_items_are_never_altered():
    class Color(enum.IntEnum):
        RED = 1
        BLUE = 2

    for items in (["a\x00", "b"], [b"a\x00", b"b"], [Color.RED, Color.BLUE],
                  [2**63 + 1, -1], [2**63, 1], [2**70, 3, -5]):
        got = meryemPer(items).tolist()
        assert got == [list(p) for p in itertools.permutations(items)]
        assert all(type(x) is type(items[0]) for row in got for x in row)


def test_items_numpy_rows():
    X = np.arange(12).reshape(4, 3)
    for order in ("F", "C"):
        out = meryemPer(X, order=order)
        assert out.shape == (24, 4, 3)
        for k, p in enumerate(itertools.permutations(range(4))):
            assert np.array_equal(out[k], X[list(p)])


def test_windows_against_factorial_base():
    rng = random.Random(7)
    for n in (1, 2, 5, 9, 12, 15, 20):
        total = math.factorial(n)
        windows = [(0, 1), (total - 1, total), (0, min(total, 50))]
        for _ in range(6):
            a = rng.randrange(total)
            windows.append((a, min(total, a + rng.randrange(1, 3000))))
        for a, b in windows:
            ref = [kth_perm(n, r) for r in range(a, b)]
            assert meryemPer(n, a, b).tolist() == ref, (n, a, b)
            cols = meryemPer(n, a, b, order="F")
            assert cols.tolist() == ref, (n, a, b)
            assert b - a < 2 or cols.flags.f_contiguous
            assert meryemSign(n, a, b).tolist() == [inv_sign(p) for p in ref], (n, a, b)
        assert meryemPer(n, 3 % total, 3 % total).shape == (0, n)


def test_large_window_multithreaded():
    n, a = 12, 123_456_789
    b = a + 3_000_000
    one = meryemPer(n, a, b, threads=1)
    four = meryemPer(n, a, b, threads=4)
    assert np.array_equal(one, four)
    assert np.array_equal(meryemPer(n, a, b, order="F", threads=1), one)
    assert np.array_equal(meryemPer(n, a, b, order="F", threads=4), one)
    s1, s4 = meryemSign(n, a, b, threads=1), meryemSign(n, a, b, threads=4)
    assert np.array_equal(s1, s4)
    rng = random.Random(11)
    for i in rng.sample(range(b - a), 300):
        assert one[i].tolist() == kth_perm(n, a + i)
        assert int(s1[i]) == inv_sign(kth_perm(n, a + i))


def test_chain_reference_generator():
    for n in range(1, 9):
        out = np.empty((math.factorial(n), n), dtype=np.uint8)
        _core.chain_rows(n, out)
        assert np.array_equal(out, ref_rows(n))


# ------------------------------------------------------------------ meryemSign


@pytest.mark.parametrize("n", range(0, 10))
def test_sign_exhaustive(n):
    ref = [inv_sign(p) for p in itertools.permutations(range(n))]
    got = meryemSign(n)
    assert got.dtype == np.int8
    assert got.tolist() == ref
    assert meryemSign(list(range(n))).tolist() == ref


def test_sign_exhaustive_n10_vectorized_reference():
    P = ref_rows(10).astype(np.int16)
    inv = np.zeros(len(P), dtype=np.int16)
    for i in range(10):
        for j in range(i + 1, 10):
            inv += P[:, i] > P[:, j]
    assert np.array_equal(meryemSign(10), np.where(inv % 2 == 1, -1, 1).astype(np.int8))


def test_sign_sums_to_zero():
    for n in range(2, 12):
        assert int(meryemSign(n).sum(dtype=np.int64)) == 0


# ------------------------------------------------------------------ berkeDet


def test_det_integer_small_exhaustive_reference():
    rng = random.Random(3)
    for n in range(1, 8):
        for _ in range(4):
            M = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(n)]
            d = berkeDet(M)
            assert isinstance(d, int)
            assert d == leibniz(M) == bareiss(M)


def test_det_integer_larger_vs_bareiss():
    rng = random.Random(5)
    for n in (8, 9, 10, 11):
        M = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(n)]
        assert berkeDet(M) == bareiss(M)
        assert berkeDet(M, threads=1) == berkeDet(M, threads=4)


def test_det_larger_entries_exact_and_streaming():
    rng = random.Random(8)
    for n, bound in ((11, 30), (12, 30), (8, 2**7), (6, 2**10), (3, 2**20)):
        M = [[rng.randint(-bound, bound) for _ in range(n)] for _ in range(n)]
        assert berkeDet(M) == bareiss(M), (n, bound)


def test_det_totals_beyond_64_bits():
    n = 10  # every term is 63**10 ~ 2**60 in size, so the sums pass 2**64
    for sign in (1, -1):
        M = np.full((n, n), 63 * sign, dtype=np.int64)
        np.fill_diagonal(M, 62 * sign)
        assert berkeDet(M) == bareiss(M.tolist())
    # Sylvester-Hadamard matrix of order 8 scaled by 128: |det| = 128**8 * 8**4 = 2**68,
    # while the product of row maxima is 2**56, inside the C fast path
    H2 = np.array([[1, 1], [1, -1]], dtype=np.int64)
    M = 128 * np.kron(H2, np.kron(H2, H2))
    for matrix in (M, M[[1, 0, 2, 3, 4, 5, 6, 7]]):
        value = berkeDet(matrix)
        assert value == bareiss(matrix.tolist()) and abs(value) == 2**68


def test_det_huge_integers_exact():
    rng = random.Random(9)
    for n in range(1, 7):
        M = [[rng.randint(-(2**70), 2**70) for _ in range(n)] for _ in range(n)]
        assert berkeDet(M) == bareiss(M)
    U = np.array([[2**63 + 5, 3], [7, 2**63 + 1]], dtype=np.uint64)
    assert berkeDet(U) == bareiss(U.tolist())


def test_det_special_integer_matrices():
    for n in range(1, 10):
        assert berkeDet(np.eye(n, dtype=np.int64)) == 1
        assert berkeDet(np.zeros((n, n), dtype=np.int64)) == 0
        for p in [kth_perm(n, math.factorial(n) - 1)]:
            P = np.zeros((n, n), dtype=np.int64)
            P[np.arange(n), p] = 1
            assert berkeDet(P) == inv_sign(p)
    assert berkeDet(np.array([[True, True], [False, True]])) == 1


def test_det_float_vs_lu():
    rng = np.random.default_rng(12)
    for n in range(1, 10):
        M = rng.standard_normal((n, n))
        d = berkeDet(M)
        assert isinstance(d, float)
        assert math.isclose(d, float(np.linalg.det(M)), rel_tol=1e-9, abs_tol=1e-9)
    assert berkeDet(np.eye(3, dtype=np.float32)) == 1.0
    assert math.isnan(berkeDet([[0.0, 1.0], [1.0, math.inf]]))
    assert berkeDet(np.eye(9)) == 1.0 and berkeDet(np.zeros((9, 9))) == 0.0
    empty = berkeDet(np.zeros((0, 0)))
    assert empty == 1.0 and isinstance(empty, float)
    raw = bytearray(8 * 16 + 1)
    unaligned = np.frombuffer(raw, dtype=np.float64, offset=1).reshape(4, 4)
    unaligned.flags.writeable = False
    M = rng.standard_normal((4, 4))
    raw[1:] = M.tobytes()
    assert berkeDet(unaligned) == berkeDet(M)


def test_det_exact_generic_types():
    rng = random.Random(4)
    F = [[Fraction(rng.randint(-5, 5), rng.randint(1, 5)) for _ in range(4)] for _ in range(4)]
    assert berkeDet(F) == leibniz(F)
    D = [[Decimal(rng.randint(-99, 99)) / 10 for _ in range(4)] for _ in range(4)]
    assert berkeDet(D) == leibniz(D)
    C = [[complex(rng.random(), rng.random()) for _ in range(3)] for _ in range(3)]
    assert cmath.isclose(berkeDet(C), complex(np.linalg.det(np.array(C))), rel_tol=1e-9)


def test_det_symbolic_if_sympy_available():
    sympy = pytest.importorskip("sympy")
    a, b, c, d = sympy.symbols("a b c d")
    assert sympy.expand(berkeDet([[a, b], [c, d]]) - (a * d - b * c)) == 0


def test_det_edges():
    assert berkeDet([]) == 1
    assert berkeDet(np.zeros((0, 0), dtype=np.int64)) == 1
    assert berkeDet([[7]]) == 7
    with pytest.raises(ValueError):
        berkeDet([], threads=0)
    with pytest.raises(ValueError):
        berkeDet([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        berkeDet(np.zeros((21, 21), dtype=np.int64))


# ------------------------------------------------------------------ errors


def test_bad_arguments():
    with pytest.raises(ValueError):
        meryemPer(-1)
    with pytest.raises(ValueError):
        meryemPer(21)
    with pytest.raises(ValueError):
        meryemPer(list(range(21)))
    with pytest.raises(ValueError):
        meryemPer(3, 0, 7)
    with pytest.raises(ValueError):
        meryemPer(3, 4, 2)
    with pytest.raises(ValueError):
        meryemPer(3, order="X")
    with pytest.raises(ValueError):
        meryemPer(3, threads=0)
    with pytest.raises(TypeError):
        meryemPer(3.5)
    with pytest.raises(ValueError):
        meryemSign(4, 0, 25)


def test_memory_guard():
    with pytest.raises(MemoryError):
        meryemPer(10, max_bytes=1000)
    with pytest.raises(MemoryError):
        meryemSign(10, max_bytes=1000)
    with pytest.raises(MemoryError):
        meryemPer(13)  # 81 GB at the default 8 GiB limit
    assert meryemPer(10, 0, 10, max_bytes=1000).shape == (10, 10)


def test_private_det_kernel_verifies_its_overflow_bound():
    cols, sm = pkg._tables(1)
    A = np.full(4, 2**30, dtype=np.int64)  # 2 x 2, row maxima product 2**60
    with pytest.raises(ValueError):
        _core.det_i64(2, 1, A, cols, sm, 0, 2, 512, np.zeros(2, dtype=np.int64))
    out = np.zeros(2, dtype=np.int64)
    _core.det_i64(2, 1, A, cols, sm, 0, 2, 3, out)  # 3 * 2**60 < 2**62: accepted
    assert out.tolist() == [0, 0]


def test_private_kernels_reject_wrapping_sizes():
    cols, sm = pkg._tables(1)
    A = np.zeros(400, dtype=np.int64)
    with pytest.raises(ValueError):
        _core.rows_range(20, 1, cols, sm, 0, 922337203685477581, bytearray(4), None)
    with pytest.raises(ValueError):
        _core.cols_range(4, 1, cols, 0, 1, bytearray(64), 2**62, 0)
    with pytest.raises(ValueError):
        _core.det_f64(20, 1, A, cols, sm, 0, 2**61 + 1, True, bytearray(8))
    with pytest.raises(ValueError):
        _core.chain_rows(20, bytearray(64))
