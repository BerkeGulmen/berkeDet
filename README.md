# berkeDet

Lexicographic permutations, their signs, and exact determinants — built by
**block replication and index mapping** instead of step-by-step generation.

```
pip install berkeDet
```

```python
from berkeDet import berkeDet, meryemPer, meryemSign

meryemPer("ABC")                     # all 6 permutations, same order as itertools.permutations
meryemPer(11)                        # all 39,916,800 permutations of 0..10, one NumPy array
meryemSign(11)                       # their signs (+1 / -1), same order
meryemPer(20, start=10**18, stop=10**18 + 1_000_000)   # a million permutations of 20 items,
                                                       # starting at rank 10^18, computed directly
berkeDet([[2, 0, 1], [1, 3, 2], [1, 1, 4]])            # 18, exact
```

The three tools are independent: use whichever you need.

Free for personal, educational, research and other noncommercial use;
commercial use needs a license (see [License](#license)).

## The tools

### `meryemPer(items, start=None, stop=None, *, order="F", threads=None, max_bytes=8 GiB)`

All permutations in lexicographic order, as one NumPy array.

- `items` — an integer `n` gives the permutations of `0, 1, …, n-1` as a
  `uint8` array of shape `(n!, n)`. A string, list, tuple or NumPy array gives
  the permutations of its items, in exactly the order of
  `itertools.permutations(items)`. Up to 20 items.
- `start`, `stop` — only the permutations with lexicographic ranks
  `start ≤ rank < stop`. They are computed directly, without generating the
  ones before them, so every one of the 20! ≈ 2.4 × 10¹⁸ permutations of 20
  items is reachable.
- `order` — memory layout; the values are the same either way. `"F"` (default)
  stores each position contiguously, which is how the algorithm builds the
  listing and the fastest layout. `"C"` stores each permutation contiguously,
  for code that needs raw row-major memory.
- `threads` — default: all CPU cores.
- `max_bytes` — the call refuses (with `MemoryError`) to allocate a result
  larger than this; default 8 GiB. All 13! permutations would need 81 GB, so
  request windows or raise the limit deliberately.

### `meryemSign(items, start=None, stop=None, *, threads=None, max_bytes=8 GiB)`

The signs (+1 or −1, as `int8`) of the same permutations, in the same order.
The complete list is built by block replication with alternating flips, with
no inversion counting. Windows work as in `meryemPer`; a window computes the
sign of its short shared prefix once per block of up to 10! consecutive signs.
`threads` and `max_bytes` work as in `meryemPer`.

### `berkeDet(matrix, *, threads=None)`

The determinant by the Leibniz formula, streamed through `meryemPer` and
`meryemSign`.

- Integer matrices give the **exact** integer determinant, whatever the size
  of the entries. While the product of the row maxima stays below 2⁶², the sum
  runs in C with an exact 128-bit accumulator; beyond that it runs in Python's
  unbounded integers (exact, slower).
- Floating-point matrices are computed in `float64` and give a `float`.
- Other entries are summed in their own Python arithmetic (a slower path):
  exact for `Fraction`, symbolic for SymPy expressions, and as precise as the
  type for `Decimal` (context precision) and complex numbers (floating point).

The Leibniz formula has `n!` terms, so the cost grows as `n · n!`. For
numerical work on larger matrices, `numpy.linalg.det` (an `O(n³)` method) is
the right tool; `berkeDet` gives exact answers for small matrices and shows the
enumeration at work.

## How it works

Classical permutation generators — `itertools.permutations`,
`std::next_permutation` (Narayana Pandita), Steinhaus–Johnson–Trotter, Heap —
are *chains*: each permutation is computed from the one before it.

`meryemPer` instead builds the whole listing of `n` items from the listing of
`n − 1` items. Block `f` is the previous listing with every value `x` replaced
by `x + (x ≥ f)`, preceded by `f` (blocks counted from `f = 0`). The blocks
are independent of each other, so the work runs as long contiguous vector
operations and splits across CPU threads. `meryemSign` uses the same block
structure: block `f` is the previous sign list multiplied by `(−1)^f`.

Windows are served from a cached table of up to 10! lines plus a short prefix
per window of 10! consecutive permutations. `berkeDet` sums runs of terms in
64-bit integers, with each run short enough that it cannot overflow (the
block-size law of the paper), and carries the runs into an exact 128-bit
total.

## Performance

Measured with `benchmarks/bench.py` on a 2-core Intel Xeon at 2.10 GHz cloud
machine (Python 3.11, NumPy 2.4). All permutations of 11 items (39,916,800),
lexicographic order, as a NumPy array:

| Method | Seconds | vs. chain in C |
|---|---:|---:|
| `itertools.permutations` → NumPy array | 10.64 | 0.04× |
| Chain generator in C (Narayana / `next_permutation`) | 0.415 | 1.00× |
| `meryemPer`, row layout (`order="C"`), 1 thread | 0.343 | 1.21× |
| `meryemPer`, row layout, 2 threads | 0.258 | 1.61× |
| `meryemPer`, column layout (default), 1 thread | 0.130 | 3.20× |
| `meryemPer`, column layout, 2 threads | 0.077 | 5.40× |

The machine is shared, so the same measurement varies between runs. Across
repeated runs of this code, the column layout measured 3.1–3.9× on one thread
and 4.8–5.9× on two; the row layout measured 0.9–1.2× on one thread (so on a
single core it is on par with a well-written chain generator) and 1.5–2.1× on
two. With 10 items the column layout measured 6–17×, but those timings are
under 10 ms and noisy. The chain generator is written in the same C as
`berkeDet`, so the comparison is between methods, not languages. The large
gap to `itertools` mostly measures Python's cost of creating one tuple per
permutation; any array-producing C generator removes most of it. Scaling
beyond two cores has not been measured here — run `benchmarks/bench.py` on
your own machine.

Also measured: all 479,001,600 signs of 12 items in 0.13 s, and the exact
determinant of a 12 × 12 integer matrix with entries in [−4, 4] (479 million
Leibniz terms) in 1.1 s.

**When `itertools` is the better choice.** If you walk through permutations in
a Python loop and do Python work on each one, the loop itself is the
bottleneck, and `itertools.permutations` is simple and lazy. `berkeDet` is for
code that consumes whole arrays (NumPy, GPUs, compiled code), for signs, and
for rank windows.

## Correctness

The test suite checks every result against an independent reference: every
permutation and sign up to 10 items against `itertools` and inversion counting;
windows up to rank 20! − 1 against factorial-number-system decoding;
determinants against fraction-free (Bareiss) elimination, plus `Fraction`,
`Decimal`, complex and SymPy inputs. The C layer checks every buffer size,
range and overflow bound before it touches memory. Every wheel is built and tested on its
own platform before release.

## Citing

If you use berkeDet, please cite the paper:

> Gülmen, B., Gülmen, M., & Gülmen, Ö. (2026). *berkeDet: data-parallel
> lexicographic permutation generation and streaming exact determinant
> evaluation via block replication and index mapping* (Version 4). Zenodo.
> https://doi.org/10.5281/zenodo.21709734

Project website: https://detsignper.org

## License

berkeDet is free for noncommercial use under the
[PolyForm Noncommercial License 1.0.0](https://github.com/BerkeGulmen/berkeDet/blob/main/LICENSE):
personal study, research and experiment, teaching, hobby projects, and use by
educational institutions, public research organizations, charities and
government institutions.

Any commercial use, for example in a company's products, services or internal
work, requires a separate commercial license from the authors:
**gulmenberkemeryem@gmail.com**

The license text is authoritative; this summary is only a guide.

© 2026 Berke Gülmen, Meryem Gülmen, Ömer Gülmen
