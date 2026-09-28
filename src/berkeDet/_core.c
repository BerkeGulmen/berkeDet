/*
 * _core.c -- C kernels of the berkeDet package.
 *
 * meryemPer  all permutations of {0..n-1} in lexicographic order, built by
 *            block replication and index mapping.  The index map of block f
 *            is g_f(x) = x + (x >= f), so every block is one vector-friendly
 *            pass over contiguous memory.
 * meryemSign all signs in the same order, by block replication with flips.
 * berkeDet   exact Leibniz determinant, streamed in windows of m! lines.
 *
 * Portable C99 for gcc, clang and MSVC.  No OpenMP: the Python layer splits
 * the work across threads, and every kernel releases the GIL.
 *
 * Authors: Berke Gulmen, Meryem Gulmen, Omer Gulmen.
 * Free for noncommercial use under the PolyForm Noncommercial License 1.0.0;
 * commercial licenses: gulmenberkemeryem@gmail.com
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 20   /* 20! < 2^64: every rank fits in uint64_t */
#define MAX_M 12   /* largest suffix table the kernels accept   */
#define TILE 512   /* lines per inner tile (fits in L1 cache)   */
#define PAD 256    /* lookup arrays hold any uint8_t index        */

static const uint64_t FACT[MAX_N + 1] = {
    1ULL, 1ULL, 2ULL, 6ULL, 24ULL, 120ULL, 720ULL, 5040ULL, 40320ULL,
    362880ULL, 3628800ULL, 39916800ULL, 479001600ULL, 6227020800ULL,
    87178291200ULL, 1307674368000ULL, 20922789888000ULL,
    355687428096000ULL, 6402373705728000ULL, 121645100408832000ULL,
    2432902008176640000ULL};

/* ------------------------------------------------------------------ */
/* Kernels (no Python API calls: safe to run without the GIL)          */
/* ------------------------------------------------------------------ */

/* Block f, rows r0..r1-1, of layer b (column-major, b! rows) from the
   column-major layer b-1 in `prev` ((b-1)! rows). */
static void fill_block(const uint8_t *prev, int b, uint8_t *cur, int f,
                       size_t r0, size_t r1)
{
    const size_t P = (size_t)FACT[b - 1];
    const size_t L = (size_t)b * P;
    const uint8_t ff = (uint8_t)f;
    int j;
    memset(cur + (size_t)f * P + r0, ff, r1 - r0);
    for (j = 1; j < b; j++) {
        const uint8_t *src = prev + (size_t)(j - 1) * P;
        uint8_t *dst = cur + (size_t)j * L + (size_t)f * P;
        size_t r;
        for (r = r0; r < r1; r++)
            dst[r] = (uint8_t)(src[r] + (src[r] >= ff));
    }
}

/* Whole column-major table of S_m: out[j*m! + r] = element j of the r-th
   permutation.  Returns -1 when scratch memory cannot be allocated. */
static int build_cols(int m, uint8_t *out)
{
    size_t tmp;
    uint8_t *a, *c, *prev, *spare;
    int b, f;
    if (m == 1) {
        out[0] = 0;
        return 0;
    }
    tmp = (size_t)(m - 1) * (size_t)FACT[m - 1];
    a = (uint8_t *)malloc(tmp);
    c = (uint8_t *)malloc(tmp);
    if (a == NULL || c == NULL) {
        free(a);
        free(c);
        return -1;
    }
    prev = a;
    spare = c;
    prev[0] = 0; /* layer 1 = [0] */
    for (b = 2; b <= m; b++) {
        uint8_t *cur = (b == m) ? out : spare;
        for (f = 0; f < b; f++)
            fill_block(prev, b, cur, f, 0, (size_t)FACT[b - 1]);
        if (b < m) {
            spare = prev;
            prev = cur;
        }
    }
    free(a);
    free(c);
    return 0;
}

/* All n! signs in lexicographic order, by block replication with flips. */
static void build_signs(int n, int8_t *out)
{
    size_t size = 1;
    int e, d;
    out[0] = 1;
    for (e = 1; e < n; e++) {
        size_t pos = size;
        for (d = 0; d < e; d++) {
            int8_t *blk = out + pos;
            if (d % 2 == 0) {
                size_t i;
                for (i = 0; i < size; i++)
                    blk[i] = (int8_t)(-out[i]);
            } else {
                memcpy(blk, out, size);
            }
            pos += size;
        }
        size = pos;
    }
}

/* Window q: its prefix (the first n-m elements, shared by m! consecutive
   lines), the increasing list of remaining elements, and the sign factor
   contributed by the prefix. */
static void unrank_prefix(int n, int m, uint64_t q, uint8_t *pre,
                          uint8_t *rem, int *psign)
{
    uint8_t avail[MAX_N];
    int na = n, w = n - m, i, k, inv = 0;
    for (i = 0; i < n; i++)
        avail[i] = (uint8_t)i;
    for (i = 0; i < w; i++) {
        uint64_t f = FACT[n - 1 - i] / FACT[m];
        int d = (int)(q / f);
        q %= f;
        pre[i] = avail[d];
        memmove(avail + d, avail + d + 1, (size_t)(na - d - 1));
        na--;
    }
    memcpy(rem, avail, (size_t)m);
    for (i = 0; i < w; i++) {
        int smaller_in_pre = 0;
        for (k = 0; k < w; k++)
            if (pre[k] < pre[i])
                smaller_in_pre++;
        inv += pre[i] - smaller_in_pre; /* smaller elements in the suffix */
        for (k = i + 1; k < w; k++)
            if (pre[i] > pre[k])
                inv++;
    }
    *psign = (inv & 1) ? -1 : 1;
}

/* Lines start .. start+count-1 of the lexicographic listing of S_n, written
   as rows (count x n, row-major) and/or signs; either output may be NULL.
   PmT is the column-major table of S_m, Sm its signs. */
static void rows_range(int n, int m, const uint8_t *PmT, const int8_t *Sm,
                       uint64_t start, uint64_t count, uint8_t *rows,
                       int8_t *signs)
{
    const uint64_t Mf = FACT[m];
    const uint64_t end = start + count;
    const int w = n - m;
    uint8_t pre[MAX_N], rem[PAD] = {0};
    int psign;
    uint64_t R = start;
    while (R < end) {
        const uint64_t q = R / Mf, base = q * Mf;
        const uint64_t s0 = R - base;
        const uint64_t s1 = (end - base < Mf) ? end - base : Mf;
        unrank_prefix(n, m, q, pre, rem, &psign);
        if (rows != NULL) {
            uint64_t t0;
            for (t0 = s0; t0 < s1; t0 += TILE) {
                const size_t cnt = (size_t)((s1 - t0 < TILE) ? s1 - t0 : TILE);
                uint8_t *blk = rows + (size_t)(base + t0 - start) * (size_t)n;
                size_t t;
                int i, j;
                for (t = 0; t < cnt; t++)
                    for (i = 0; i < w; i++)
                        blk[t * (size_t)n + (size_t)i] = pre[i];
                if (w == 1) {
                    /* one removed element f: rem[x] = x + (x >= f), which
                       compilers turn into vector instructions (measured
                       faster than the table lookup for this case) */
                    const uint8_t ff = pre[0];
                    for (j = 0; j < m; j++) {
                        const uint8_t *src = PmT + (size_t)j * (size_t)Mf + t0;
                        uint8_t *dst = blk + 1 + j;
                        for (t = 0; t < cnt; t++) {
                            const uint8_t x = src[t];
                            dst[t * (size_t)n] = (uint8_t)(x + (x >= ff));
                        }
                    }
                } else {
                    for (j = 0; j < m; j++) {
                        const uint8_t *src = PmT + (size_t)j * (size_t)Mf + t0;
                        uint8_t *dst = blk + w + j;
                        for (t = 0; t < cnt; t++)
                            dst[t * (size_t)n] = rem[src[t]];
                    }
                }
            }
        }
        if (signs != NULL) {
            int8_t *dst = signs + (size_t)(base + s0 - start);
            const int8_t *src = Sm + s0;
            const size_t cnt = (size_t)(s1 - s0);
            if (psign > 0) {
                memcpy(dst, src, cnt);
            } else {
                size_t t;
                for (t = 0; t < cnt; t++)
                    dst[t] = (int8_t)(-src[t]);
            }
        }
        R = base + s1;
    }
}

/* Lines start .. start+count-1 in column layout: element j of line
   start+i goes to out[j*stride + offset + i].  Every write is contiguous, and
   for prefixes of one or two elements the index map is pure arithmetic
   (x + (x >= f)), which compilers vectorize. */
static void cols_range(int n, int m, const uint8_t *PmT, uint64_t start,
                       uint64_t count, uint8_t *out, size_t stride,
                       size_t offset)
{
    const uint64_t Mf = FACT[m];
    const uint64_t end = start + count;
    const int w = n - m;
    uint8_t pre[MAX_N], rem[PAD] = {0};
    int psign;
    uint64_t R = start;
    while (R < end) {
        const uint64_t q = R / Mf, base = q * Mf;
        const uint64_t s0 = R - base;
        const uint64_t s1 = (end - base < Mf) ? end - base : Mf;
        const size_t cnt = (size_t)(s1 - s0);
        const size_t o = offset + (size_t)(base + s0 - start);
        int i, j;
        size_t t;
        unrank_prefix(n, m, q, pre, rem, &psign);
        for (i = 0; i < w; i++)
            memset(out + (size_t)i * stride + o, pre[i], cnt);
        for (j = 0; j < m; j++) {
            const uint8_t *src = PmT + (size_t)j * (size_t)Mf + s0;
            uint8_t *dst = out + (size_t)(w + j) * stride + o;
            if (w == 0) {
                memcpy(dst, src, cnt);
            } else if (w == 1) {
                const uint8_t f = pre[0];
                for (t = 0; t < cnt; t++) {
                    const uint8_t x = src[t];
                    dst[t] = (uint8_t)(x + (x >= f));
                }
            } else if (w == 2) {
                const uint8_t lo = (pre[0] < pre[1]) ? pre[0] : pre[1];
                const uint8_t hi = (pre[0] < pre[1]) ? pre[1] : pre[0];
                for (t = 0; t < cnt; t++) {
                    const uint8_t x = src[t];
                    const uint8_t y = (uint8_t)(x + (x >= lo));
                    dst[t] = (uint8_t)(y + (y >= hi));
                }
            } else {
                for (t = 0; t < cnt; t++)
                    dst[t] = rem[src[t]];
            }
        }
        R = base + s1;
    }
}

/* Add a signed 64-bit value to a 128-bit two's-complement accumulator held
   as (hi, lo).  Portable: MSVC has no __int128. */
static void add128(int64_t *hi, uint64_t *lo, int64_t x)
{
    const uint64_t ux = (uint64_t)x;
    const uint64_t sum = *lo + ux;
    *hi += (int64_t)(sum < *lo) - (int64_t)(x < 0);
    *lo = sum;
}

/* Exact sum of the Leibniz terms of windows q_lo .. q_hi-1 into a 128-bit
   total (out[0] = low 64 bits, out[1] = high 64 bits).  The caller guarantees
   pmax = prod_i max(1, max_j |a_ij|) < 2^62, so every product fits in 64 bits,
   and B * pmax < 2^62, so each run of B terms is summed in 64 bits before it
   is added to the 128-bit total (|total| <= 20! * 2^62 < 2^124). */
static void det_i64(int n, int m, const int64_t *A, const uint8_t *PmT,
                    const int8_t *Sm, uint64_t q_lo, uint64_t q_hi, size_t B,
                    int64_t *out)
{
    const uint64_t Mf = FACT[m];
    const int w = n - m;
    uint8_t pre[MAX_N], rem[PAD] = {0};
    int psign;
    int64_t T[MAX_M * MAX_M + PAD] = {0};
    int64_t prod[TILE];
    int64_t hi = 0;
    uint64_t lo = 0;
    uint64_t q;
    for (q = q_lo; q < q_hi; q++) {
        int64_t pp = 1;
        int i, j, k;
        uint64_t t0;
        unrank_prefix(n, m, q, pre, rem, &psign);
        for (i = 0; i < w; i++)
            pp *= A[(size_t)i * n + pre[i]];
        if (pp == 0) /* exact integers: every term of the window is 0 */
            continue;
        for (j = 0; j < m; j++)
            for (k = 0; k < m; k++)
                T[j * m + k] = A[(size_t)(w + j) * n + rem[k]];
        for (t0 = 0; t0 < Mf; t0 += TILE) {
            const size_t cnt = (size_t)((Mf - t0 < TILE) ? Mf - t0 : TILE);
            const int8_t *sg = Sm + t0;
            size_t t, b0;
            for (t = 0; t < cnt; t++)
                prod[t] = pp;
            for (j = 0; j < m; j++) {
                const uint8_t *src = PmT + (size_t)j * (size_t)Mf + t0;
                const int64_t *Tj = T + j * m;
                for (t = 0; t < cnt; t++)
                    prod[t] *= Tj[src[t]];
            }
            for (b0 = 0; b0 < cnt; b0 += B) {
                const size_t b1 = (cnt - b0 < B) ? cnt : b0 + B;
                int64_t run = 0;
                for (t = b0; t < b1; t++)
                    run += (int64_t)sg[t] * prod[t];
                add128(&hi, &lo, (psign > 0) ? run : -run);
            }
        }
    }
    out[0] = (int64_t)lo;
    out[1] = hi;
}

/* Floating-point version of det_i64. */
static void det_f64(int n, int m, const double *A, const uint8_t *PmT,
                    const int8_t *Sm, uint64_t q_lo, uint64_t q_hi,
                    int all_finite, double *partials)
{
    const uint64_t Mf = FACT[m];
    const int w = n - m;
    uint8_t pre[MAX_N], rem[PAD] = {0};
    int psign;
    double T[MAX_M * MAX_M + PAD] = {0};
    double prod[TILE];
    uint64_t q;
    for (q = q_lo; q < q_hi; q++) {
        double pp = 1.0, acc = 0.0;
        int i, j, k;
        unrank_prefix(n, m, q, pre, rem, &psign);
        for (i = 0; i < w; i++)
            pp *= A[(size_t)i * n + pre[i]];
        /* a zero prefix empties the window, unless some entry is inf or NaN
           (0 * inf must stay NaN), so the caller tells us which case holds */
        if (pp != 0.0 || !all_finite) {
            uint64_t t0;
            for (j = 0; j < m; j++)
                for (k = 0; k < m; k++)
                    T[j * m + k] = A[(size_t)(w + j) * n + rem[k]];
            for (t0 = 0; t0 < Mf; t0 += TILE) {
                const size_t cnt = (size_t)((Mf - t0 < TILE) ? Mf - t0 : TILE);
                const int8_t *sg = Sm + t0;
                size_t t;
                for (t = 0; t < cnt; t++)
                    prod[t] = pp;
                for (j = 0; j < m; j++) {
                    const uint8_t *src = PmT + (size_t)j * (size_t)Mf + t0;
                    const double *Tj = T + j * m;
                    for (t = 0; t < cnt; t++)
                        prod[t] *= Tj[src[t]];
                }
                for (t = 0; t < cnt; t++)
                    acc += (double)sg[t] * prod[t];
            }
        }
        partials[q - q_lo] = (psign > 0) ? acc : -acc;
    }
}

/* Reference chain generator (Narayana Pandita / std::next_permutation) in
   the same C, used only by the benchmarks for a fair comparison. */
static void chain_rows(int n, uint8_t *out)
{
    uint8_t a[MAX_N];
    const uint64_t N = FACT[n];
    uint64_t r;
    int i;
    for (i = 0; i < n; i++)
        a[i] = (uint8_t)i;
    for (r = 0; r < N; r++) {
        int x, y, lo, hi;
        uint8_t t;
        memcpy(out + (size_t)r * (size_t)n, a, (size_t)n);
        x = n - 2;
        while (x >= 0 && a[x] > a[x + 1])
            x--;
        if (x < 0)
            break;
        y = n - 1;
        while (a[y] < a[x])
            y--;
        t = a[x];
        a[x] = a[y];
        a[y] = t;
        for (lo = x + 1, hi = n - 1; lo < hi; lo++, hi--) {
            t = a[lo];
            a[lo] = a[hi];
            a[hi] = t;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Python wrappers                                                     */
/* ------------------------------------------------------------------ */

/* The buffer must hold count * size bytes.  Compared by division, so the
   product can never wrap around. */
static int need(const Py_buffer *b, uint64_t count, uint64_t size,
                const char *what)
{
    if (size != 0 && count > (uint64_t)b->len / size) {
        PyErr_Format(PyExc_ValueError, "%s buffer is too small", what);
        return -1;
    }
    return 0;
}

static int bad_range(long long v, long long lo, long long hi, const char *what)
{
    if (v < lo || v > hi) {
        PyErr_Format(PyExc_ValueError, "%s=%lld is outside [%lld, %lld]", what,
                     v, lo, hi);
        return 1;
    }
    return 0;
}

/* optional writable buffer: None -> buf.obj == NULL */
static int get_optional(PyObject *o, Py_buffer *v)
{
    if (o == Py_None) {
        v->obj = NULL;
        v->buf = NULL;
        v->len = 0;
        return 0;
    }
    return PyObject_GetBuffer(o, v, PyBUF_WRITABLE | PyBUF_C_CONTIGUOUS);
}

static PyObject *py_cols_full(PyObject *self, PyObject *args)
{
    int m, rc;
    Py_buffer out;
    (void)self;
    if (!PyArg_ParseTuple(args, "iw*", &m, &out))
        return NULL;
    if (bad_range(m, 1, MAX_M, "m") ||
        need(&out, FACT[m], (uint64_t)m, "out")) {
        PyBuffer_Release(&out);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    rc = build_cols(m, (uint8_t *)out.buf);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&out);
    if (rc != 0)
        return PyErr_NoMemory();
    Py_RETURN_NONE;
}

static PyObject *py_cols_range(PyObject *self, PyObject *args)
{
    int n, m;
    long long start, count, stride, offset;
    Py_buffer PmT, out;
    PyObject *res = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiy*LLw*LL", &n, &m, &PmT, &start, &count,
                          &out, &stride, &offset))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") ||
        bad_range(m, 1, (n < MAX_M) ? n : MAX_M, "m") ||
        bad_range(start, 0, (long long)FACT[n], "start") ||
        bad_range(count, 0, (long long)(FACT[n] - (uint64_t)start), "count") ||
        bad_range(stride, 0, PY_SSIZE_T_MAX, "stride") ||
        bad_range(offset, 0, stride - count, "offset") ||
        need(&PmT, FACT[m], (uint64_t)m, "PmT") ||
        need(&out, (uint64_t)stride, (uint64_t)n, "out"))
        goto done;
    Py_BEGIN_ALLOW_THREADS
    cols_range(n, m, (const uint8_t *)PmT.buf, (uint64_t)start,
               (uint64_t)count, (uint8_t *)out.buf, (size_t)stride,
               (size_t)offset);
    Py_END_ALLOW_THREADS
    res = Py_None;
    Py_INCREF(res);
done:
    PyBuffer_Release(&PmT);
    PyBuffer_Release(&out);
    return res;
}

static PyObject *py_rows_range(PyObject *self, PyObject *args)
{
    int n, m;
    long long start, count;
    Py_buffer PmT, Sm, rows, signs;
    PyObject *rows_obj, *signs_obj, *res = NULL;
    (void)self;
    rows.obj = NULL;
    signs.obj = NULL;
    if (!PyArg_ParseTuple(args, "iiy*y*LLOO", &n, &m, &PmT, &Sm, &start,
                          &count, &rows_obj, &signs_obj))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") ||
        bad_range(m, 1, (n < MAX_M) ? n : MAX_M, "m") ||
        bad_range(start, 0, (long long)FACT[n], "start") ||
        bad_range(count, 0, (long long)(FACT[n] - (uint64_t)start), "count") ||
        need(&PmT, FACT[m], (uint64_t)m, "PmT") ||
        need(&Sm, FACT[m], 1, "Sm"))
        goto done;
    if (get_optional(rows_obj, &rows) < 0 || get_optional(signs_obj, &signs) < 0)
        goto done;
    if ((rows.obj != NULL && need(&rows, (uint64_t)count, (uint64_t)n, "rows")) ||
        (signs.obj != NULL && need(&signs, (uint64_t)count, 1, "signs")))
        goto done;
    Py_BEGIN_ALLOW_THREADS
    rows_range(n, m, (const uint8_t *)PmT.buf, (const int8_t *)Sm.buf,
               (uint64_t)start, (uint64_t)count, (uint8_t *)rows.buf,
               (int8_t *)signs.buf);
    Py_END_ALLOW_THREADS
    res = Py_None;
    Py_INCREF(res);
done:
    PyBuffer_Release(&PmT);
    PyBuffer_Release(&Sm);
    if (rows.obj != NULL)
        PyBuffer_Release(&rows);
    if (signs.obj != NULL)
        PyBuffer_Release(&signs);
    return res;
}

static PyObject *py_signs_full(PyObject *self, PyObject *args)
{
    int n;
    Py_buffer out;
    (void)self;
    if (!PyArg_ParseTuple(args, "iw*", &n, &out))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") || need(&out, FACT[n], 1, "out")) {
        PyBuffer_Release(&out);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    build_signs(n, (int8_t *)out.buf);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&out);
    Py_RETURN_NONE;
}

/* The 64-bit kernel is exact only if every product and every run of B terms
   stays below 2^62; verify that from the matrix itself. */
static int det_bound_ok(const int64_t *A, int n, long long B)
{
    const uint64_t LIM = (uint64_t)1 << 62;
    uint64_t pmax = 1;
    int i, j;
    for (i = 0; i < n; i++) {
        uint64_t rmax = 1;
        for (j = 0; j < n; j++) {
            const int64_t v = A[(size_t)i * n + j];
            const uint64_t a = (v < 0) ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
            if (a > rmax)
                rmax = a;
        }
        if (rmax >= LIM || pmax > (LIM - 1) / rmax)
            goto too_large;
        pmax *= rmax;
    }
    if ((uint64_t)B > (LIM - 1) / pmax)
        goto too_large;
    return 1;
too_large:
    PyErr_SetString(PyExc_ValueError,
                    "entries too large for the 64-bit kernel with this run length");
    return 0;
}

static PyObject *py_det_i64(PyObject *self, PyObject *args)
{
    int n, m;
    long long q_lo, q_hi, B;
    Py_buffer A, PmT, Sm, out;
    PyObject *res = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiy*y*y*LLLw*", &n, &m, &A, &PmT, &Sm, &q_lo,
                          &q_hi, &B, &out))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") ||
        bad_range(m, 1, (n < MAX_M) ? n : MAX_M, "m") ||
        bad_range(q_lo, 0, (long long)(FACT[n] / FACT[m]), "q_lo") ||
        bad_range(q_hi, q_lo, (long long)(FACT[n] / FACT[m]), "q_hi") ||
        bad_range(B, 1, TILE, "B") ||
        need(&A, (uint64_t)n * (uint64_t)n, sizeof(int64_t), "A") ||
        need(&PmT, FACT[m], (uint64_t)m, "PmT") ||
        need(&Sm, FACT[m], 1, "Sm") ||
        need(&out, 2, sizeof(int64_t), "out") ||
        !det_bound_ok((const int64_t *)A.buf, n, B))
        goto done;
    Py_BEGIN_ALLOW_THREADS
    det_i64(n, m, (const int64_t *)A.buf, (const uint8_t *)PmT.buf,
            (const int8_t *)Sm.buf, (uint64_t)q_lo, (uint64_t)q_hi, (size_t)B,
            (int64_t *)out.buf);
    Py_END_ALLOW_THREADS
    res = Py_None;
    Py_INCREF(res);
done:
    PyBuffer_Release(&A);
    PyBuffer_Release(&PmT);
    PyBuffer_Release(&Sm);
    PyBuffer_Release(&out);
    return res;
}

static PyObject *py_det_f64(PyObject *self, PyObject *args)
{
    int n, m;
    int all_finite;
    long long q_lo, q_hi;
    Py_buffer A, PmT, Sm, partials;
    PyObject *res = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "iiy*y*y*LLpw*", &n, &m, &A, &PmT, &Sm, &q_lo,
                          &q_hi, &all_finite, &partials))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") ||
        bad_range(m, 1, (n < MAX_M) ? n : MAX_M, "m") ||
        bad_range(q_lo, 0, (long long)(FACT[n] / FACT[m]), "q_lo") ||
        bad_range(q_hi, q_lo, (long long)(FACT[n] / FACT[m]), "q_hi") ||
        need(&A, (uint64_t)n * (uint64_t)n, sizeof(double), "A") ||
        need(&PmT, FACT[m], (uint64_t)m, "PmT") ||
        need(&Sm, FACT[m], 1, "Sm") ||
        need(&partials, (uint64_t)(q_hi - q_lo), sizeof(double), "partials"))
        goto done;
    Py_BEGIN_ALLOW_THREADS
    det_f64(n, m, (const double *)A.buf, (const uint8_t *)PmT.buf,
            (const int8_t *)Sm.buf, (uint64_t)q_lo, (uint64_t)q_hi, all_finite,
            (double *)partials.buf);
    Py_END_ALLOW_THREADS
    res = Py_None;
    Py_INCREF(res);
done:
    PyBuffer_Release(&A);
    PyBuffer_Release(&PmT);
    PyBuffer_Release(&Sm);
    PyBuffer_Release(&partials);
    return res;
}

static PyObject *py_chain_rows(PyObject *self, PyObject *args)
{
    int n;
    Py_buffer out;
    (void)self;
    if (!PyArg_ParseTuple(args, "iw*", &n, &out))
        return NULL;
    if (bad_range(n, 1, MAX_N, "n") ||
        need(&out, FACT[n], (uint64_t)n, "out")) {
        PyBuffer_Release(&out);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    chain_rows(n, (uint8_t *)out.buf);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&out);
    Py_RETURN_NONE;
}

static PyMethodDef core_methods[] = {
    {"cols_full", py_cols_full, METH_VARARGS,
     "cols_full(m, out): column-major table of S_m into out."},
    {"cols_range", py_cols_range, METH_VARARGS,
     "cols_range(n, m, PmT, start, count, out, stride, offset): a window of "
     "lines in column layout."},
    {"rows_range", py_rows_range, METH_VARARGS,
     "rows_range(n, m, PmT, Sm, start, count, rows, signs): a window of lines."},
    {"signs_full", py_signs_full, METH_VARARGS,
     "signs_full(n, out): all n! signs in lexicographic order."},
    {"det_i64", py_det_i64, METH_VARARGS,
     "det_i64(n, m, A, PmT, Sm, q_lo, q_hi, B, out): exact 128-bit sum."},
    {"det_f64", py_det_f64, METH_VARARGS,
     "det_f64(n, m, A, PmT, Sm, q_lo, q_hi, all_finite, partials): float window sums."},
    {"chain_rows", py_chain_rows, METH_VARARGS,
     "chain_rows(n, out): reference chain generator (benchmarks only)."},
    {NULL, NULL, 0, NULL}};

static PyModuleDef_Slot core_slots[] = {
#ifdef Py_mod_gil
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

static struct PyModuleDef core_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Private C kernels of berkeDet; use the functions of the berkeDet package.",
    0,
    core_methods,
    core_slots,
    NULL,
    NULL,
    NULL};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModuleDef_Init(&core_module);
}
