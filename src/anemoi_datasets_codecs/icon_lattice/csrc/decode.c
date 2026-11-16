/* lattice_codec decoder: payload parsing, structure step kernels, field decode, chunk decode over a field DAG.
 *
 * Field decode (mode LATTICE): build the rANS tables, then for every step of the structure (in order):
 *   DELTA:  v[t_i] = v[t_{i-1}] + r_i (v[t_{-1}] = 0)
 *   GATHER: v0 = v[src_0], acc = sum_{j>=1} W_j (v[src_j] - v0)          (W: fixed point, 8 fractional bits)
 *           pred_sp = v0 + ((acc + 128) >> 8)
 *           pred = pred_sp + sum_c ((C_c * d_c[t] + 2^(s_c - 1)) >> s_c)  (cross terms: detail d_c of field c)
 *           v[t] = pred + r;  detail[t] = v[t] - pred_sp (kept when another field uses it)
 *   residual r: symbol < 255 -> zigzag; 255 -> next value of the escape stream (int16 or int32).
 *   context (optional, <= 4 buckets): spread = max - min of the first min(k, 4) sources; bucket = number of the
 *   step's thresholds below the spread; table = tmap[bucket].
 * then out[c] = clip(offset + step * (float)v[c]) (two IEEE roundings, -ffp-contract=off).
 *
 * Chunk decode: one call, one OpenMP team; every decodable field is a task with depend clauses on its sources
 * (cross-variable details, recipe inputs), created in a topological order (longest remaining chain first).
 * LOSSLESS fields are decoded by the caller beforehand (they are already in `out`). */
#include "simd.h"
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>

#include "lc.h"

#define LATTICE_HEADER 56
#define CHUNK_MAGIC "LATC"
#define CHUNK_VERSION 3
#define RECORD_BYTES 16
#define MAX_DEP 16
#define MAX_FEAT 4
#define MASK_SENTINEL INT32_MIN

/* ------------------------------------------------------------------------------------------- buffer pool */
/* Large buffers (values, details, scratch) are recycled across calls: fresh pages would cost more than the decode. */
#define POOL_MAX 1024
static pthread_mutex_t pool_lock = PTHREAD_MUTEX_INITIALIZER;
static void *pool_ptr[POOL_MAX];
static size_t pool_size[POOL_MAX];
static int pool_n;

static void *buf_get(size_t bytes, size_t *got) {
    bytes = (bytes + 4095) & ~(size_t)4095;
    pthread_mutex_lock(&pool_lock);
    int best = -1;
    for (int i = 0; i < pool_n; i++)
        if (pool_size[i] >= bytes && (best < 0 || pool_size[i] < pool_size[best])) best = i;
    if (best >= 0) {
        void *p = pool_ptr[best];
        *got = pool_size[best];
        pool_ptr[best] = pool_ptr[pool_n - 1];
        pool_size[best] = pool_size[pool_n - 1];
        pool_n--;
        pthread_mutex_unlock(&pool_lock);
        return p;
    }
    pthread_mutex_unlock(&pool_lock);
    *got = bytes;
    return aligned_alloc(64, bytes);
}

static void buf_put(void *p, size_t bytes) {
    if (!p) return;
    pthread_mutex_lock(&pool_lock);
    if (pool_n < POOL_MAX) {
        pool_ptr[pool_n] = p;
        pool_size[pool_n] = bytes;
        pool_n++;
        p = NULL;
    }
    pthread_mutex_unlock(&pool_lock);
    free(p);
}

EXPORT void lc_release_buffers(void) {
    pthread_mutex_lock(&pool_lock);
    for (int i = 0; i < pool_n; i++) free(pool_ptr[i]);
    pool_n = 0;
    pthread_mutex_unlock(&pool_lock);
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* --------------------------------------------------------------------------------------- payload parsing */
typedef struct {
    int vt, et, n_steps, n_tables, n_cross, flags, pbits; /* pbits: probability precision (10 or 12) */
    int rsh; /* residual-domain reconstruction: decoded residuals are multiplied by 2^rsh */
    float offset, step, clip_lo, clip_hi;
    uint32_t n_esc, rans_bytes, tables_bytes, steps_bytes;
    int32_t cross_src[MAX_CROSS], cross_shift[MAX_CROSS];
    uint8_t n_ctx[MAX_STEPS], tmap[MAX_STEPS][MAX_CTX];
    int32_t thr[MAX_STEPS][MAX_CTX - 1];
    int32_t w[MAX_STEPS][MAX_K];
    int32_t cc[MAX_STEPS][MAX_CROSS];
    const uint8_t *tables, *rans, *esc;
    /* per step and range: rANS stream bytes / escape count, and their offsets in the field's streams */
    uint32_t rbytes[MAX_STEPS][MAX_RANGES], resc[MAX_STEPS][MAX_RANGES];
    uint32_t roff[MAX_STEPS][MAX_RANGES], eoff[MAX_STEPS][MAX_RANGES];
    uint32_t serial_bytes; /* the stream of all single-range steps (first in the rANS section) */
    /* optional mask term (flags bit 2): cells where field mask_src < thr are predicted by
     * rint((c0 + sum_j coef_j A_j (B_j or 1) - offset) / step) and coded with table mask_table */
    int has_mask, mask_src, n_feat, mask_table;
    float mthr;
    double c0, coef[MAX_FEAT];
    int32_t fa[MAX_FEAT], fb[MAX_FEAT];
} lfield_t;

/* target ranges [lo, hi) of step s (see RANGE_T / RANGE_G); returns their number */
static int step_ranges(const lc_struct_t *S, int s, int64_t *lo, int64_t *hi) {
    const lc_step_t *st = &S->steps[s];
    const lc_group_t *G = S->group ? &S->group[s] : NULL;
    if (st->kind == STEP_DELTA) {
        lo[0] = 0;
        hi[0] = st->n;
        return 1;
    }
    int64_t unit = 1, size = RANGE_T, m = st->n;
    if (G && G->ng > 0 && st->k == 3) {
        unit = 3;
        size = RANGE_G;
        m = G->ng;
    }
    int r = 0;
    for (int64_t a = 0; a < m && r < MAX_RANGES; a += size, r++) {
        lo[r] = a * unit;
        hi[r] = (a + size < m ? a + size : m) * unit;
    }
    if (r == 0) {
        lo[0] = hi[0] = 0;
        r = 1;
    }
    return r;
}

static int parse_lattice(const uint8_t *p, size_t n, const lc_struct_t *S, lfield_t *f) {
    if (n < LATTICE_HEADER) return LC_ERR_FORMAT;
    f->vt = p[0];
    f->et = p[1];
    f->n_steps = p[2];
    f->n_tables = p[3];
    f->n_cross = p[4];
    f->flags = p[5];
    f->pbits = p[6] ? p[6] : PROB_BITS;
    if (f->pbits != 10 && f->pbits != 12) return LC_ERR_FORMAT;
    f->rsh = p[7];
    if (f->rsh > 8) return LC_ERR_FORMAT;
    if ((f->vt != 2 && f->vt != 4) || (f->et != 1 && f->et != 2 && f->et != 4) || f->n_tables < 1 || f->n_cross > MAX_CROSS)
        return LC_ERR_FORMAT;
    if (f->n_steps != S->n_steps) return LC_ERR_STRUCT;
    f->offset = rdf32(p + 8);
    f->step = rdf32(p + 12);
    f->clip_lo = rdf32(p + 16);
    f->clip_hi = rdf32(p + 20);
    f->n_esc = rd32(p + 24);
    f->rans_bytes = rd32(p + 28);
    f->tables_bytes = rd32(p + 32);
    f->steps_bytes = rd32(p + 36);
    for (int c = 0; c < MAX_CROSS; c++) {
        f->cross_src[c] = c < f->n_cross ? (int32_t)rd16(p + 40 + 2 * c) : -1;
        f->cross_shift[c] = c < f->n_cross ? p[48 + c] : 0;
        if (f->cross_shift[c] > 30) return LC_ERR_FORMAT;
    }
    size_t off = LATTICE_HEADER;
    f->has_mask = (f->flags & 4) != 0;
    if (f->has_mask) { /* u16 src, u8 n_feat, u8 table, f32 thr, f64 c0, n_feat x (u16 a, u16 b, u32 0, f64 coef) */
        if (off + 16 > n) return LC_ERR_FORMAT;
        f->mask_src = rd16(p + off);
        f->n_feat = p[off + 2];
        f->mask_table = p[off + 3];
        f->mthr = rdf32(p + off + 4);
        f->c0 = rdf64(p + off + 8);
        if (f->n_feat > MAX_FEAT || f->mask_table >= f->n_tables || off + 16 + 16 * (size_t)f->n_feat > n)
            return LC_ERR_FORMAT;
        for (int j = 0; j < f->n_feat; j++) {
            const uint8_t *q = p + off + 16 + 16 * j;
            f->fa[j] = rd16(q);
            f->fb[j] = rd16(q + 2) == 0xffff ? -1 : rd16(q + 2);
            f->coef[j] = rdf64(q + 8);
        }
        off += 16 + 16 * (size_t)f->n_feat;
    }
    if (off + f->steps_bytes > n) return LC_ERR_FORMAT;
    const uint8_t *q = p + off, *qe = q + f->steps_bytes;
    uint64_t rsum = 0, esum = 0;
    for (int s = 0; s < f->n_steps; s++) {
        const lc_step_t *st = &S->steps[s];
        if (q + 1 > qe) return LC_ERR_FORMAT;
        int nc = *q++;
        if (nc < 1 || nc > MAX_CTX || (st->kind == STEP_DELTA && nc != 1)) return LC_ERR_FORMAT;
        f->n_ctx[s] = (uint8_t)nc;
        int64_t rlo[MAX_RANGES], rhi[MAX_RANGES];
        int nr = step_ranges(S, s, rlo, rhi);
        int need = nc + 4 * (nc - 1) + (st->kind == STEP_GATHER ? 2 * (st->k - 1) + 4 * f->n_cross : 0) + 8 * nr;
        if (q + need > qe) return LC_ERR_FORMAT;
        for (int b = 0; b < nc; b++) {
            f->tmap[s][b] = *q++;
            if (f->tmap[s][b] >= f->n_tables) return LC_ERR_FORMAT;
        }
        for (int b = 0; b < nc - 1; b++, q += 4) f->thr[s][b] = rd32s(q);
        f->w[s][0] = 0;
        if (st->kind == STEP_GATHER) {
            for (int j = 1; j < st->k; j++, q += 2) f->w[s][j] = rd16s(q);
            for (int c = 0; c < f->n_cross; c++, q += 4) f->cc[s][c] = rd32s(q);
        } else {
            for (int c = 0; c < f->n_cross; c++) f->cc[s][c] = 0;
        }
        for (int r = 0; r < nr; r++, q += 8) { /* single-range steps: 0 bytes (they live in the serial stream) */
            f->rbytes[s][r] = rd32(q);
            f->resc[s][r] = rd32(q + 4);
            if ((nr == 1) != (f->rbytes[s][r] == 0)) return LC_ERR_FORMAT;
            f->roff[s][r] = rsum; /* relative to the end of the serial stream */
            f->eoff[s][r] = esum;
            rsum += f->rbytes[s][r];
            esum += f->resc[s][r];
            if (rsum > f->rans_bytes || esum > f->n_esc) return LC_ERR_FORMAT;
        }
    }
    if (q != qe || esum != f->n_esc) return LC_ERR_FORMAT;
    f->serial_bytes = f->rans_bytes - (uint32_t)rsum;
    for (int s = 0; s < f->n_steps; s++)
        for (int r = 0; r < MAX_RANGES; r++) f->roff[s][r] += f->serial_bytes;
    off += f->steps_bytes;
    if (off + f->tables_bytes > n) return LC_ERR_FORMAT;
    f->tables = p + off;
    off = (off + f->tables_bytes + 3) & ~(size_t)3;
    if (off + f->rans_bytes + (size_t)f->n_esc * (size_t)f->et > n) return LC_ERR_FORMAT;
    f->rans = p + off;
    f->esc = p + off + f->rans_bytes;
    return LC_OK;
}

INLINE int varint(const uint8_t **q, const uint8_t *end, uint32_t *v) {
    uint32_t x = 0;
    for (int sh = 0; sh < 35; sh += 7) {
        if (*q >= end) return -1;
        uint8_t b = *(*q)++;
        x |= (uint32_t)(b & 127) << sh;
        if (!(b & 128)) {
            *v = x;
            return 0;
        }
    }
    return -1;
}

/* Tables: per table u8 m, varint freq[0..m-1], varint freq[255]; frequencies sum to 4096. */
static int build_tables(const lfield_t *f, uint32_t *tabs) {
    const uint8_t *q = f->tables, *end = f->tables + f->tables_bytes;
    uint32_t freqs[ALPHABET];
    for (int t = 0; t < f->n_tables; t++) {
        if (q >= end) return LC_ERR_FORMAT;
        int m = *q++;
        memset(freqs, 0, sizeof freqs);
        uint32_t sum = 0;
        for (int s = 0; s < m; s++) {
            if (varint(&q, end, &freqs[s])) return LC_ERR_FORMAT;
            sum += freqs[s];
        }
        if (varint(&q, end, &freqs[ESCAPE])) return LC_ERR_FORMAT;
        sum += freqs[ESCAPE];
        if (sum != (1u << f->pbits)) return LC_ERR_FORMAT;
        lc_build_table(freqs, tabs + ((size_t)t << f->pbits), f->pbits);
    }
    return q == end ? LC_OK : LC_ERR_FORMAT;
}

/* ---------------------------------------------------------------------------------------------- kernels */
typedef struct {
    const uint8_t *p;
    uint32_t left;
    int et;
    int32_t mul;  /* 2^rsh: decoded residuals (symbols and escapes) are scaled by it */
    __m128i shv;  /* rsh as a shift count */
} esc_t;

/* et 2 / 4: int16 / int32 values (left counts values); et 1: LEB128 varints of zigzag(r) - 255 (left counts bytes) */
INLINE int esc_next(esc_t *e, int32_t *r) {
    if (!e->left) return -1;
    if (e->et == 1) {
        uint32_t v = 0;
        for (int sh = 0;; sh += 7) {
            if (!e->left || sh > 28) return -1;
            uint8_t b = *e->p++;
            e->left--;
            v |= (uint32_t)(b & 127) << sh;
            if (!(b & 128)) break;
        }
        *r = wmul32(zigzag_dec(v + 255), e->mul);
        return 0;
    }
    *r = wmul32(e->et == 2 ? (int32_t)rd16s(e->p) : rd32s(e->p), e->mul);
    e->p += e->et;
    e->left--;
    return 0;
}

typedef struct {
    int nc;                  /* cross terms */
    const int16_t *d[MAX_CROSS]; /* details are int16 (saturated) */
    __m128i sh[MAX_CROSS];   /* shift counts */
    int32_t shs[MAX_CROSS];
    int32_t c[MAX_CROSS];
} cross_t;

INLINE __m256i zz8(const uint8_t *s) {
    __m256i r = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)s));
    return _mm256_xor_si256(_mm256_srli_epi32(r, 1),
                            _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_and_si256(r, _mm256_set1_epi32(1))));
}

INLINE int has_escape8(const uint8_t *s) {
    return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)s), _mm_set1_epi8((char)-1))) & 0xff;
}

#define LD8(V, I) _mm256_setr_epi32(V[(I)[0]], V[(I)[1]], V[(I)[2]], V[(I)[3]], V[(I)[4]], V[(I)[5]], V[(I)[6]], V[(I)[7]])

INLINE int16_t sat16(int32_t x) { return (int16_t)(x < -32768 ? -32768 : x > 32767 ? 32767 : x); }

/* 8 int32 -> 8 saturated int16 */
INLINE void store_i16x8(int16_t *p, __m256i v) {
    _mm_storeu_si128((__m128i *)p, _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

INLINE int32_t cross_scalar(const cross_t *x, int32_t t) {
    int32_t s = 0;
    for (int c = 0; c < x->nc; c++) {
        int64_t v = (int64_t)x->c[c] * x->d[c][t];
        s += (int32_t)((v + (x->shs[c] ? (1ll << (x->shs[c] - 1)) : 0)) >> x->shs[c]);
    }
    return s;
}

/* Value type VT (int16_t / int32_t) and K (compile-time source count, 0 = runtime k). Modes:
 *   fused (ctx == NULL): residuals from sym, write v[t] (and detail);
 *   predict (ctx != NULL): store pred[i], dp[i] = pred - pred_sp, tid[i] = tmap[bucket]. */
#define DEF_GATHER(VT, KK)                                                                                            \
    static int gather_##VT##_##KK(VT *v, const lc_step_t *st, const int32_t *w, const cross_t *x, const uint8_t *sym, \
                                  esc_t *esc, int16_t *dout, int n_ctx, const int32_t *thr, const uint8_t *tmap,       \
                                  int32_t *pred, int32_t *dp, uint8_t *tid, int64_t lo, int64_t hi) {                  \
        const int K = KK ? KK : st->k;                                                                                 \
        const int64_t n = st->n;                                                                                       \
        const int32_t *src = st->srcp; const int64_t p0 = st->p0;                                                                      \
        const int predict = tid != NULL;                                                                               \
        int64_t i = lo;                                                                                                \
        __m256i vw[MAX_K], vthr[MAX_CTX - 1], vcc[MAX_CROSS];                                                          \
        for (int j = 1; j < K; j++) vw[j] = _mm256_set1_epi32(w[j]);                                                   \
        for (int b = 0; b < n_ctx - 1; b++) vthr[b] = _mm256_set1_epi32(thr[b]);                                       \
        for (int c = 0; c < x->nc; c++) vcc[c] = _mm256_set1_epi32(x->c[c]);                                           \
        const __m256i r128 = _mm256_set1_epi32(128);                                                                   \
        for (; i + 8 <= hi; i += 8) {                                                                                   \
            __m256i v0 = LD8(v, src + i), acc = r128, hi = v0, lo = v0;                                                \
            for (int j = 1; j < K; j++) {                                                                              \
                __m256i xj = LD8(v, src + (size_t)j * n + i);                                                          \
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vw[j], _mm256_sub_epi32(xj, v0)));                      \
                if (predict && j < 4) {                                                                                \
                    hi = _mm256_max_epi32(hi, xj);                                                                     \
                    lo = _mm256_min_epi32(lo, xj);                                                                     \
                }                                                                                                      \
            }                                                                                                          \
            __m256i psp = _mm256_add_epi32(v0, _mm256_srai_epi32(acc, 8)), pr = psp;                                  \
            for (int c = 0; c < x->nc; c++) {                                                                          \
                __m256i d = _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(x->d[c] + p0 + i)));                                                                       \
                __m256i m = _mm256_mullo_epi32(vcc[c], d);                                                             \
                if (x->shs[c]) m = _mm256_add_epi32(m, _mm256_set1_epi32(1 << (x->shs[c] - 1)));                      \
                pr = _mm256_add_epi32(pr, _mm256_sra_epi32(m, x->sh[c]));                                              \
            }                                                                                                          \
            if (predict) {                                                                                             \
                __m256i sp = _mm256_sub_epi32(hi, lo), b = _mm256_setzero_si256();                                    \
                for (int q = 0; q < n_ctx - 1; q++) b = _mm256_sub_epi32(b, _mm256_cmpgt_epi32(sp, vthr[q]));         \
                int32_t bb[8];                                                                                         \
                _mm256_storeu_si256((__m256i *)bb, b);                                                                 \
                for (int q = 0; q < 8; q++) tid[i + q] = tmap[bb[q]];                                                  \
                _mm256_storeu_si256((__m256i *)(pred + i), pr);                                                        \
                if (dp) _mm256_storeu_si256((__m256i *)(dp + i), _mm256_sub_epi32(pr, psp));                           \
                continue;                                                                                              \
            }                                                                                                          \
            int32_t o[8], ps[8];                                                                                       \
            __m256i val = _mm256_add_epi32(pr, _mm256_sll_epi32(zz8(sym + i), esc->shv));                                                          \
            _mm256_storeu_si256((__m256i *)o, val);                                                                    \
            if (has_escape8(sym + i)) {                                                                                \
                _mm256_storeu_si256((__m256i *)ps, pr);                                                                \
                for (int q = 0; q < 8; q++)                                                                            \
                    if (sym[i + q] == ESCAPE) {                                                                        \
                        int32_t r;                                                                                     \
                        if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                   \
                        o[q] = wadd32(ps[q], r);                                                                            \
                    }                                                                                                  \
            }                                                                                                          \
            for (int q = 0; q < 8; q++) v[p0 + i + q] = (VT)o[q];                                                        \
            if (dout) {                                                                                                \
                _mm256_storeu_si256((__m256i *)ps, psp);                                                               \
                store_i16x8(dout + p0 + i, _mm256_sub_epi32(_mm256_loadu_si256((const __m256i *)o), _mm256_loadu_si256((const __m256i *)ps)));                                             \
            }                                                                                                          \
        }                                                                                                              \
        for (; i < hi; i++) {                                                                                           \
            int32_t v0 = v[src[i]], hi = v0, lo = v0;                                                                  \
            int64_t acc = 128;                                                                                         \
            for (int j = 1; j < K; j++) {                                                                              \
                int32_t xj = v[src[(size_t)j * n + i]];                                                                \
                acc += (int64_t)w[j] * (xj - v0);                                                                      \
                if (j < 4) {                                                                                           \
                    hi = xj > hi ? xj : hi;                                                                            \
                    lo = xj < lo ? xj : lo;                                                                            \
                }                                                                                                      \
            }                                                                                                          \
            int32_t psp = v0 + (int32_t)(acc >> 8), pr = psp + cross_scalar(x, (int32_t)(p0 + i));                                  \
            if (predict) {                                                                                             \
                int b = 0;                                                                                             \
                for (int q = 0; q < n_ctx - 1; q++) b += (hi - lo) > thr[q];                                           \
                tid[i] = tmap[b];                                                                                      \
                pred[i] = pr;                                                                                          \
                if (dp) dp[i] = pr - psp;                                                                              \
                continue;                                                                                              \
            }                                                                                                          \
            int32_t r;                                                                                                 \
            if (sym[i] == ESCAPE) {                                                                                    \
                if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                           \
            } else                                                                                                     \
                r = zigzag_dec(sym[i]) * esc->mul;                                                                                \
            v[p0 + i] = (VT)wadd32(pr, r);                                                                                \
            if (dout) dout[p0 + i] = sat16(wsub32(wadd32(pr, r), psp));                                                                \
        }                                                                                                              \
        return LC_OK;                                                                                                  \
    }

/* The common case: no cross terms, no detail output, no contexts (fused, K compile-time). */
#define DEF_PLAIN(VT, KK)                                                                                             \
    static int plain_##VT##_##KK(VT *restrict v, const lc_step_t *st, const int32_t *w, const uint8_t *restrict sym,   \
                                 esc_t *esc, int64_t lo, int64_t hi) {                                                                         \
        const int64_t n = st->n;                                                                                       \
        const int32_t *restrict src = st->srcp; const int64_t p0 = st->p0;                                                    \
        __m256i vw[KK];                                                                                                \
        for (int j = 1; j < KK; j++) vw[j] = _mm256_set1_epi32(w[j]);                                                  \
        const __m256i r128 = _mm256_set1_epi32(128);                                                                   \
        int64_t i = lo;                                                                                                 \
        for (; i + 8 <= hi; i += 8) {                                                                                   \
            __m256i v0 = LD8(v, src + i), acc = r128;                                                                  \
            for (int j = 1; j < KK; j++)                                                                               \
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(vw[j], _mm256_sub_epi32(LD8(v, src + (size_t)j * n + i), v0))); \
            __m256i pr = _mm256_add_epi32(v0, _mm256_srai_epi32(acc, 8));                                              \
            int32_t o[8] __attribute__((aligned(32)));                                                                 \
            _mm256_store_si256((__m256i *)o, _mm256_add_epi32(pr, _mm256_sll_epi32(zz8(sym + i), esc->shv)));                                      \
            if (__builtin_expect(has_escape8(sym + i), 0)) {                                                           \
                int32_t ps[8] __attribute__((aligned(32)));                                                            \
                _mm256_store_si256((__m256i *)ps, pr);                                                                 \
                for (int q = 0; q < 8; q++)                                                                            \
                    if (sym[i + q] == ESCAPE) {                                                                        \
                        int32_t r;                                                                                     \
                        if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                   \
                        o[q] = wadd32(ps[q], r);                                                                            \
                    }                                                                                                  \
            }                                                                                                          \
            v[p0 + i] = (VT)o[0];                                                                                        \
            v[p0 + i + 1] = (VT)o[1];                                                                                    \
            v[p0 + i + 2] = (VT)o[2];                                                                                    \
            v[p0 + i + 3] = (VT)o[3];                                                                                    \
            v[p0 + i + 4] = (VT)o[4];                                                                                    \
            v[p0 + i + 5] = (VT)o[5];                                                                                    \
            v[p0 + i + 6] = (VT)o[6];                                                                                    \
            v[p0 + i + 7] = (VT)o[7];                                                                                    \
        }                                                                                                              \
        for (; i < hi; i++) {                                                                                           \
            int32_t v0 = v[src[i]];                                                                                    \
            int64_t acc = 128;                                                                                         \
            for (int j = 1; j < KK; j++) acc += (int64_t)w[j] * (v[src[(size_t)j * n + i]] - v0);                      \
            int32_t r;                                                                                                 \
            if (sym[i] == ESCAPE) {                                                                                    \
                if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                           \
            } else                                                                                                     \
                r = zigzag_dec(sym[i]) * esc->mul;                                                                                \
            v[p0 + i] = (VT)wadd32(v0 + (int32_t)(acc >> 8), r);                                                           \
        }                                                                                                              \
        return LC_OK;                                                                                                  \
    }

#define DEF_APPLY(VT)                                                                                                 \
    static int apply_##VT(VT *v, const lc_step_t *st, const int32_t *pred, const int32_t *dp, const uint8_t *sym,    \
                          esc_t *esc, int16_t *dout, int64_t lo, int64_t hi) {                                        \
                const int64_t p0 = st->p0;                                                                                     \
        int64_t i = lo;                                                                                               \
        for (; i + 8 <= hi; i += 8) {                                                                                  \
            int32_t o[8], r[8];                                                                                       \
            __m256i rr = _mm256_sll_epi32(zz8(sym + i), esc->shv);                                                                                \
            _mm256_storeu_si256((__m256i *)r, rr);                                                                    \
            if (has_escape8(sym + i))                                                                                 \
                for (int q = 0; q < 8; q++)                                                                           \
                    if (sym[i + q] == ESCAPE && esc_next(esc, &r[q])) return LC_ERR_ESCAPE;                           \
            rr = _mm256_loadu_si256((const __m256i *)r);                                                              \
            _mm256_storeu_si256((__m256i *)o, _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(pred + i)), rr)); \
            for (int q = 0; q < 8; q++) v[p0 + i + q] = (VT)o[q];                                                       \
            if (dout) {                                                                                               \
                _mm256_storeu_si256((__m256i *)o, _mm256_add_epi32(rr, _mm256_loadu_si256((const __m256i *)(dp + i)))); \
                store_i16x8(dout + p0 + i, _mm256_loadu_si256((const __m256i *)o));                                                    \
            }                                                                                                         \
        }                                                                                                             \
        for (; i < hi; i++) {                                                                                          \
            int32_t r;                                                                                                \
            if (sym[i] == ESCAPE) {                                                                                   \
                if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                          \
            } else                                                                                                    \
                r = zigzag_dec(sym[i]) * esc->mul;                                                                               \
            v[p0 + i] = (VT)wadd32(pred[i], r);                                                                           \
            if (dout) dout[p0 + i] = sat16(wadd32(r, dp[i]));                                                                      \
        }                                                                                                             \
        return LC_OK;                                                                                                 \
    }

#define DEF_DELTA(VT)                                                                                                 \
    static int delta_##VT(VT *v, const lc_step_t *st, const uint8_t *sym, esc_t *esc, int16_t *dout) {               \
        int32_t prev = 0;                                                                                             \
        for (int64_t i = 0; i < st->n; i++) {                                                                         \
            int32_t r;                                                                                                \
            if (sym[i] == ESCAPE) {                                                                                   \
                if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                          \
            } else                                                                                                    \
                r = zigzag_dec(sym[i]) * esc->mul;                                                                               \
            prev = wadd32(prev, r);                                                                                          \
            v[st->p0 + i] = (VT)prev;                                                                                   \
            if (dout) dout[st->p0 + i] = sat16(r);                                                                             \
        }                                                                                                             \
        return LC_OK;                                                                                                 \
    }

/* Dequantise into the native-order output: out = clip(offset + step * (float)v). Non-temporal stores. */
#define DEF_DEQUANT(VT, CVT)                                                                                          \
    static void dequant_##VT(const VT *v, int64_t n, float off, float step, int flags, float lo, float hi,          \
                             float *out) {                                                                            \
        const __m256 vs = _mm256_set1_ps(step), vo = _mm256_set1_ps(off), vlo = _mm256_set1_ps(lo),                  \
                     vhi = _mm256_set1_ps(hi);                                                                        \
        int64_t i = 0;                                                                                                \
        for (; i < n && ((uintptr_t)(out + i) & 31); i++) {                                                           \
            float y = off + step * (float)v[i];                                                                       \
            if ((flags & 1) && y < lo) y = lo;                                                                        \
            if ((flags & 2) && y > hi) y = hi;                                                                        \
            out[i] = y;                                                                                               \
        }                                                                                                             \
        for (; i + 8 <= n; i += 8) {                                                                                  \
            __m256 y = _mm256_add_ps(vo, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(CVT(v + i))));                          \
            if (flags & 1) y = _mm256_max_ps(y, vlo);                                                                 \
            if (flags & 2) y = _mm256_min_ps(y, vhi);                                                                 \
            _mm256_stream_ps(out + i, y);                                                                             \
        }                                                                                                             \
        for (; i < n; i++) {                                                                                          \
            float y = off + step * (float)v[i];                                                                       \
            if ((flags & 1) && y < lo) y = lo;                                                                        \
            if ((flags & 2) && y > hi) y = hi;                                                                        \
            out[i] = y;                                                                                               \
        }                                                                                                             \
        _mm_sfence();                                                                                                 \
    }

#define DEF_DEQUANT_POS(VT)                                                                                           \
    static void dequant_pos_##VT(const VT *v, const int32_t *pos, int64_t n, float off, float step, int flags,      \
                                 float lo, float hi, float *out) {                                                    \
        const __m256 vs = _mm256_set1_ps(step), vo = _mm256_set1_ps(off), vlo = _mm256_set1_ps(lo),                  \
                     vhi = _mm256_set1_ps(hi);                                                                        \
        int64_t i = 0;                                                                                                \
        for (; i < n && ((uintptr_t)(out + i) & 31); i++) {                                                           \
            float y = off + step * (float)v[pos[i]];                                                                  \
            if ((flags & 1) && y < lo) y = lo;                                                                        \
            if ((flags & 2) && y > hi) y = hi;                                                                        \
            out[i] = y;                                                                                               \
        }                                                                                                             \
        for (; i + 8 <= n; i += 8) {                                                                                  \
            __m256 y = _mm256_add_ps(vo, _mm256_mul_ps(vs, _mm256_cvtepi32_ps(LD8(v, pos + i))));                     \
            if (flags & 1) y = _mm256_max_ps(y, vlo);                                                                 \
            if (flags & 2) y = _mm256_min_ps(y, vhi);                                                                 \
            _mm256_stream_ps(out + i, y);                                                                             \
        }                                                                                                             \
        for (; i < n; i++) {                                                                                          \
            float y = off + step * (float)v[pos[i]];                                                                  \
            if ((flags & 1) && y < lo) y = lo;                                                                        \
            if ((flags & 2) && y > hi) y = hi;                                                                        \
            out[i] = y;                                                                                               \
        }                                                                                                             \
        _mm_sfence();                                                                                                 \
    }
DEF_DEQUANT_POS(int16_t)
DEF_DEQUANT_POS(int32_t)

#define CVT16(p) _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(p)))
#define CVT32(p) _mm256_loadu_si256((const __m256i *)(p))

DEF_GATHER(int16_t, 0)
DEF_GATHER(int16_t, 2)
DEF_GATHER(int16_t, 3)
DEF_GATHER(int16_t, 4)
DEF_GATHER(int32_t, 0)
DEF_GATHER(int32_t, 2)
DEF_GATHER(int32_t, 3)
DEF_GATHER(int32_t, 4)
DEF_PLAIN(int16_t, 2)
DEF_PLAIN(int16_t, 3)
DEF_PLAIN(int16_t, 4)
DEF_PLAIN(int32_t, 2)
DEF_PLAIN(int32_t, 3)
DEF_PLAIN(int32_t, 4)
DEF_APPLY(int16_t)
DEF_APPLY(int32_t)
DEF_DELTA(int16_t)
DEF_DELTA(int32_t)
DEF_DEQUANT(int16_t, CVT16)
DEF_DEQUANT(int32_t, CVT32)

typedef int (*gather_f)(void *, const lc_step_t *, const int32_t *, const cross_t *, const uint8_t *, esc_t *,
                        int16_t *, int, const int32_t *, const uint8_t *, int32_t *, int32_t *, uint8_t *, int64_t,
                        int64_t);

static gather_f pick_gather(int vt, int k) {
    if (vt == 2)
        return (gather_f)(k == 2 ? gather_int16_t_2 : k == 3 ? gather_int16_t_3 : k == 4 ? gather_int16_t_4
                                                                                        : gather_int16_t_0);
    return (gather_f)(k == 2 ? gather_int32_t_2 : k == 3 ? gather_int32_t_3 : k == 4 ? gather_int32_t_4
                                                                                    : gather_int32_t_0);
}

/* prof: [0] tables, [1] entropy, [2] steps (kernels, entropy excluded), [3] dequantisation, [4] mask predictor,
 * [5 + s] step s including its entropy. Single-threaded diagnostics (lc_profile(1, NULL) then decode at 1 thread). */
#define PROF_N (5 + MAX_STEPS)
static double prof[PROF_N];
static int prof_on;
static double prof_rans_inner; /* entropy time measured inside fused kernels */

#include "group.h"

typedef int (*plain_f)(void *, const lc_step_t *, const int32_t *, const uint8_t *, esc_t *, int64_t, int64_t);

/* Mask term: pm[i] = rint((c0 + sum_j coef_j A_j[i] (B_j[i] or 1) - off) / step) where msrc[i] < thr, else
 * MASK_SENTINEL (float64, never fused: the encoder calls this same function). */
EXPORT void lc_mask_predict(int64_t n, const float *msrc, float thr, double c0, int nf, const float *const *A,
                            const float *const *B, const double *coef, float off, float step, int vt, int32_t *pm) {
    const double lo = vt == 2 ? -32768.0 : -1073741824.0, hi = vt == 2 ? 32767.0 : 1073741824.0;
    for (int64_t i = 0; i < n; i++) {
        if (!(msrc[i] < thr)) {
            pm[i] = MASK_SENTINEL;
            continue;
        }
        double y = c0;
        for (int j = 0; j < nf; j++) {
            double a = (double)A[j][i];
            if (B[j]) a = a * (double)B[j][i];
            y = y + coef[j] * a;
        }
        double q = nearbyint((y - (double)off) / (double)step);
        if (!(q >= lo)) q = lo;
        if (q > hi) q = hi;
        pm[i] = (int32_t)q;
    }
}

/* masked targets take the mask prediction and the mask table (after a predict pass over [lo, hi)) */
static void mask_fix(const int32_t *t, const int32_t *pm, int mtab, int32_t *pred, int32_t *dp, uint8_t *tid,
                     int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; i++) {
        int32_t m = pm[t[i]];
        if (m != MASK_SENTINEL) {
            if (dp) dp[i] += m - pred[i];
            pred[i] = m;
            tid[i] = (uint8_t)mtab;
        }
    }
}

static plain_f pick_plain(int vt, int k) {
    if (vt == 2) return (plain_f)(k == 2 ? plain_int16_t_2 : k == 3 ? plain_int16_t_3 : k == 4 ? plain_int16_t_4 : NULL);
    return (plain_f)(k == 2 ? plain_int32_t_2 : k == 3 ? plain_int32_t_3 : k == 4 ? plain_int32_t_4 : NULL);
}

/* ---------------------------------------------------------------------------------------- field decode */
/* Stage profile (single-threaded diagnostics only): tables, entropy, step kernels, dequantisation (seconds). */

EXPORT void lc_profile(int on, double *out) {
    if (out) memcpy(out, prof, sizeof prof);
    memset(prof, 0, sizeof prof);
    prof_on = on;
}
typedef struct {
    uint8_t *mem;
    size_t bytes;
    uint8_t *sym, *tid;
    int32_t *pred, *dp, *pm;
    uint32_t *tabs;
    void *vals;
} scratch_t;

static int scratch_init(scratch_t *sc, const lc_struct_t *S) {
    size_t m = (size_t)S->max_n + 64, nc = (size_t)S->n_cells + 64;
    size_t need = 2 * m + 8 * m + 4 * (size_t)PROB_SCALE * MAX_TABLES + 8 * nc + 7 * 64;
    sc->mem = buf_get(need, &sc->bytes);
    if (!sc->mem) return LC_ERR_MEMORY;
    uint8_t *p = sc->mem;
#define TAKE(dst, type, count)                                \
    do {                                                      \
        dst = (type *)p;                                      \
        p += (((count) * sizeof(type)) + 63) & ~(size_t)63;   \
    } while (0)
    TAKE(sc->tabs, uint32_t, (size_t)PROB_SCALE * MAX_TABLES);
    TAKE(sc->vals, int32_t, nc);
    TAKE(sc->pm, int32_t, nc);
    TAKE(sc->pred, int32_t, m);
    TAKE(sc->dp, int32_t, m);
    TAKE(sc->sym, uint8_t, m);
    TAKE(sc->tid, uint8_t, m);
#undef TAKE
    return LC_OK;
}

static void scratch_free(scratch_t *sc) {
    buf_put(sc->mem, sc->bytes);
    sc->mem = NULL;
}

/* Decode one LATTICE field. dsrc[c]: detail arrays of the cross sources; dout: this field's detail (or NULL);
 * out: float output (or NULL); ival: int32 copy of the values (or NULL, for the encoder's checks). */
/* One range [lo, hi) of step s: its own rANS stream and escape slice; ranges of a step are independent. */
static int decode_range(const lfield_t *f, const lc_struct_t *S, scratch_t *sc, int s, int r, int64_t lo, int64_t hi,
                        const cross_t *xs, const int32_t *pm, int16_t *dout, float *fo, const deq_t *dq, void *v,
                        double *tr, rstate_t *serial) {
    const lc_step_t *st = &S->steps[s];
    const lc_group_t *G = S->group ? &S->group[s] : NULL;
    rstate_t own, *rsp = serial;
    int rc = LC_OK;
    if (f->rbytes[s][r]) { /* a range of a multi-range step: its own stream */
        rsp = &own;
        if ((rc = rans_init(&own, f->rans + f->roff[s][r], f->rbytes[s][r]))) return rc;
    }
#define rs (*rsp)
    esc_t esc = {f->esc + (size_t)f->eoff[s][r] * (size_t)f->et, f->resc[s][r], f->et, 1 << f->rsh,
                 _mm_cvtsi32_si128(f->rsh)};
    double ta = tr && prof_on ? now() : 0;
    if (st->kind == STEP_DELTA) {
        rc = rans_segment(&rs, (size_t)st->n, sc->tabs + ((size_t)f->tmap[s][0] << f->pbits), sc->sym, f->pbits);
        if (tr && prof_on) *tr += now() - ta;
        if (!rc) rc = f->vt == 2 ? delta_int16_t(v, st, sc->sym, &esc, dout) : delta_int32_t(v, st, sc->sym, &esc, dout);
        return rc ? rc : (esc.left ? LC_ERR_ESCAPE : LC_OK);
    }
    /* parent triples: neighbours loaded once per group (the mask term needs native cells: generic path then) */
    const int grp = G && G->ng > 0 && st->k == 3 && (!pm || G->c0 >= 0);
    if (f->n_ctx[s] == 1 && !pm) {
        rc = rans_segment(&rs, (size_t)(hi - lo), sc->tabs + ((size_t)f->tmap[s][0] << f->pbits), sc->sym + lo, f->pbits);
        if (tr && prof_on) *tr += now() - ta;
        if (rc) return rc;
        if (grp) {
            rc = f->vt == 2 ? group_int16_t(v, st, G, f->w[s], xs, sc->sym, &esc, dout, fo, dq, 1, NULL, NULL, NULL,
                                            NULL, NULL, lo / 3, hi / 3)
                            : group_int32_t(v, st, G, f->w[s], xs, sc->sym, &esc, dout, fo, dq, 1, NULL, NULL, NULL,
                                            NULL, NULL, lo / 3, hi / 3);
        } else {
            plain_f pf = xs->nc || dout ? NULL : pick_plain(f->vt, st->k);
            if (pf) rc = pf(v, st, f->w[s], sc->sym, &esc, lo, hi);
            else
                rc = pick_gather(f->vt, st->k)(v, st, f->w[s], xs, sc->sym, &esc, dout, 1, NULL, NULL, NULL, NULL,
                                               NULL, lo, hi);
        }
        return rc ? rc : (esc.left ? LC_ERR_ESCAPE : LC_OK);
    }
    /* contexts: predict -> rANS with per-symbol tables -> apply, in blocks that stay in L1 */
    int32_t *dp = dout ? sc->dp : NULL;
    if (grp) {
        rc = f->vt == 2 ? group_ctx_int16_t(v, st, G, f->w[s], xs, f->n_ctx[s], f->thr[s], f->tmap[s], &rs, sc->tabs,
                                            &esc, dout, fo, dq, pm, f->mask_table, lo / 3, hi / 3, f->pbits)
                        : group_ctx_int32_t(v, st, G, f->w[s], xs, f->n_ctx[s], f->thr[s], f->tmap[s], &rs, sc->tabs,
                                            &esc, dout, fo, dq, pm, f->mask_table, lo / 3, hi / 3, f->pbits);
    } else {
        gather_f g = pick_gather(f->vt, st->k);
        const int64_t B = 512;
        for (int64_t i0 = lo; i0 < hi && !rc; i0 += B) {
            int64_t i1 = i0 + B < hi ? i0 + B : hi;
            rc = g(v, st, f->w[s], xs, NULL, &esc, NULL, f->n_ctx[s], f->thr[s], f->tmap[s], sc->pred, dp, sc->tid,
                   i0, i1);
            if (pm) mask_fix(st->t, pm, f->mask_table, sc->pred, dp, sc->tid, i0, i1);
            double t1 = tr && prof_on ? now() : 0;
            if (!rc) rc = rans_segment_tid(&rs, (size_t)(i1 - i0), sc->tabs, sc->tid + i0, sc->sym + i0, f->pbits);
            if (tr && prof_on) *tr += now() - t1;
            if (!rc)
                rc = f->vt == 2 ? apply_int16_t(v, st, sc->pred, dp, sc->sym, &esc, dout, i0, i1)
                                : apply_int32_t(v, st, sc->pred, dp, sc->sym, &esc, dout, i0, i1);
        }
    }
    return rc ? rc : (esc.left ? LC_ERR_ESCAPE : LC_OK);
#undef rs
}

/* mask inputs: [mask source, A_0..A_3, B_0..B_3] float arrays (decoded fields) */
static int decode_lattice(const lfield_t *f, const lc_struct_t *S, scratch_t *sc, const int16_t *const *dsrc,
                          const float *const *mf, int16_t *dout, float *out, int32_t *ival, int par) {
    double tp0 = prof_on ? now() : 0, tr = 0, tstep = 0;
    prof_rans_inner = 0;
    int rc = build_tables(f, sc->tabs);
    if (rc) return rc;
    if (prof_on) prof[0] += now() - tp0;
    rstate_t serial;
    if ((rc = rans_init(&serial, f->rans, f->serial_bytes))) return rc;
    cross_t x;
    x.nc = f->n_cross;
    for (int c = 0; c < f->n_cross; c++) {
        if (!dsrc[c]) return LC_ERR_MISSING;
        x.d[c] = dsrc[c];
        x.shs[c] = f->cross_shift[c];
        x.sh[c] = _mm_cvtsi32_si128(f->cross_shift[c]);
    }
    void *v = sc->vals;
    const int32_t *pm = NULL;
    if (f->has_mask) {
        if (!mf || !mf[0]) return LC_ERR_MISSING;
        for (int j = 0; j < f->n_feat; j++)
            if (!mf[1 + j] || (f->fb[j] >= 0 && !mf[1 + MAX_FEAT + j])) return LC_ERR_MISSING;
        lc_mask_predict(S->n_cells, mf[0], f->mthr, f->c0, f->n_feat, mf + 1, mf + 1 + MAX_FEAT, f->coef, f->offset,
                        f->step, f->vt, sc->pm);
        pm = sc->pm;
        if (prof_on) prof[4] += now() - tp0 - prof[0];
    }
    /* the last step's regular groups write the float output directly (not when the values are wanted) */
    const lc_group_t *GL = S->group ? &S->group[S->n_steps - 1] : NULL;
    const int fuse = out && !ival && GL && GL->ng > 0 && GL->c0 >= 0;
    int64_t fused_c0 = S->n_cells;
    const deq_t dq = {f->offset, f->step, f->clip_lo, f->clip_hi, f->flags};
    for (int s = 0; s < S->n_steps && !rc; s++) {
        if (prof_on) {
            double t = now();
            if (s > 0) prof[5 + s - 1] += t - tstep;
            tstep = t;
        }
        cross_t xs = x;
        int any = 0;
        for (int c = 0; c < x.nc; c++) any |= (xs.c[c] = f->cc[s][c]) != 0;
        if (!any) xs.nc = 0; /* coefficients of 0 cost nothing */
        const int last = s == S->n_steps - 1;
        float *fo = fuse && last ? out : NULL;
        if (fo) fused_c0 = GL->c0;
        int64_t lo[MAX_RANGES], hi[MAX_RANGES];
        const int nr = step_ranges(S, s, lo, hi);
        int err = 0;
        if (par && nr > 1) {
#pragma omp taskloop grainsize(1) shared(err)
            for (int r = 0; r < nr; r++) {
                int e = decode_range(f, S, sc, s, r, lo[r], hi[r], &xs, pm, dout, fo, &dq, v, NULL, NULL);
                if (e) {
#pragma omp atomic write
                    err = e;
                }
            }
        } else {
            for (int r = 0; r < nr && !err; r++)
                err = decode_range(f, S, sc, s, r, lo[r], hi[r], &xs, pm, dout, fo, &dq, v, &tr, &serial);
        }
        rc = err;
    }
    if (rc) return rc;
    double td = prof_on ? now() : 0;
    if (prof_on) {
        prof[5 + S->n_steps - 1] += td - tstep;
        tr += prof_rans_inner;
        prof[1] += tr;
        prof[2] += td - tp0 - tr;
    }
    if (out) { /* cells [0, fused_c0) were not written by the group kernel: gather them from decode order */
        if (f->vt == 2)
            dequant_pos_int16_t(v, S->pos, fused_c0, f->offset, f->step, f->flags, f->clip_lo, f->clip_hi, out);
        else
            dequant_pos_int32_t(v, S->pos, fused_c0, f->offset, f->step, f->flags, f->clip_lo, f->clip_hi, out);
    }
    if (prof_on) prof[3] += now() - td;
    if (ival) {
        if (f->vt == 2)
            for (int64_t i = 0; i < S->n_cells; i++) ival[i] = ((const int16_t *)v)[S->pos[i]];
        else
            for (int64_t i = 0; i < S->n_cells; i++) ival[i] = ((const int32_t *)v)[S->pos[i]];
    }
    return LC_OK;
}

/* Encoder helper: decode one LATTICE payload (cross details given), return values, detail and float output. */
EXPORT int lc_decode_field(const uint8_t *payload, size_t n, const lc_struct_t *S, const int32_t *const *dsrc,
                           const float *const *mf, float *out, int32_t *ival, int32_t *dout) {
    lfield_t *f = malloc(sizeof(lfield_t));
    if (!f) return LC_ERR_MEMORY;
    int rc = parse_lattice(payload, n, S, f);
    scratch_t sc = {0};
    if (!rc) rc = scratch_init(&sc, S);
    /* the encoder works in native order: details are converted to / from decode order here */
    const int64_t nc = S->n_cells;
    int16_t *dp[MAX_CROSS] = {0}, *dpo = NULL;
    for (int c = 0; !rc && c < f->n_cross; c++) {
        if (!dsrc || !dsrc[c]) {
            rc = LC_ERR_MISSING;
            break;
        }
        if (!(dp[c] = malloc(2 * (size_t)nc))) rc = LC_ERR_MEMORY;
        else
            for (int64_t i = 0; i < nc; i++) dp[c][S->pos[i]] = sat16(dsrc[c][i]);
    }
    if (!rc && dout && !(dpo = malloc(2 * (size_t)nc))) rc = LC_ERR_MEMORY;
    if (!rc) rc = decode_lattice(f, S, &sc, (const int16_t *const *)dp, mf, dpo, out, ival, 0);
    if (!rc && dout)
        for (int64_t i = 0; i < nc; i++) dout[i] = dpo[S->pos[i]];
    for (int c = 0; c < MAX_CROSS; c++) free(dp[c]);
    free(dpo);
    if (sc.mem) scratch_free(&sc);
    free(f);
    return rc;
}

/* ------------------------------------------------------------------------------------------------ recipes */
/* RECIPE payload (64-byte header): u8 id, u8 n_src, u16 0, u16 src[4] (field indices: sin_lat, cos_lat, sin_lon,
 * cos_lon; 0xffff = unused), u32 0, f64 p[4], u32 n_patch, u32 0, i64 date (s since 1970, informational), then
 * n_patch x (u32 cell, f32 value): cells where the formula is further than the bound from the data.
 * Formulas (float64, no fused operations), on the decoded float32 static fields:
 *   1 cos_local_time = p0 cos_lon - p1 sin_lon                       (p0, p1 = cos, sin of 2 pi hour / 24)
 *   2 sin_local_time = p1 cos_lon + p0 sin_lon
 *   3 insolation = max(0, p0 sin_lat + p1 cos_lat (p2 cos_lon - p3 sin_lon))  (p0, p1 = sin, cos declination;
 *                                                                    p2, p3 = cos, sin of the hour angle offset) */
#define RECIPE_HEADER 64

EXPORT void lc_recipe_eval(int id, const double *p, const float *sla, const float *cla, const float *slo,
                           const float *clo, int64_t lo, int64_t hi, float *out) {
    if (id == 1)
        for (int64_t i = lo; i < hi; i++) out[i] = (float)(p[0] * (double)clo[i] - p[1] * (double)slo[i]);
    else if (id == 2)
        for (int64_t i = lo; i < hi; i++) out[i] = (float)(p[1] * (double)clo[i] + p[0] * (double)slo[i]);
    else if (id == 3)
        for (int64_t i = lo; i < hi; i++) {
            double a = p[2] * (double)clo[i] - p[3] * (double)slo[i];
            double y = p[0] * (double)sla[i] + p[1] * (double)cla[i] * a;
            out[i] = (float)(y > 0.0 ? y : 0.0);
        }
}

static int recipe_srcs(const uint8_t *p, size_t n, uint32_t n_fields, int32_t src[4]) {
    if (n < RECIPE_HEADER) return LC_ERR_FORMAT;
    int id = p[0];
    if (id < 1 || id > 3) return LC_ERR_FORMAT;
    for (int j = 0; j < 4; j++) {
        src[j] = rd16(p + 4 + 2 * j);
        if (src[j] == 0xffff) src[j] = -1;
        else if ((uint32_t)src[j] >= n_fields) return LC_ERR_FORMAT;
    }
    if ((id == 3 && (src[0] < 0 || src[1] < 0)) || src[2] < 0 || src[3] < 0) return LC_ERR_FORMAT;
    uint32_t np = rd32(p + 48);
    if (RECIPE_HEADER + 8 * (size_t)np > n) return LC_ERR_FORMAT;
    return LC_OK;
}

static int decode_recipe(const uint8_t *p, size_t n, const float *chunk_out, int64_t nc, uint32_t n_fields,
                         float *out) {
    int32_t src[4];
    int rc = recipe_srcs(p, n, n_fields, src);
    if (rc) return rc;
    double prm[4];
    for (int j = 0; j < 4; j++) prm[j] = rdf64(p + 16 + 8 * j);
    const float *f[4];
    for (int j = 0; j < 4; j++) f[j] = src[j] >= 0 ? chunk_out + (size_t)src[j] * (size_t)nc : NULL;
    lc_recipe_eval(p[0], prm, f[0], f[1], f[2], f[3], 0, nc, out);
    uint32_t np = rd32(p + 48);
    for (uint32_t k = 0; k < np; k++) {
        uint32_t cell = rd32(p + RECIPE_HEADER + 8 * k);
        if (cell >= (uint64_t)nc) return LC_ERR_FORMAT;
        out[cell] = rdf32(p + RECIPE_HEADER + 8 * k + 4);
    }
    return LC_OK;
}

/* --------------------------------------------------------------------------------------- chunk decode */
/* Chunk: "LATC" | u16 version | u16 flags | u32 n_fields | u32 n_dates | u32 header_bytes | u32 reserved |
 * u8 structure_sha[16] | i64 dates[n_dates] | n_fields x (u8 mode, u8 fallback, u16 0, u32 0, u64 nbytes) |
 * zero padding to header_bytes | payloads. */
typedef struct {
    uint32_t n_fields, n_dates;
    const uint8_t **payload;
    uint64_t *nbytes;
    uint8_t *mode;
} chunk_t;

static int parse_chunk(const uint8_t *buf, size_t len, chunk_t *c) {
    if (len < 40 || memcmp(buf, CHUNK_MAGIC, 4) || rd16(buf + 4) != CHUNK_VERSION) return LC_ERR_FORMAT;
    c->n_fields = rd32(buf + 8);
    c->n_dates = rd32(buf + 12);
    uint64_t hb = rd32(buf + 16), rec = 40 + 8 * (uint64_t)c->n_dates;
    if (rec + RECORD_BYTES * (uint64_t)c->n_fields > hb || hb > len) return LC_ERR_FORMAT;
    c->payload = malloc(sizeof(uint8_t *) * c->n_fields + 1);
    c->nbytes = malloc(8 * (size_t)c->n_fields + 1);
    c->mode = malloc(c->n_fields + 1);
    if (!c->payload || !c->nbytes || !c->mode) return LC_ERR_MEMORY;
    uint64_t off = hb;
    for (uint32_t i = 0; i < c->n_fields; i++) {
        const uint8_t *r = buf + rec + RECORD_BYTES * (uint64_t)i;
        c->mode[i] = r[0];
        c->nbytes[i] = rd64(r + 8);
        if (c->nbytes[i] > len - off) return LC_ERR_FORMAT;
        c->payload[i] = buf + off;
        off += c->nbytes[i];
    }
    return off == len ? LC_OK : LC_ERR_FORMAT;
}

static void chunk_free(chunk_t *c) {
    free(c->payload);
    free(c->nbytes);
    free(c->mode);
}

/* Decode a chunk into out (n_fields x n_cells float32, native order). static_ptr[i]: the array of STATIC_REF field i;
 * ext_detail[i]: int16 detail array (decode order) of a non-LATTICE field used as a cross source (static fields);
 * times[i] (optional):
 * seconds per field; status[i] (optional): per-field return code. Returns 0 or the first error. */
EXPORT int lc_decode_chunk(const uint8_t *buf, size_t len, const lc_struct_t *S, float *out, int n_threads,
                           const float *const *static_ptr, const int16_t *const *ext_detail, double *times,
                           int32_t *status) {
    chunk_t ch = {0};
    int rc = parse_chunk(buf, len, &ch);
    if (rc) {
        chunk_free(&ch);
        return rc;
    }
    const uint32_t nf = ch.n_fields;
    const int64_t nc = S->n_cells;
    size_t lf_bytes = 0, aux_bytes = 0;
    lfield_t *lf = buf_get(sizeof(lfield_t) * (nf + 1), &lf_bytes);
    /* aux: deps (nf x MAX_DEP int32), need_detail, n_dep, order, depth, indeg, dflag, results */
    size_t aux_need = (size_t)nf * (4 * MAX_DEP + 4 * 6 + 8) + 256;
    uint8_t *aux = buf_get(aux_need, &aux_bytes);
    int16_t **detail = calloc(nf + 1, sizeof(int16_t *));
    size_t *detail_bytes = calloc(nf + 1, sizeof(size_t));
    if (!lf || !aux || !detail || !detail_bytes) {
        rc = LC_ERR_MEMORY;
        goto done;
    }
    int32_t *deps = (int32_t *)aux, *ndep = deps + (size_t)nf * MAX_DEP, *need = ndep + nf, *order = need + nf,
            *depth = order + nf, *indeg = depth + nf, *res = indeg + nf;
    char *dflag = (char *)(res + nf);
    (void)dflag; /* only addressed in depend clauses */
    for (uint32_t i = 0; i < nf; i++) ndep[i] = need[i] = depth[i] = indeg[i] = res[i] = 0;
    for (uint32_t i = 0; i < nf; i++) {
        int m = ch.mode[i];
        if (m == MODE_LATTICE) {
            if ((rc = parse_lattice(ch.payload[i], ch.nbytes[i], S, &lf[i]))) goto done;
            for (int c = 0; c < lf[i].n_cross; c++) {
                int32_t s = lf[i].cross_src[c];
                if ((uint32_t)s >= nf || (uint32_t)s == i) {
                    rc = LC_ERR_DAG;
                    goto done;
                }
                if (ch.mode[s] == MODE_LATTICE) {
                    deps[i * MAX_DEP + ndep[i]++] = s;
                    need[s] = 1;
                } else if (!ext_detail || !ext_detail[s]) {
                    rc = LC_ERR_MISSING;
                    goto done;
                }
            }
            if (lf[i].has_mask) { /* the mask source and the features must be decoded first (floats in out) */
                int32_t ms[1 + 2 * MAX_FEAT];
                int nm = 0;
                ms[nm++] = lf[i].mask_src;
                for (int j = 0; j < lf[i].n_feat; j++) {
                    ms[nm++] = lf[i].fa[j];
                    if (lf[i].fb[j] >= 0) ms[nm++] = lf[i].fb[j];
                }
                for (int j = 0; j < nm; j++) {
                    if ((uint32_t)ms[j] >= nf || (uint32_t)ms[j] == i) {
                        rc = LC_ERR_DAG;
                        goto done;
                    }
                    if (ch.mode[ms[j]] == MODE_LOSSLESS) continue; /* decoded by the caller beforehand */
                    int dup = 0;
                    for (int q = 0; q < ndep[i]; q++) dup |= deps[i * MAX_DEP + q] == ms[j];
                    if (!dup) deps[i * MAX_DEP + ndep[i]++] = ms[j];
                }
            }
        } else if (m == MODE_RECIPE) {
            int32_t src[4];
            if ((rc = recipe_srcs(ch.payload[i], ch.nbytes[i], nf, src))) goto done;
            for (int j = 0; j < 4; j++)
                if (src[j] >= 0 && ch.mode[src[j]] != MODE_LOSSLESS) {
                    int dup = 0;
                    for (int q = 0; q < ndep[i]; q++) dup |= deps[i * MAX_DEP + q] == src[j];
                    if (!dup) deps[i * MAX_DEP + ndep[i]++] = src[j];
                }
        } else if (m == MODE_STATIC_REF) {
            if (!static_ptr || !static_ptr[i]) {
                rc = LC_ERR_MISSING;
                goto done;
            }
        } else if (m == MODE_CONSTANT) {
            if (ch.nbytes[i] < 4) {
                rc = LC_ERR_FORMAT;
                goto done;
            }
        } else if (m != MODE_LOSSLESS) {
            rc = LC_ERR_FORMAT;
            goto done;
        }
    }
    for (uint32_t i = 0; i < nf; i++)
        for (int q = 0; q < ndep[i]; q++) {
            int32_t s = deps[i * MAX_DEP + q];
            if (ch.mode[s] == MODE_RECIPE) { /* recipes read statics only */
                rc = LC_ERR_DAG;
                goto done;
            }
            indeg[i]++;
        }
    /* depth = longest chain of dependants below a field (critical path first); Kahn with that priority. */
    {
        int32_t *lvl = order; /* temporarily: topological levels */
        int32_t *q = calloc(nf + 1, sizeof(int32_t)), *deg = calloc(nf + 1, sizeof(int32_t));
        if (!q || !deg) {
            free(q);
            free(deg);
            rc = LC_ERR_MEMORY;
            goto done;
        }
        uint32_t head = 0, tail = 0;
        for (uint32_t i = 0; i < nf; i++) {
            deg[i] = indeg[i];
            if (!deg[i]) q[tail++] = (int32_t)i;
        }
        while (head < tail) { /* plain topological order into q */
            int32_t u = q[head++];
            for (uint32_t i = 0; i < nf; i++) /* nf ~ 100: quadratic is fine */
                for (int k = 0; k < ndep[i]; k++)
                    if (deps[i * MAX_DEP + k] == u && --deg[i] == 0) q[tail++] = (int32_t)i;
        }
        if (tail != nf) {
            free(q);
            free(deg);
            rc = LC_ERR_DAG;
            goto done;
        }
        for (int64_t a = (int64_t)nf - 1; a >= 0; a--) { /* depth = 1 + max depth of dependants */
            int32_t u = q[a];
            int32_t d = 0;
            for (uint32_t i = 0; i < nf; i++)
                for (int k = 0; k < ndep[i]; k++)
                    if (deps[i * MAX_DEP + k] == u && depth[i] + 1 > d) d = depth[i] + 1;
            depth[u] = d;
        }
        /* stable sort of the topological order by depth descending keeps it topological (a source always has a
         * larger depth than its dependants) */
        for (uint32_t i = 0; i < nf; i++) lvl[i] = q[i];
        for (uint32_t i = 1; i < nf; i++) {
            int32_t u = lvl[i];
            int64_t j = (int64_t)i - 1;
            while (j >= 0 && depth[lvl[j]] < depth[u]) {
                lvl[j + 1] = lvl[j];
                j--;
            }
            lvl[j + 1] = u;
        }
        free(q);
        free(deg);
    }
    for (uint32_t i = 0; i < nf; i++)
        if (need[i]) {
            detail[i] = buf_get(2 * (size_t)nc, &detail_bytes[i]);
            if (!detail[i]) {
                rc = LC_ERR_MEMORY;
                goto done;
            }
        }
    if (n_threads < 1) n_threads = 1;
    scratch_t *sc = calloc((size_t)n_threads, sizeof(scratch_t));
    if (!sc) {
        rc = LC_ERR_MEMORY;
        goto done;
    }
#pragma omp parallel num_threads(n_threads)
#pragma omp single
    {
        for (uint32_t o = 0; o < nf; o++) {
            int32_t i = order[o];
            int m = ch.mode[i];
            if (m == MODE_LOSSLESS) continue;
            int32_t d[MAX_DEP];
            for (int k = 0; k < MAX_DEP; k++) d[k] = k < ndep[i] ? deps[i * MAX_DEP + k] : (int32_t)nf;
#pragma omp task firstprivate(i, m) depend(out : dflag[i]) \
    depend(in : dflag[d[0]], dflag[d[1]], dflag[d[2]], dflag[d[3]], dflag[d[4]], dflag[d[5]], dflag[d[6]], \
               dflag[d[7]], dflag[d[8]], dflag[d[9]], dflag[d[10]], dflag[d[11]], dflag[d[12]], dflag[d[13]],  \
               dflag[d[14]], dflag[d[15]])
            {
                double t0 = times ? now() : 0;
                int tn = omp_get_thread_num();
                float *fo = out + (size_t)i * (size_t)nc;
                const uint8_t *p = ch.payload[i];
                int r = LC_OK;
                if (m == MODE_LATTICE) {
                    if (!sc[tn].mem) r = scratch_init(&sc[tn], S);
                    const int16_t *ds[MAX_CROSS] = {0};
                    for (int c = 0; c < lf[i].n_cross; c++) {
                        int32_t s = lf[i].cross_src[c];
                        ds[c] = ch.mode[s] == MODE_LATTICE ? detail[s] : ext_detail[s];
                    }
                    const float *mf[1 + 2 * MAX_FEAT] = {0};
                    if (lf[i].has_mask) {
                        mf[0] = out + (size_t)lf[i].mask_src * (size_t)nc;
                        for (int j = 0; j < lf[i].n_feat; j++) {
                            mf[1 + j] = out + (size_t)lf[i].fa[j] * (size_t)nc;
                            if (lf[i].fb[j] >= 0) mf[1 + MAX_FEAT + j] = out + (size_t)lf[i].fb[j] * (size_t)nc;
                        }
                    }
                    if (!r) r = decode_lattice(&lf[i], S, &sc[tn], ds, mf, detail[i], fo, NULL, n_threads > 1);
                } else if (m == MODE_CONSTANT) {
                    float c0 = rdf32(p);
                    for (int64_t k = 0; k < nc; k++) fo[k] = c0;
                } else if (m == MODE_STATIC_REF) {
                    memcpy(fo, static_ptr[i], 4 * (size_t)nc);
                } else if (m == MODE_RECIPE) {
                    r = decode_recipe(p, ch.nbytes[i], out, nc, nf, fo);
                }
                res[i] = r;
                if (times) times[i] = now() - t0;
            }
        }
    }
    for (int k = 0; k < n_threads; k++)
        if (sc[k].mem) scratch_free(&sc[k]);
    free(sc);
    for (uint32_t i = 0; i < nf; i++) {
        if (status) status[i] = res[i];
        if (res[i] && !rc) rc = res[i];
    }
done:
    if (detail)
        for (uint32_t i = 0; i < nf; i++) buf_put(detail[i], detail_bytes[i]);
    free(detail);
    free(detail_bytes);
    buf_put(lf, lf_bytes);
    buf_put(aux, aux_bytes);
    chunk_free(&ch);
    return rc;
}

/* Structure validation helper (tests): every cell is a target exactly once and every source is decoded earlier. */
EXPORT int lc_struct_check(const lc_struct_t *S) {
    uint8_t *done = calloc((size_t)S->n_cells, 1);
    if (!done) return LC_ERR_MEMORY;
    int rc = LC_OK;
    for (int s = 0; s < S->n_steps && !rc; s++) {
        const lc_step_t *st = &S->steps[s];
        if (st->n > S->max_n || st->k > MAX_K || (st->kind == STEP_GATHER && st->k < 1)) rc = LC_ERR_STRUCT;
        for (int64_t i = 0; i < st->n && !rc; i++)
            for (int j = 0; j < (st->kind == STEP_GATHER ? st->k : 0); j++) {
                int32_t c = st->src[(size_t)j * st->n + i];
                if (c < 0 || c >= S->n_cells || !done[c]) rc = LC_ERR_STRUCT;
            }
        for (int64_t i = 0; i < st->n && !rc; i++) {
            int32_t c = st->t[i];
            if (c < 0 || c >= S->n_cells || done[c]) rc = LC_ERR_STRUCT;
            else done[c] = 1;
        }
    }
    for (int64_t c = 0; c < S->n_cells && !rc; c++)
        if (!done[c]) rc = LC_ERR_STRUCT;
    free(done);
    return rc;
}
