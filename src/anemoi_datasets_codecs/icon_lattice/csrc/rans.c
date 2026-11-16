/* Interleaved static rANS (32 lanes, 32-bit states, 16-bit renormalisation words, 12-bit frequencies).
 *
 * Stream: u32 lanes (= 32) | u32 state[32] | u16 words[] | 16 zero bytes. The symbols of a field form segments (one
 * per structure step); within a segment symbol i is coded by lane i % 32, so every segment starts at lane 0 and a
 * decoder can process a segment as soon as its tables (contexts) are known. The encoder runs backwards over all
 * symbols and writes words backwards, so the decoder reads them strictly forwards. Decoding a symbol:
 * e = table[x & 4095]; sym = e & 255; x = freq(e) * (x >> 12) + bias(e); if x < 2^16: x = x << 16 | next word.
 * References: J. Duda, "Asymmetric numeral systems", arXiv:1311.2540 (2013); F. Giesen, "Interleaved entropy coders",
 * arXiv:1402.3392 (2014), and his ryg_rans. */
#include "simd.h"
#include <stdlib.h>

#include "lc.h"

/* Normalise counts to frequencies summing to `total` (a power of two). Every symbol with a non-zero count and the
 * escape symbol get >= 1; no symbol gets `total`. Returns 0, or -1 if all counts are zero. */
EXPORT int lc_normalize_freqs(const uint32_t *counts, uint32_t *freqs, uint32_t total) {
    uint64_t sum = 0;
    for (int s = 0; s < ALPHABET; s++) sum += counts[s];
    if (sum == 0) return -1;
    uint32_t got = 0;
    for (int s = 0; s < ALPHABET; s++) {
        uint32_t f = 0;
        if (counts[s] || s == ESCAPE) {
            f = (uint32_t)(((uint64_t)counts[s] * total + sum / 2) / sum);
            if (f == 0) f = 1;
        }
        freqs[s] = f;
        got += f;
    }
    while (got != total) { /* fix the sum on the largest frequencies */
        int best = 0;
        for (int s = 1; s < ALPHABET; s++)
            if (freqs[s] > freqs[best]) best = s;
        if (got < total) {
            freqs[best] += total - got;
            got = total;
        } else {
            uint32_t take = got - total, room = freqs[best] - 1;
            if (take > room / 2 + 1) take = room / 2 + 1;
            if (take > room) take = room;
            freqs[best] -= take;
            got -= take;
        }
    }
    for (int s = 0; s < ALPHABET; s++)
        if (freqs[s] >= total) {
            freqs[s] = total - 1;
            freqs[s == ESCAPE ? 0 : ESCAPE] += 1;
        }
    return 0;
}

/* Decode table of 2^bits entries (bits = 10 or 12): sym (bits 0-7) | bias = slot - start (bits 8-19) |
 * freq (bits 20-31). */
EXPORT void lc_build_table(const uint32_t *freqs, uint32_t *table, int bits) {
    const uint32_t scale = 1u << bits;
    uint32_t start = 0;
    for (uint32_t s = 0; s < ALPHABET && start < scale; s++) {
        uint32_t f = freqs[s];
        if (start + f > scale) f = scale - start; /* malformed tables: never write out of bounds */
        for (uint32_t k = 0; k < f; k++) table[start + k] = s | (k << 8) | (freqs[s] << 20);
        start += f;
    }
    for (; start < scale; start++) table[start] = ESCAPE | (1u << 20); /* malformed (sum < scale) */
}

/* Encode symbols; symbol i uses frequency table tid[i] (freqs: ntab x 256); lanes restart at every segment
 * [seg[s], seg[s + 1]). Returns the stream size (padding included) or 0 on error (cap too small, zero frequency). */
EXPORT size_t lc_rans_encode(const uint8_t *syms, const uint8_t *tid, const uint64_t *seg, int nseg,
                             const uint32_t *freqs, int ntab, uint8_t *out, size_t cap, int bits) {
    size_t n = seg[nseg];
    if (cap < 4 + 4 * LANES + 2 * n + 16) return 0;
    uint32_t *cum = malloc(sizeof(uint32_t) * ALPHABET * (size_t)ntab);
    uint16_t *words = malloc(2 * n + 2);
    if (!cum || !words) {
        free(cum);
        free(words);
        return 0;
    }
    for (int tb = 0; tb < ntab; tb++)
        for (uint32_t s = 0, c = 0; s < ALPHABET; s++) {
            cum[tb * ALPHABET + s] = c;
            c += freqs[tb * ALPHABET + s];
        }
    uint16_t *end = words + n, *p = end;
    uint32_t x[LANES];
    for (int j = 0; j < LANES; j++) x[j] = RANS_L;
    for (int sg = nseg - 1; sg >= 0; sg--)
        for (size_t i = seg[sg + 1]; i-- > seg[sg];) {
            uint32_t j = (uint32_t)((i - seg[sg]) % LANES), tb = tid[i], s = syms[i];
            uint32_t f = tb < (uint32_t)ntab ? freqs[tb * ALPHABET + s] : 0;
            if (!f) {
                free(words);
                free(cum);
                return 0;
            }
            if (x[j] >= (f << (32 - bits))) { /* x_max = ((L >> bits) << 16) * f */
                *--p = (uint16_t)x[j];
                x[j] >>= 16;
            }
            x[j] = ((x[j] / f) << bits) + (x[j] % f) + cum[tb * ALPHABET + s];
        }
    uint32_t lanes = LANES;
    size_t nw = (size_t)(end - p), size = 4 + 4 * LANES + 2 * nw;
    memcpy(out, &lanes, 4);
    memcpy(out + 4, x, 4 * LANES);
    memcpy(out + 4 + 4 * LANES, p, 2 * nw);
    memset(out + size, 0, 16);
    free(words);
    free(cum);
    return size + 16;
}

int rans_init(rstate_t *st, const uint8_t *in, size_t nbytes) {
    if (nbytes < 4 + 4 * LANES + 16 || rd32(in) != LANES) return LC_ERR_RANS;
    memcpy(st->x, in + 4, 4 * LANES);
    st->p = in + 4 + 4 * LANES;
    st->end = in + nbytes;
    return LC_OK;
}

/* scalar step with a bounds check (the vector loops stop 64 bytes before the end) */
INLINE int step_scalar(uint32_t *x, rstate_t *st, const uint32_t *table, uint8_t *out, const int B) {
    uint32_t e = table[*x & ((1u << B) - 1)];
    *out = (uint8_t)e;
    uint32_t y = (e >> 20) * (*x >> B) + ((e >> 8) & 4095u);
    if (y < RANS_L) {
        if (st->p + 2 > st->end) return -1;
        y = (y << 16) | rd16(st->p);
        st->p += 2;
    }
    *x = y;
    return 0;
}

static __m256i renorm_perm[256]; /* lane j takes word popcount(mask & ((1 << j) - 1)) */

__attribute__((constructor)) static void init_renorm_perm(void) {
    for (int m = 0; m < 256; m++) {
        uint32_t idx[8];
        for (int j = 0; j < 8; j++) idx[j] = (m >> j) & 1 ? (uint32_t)__builtin_popcount(m & ((1 << j) - 1)) : 0;
        renorm_perm[m] = _mm256_loadu_si256((const __m256i *)idx);
    }
}

/* 8 lanes; table entries from tabs + (tid << 12) when tid != NULL, else from tabs. Scalar loads (no vpgatherdd,
 * slow on Zen 2/3). */
INLINE __m256i step8(__m256i x, const uint8_t **p, const uint32_t *tabs, const uint8_t *tid, uint8_t *out,
                     const int B) {
    const __m256i mask = _mm256_set1_epi32((1 << B) - 1), bmask = _mm256_set1_epi32(4095);
    __m256i idx = _mm256_and_si256(x, mask);
    if (tid) idx = _mm256_or_si256(idx, _mm256_slli_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)tid)), B));
    uint32_t s[8] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)s, idx);
    __m256i e = _mm256_setr_epi32((int)tabs[s[0]], (int)tabs[s[1]], (int)tabs[s[2]], (int)tabs[s[3]],
                                  (int)tabs[s[4]], (int)tabs[s[5]], (int)tabs[s[6]], (int)tabs[s[7]]);
    x = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(e, 20), _mm256_srli_epi32(x, B)),
                         _mm256_and_si256(_mm256_srli_epi32(e, 8), bmask));
    const __m256i pick = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i sb = _mm256_shuffle_epi8(e, pick);
    uint32_t lo = (uint32_t)_mm256_extract_epi32(sb, 0), hi = (uint32_t)_mm256_extract_epi32(sb, 4);
    memcpy(out, &lo, 4);
    memcpy(out + 4, &hi, 4);
    __m256i need = _mm256_cmpeq_epi32(_mm256_srli_epi32(x, 16), _mm256_setzero_si256());
    int m = _mm256_movemask_ps(_mm256_castsi256_ps(need));
    __m256i w = _mm256_permutevar8x32_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)*p)), renorm_perm[m]);
    x = _mm256_blendv_epi8(x, _mm256_or_si256(_mm256_slli_epi32(x, 16), w), need);
    *p += 2 * __builtin_popcount(m);
    return x;
}

INLINE int segment(rstate_t *st, size_t n, const uint32_t *tabs, const uint8_t *tid, uint8_t *out, const int B) {
    __m256i x[4];
    for (int r = 0; r < 4; r++) x[r] = _mm256_loadu_si256((const __m256i *)(st->x + 8 * r));
    const uint8_t *p = st->p, *safe = st->end - 64; /* a batch of 32 reads at most 64 bytes */
    size_t i = 0;
    for (; i + LANES <= n && p <= safe; i += LANES)
        for (int r = 0; r < 4; r++)
            x[r] = step8(x[r], &p, tabs, tid ? tid + i + 8 * r : NULL, out + i + 8 * r, B);
    for (int r = 0; r < 4; r++) _mm256_storeu_si256((__m256i *)(st->x + 8 * r), x[r]);
    st->p = p;
    for (uint32_t j = 0; i < n; i++, j = (j + 1) % LANES) {
        const uint32_t *table = tid ? tabs + ((uint32_t)tid[i] << B) : tabs;
        if (step_scalar(&st->x[j], st, table, out + i, B)) return LC_ERR_RANS;
    }
    return LC_OK;
}

/* Note: a segment always restarts at lane 0; the vector loop leaves i at a multiple of 32, so the scalar tail lane
 * index j = i % 32 starts at 0 too. */
int rans_segment(rstate_t *st, size_t n, const uint32_t *table, uint8_t *out, int bits) {
    return bits == 10 ? segment(st, n, table, NULL, out, 10) : segment(st, n, table, NULL, out, 12);
}

int rans_segment_tid(rstate_t *st, size_t n, const uint32_t *tabs, const uint8_t *tid, uint8_t *out, int bits) {
    return bits == 10 ? segment(st, n, tabs, tid, out, 10) : segment(st, n, tabs, tid, out, 12);
}

/* Decode a whole single-segment stream with one frequency table (tests). */
EXPORT int lc_rans_decode_test(const uint8_t *in, size_t nbytes, const uint64_t *seg, int nseg, const uint32_t *freqs,
                               const uint8_t *tid, uint8_t *out, int bits) {
    rstate_t st;
    int ntab = 0;
    for (size_t i = 0; i < seg[nseg]; i++)
        if (tid[i] + 1 > ntab) ntab = tid[i] + 1;
    uint32_t *tabs = malloc(4 * PROB_SCALE * (size_t)(ntab ? ntab : 1));
    if (!tabs) return LC_ERR_MEMORY;
    for (int t = 0; t < ntab; t++) lc_build_table(freqs + t * ALPHABET, tabs + ((size_t)t << bits), bits);
    int rc = rans_init(&st, in, nbytes);
    for (int s = 0; s < nseg && rc == 0; s++)
        rc = rans_segment_tid(&st, seg[s + 1] - seg[s], tabs, tid + seg[s], out + seg[s], bits);
    free(tabs);
    return rc;
}
