/* lattice_codec: shared declarations of the C kernels (rANS, structure steps, chunk decoder).
 *
 * Build: see setup.py. The kernels are compiled once per ISA variant (variant.c: AVX2, and a portable scalar
 * emulation of the same intrinsics, simd.h) with -O3 -fopenmp -ffp-contract=off; dispatch.c picks one at load time.
 * All arithmetic on quantised values is integer; the only floating-point operations are the dequantisation
 * `offset + step * (float)n` (two IEEE roundings, never fused) and the recipe formulas (float64, never fused). */
#ifndef LATTICE_CODEC_H
#define LATTICE_CODEC_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* In the unity build of a variant (variant.c, LC_UNITY) every function is static: the variants are linked into one
 * library and only the dispatch layer (dispatch.c) exports symbols. */
#ifdef LC_UNITY
#define EXPORT static
#define LC_INTERNAL static
#else
#define EXPORT __attribute__((visibility("default")))
#define LC_INTERNAL
#endif
#define INLINE static inline __attribute__((always_inline))

/* entropy coder: 32-lane interleaved static rANS, 12-bit frequencies, byte alphabet, 255 = escape */
#define PROB_BITS 12
#define PROB_SCALE (1u << PROB_BITS)
#define RANS_L (1u << 16)
#define LANES 32
#define ALPHABET 256
#define ESCAPE 255

#define MAX_K 8       /* sources per gather step */
#define MAX_CROSS 4   /* cross-variable terms per field */
#define MAX_CTX 8     /* context buckets per step */
#define MAX_STEPS 64  /* steps per structure */
#define MAX_TABLES 255
#define MAX_RANGES 16
/* Steps are split into independent ranges (own rANS stream and escape slice) of RANGE_T targets, or RANGE_G groups
 * for the regular-group step; a function of the structure only (both sides compute it). */
#define RANGE_T 131072
#define RANGE_G 43648

/* container field modes */
enum { MODE_LATTICE = 0, MODE_LOSSLESS = 1, MODE_CONSTANT = 2, MODE_STATIC_REF = 3, MODE_RECIPE = 4 };

/* structure step kinds */
enum { STEP_DELTA = 0, STEP_GATHER = 1 };

/* Error codes (negative returns). */
enum {
    LC_OK = 0,
    LC_ERR_FORMAT = -1,   /* malformed container / payload */
    LC_ERR_STRUCT = -2,   /* payload does not match the structure */
    LC_ERR_RANS = -3,     /* entropy stream overrun */
    LC_ERR_ESCAPE = -4,   /* escape stream overrun */
    LC_ERR_MEMORY = -5,
    LC_ERR_MISSING = -6,  /* a static array or cross-variable detail is missing */
    LC_ERR_DAG = -7,      /* dependency cycle or dependency on a non-decodable field */
};

/* One step of a structure: n targets t[i] (native cell indices) predicted from k sources src[j * n + i] (SoA),
 * all decoded by earlier steps. DELTA steps (k = 0): t[i] predicted from t[i - 1] (t[0] from 0). */
typedef struct {
    int32_t kind, k;
    int64_t n;
    const int32_t *t;   /* native target cells */
    const int32_t *src; /* native source cells (k x n) */
    int64_t p0;         /* decode-order position of the first target (targets are p0 .. p0 + n - 1) */
    const int32_t *srcp; /* sources as decode-order positions (k x n): what the kernels read */
} lc_step_t;

/* Optional fast path per step (a decoding hint derived from the structure, no change of semantics): the step's
 * targets come in ng triples (corners of one parent, source 0 = the parent's central, sources 1-2 drawn from 3
 * neighbouring centrals), so the kernel loads the 3 neighbours once per group. When c0 >= 0 the triples are also
 * the native cells c0 + 4 q + {0, 1, 3} with the central at c0 + 4 q + 2 (the regular ICON groups in native order,
 * c0 + 4 ng = n_cells) and the last step writes 4 floats per group contiguously. ng = 0: no groups. */
typedef struct {
    int32_t step, pad;
    int64_t i0, c0, ng;
    const int32_t *gpos; /* decode-order position of each group's central cell */
    /* the 3 neighbouring centrals of each group (positions) and, per group, which pair each corner uses:
     * corner 0 = (nb0, nb1); corner j = 1, 2: (nb[pat >> (4j - 4) & 3], nb[pat >> (4j - 2) & 3]) */
    const int32_t *nb0, *nb1, *nb2;
    const uint8_t *pat;
} lc_group_t;

typedef struct {
    int64_t n_cells;
    int32_t n_steps, max_n; /* max_n: the largest step */
    const lc_step_t *steps;
    const lc_group_t *group; /* per step (n_steps entries, ng = 0 where none), or NULL */
    const int32_t *pos;      /* decode-order position of every native cell (values / details live in that order) */
} lc_struct_t;

INLINE uint32_t zigzag_enc(int32_t v) { return ((uint32_t)v << 1) ^ (uint32_t)(v >> 31); }
INLINE int32_t zigzag_dec(uint32_t u) { return (int32_t)(u >> 1) ^ -(int32_t)(u & 1); }
/* int32 arithmetic that wraps (two's complement) where the plain signed operation would overflow, which only a
 * corrupt stream can make it do: the same result as the plain operation whenever that does not overflow (every
 * valid stream), and defined behaviour on corrupt input. The SIMD paths wrap already (simd.h, AVX2). */
INLINE int32_t wadd32(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
INLINE int32_t wsub32(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
INLINE int32_t wmul32(int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }
INLINE uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
INLINE uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
INLINE int32_t rd32s(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }
INLINE int16_t rd16s(const uint8_t *p) { int16_t v; memcpy(&v, p, 2); return v; }
INLINE uint64_t rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
INLINE float rdf32(const uint8_t *p) { float v; memcpy(&v, p, 4); return v; }
INLINE double rdf64(const uint8_t *p) { double v; memcpy(&v, p, 8); return v; }

/* ---- rans.c */
typedef struct {
    uint32_t x[LANES];
    const uint8_t *p, *end; /* next renormalisation word, end of the words (+16 bytes of zero padding) */
} rstate_t;

LC_INTERNAL int lc_normalize_freqs(const uint32_t *counts, uint32_t *freqs, uint32_t total);
LC_INTERNAL void lc_build_table(const uint32_t *freqs, uint32_t *table, int bits);
LC_INTERNAL int rans_init(rstate_t *st, const uint8_t *in, size_t nbytes);
/* Decode n symbols of one segment (lanes restart at 0) with one table. */
LC_INTERNAL int rans_segment(rstate_t *st, size_t n, const uint32_t *table, uint8_t *out, int bits);
/* Decode n symbols, symbol i with table tabs + (tid[i] << PROB_BITS). */
LC_INTERNAL int rans_segment_tid(rstate_t *st, size_t n, const uint32_t *tabs, const uint8_t *tid, uint8_t *out, int bits);

#endif
