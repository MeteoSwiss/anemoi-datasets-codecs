/* SIMD layer of the kernels.
 *
 * Default: the x86 AVX2 intrinsics (<immintrin.h>); the variant is compiled with -mavx2.
 * LC_SCALAR: a portable emulation of exactly the intrinsics the kernels use. The kernels are the same source in both
 * variants, so the "scalar" variant performs the same integer and IEEE float32 operations in the same order and
 * decodes bit-identically (verified by the tests); it is the fallback for x86 CPUs without AVX2 and the only variant
 * on other platforms. Lane-wise arithmetic uses GCC / clang generic vectors (8 x 32 bit), which the compiler lowers
 * to whatever the baseline ISA has (SSE2 on x86-64, NEON on aarch64) or to scalar code; shuffles and byte-wise
 * operations are plain loops. Loads and stores read and write the same number of bytes as the real instructions, so
 * the buffer padding the AVX2 code relies on is exercised identically.
 *
 * Semantics follow the Intel Intrinsics Guide; only what the kernels need is implemented. */
#ifndef LATTICE_CODEC_SIMD_H
#define LATTICE_CODEC_SIMD_H

#ifndef LC_SCALAR
#include <immintrin.h>
#else

#include <stdint.h>
#include <string.h>

/* aligned(4): the emulated vectors may live anywhere (the kernels load them from unaligned byte pointers) */
typedef int32_t lc_v8si __attribute__((vector_size(32), aligned(4)));
typedef uint32_t lc_v8su __attribute__((vector_size(32), aligned(4)));
typedef float lc_v8sf __attribute__((vector_size(32), aligned(4)));

typedef union {
    lc_v8si vi;
    lc_v8su vu;
    lc_v8sf vf;
    int8_t b[32];
    uint8_t ub[32];
    int16_t h[16];
    uint16_t uh[16];
    int32_t i[8];
    uint32_t u[8];
    float f[8];
    double d[4];
} lc_v256;

typedef union {
    int8_t b[16];
    uint8_t ub[16];
    int16_t h[8];
    uint16_t uh[8];
    int32_t i[4];
    uint32_t u[4];
    uint64_t q[2];
} lc_v128;

/* The kernels name the x86 vector types; without <immintrin.h> they are free to define. */
typedef lc_v256 __m256i;
typedef lc_v256 __m256;
typedef lc_v256 __m256d;
typedef lc_v128 __m128i;

#define SIMD_INLINE static inline __attribute__((always_inline))
#define FOR8 for (int k_ = 0; k_ < 8; k_++)

/* ---- 256-bit integer */
SIMD_INLINE lc_v256 _mm256_setzero_si256(void) {
    lc_v256 r;
    r.vi = (lc_v8si){0};
    return r;
}
SIMD_INLINE lc_v256 _mm256_set1_epi32(int32_t a) {
    lc_v256 r;
    r.vi = (lc_v8si){a, a, a, a, a, a, a, a};
    return r;
}
SIMD_INLINE lc_v256 _mm256_setr_epi32(int32_t a0, int32_t a1, int32_t a2, int32_t a3, int32_t a4, int32_t a5,
                                      int32_t a6, int32_t a7) {
    lc_v256 r;
    r.vi = (lc_v8si){a0, a1, a2, a3, a4, a5, a6, a7};
    return r;
}
SIMD_INLINE lc_v256 _mm256_setr_epi8(char a0, char a1, char a2, char a3, char a4, char a5, char a6, char a7, char a8,
                                     char a9, char a10, char a11, char a12, char a13, char a14, char a15, char a16,
                                     char a17, char a18, char a19, char a20, char a21, char a22, char a23, char a24,
                                     char a25, char a26, char a27, char a28, char a29, char a30, char a31) {
    const char v[32] = {a0,  a1,  a2,  a3,  a4,  a5,  a6,  a7,  a8,  a9,  a10, a11, a12, a13, a14, a15,
                        a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31};
    lc_v256 r;
    memcpy(&r, v, 32);
    return r;
}
SIMD_INLINE lc_v256 _mm256_loadu_si256(const lc_v256 *p) {
    lc_v256 r;
    memcpy(&r, (const void *)p, 32);
    return r;
}
SIMD_INLINE void _mm256_storeu_si256(lc_v256 *p, lc_v256 a) { memcpy((void *)p, &a, 32); }
SIMD_INLINE void _mm256_store_si256(lc_v256 *p, lc_v256 a) { memcpy((void *)p, &a, 32); }

/* wrap-around arithmetic on unsigned lanes (no signed overflow) */
SIMD_INLINE lc_v256 _mm256_add_epi32(lc_v256 a, lc_v256 b) {
    a.vu = a.vu + b.vu;
    return a;
}
SIMD_INLINE lc_v256 _mm256_sub_epi32(lc_v256 a, lc_v256 b) {
    a.vu = a.vu - b.vu;
    return a;
}
SIMD_INLINE lc_v256 _mm256_mullo_epi32(lc_v256 a, lc_v256 b) {
    a.vu = a.vu * b.vu;
    return a;
}
SIMD_INLINE lc_v256 _mm256_and_si256(lc_v256 a, lc_v256 b) {
    a.vu = a.vu & b.vu;
    return a;
}
SIMD_INLINE lc_v256 _mm256_or_si256(lc_v256 a, lc_v256 b) {
    a.vu = a.vu | b.vu;
    return a;
}
SIMD_INLINE lc_v256 _mm256_xor_si256(lc_v256 a, lc_v256 b) {
    a.vu = a.vu ^ b.vu;
    return a;
}
/* shifts: counts > 31 give 0 (logical) or the sign (arithmetic), as the instructions do */
SIMD_INLINE lc_v256 lc_srl_(lc_v256 a, uint64_t n) {
    if (n > 31) return _mm256_setzero_si256();
    a.vu = a.vu >> (uint32_t)n;
    return a;
}
SIMD_INLINE lc_v256 lc_sll_(lc_v256 a, uint64_t n) {
    if (n > 31) return _mm256_setzero_si256();
    a.vu = a.vu << (uint32_t)n;
    return a;
}
SIMD_INLINE lc_v256 lc_sra_(lc_v256 a, uint64_t n) { /* arithmetic: ~(~x >> n) for negative lanes */
    if (n > 31) n = 31;
    lc_v8su neg = (lc_v8su)(a.vi < (lc_v8si){0});
    a.vu = (((~a.vu >> (uint32_t)n) ^ ~(lc_v8su){0}) & neg) | ((a.vu >> (uint32_t)n) & ~neg);
    return a;
}
SIMD_INLINE lc_v256 _mm256_srli_epi32(lc_v256 a, int n) { return lc_srl_(a, (uint64_t)(unsigned)n); }
SIMD_INLINE lc_v256 _mm256_slli_epi32(lc_v256 a, int n) { return lc_sll_(a, (uint64_t)(unsigned)n); }
SIMD_INLINE lc_v256 _mm256_srai_epi32(lc_v256 a, int n) { return lc_sra_(a, (uint64_t)(unsigned)n); }
SIMD_INLINE lc_v256 _mm256_sll_epi32(lc_v256 a, lc_v128 c) { return lc_sll_(a, c.q[0]); } /* count: low 64 bits */
SIMD_INLINE lc_v256 _mm256_sra_epi32(lc_v256 a, lc_v128 c) { return lc_sra_(a, c.q[0]); }
SIMD_INLINE lc_v256 _mm256_cmpeq_epi32(lc_v256 a, lc_v256 b) {
    a.vi = (lc_v8si)(a.vi == b.vi); /* -1 / 0 per lane */
    return a;
}
SIMD_INLINE lc_v256 _mm256_cmpgt_epi32(lc_v256 a, lc_v256 b) {
    a.vi = (lc_v8si)(a.vi > b.vi);
    return a;
}
SIMD_INLINE lc_v256 lc_select_(lc_v8su m, lc_v256 a, lc_v256 b) { /* m ? a : b, lane-wise (m all ones / zeros) */
    a.vu = (a.vu & m) | (b.vu & ~m);
    return a;
}
SIMD_INLINE lc_v256 _mm256_max_epi32(lc_v256 a, lc_v256 b) { return lc_select_((lc_v8su)(a.vi > b.vi), a, b); }
SIMD_INLINE lc_v256 _mm256_min_epi32(lc_v256 a, lc_v256 b) { return lc_select_((lc_v8su)(a.vi < b.vi), a, b); }
SIMD_INLINE lc_v256 _mm256_blendv_epi8(lc_v256 a, lc_v256 b, lc_v256 m) { /* per byte: top bit of m ? b : a */
    lc_v8su t = ((m.vu & (lc_v8su){0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u, 0x80808080u,
                                     0x80808080u, 0x80808080u}) >> 7) * 0xffu;
    a.vu = (a.vu & ~t) | (b.vu & t);
    return a;
}
SIMD_INLINE lc_v256 _mm256_permutevar8x32_epi32(lc_v256 a, lc_v256 idx) {
    lc_v256 r;
    FOR8 r.u[k_] = a.u[idx.u[k_] & 7];
    return r;
}
SIMD_INLINE lc_v256 _mm256_shuffle_epi8(lc_v256 a, lc_v256 b) { /* within each 128-bit lane */
    lc_v256 r;
    for (int k = 0; k < 32; k++) r.ub[k] = (b.ub[k] & 0x80) ? 0 : a.ub[(k & 16) | (b.ub[k] & 15)];
    return r;
}
#define _mm256_extract_epi32(a, n) ((a).i[(n) & 7])
SIMD_INLINE lc_v256 _mm256_cvtepu8_epi32(lc_v128 a) {
    lc_v256 r;
    r.vi = (lc_v8si){a.ub[0], a.ub[1], a.ub[2], a.ub[3], a.ub[4], a.ub[5], a.ub[6], a.ub[7]};
    return r;
}
SIMD_INLINE lc_v256 _mm256_cvtepi16_epi32(lc_v128 a) {
    lc_v256 r;
    r.vi = (lc_v8si){a.h[0], a.h[1], a.h[2], a.h[3], a.h[4], a.h[5], a.h[6], a.h[7]};
    return r;
}
SIMD_INLINE lc_v256 _mm256_cvtepu16_epi32(lc_v128 a) {
    lc_v256 r;
    r.vi = (lc_v8si){a.uh[0], a.uh[1], a.uh[2], a.uh[3], a.uh[4], a.uh[5], a.uh[6], a.uh[7]};
    return r;
}
SIMD_INLINE lc_v128 _mm256_castsi256_si128(lc_v256 a) {
    lc_v128 r;
    memcpy(&r, &a, 16);
    return r;
}
SIMD_INLINE lc_v128 _mm256_extracti128_si256(lc_v256 a, int n) {
    lc_v128 r;
    memcpy(&r, a.ub + ((n & 1) ? 16 : 0), 16);
    return r;
}

/* ---- 256-bit float (IEEE single, round to nearest: the same operations as the vector instructions) */
SIMD_INLINE lc_v256 _mm256_castsi256_ps(lc_v256 a) { return a; }
SIMD_INLINE lc_v256 _mm256_castps_pd(lc_v256 a) { return a; }
SIMD_INLINE lc_v256 _mm256_castpd_ps(lc_v256 a) { return a; }
SIMD_INLINE int _mm256_movemask_ps(lc_v256 a) {
    int m = 0;
    FOR8 m |= (int)(a.u[k_] >> 31) << k_;
    return m;
}
SIMD_INLINE lc_v256 _mm256_set1_ps(float a) {
    lc_v256 r;
    r.vf = (lc_v8sf){a, a, a, a, a, a, a, a};
    return r;
}
SIMD_INLINE lc_v256 _mm256_add_ps(lc_v256 a, lc_v256 b) {
    a.vf = a.vf + b.vf;
    return a;
}
SIMD_INLINE lc_v256 _mm256_mul_ps(lc_v256 a, lc_v256 b) {
    a.vf = a.vf * b.vf;
    return a;
}
/* maxps / minps: the second operand unless a > b (a < b); no NaN reaches the kernels */
SIMD_INLINE lc_v256 _mm256_max_ps(lc_v256 a, lc_v256 b) { return lc_select_((lc_v8su)(a.vf > b.vf), a, b); }
SIMD_INLINE lc_v256 _mm256_min_ps(lc_v256 a, lc_v256 b) { return lc_select_((lc_v8su)(a.vf < b.vf), a, b); }
SIMD_INLINE lc_v256 _mm256_cvtepi32_ps(lc_v256 a) {
#if defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 9)
    a.vf = __builtin_convertvector(a.vi, lc_v8sf);
#else
    FOR8 a.f[k_] = (float)a.i[k_];
#endif
    return a;
}
SIMD_INLINE void _mm256_storeu_ps(float *p, lc_v256 a) { memcpy(p, a.f, 32); }
SIMD_INLINE void _mm256_stream_ps(float *p, lc_v256 a) { memcpy(p, a.f, 32); }
SIMD_INLINE lc_v256 _mm256_unpacklo_ps(lc_v256 a, lc_v256 b) {
    lc_v256 r;
    for (int l = 0; l < 8; l += 4) {
        r.u[l] = a.u[l], r.u[l + 1] = b.u[l], r.u[l + 2] = a.u[l + 1], r.u[l + 3] = b.u[l + 1];
    }
    return r;
}
SIMD_INLINE lc_v256 _mm256_unpackhi_ps(lc_v256 a, lc_v256 b) {
    lc_v256 r;
    for (int l = 0; l < 8; l += 4) {
        r.u[l] = a.u[l + 2], r.u[l + 1] = b.u[l + 2], r.u[l + 2] = a.u[l + 3], r.u[l + 3] = b.u[l + 3];
    }
    return r;
}
SIMD_INLINE lc_v256 _mm256_unpacklo_pd(lc_v256 a, lc_v256 b) { /* 64-bit moves: no float arithmetic */
    lc_v256 r;
    memcpy(r.ub, a.ub, 8), memcpy(r.ub + 8, b.ub, 8), memcpy(r.ub + 16, a.ub + 16, 8), memcpy(r.ub + 24, b.ub + 16, 8);
    return r;
}
SIMD_INLINE lc_v256 _mm256_unpackhi_pd(lc_v256 a, lc_v256 b) {
    lc_v256 r;
    memcpy(r.ub, a.ub + 8, 8), memcpy(r.ub + 8, b.ub + 8, 8), memcpy(r.ub + 16, a.ub + 24, 8);
    memcpy(r.ub + 24, b.ub + 24, 8);
    return r;
}
SIMD_INLINE lc_v256 _mm256_permute2f128_ps(lc_v256 a, lc_v256 b, int imm) {
    lc_v256 r;
    for (int h = 0; h < 2; h++) {
        int sel = imm >> (4 * h);
        if (sel & 8) memset(r.ub + 16 * h, 0, 16);
        else memcpy(r.ub + 16 * h, ((sel & 2) ? b.ub : a.ub) + 16 * (sel & 1), 16);
    }
    return r;
}

/* ---- 128-bit */
SIMD_INLINE lc_v128 _mm_loadu_si128(const lc_v128 *p) {
    lc_v128 r;
    memcpy(&r, (const void *)p, 16);
    return r;
}
SIMD_INLINE lc_v128 _mm_loadl_epi64(const lc_v128 *p) {
    lc_v128 r;
    memcpy(&r, (const void *)p, 8);
    memset(r.ub + 8, 0, 8);
    return r;
}
SIMD_INLINE void _mm_storeu_si128(lc_v128 *p, lc_v128 a) { memcpy((void *)p, &a, 16); }
SIMD_INLINE lc_v128 _mm_cvtsi32_si128(int32_t a) {
    lc_v128 r;
    memset(&r, 0, sizeof r);
    r.i[0] = a;
    return r;
}
SIMD_INLINE lc_v128 _mm_set1_epi8(char a) {
    lc_v128 r;
    memset(&r, (unsigned char)a, sizeof r);
    return r;
}
SIMD_INLINE lc_v128 _mm_or_si128(lc_v128 a, lc_v128 b) {
    for (int k = 0; k < 4; k++) a.u[k] |= b.u[k];
    return a;
}
SIMD_INLINE lc_v128 _mm_cmpeq_epi8(lc_v128 a, lc_v128 b) {
    for (int k = 0; k < 16; k++) a.ub[k] = a.ub[k] == b.ub[k] ? 0xff : 0;
    return a;
}
SIMD_INLINE int _mm_movemask_epi8(lc_v128 a) {
    int m = 0;
    for (int k = 0; k < 16; k++) m |= (a.ub[k] >> 7) << k;
    return m;
}
SIMD_INLINE lc_v128 _mm_shuffle_epi8(lc_v128 a, lc_v128 b) {
    lc_v128 r;
    for (int k = 0; k < 16; k++) r.ub[k] = (b.ub[k] & 0x80) ? 0 : a.ub[b.ub[k] & 15];
    return r;
}
SIMD_INLINE int16_t lc_sat16_(int32_t x) { return (int16_t)(x < -32768 ? -32768 : x > 32767 ? 32767 : x); }
SIMD_INLINE lc_v128 _mm_packs_epi32(lc_v128 a, lc_v128 b) {
    lc_v128 r;
    for (int k = 0; k < 4; k++) r.h[k] = lc_sat16_(a.i[k]), r.h[4 + k] = lc_sat16_(b.i[k]);
    return r;
}
SIMD_INLINE void _mm_sfence(void) {}

#undef FOR8
#endif /* LC_SCALAR */
#endif
