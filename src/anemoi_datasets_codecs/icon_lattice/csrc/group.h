/* Regular-group kernels (lc_group_t), included by decode.c after its helpers (esc_t, cross_t, cross_scalar).
 *
 * 8 groups per iteration. Modes as DEF_GATHER: fused (sym != NULL: values to v, or dequantised floats straight to
 * `out` when `out` != NULL, 32 consecutive floats per 8 groups) or predict (tid != NULL: pred / dp / tid at the
 * target positions; unused by decode.c, which takes group_ctx for contexts). Same integer arithmetic as the
 * generic kernels (results are bit-identical). */

typedef struct {
    float off, step, lo, hi;
    int flags;
} deq_t;

INLINE float deq1(const deq_t *d, int32_t n) {
    float y = d->off + d->step * (float)n;
    if ((d->flags & 1) && y < d->lo) y = d->lo;
    if ((d->flags & 2) && y > d->hi) y = d->hi;
    return y;
}

INLINE __m256 deq8(const deq_t *d, __m256i n) {
    __m256 y = _mm256_add_ps(_mm256_set1_ps(d->off), _mm256_mul_ps(_mm256_set1_ps(d->step), _mm256_cvtepi32_ps(n)));
    if (d->flags & 1) y = _mm256_max_ps(y, _mm256_set1_ps(d->lo));
    if (d->flags & 2) y = _mm256_min_ps(y, _mm256_set1_ps(d->hi));
    return y;
}

/* 8 groups x (c0, c1, P, c3) -> 32 consecutive floats */
INLINE void store_groups8(float *o, __m256 f0, __m256 f1, __m256 f2, __m256 f3) {
    __m256 t0 = _mm256_unpacklo_ps(f0, f1), t1 = _mm256_unpackhi_ps(f0, f1);
    __m256 t2 = _mm256_unpacklo_ps(f2, f3), t3 = _mm256_unpackhi_ps(f2, f3);
    __m256 u0 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
    __m256 u1 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t0), _mm256_castps_pd(t2)));
    __m256 u2 = _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
    __m256 u3 = _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(t1), _mm256_castps_pd(t3)));
    __m256 a = _mm256_permute2f128_ps(u0, u1, 0x20), b = _mm256_permute2f128_ps(u2, u3, 0x20);
    __m256 c = _mm256_permute2f128_ps(u0, u1, 0x31), d = _mm256_permute2f128_ps(u2, u3, 0x31);
    if (((uintptr_t)o & 31) == 0) {
        _mm256_stream_ps(o, a);
        _mm256_stream_ps(o + 8, b);
        _mm256_stream_ps(o + 16, c);
        _mm256_stream_ps(o + 24, d);
    } else {
        _mm256_storeu_ps(o, a);
        _mm256_storeu_ps(o + 8, b);
        _mm256_storeu_ps(o + 16, c);
        _mm256_storeu_ps(o + 24, d);
    }
}

static const int8_t SYM3_A[3][16] = {{0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                     {1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
                                     {2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
static const int8_t SYM3_B[3][16] = {{-1, -1, -1, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1},
                                     {-1, -1, -1, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1},
                                     {-1, -1, -1, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1}};

/* residuals of corner j of 8 groups from 24 stride-3 symbol bytes */
INLINE __m256i sym3(const uint8_t *s, int j) {
    __m128i a = _mm_loadu_si128((const __m128i *)s), b = _mm_loadu_si128((const __m128i *)(s + 8));
    __m128i x = _mm_or_si128(_mm_shuffle_epi8(a, _mm_loadu_si128((const __m128i *)SYM3_A[j])),
                             _mm_shuffle_epi8(b, _mm_loadu_si128((const __m128i *)SYM3_B[j])));
    __m256i r = _mm256_cvtepu8_epi32(x);
    return _mm256_xor_si256(_mm256_srli_epi32(r, 1),
                            _mm256_sub_epi32(_mm256_setzero_si256(), _mm256_and_si256(r, _mm256_set1_epi32(1))));
}

INLINE int has_escape24(const uint8_t *s) {
    __m128i e = _mm_set1_epi8((char)-1);
    return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)s), e)) |
           (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)(s + 16)), e)) & 0xff);
}

/* corner j of 8 groups from a stride-3 array; cell (B) + 4 k + (O) of 8 groups */
#define LDS(V, I, J)                                                                                                 \
    _mm256_setr_epi32(V[(I)[J]], V[(I)[3 + J]], V[(I)[6 + J]], V[(I)[9 + J]], V[(I)[12 + J]], V[(I)[15 + J]],        \
                      V[(I)[18 + J]], V[(I)[21 + J]])
#define LDV(P, J) _mm256_setr_epi32((P)[J], (P)[3 + J], (P)[6 + J], (P)[9 + J], (P)[12 + J], (P)[15 + J], (P)[18 + J], (P)[21 + J])
#define LDC(V, B, O)                                                                                                 \
    _mm256_setr_epi32(V[(B) + (O)], V[(B) + 4 + (O)], V[(B) + 8 + (O)], V[(B) + 12 + (O)], V[(B) + 16 + (O)],         \
                      V[(B) + 20 + (O)], V[(B) + 24 + (O)], V[(B) + 28 + (O)])


/* corner j's two neighbour values from the group's 3 neighbours and its pattern byte (per lane) */
INLINE void corner_pair(__m256i n0, __m256i n1, __m256i n2, __m256i pat, int j, __m256i *a, __m256i *b) {
    if (j == 0) {
        *a = n0;
        *b = n1;
        return;
    }
    const __m256i three = _mm256_set1_epi32(3), one = _mm256_set1_epi32(1), two = _mm256_set1_epi32(2);
    __m256i ca = _mm256_and_si256(_mm256_srli_epi32(pat, 4 * j - 4), three);
    __m256i cb = _mm256_and_si256(_mm256_srli_epi32(pat, 4 * j - 2), three);
    *a = _mm256_blendv_epi8(_mm256_blendv_epi8(n0, n1, _mm256_cmpeq_epi32(ca, one)), n2, _mm256_cmpeq_epi32(ca, two));
    *b = _mm256_blendv_epi8(_mm256_blendv_epi8(n0, n1, _mm256_cmpeq_epi32(cb, one)), n2, _mm256_cmpeq_epi32(cb, two));
}

static const int SLOT[3] = {0, 1, 3};

#define DEF_GROUP(VT)                                                                                                 \
    static int group_##VT(VT *v, const lc_step_t *st, const lc_group_t *G, const int32_t *w, const cross_t *x,       \
                          const uint8_t *sym, esc_t *esc, int16_t *dout, float *out, const deq_t *dq, int n_ctx,      \
                          const int32_t *thr, const uint8_t *tmap, int32_t *pred, int32_t *dp, uint8_t *tid,          \
                          int64_t qlo, int64_t qhi) {                                                                 \
        const int64_t n = st->n, i0 = G->i0, c0 = G->c0, p0 = st->p0;                                                  \
        const int32_t *s1 = st->srcp + n + i0, *s2 = st->srcp + 2 * n + i0;                                            \
        const int predict = tid != NULL;                                                                              \
        const __m256i w1 = _mm256_set1_epi32(w[1]), w2 = _mm256_set1_epi32(w[2]), r128 = _mm256_set1_epi32(128);      \
        __m256i vthr[MAX_CTX - 1], vcc[MAX_CROSS];                                                                    \
        for (int b = 0; b < n_ctx - 1; b++) vthr[b] = _mm256_set1_epi32(thr[b]);                                      \
        for (int c = 0; c < x->nc; c++) vcc[c] = _mm256_set1_epi32(x->c[c]);                                          \
        int64_t q = qlo;                                                                                                \
        for (; q + 8 <= qhi; q += 8) {                                                                                 \
            const int64_t cb = c0 + 4 * q;                                                                            \
            const __m256i N0 = LD8(v, G->nb0 + q), N1 = LD8(v, G->nb1 + q), N2 = LD8(v, G->nb2 + q);                 \
            const __m256i PT = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(G->pat + q)));                  \
            __m256i P = LD8(v, G->gpos + q), psp[3], pr[3];                                                                 \
            int32_t bb[3][8];                                                                                         \
            for (int j = 0; j < 3; j++) {                                                                             \
                __m256i A, B;                                                                                         \
                corner_pair(N0, N1, N2, PT, j, &A, &B);                                                               \
                __m256i acc = _mm256_add_epi32(r128, _mm256_mullo_epi32(w1, _mm256_sub_epi32(A, P)));                 \
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w2, _mm256_sub_epi32(B, P)));                          \
                psp[j] = pr[j] = _mm256_add_epi32(P, _mm256_srai_epi32(acc, 8));                                      \
                for (int c = 0; c < x->nc; c++) {                                                                     \
                    __m256i m = _mm256_mullo_epi32(vcc[c], LDV(x->d[c] + p0 + 3 * q, j));                                \
                    if (x->shs[c]) m = _mm256_add_epi32(m, _mm256_set1_epi32(1 << (x->shs[c] - 1)));                  \
                    pr[j] = _mm256_add_epi32(pr[j], _mm256_sra_epi32(m, x->sh[c]));                                   \
                }                                                                                                     \
                if (predict) {                                                                                        \
                    __m256i sp = _mm256_sub_epi32(_mm256_max_epi32(P, _mm256_max_epi32(A, B)),                         \
                                                  _mm256_min_epi32(P, _mm256_min_epi32(A, B)));                        \
                    __m256i bk = _mm256_setzero_si256();                                                              \
                    for (int b = 0; b < n_ctx - 1; b++) bk = _mm256_sub_epi32(bk, _mm256_cmpgt_epi32(sp, vthr[b]));   \
                    _mm256_storeu_si256((__m256i *)bb[j], bk);                                                        \
                }                                                                                                     \
            }                                                                                                         \
            if (predict) {                                                                                            \
                int32_t p3[3][8], d3[3][8];                                                                           \
                for (int j = 0; j < 3; j++) {                                                                         \
                    _mm256_storeu_si256((__m256i *)p3[j], pr[j]);                                                     \
                    _mm256_storeu_si256((__m256i *)d3[j], _mm256_sub_epi32(pr[j], psp[j]));                           \
                }                                                                                                     \
                for (int k = 0; k < 8; k++)                                                                           \
                    for (int j = 0; j < 3; j++) {                                                                     \
                        int64_t i = i0 + 3 * (q + k) + j;                                                             \
                        pred[i] = p3[j][k];                                                                           \
                        tid[i] = tmap[bb[j][k]];                                                                      \
                        if (dp) dp[i] = d3[j][k];                                                                     \
                    }                                                                                                 \
                continue;                                                                                             \
            }                                                                                                         \
            const uint8_t *sy = sym + 3 * q;                                                                          \
            __m256i val[3];                                                                                           \
            for (int j = 0; j < 3; j++) val[j] = _mm256_add_epi32(pr[j], _mm256_sll_epi32(sym3(sy, j), esc->shv));                                \
            if (__builtin_expect(has_escape24(sy), 0)) {                                                              \
                int32_t vv[3][8], pp[3][8];                                                                           \
                for (int j = 0; j < 3; j++) {                                                                         \
                    _mm256_storeu_si256((__m256i *)vv[j], val[j]);                                                    \
                    _mm256_storeu_si256((__m256i *)pp[j], pr[j]);                                                     \
                }                                                                                                     \
                for (int k = 0; k < 8; k++)                                                                           \
                    for (int j = 0; j < 3; j++)                                                                       \
                        if (sy[3 * k + j] == ESCAPE) {                                                                \
                            int32_t r;                                                                                \
                            if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                              \
                            vv[j][k] = wadd32(pp[j][k], r);                                                              \
                        }                                                                                             \
                for (int j = 0; j < 3; j++) val[j] = _mm256_loadu_si256((const __m256i *)vv[j]);                      \
            }                                                                                                         \
            if (dout)                                                                                                 \
                for (int j = 0; j < 3; j++) {                                                                         \
                    int32_t dd[8];                                                                                    \
                    _mm256_storeu_si256((__m256i *)dd, _mm256_sub_epi32(val[j], psp[j]));                             \
                    for (int k = 0; k < 8; k++) dout[p0 + 3 * (q + k) + j] = sat16(dd[k]);                            \
                }                                                                                                     \
            if (out) {                                                                                                \
                store_groups8(out + cb, deq8(dq, val[0]), deq8(dq, val[1]), deq8(dq, P), deq8(dq, val[2]));            \
            } else {                                                                                                  \
                int32_t vv[3][8];                                                                                     \
                for (int j = 0; j < 3; j++) _mm256_storeu_si256((__m256i *)vv[j], val[j]);                            \
                for (int k = 0; k < 8; k++) {                                                                         \
                    v[p0 + 3 * (q + k)] = (VT)vv[0][k];                                                                     \
                    v[p0 + 3 * (q + k) + 1] = (VT)vv[1][k];                                                                 \
                    v[p0 + 3 * (q + k) + 2] = (VT)vv[2][k];                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        for (; q < qhi; q++) { /* scalar tail, same arithmetic */                                                      \
            const int64_t cb = c0 + 4 * q;                                                                            \
            int32_t P = v[G->gpos[q]], vals[3] = {0, 0, 0};                                                               \
            for (int j = 0; j < 3; j++) {                                                                             \
                int64_t i = i0 + 3 * q + j;                                                                           \
                int32_t A = v[st->srcp[n + i]], B = v[st->srcp[2 * n + i]];                                             \
                int64_t acc = 128 + (int64_t)w[1] * (A - P) + (int64_t)w[2] * (B - P);                                \
                int32_t psp = P + (int32_t)(acc >> 8), pr = psp + cross_scalar(x, (int32_t)(p0 + 3 * q + j));           \
                if (predict) {                                                                                        \
                    int32_t hi = P > A ? P : A, lo = P < A ? P : A;                                                   \
                    hi = hi > B ? hi : B;                                                                             \
                    lo = lo < B ? lo : B;                                                                             \
                    int b = 0;                                                                                        \
                    for (int qq = 0; qq < n_ctx - 1; qq++) b += (hi - lo) > thr[qq];                                  \
                    pred[i] = pr;                                                                                     \
                    tid[i] = tmap[b];                                                                                 \
                    if (dp) dp[i] = pr - psp;                                                                         \
                    continue;                                                                                         \
                }                                                                                                     \
                int32_t r;                                                                                            \
                uint8_t sb = sym[3 * q + j];                                                                          \
                if (sb == ESCAPE) {                                                                                   \
                    if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                      \
                } else                                                                                                \
                    r = zigzag_dec(sb) * esc->mul;                                                                               \
                vals[j] = wadd32(pr, r);                                                                                 \
                if (dout) dout[p0 + 3 * q + j] = sat16(wsub32(vals[j], psp));                                               \
            }                                                                                                         \
            if (predict) continue;                                                                                    \
            if (out) {                                                                                                \
                out[cb] = deq1(dq, vals[0]);                                                                          \
                out[cb + 1] = deq1(dq, vals[1]);                                                                      \
                out[cb + 2] = deq1(dq, P);                                                                            \
                out[cb + 3] = deq1(dq, vals[2]);                                                                      \
            } else {                                                                                                  \
                v[p0 + 3 * q] = (VT)vals[0];                                                                                  \
                v[p0 + 3 * q + 1] = (VT)vals[1];                                                                              \
                v[p0 + 3 * q + 2] = (VT)vals[2];                                                                              \
            }                                                                                                         \
        }                                                                                                             \
        return LC_OK;                                                                                                 \
    }

DEF_GROUP(int16_t)
DEF_GROUP(int32_t)

/* Context path over regular groups, fused per block of QB groups (all buffers in L1): predictions corner-major,
 * table ids interleaved into symbol order, rANS for the block, then values / floats. */
#define GQB 64
#define DEF_GROUP_CTX(VT)                                                                                             \
    static int group_ctx_##VT(VT *v, const lc_step_t *st, const lc_group_t *G, const int32_t *w, const cross_t *x,   \
                              int n_ctx, const int32_t *thr, const uint8_t *tmap, rstate_t *rs, const uint32_t *tabs, \
                              esc_t *esc, int16_t *dout, float *out, const deq_t *dq, const int32_t *pm, int mtab,  \
                              int64_t qlo, int64_t qhi, int pbits) {                                                   \
        const int64_t n = st->n, c0 = G->c0, ng = G->ng, p0 = st->p0;                                                              \
        const int32_t *s1 = st->srcp + n, *s2 = st->srcp + 2 * n;                                                       \
        const __m256i w1 = _mm256_set1_epi32(w[1]), w2 = _mm256_set1_epi32(w[2]), r128 = _mm256_set1_epi32(128);      \
        __m256i vthr[MAX_CTX - 1], vcc[MAX_CROSS];                                                                    \
        int32_t tm[8] = {0};                                                                                          \
        for (int b = 0; b < n_ctx; b++) tm[b] = tmap[b];                                                              \
        const __m256i vtm = _mm256_loadu_si256((const __m256i *)tm);                                                  \
        for (int b = 0; b < n_ctx - 1; b++) vthr[b] = _mm256_set1_epi32(thr[b]);                                      \
        for (int c = 0; c < x->nc; c++) vcc[c] = _mm256_set1_epi32(x->c[c]);                                          \
        __m256i PR[GQB / 8][3], DP[GQB / 8][3];                                                                       \
        uint8_t tid[3 * GQB + 32] __attribute__((aligned(32))), sym[3 * GQB + 32] __attribute__((aligned(32)));       \
        (void)ng;                                                                                                     \
        for (int64_t q0 = qlo; q0 < qhi; q0 += GQB) {                                                                  \
            const int64_t q1 = q0 + GQB < qhi ? q0 + GQB : qhi, full = (q1 - q0) / 8 * 8;                             \
            for (int64_t b = 0; b < full; b += 8) {                                                                   \
                const int64_t q = q0 + b, cb = c0 + 4 * q;                                                            \
                const __m256i N0 = LD8(v, G->nb0 + q), N1 = LD8(v, G->nb1 + q), N2 = LD8(v, G->nb2 + q);             \
                const __m256i PT = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(G->pat + q)));              \
                __m256i P = LD8(v, G->gpos + q);                                                                            \
                int32_t tj[3][8];                                                                                     \
                for (int j = 0; j < 3; j++) {                                                                         \
                    __m256i A, B;                                                                                     \
                    corner_pair(N0, N1, N2, PT, j, &A, &B);                                                           \
                    __m256i acc = _mm256_add_epi32(r128, _mm256_mullo_epi32(w1, _mm256_sub_epi32(A, P)));             \
                    acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(w2, _mm256_sub_epi32(B, P)));                      \
                    __m256i psp = _mm256_add_epi32(P, _mm256_srai_epi32(acc, 8)), pr = psp;                           \
                    for (int c = 0; c < x->nc; c++) {                                                                 \
                        __m256i m = _mm256_mullo_epi32(vcc[c], LDV(x->d[c] + p0 + 3 * q, j));                            \
                        if (x->shs[c]) m = _mm256_add_epi32(m, _mm256_set1_epi32(1 << (x->shs[c] - 1)));              \
                        pr = _mm256_add_epi32(pr, _mm256_sra_epi32(m, x->sh[c]));                                     \
                    }                                                                                                 \
                    __m256i keep = _mm256_set1_epi32(-1);                                                             \
                    if (pm) {                                                                                         \
                        __m256i mv = LDC(pm, cb, SLOT[j]);                                                            \
                        keep = _mm256_cmpeq_epi32(mv, _mm256_set1_epi32(MASK_SENTINEL));                              \
                        pr = _mm256_blendv_epi8(mv, pr, keep);                                                        \
                    }                                                                                                 \
                    PR[b / 8][j] = pr;                                                                                \
                    DP[b / 8][j] = _mm256_sub_epi32(pr, psp);                                                         \
                    __m256i sp = _mm256_sub_epi32(_mm256_max_epi32(P, _mm256_max_epi32(A, B)),                         \
                                                  _mm256_min_epi32(P, _mm256_min_epi32(A, B)));                        \
                    __m256i bk = _mm256_setzero_si256();                                                              \
                    for (int qq = 0; qq < n_ctx - 1; qq++) bk = _mm256_sub_epi32(bk, _mm256_cmpgt_epi32(sp, vthr[qq])); \
                    _mm256_storeu_si256((__m256i *)tj[j], _mm256_blendv_epi8(_mm256_set1_epi32(mtab),               \
                                                                             _mm256_permutevar8x32_epi32(vtm, bk), keep)); \
                }                                                                                                     \
                uint8_t *tb = tid + 3 * b;                                                                            \
                for (int k = 0; k < 8; k++) {                                                                         \
                    tb[3 * k] = (uint8_t)tj[0][k];                                                                    \
                    tb[3 * k + 1] = (uint8_t)tj[1][k];                                                                \
                    tb[3 * k + 2] = (uint8_t)tj[2][k];                                                                \
                }                                                                                                     \
            }                                                                                                         \
            int32_t tpr[8][3], tdp[8][3];                                                                             \
            for (int64_t q = q0 + full; q < q1; q++) { /* scalar remainder (< 8 groups, last block only) */           \
                const int64_t cb = c0 + 4 * q, l = q - q0 - full;                                                     \
                int32_t P = v[G->gpos[q]];                                                                                \
                for (int j = 0; j < 3; j++) {                                                                         \
                    int64_t i = 3 * q + j;                                                                            \
                    int32_t A = v[s1[i]], B = v[s2[i]];                                                               \
                    int64_t acc = 128 + (int64_t)w[1] * (A - P) + (int64_t)w[2] * (B - P);                            \
                    int32_t psp = P + (int32_t)(acc >> 8), pr = psp + cross_scalar(x, (int32_t)(p0 + 3 * q + j));       \
                    int32_t hi = P > A ? P : A, lo = P < A ? P : A;                                                   \
                    hi = hi > B ? hi : B;                                                                             \
                    lo = lo < B ? lo : B;                                                                             \
                    int bk = 0;                                                                                       \
                    for (int qq = 0; qq < n_ctx - 1; qq++) bk += (hi - lo) > thr[qq];                                 \
                    tid[3 * (q - q0) + j] = tmap[bk];                                                                 \
                    if (pm && pm[cb + SLOT[j]] != MASK_SENTINEL) {                                                    \
                        pr = pm[cb + SLOT[j]];                                                                        \
                        tid[3 * (q - q0) + j] = (uint8_t)mtab;                                                        \
                    }                                                                                                 \
                    tpr[l][j] = pr;                                                                                   \
                    tdp[l][j] = pr - psp;                                                                             \
                }                                                                                                     \
            }                                                                                                         \
            double tr0 = prof_on ? now() : 0;                                                                        \
            int rc = rans_segment_tid(rs, (size_t)(3 * (q1 - q0)), tabs, tid, sym, pbits);                            \
            if (prof_on) prof_rans_inner += now() - tr0;                                                              \
            if (rc) return rc;                                                                                        \
            for (int64_t b = 0; b < full; b += 8) {                                                                   \
                const int64_t q = q0 + b, cb = c0 + 4 * q;                                                            \
                const uint8_t *sy = sym + 3 * b;                                                                      \
                __m256i val[3], rr[3];                                                                                \
                for (int j = 0; j < 3; j++) {                                                                         \
                    rr[j] = _mm256_sll_epi32(sym3(sy, j), esc->shv);                                                                              \
                    val[j] = _mm256_add_epi32(PR[b / 8][j], rr[j]);                                                   \
                }                                                                                                     \
                if (__builtin_expect(has_escape24(sy), 0)) {                                                          \
                    int32_t vv[3][8], r3[3][8], pp[3][8];                                                             \
                    for (int j = 0; j < 3; j++) {                                                                     \
                        _mm256_storeu_si256((__m256i *)vv[j], val[j]);                                                \
                        _mm256_storeu_si256((__m256i *)r3[j], rr[j]);                                                 \
                        _mm256_storeu_si256((__m256i *)pp[j], PR[b / 8][j]);                                          \
                    }                                                                                                 \
                    for (int k = 0; k < 8; k++)                                                                       \
                        for (int j = 0; j < 3; j++)                                                                   \
                            if (sy[3 * k + j] == ESCAPE) {                                                            \
                                int32_t r;                                                                            \
                                if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                          \
                                vv[j][k] = wadd32(pp[j][k], r);                                                          \
                                r3[j][k] = r;                                                                         \
                            }                                                                                         \
                    for (int j = 0; j < 3; j++) {                                                                     \
                        val[j] = _mm256_loadu_si256((const __m256i *)vv[j]);                                          \
                        rr[j] = _mm256_loadu_si256((const __m256i *)r3[j]);                                           \
                    }                                                                                                 \
                }                                                                                                     \
                if (dout)                                                                                             \
                    for (int j = 0; j < 3; j++) {                                                                     \
                        int32_t dd[8];                                                                                \
                        _mm256_storeu_si256((__m256i *)dd, _mm256_add_epi32(rr[j], DP[b / 8][j]));                    \
                        for (int k = 0; k < 8; k++) dout[p0 + 3 * (q + k) + j] = sat16(dd[k]);                        \
                    }                                                                                                 \
                if (out) {                                                                                            \
                    store_groups8(out + cb, deq8(dq, val[0]), deq8(dq, val[1]), deq8(dq, LD8(v, G->gpos + q)),              \
                                  deq8(dq, val[2]));                                                                  \
                } else {                                                                                              \
                    int32_t vv[3][8];                                                                                 \
                    for (int j = 0; j < 3; j++) _mm256_storeu_si256((__m256i *)vv[j], val[j]);                        \
                    for (int k = 0; k < 8; k++) {                                                                     \
                        v[p0 + 3 * (q + k)] = (VT)vv[0][k];                                                                 \
                        v[p0 + 3 * (q + k) + 1] = (VT)vv[1][k];                                                             \
                        v[p0 + 3 * (q + k) + 2] = (VT)vv[2][k];                                                             \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
            for (int64_t q = q0 + full; q < q1; q++) {                                                                \
                const int64_t cb = c0 + 4 * q, l = q - q0 - full;                                                     \
                int32_t vals[3];                                                                                      \
                for (int j = 0; j < 3; j++) {                                                                         \
                    uint8_t sb = sym[3 * (q - q0) + j];                                                               \
                    int32_t r;                                                                                        \
                    if (sb == ESCAPE) {                                                                               \
                        if (esc_next(esc, &r)) return LC_ERR_ESCAPE;                                                  \
                    } else                                                                                            \
                        r = zigzag_dec(sb) * esc->mul;                                                                           \
                    vals[j] = wadd32(tpr[l][j], r);                                                                      \
                    if (dout) dout[p0 + 3 * q + j] = sat16(wadd32(r, tdp[l][j]));                                           \
                }                                                                                                     \
                if (out) {                                                                                            \
                    out[cb] = deq1(dq, vals[0]);                                                                      \
                    out[cb + 1] = deq1(dq, vals[1]);                                                                  \
                    out[cb + 2] = deq1(dq, v[G->gpos[q]]);                                                                \
                    out[cb + 3] = deq1(dq, vals[2]);                                                                  \
                } else {                                                                                              \
                    v[p0 + 3 * q] = (VT)vals[0];                                                                              \
                    v[p0 + 3 * q + 1] = (VT)vals[1];                                                                          \
                    v[p0 + 3 * q + 2] = (VT)vals[2];                                                                          \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
        return LC_OK;                                                                                                 \
    }

DEF_GROUP_CTX(int16_t)
DEF_GROUP_CTX(int32_t)
