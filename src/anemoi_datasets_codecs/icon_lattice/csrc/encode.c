/* Encoder-side kernels (the decoder never calls them).
 *
 * lc_irls_scale: one reweighting of the encoder's IRLS fit (encoder.fit_robust) in one pass. For every row i
 *   s_i = 1 / sqrt(max(|y_i - xw_i|, 0.5)),  a[i + j m] = X[i, j] s_i (column-major),  b_i = y_i s_i
 * with xw = X @ w computed by the caller (numpy). Every operation is one IEEE double operation (-ffp-contract=off: no
 * fused multiply-add; sqrt and division are correctly rounded in both variants), in the order of the numpy expressions
 * it replaces, so the system handed to the least-squares solver is bit-identical; max() keeps a NaN like np.maximum.
 * X is addressed with element strides (sr, sc), so C- and Fortran-ordered matrices are read in place. */
#include <math.h>
#include <stdint.h>

#include "lc.h"
#include "simd.h"

#define IRLS_BLOCK 512

EXPORT void lc_irls_scale(int64_t m, int64_t n, const double *y, const double *xw, const double *X, int64_t sr,
                          int64_t sc, double *a, double *b) {
    double s[IRLS_BLOCK];
    for (int64_t i0 = 0; i0 < m; i0 += IRLS_BLOCK) {
        int64_t nb = m - i0 < IRLS_BLOCK ? m - i0 : IRLS_BLOCK, i = 0;
        const double *yb = y + i0, *xwb = xw + i0;
#ifndef LC_SCALAR
        const __m256d sign = _mm256_set1_pd(-0.0), half = _mm256_set1_pd(0.5), one = _mm256_set1_pd(1.0);
        for (; i + 4 <= nb; i += 4) {
            __m256d r = _mm256_andnot_pd(sign, _mm256_sub_pd(_mm256_loadu_pd(yb + i), _mm256_loadu_pd(xwb + i)));
            r = _mm256_blendv_pd(r, half, _mm256_cmp_pd(r, half, _CMP_LT_OQ)); /* r < 0.5 ? 0.5 : r (NaN stays) */
            _mm256_storeu_pd(s + i, _mm256_div_pd(one, _mm256_sqrt_pd(r)));
        }
#endif
        for (; i < nb; i++) {
            double r = fabs(yb[i] - xwb[i]);
            r = r < 0.5 ? 0.5 : r;
            s[i] = 1.0 / sqrt(r);
        }
        for (i = 0; i < nb; i++) b[i0 + i] = yb[i] * s[i];
        for (int64_t j = 0; j < n; j++) {
            const double *x = X + i0 * sr + j * sc;
            double *aj = a + i0 + j * m;
            if (sr == 1)
                for (i = 0; i < nb; i++) aj[i] = x[i] * s[i];
            else
                for (i = 0; i < nb; i++) aj[i] = x[i * sr] * s[i];
        }
    }
}
