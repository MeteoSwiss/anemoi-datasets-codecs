/* Exported entry points of the kernels (loaded with ctypes by native.py): forward to the variant chosen at load
 * time. The AVX2 variant is used when the CPU supports AVX2 (x86-64), else the scalar variant; the environment
 * variable ANEMOI_DATASETS_CODECS_ISA (auto | avx2 | scalar) overrides the choice, and lc_select_isa switches at run time
 * (tests, benchmarks). Both variants decode bit-identically.
 *
 * The file is also a trivial CPython extension module, so that the build tools compile and package the library like
 * any extension (stable ABI, abi3); nothing of it is called through the Python C API. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

#include "api.h"

#define EXPORT __attribute__((visibility("default")))

extern const lc_api_t lc_api_scalar;
#ifdef LC_HAVE_AVX2
extern const lc_api_t lc_api_avx2;
#endif

static const lc_api_t *api = &lc_api_scalar;

static int cpu_has_avx2(void) {
#if defined(LC_HAVE_AVX2) && (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

/* "auto" (best supported), "avx2" or "scalar". Returns 0, or -1 when the variant is unknown or not supported by
 * this CPU / build (the current one is kept). Not thread-safe with respect to concurrent decodes: switch between
 * calls. */
EXPORT int lc_select_isa(const char *name) {
    if (!name || !*name || !strcmp(name, "auto")) {
#ifdef LC_HAVE_AVX2
        api = cpu_has_avx2() ? &lc_api_avx2 : &lc_api_scalar;
#else
        api = &lc_api_scalar;
#endif
        return 0;
    }
    if (!strcmp(name, "scalar")) {
        api = &lc_api_scalar;
        return 0;
    }
#ifdef LC_HAVE_AVX2
    if (!strcmp(name, "avx2") && cpu_has_avx2()) {
        api = &lc_api_avx2;
        return 0;
    }
#endif
    return -1;
}

EXPORT const char *lc_isa(void) { return api->name; }

/* comma-separated variants usable on this CPU, best first */
EXPORT const char *lc_available_isas(void) {
#ifdef LC_HAVE_AVX2
    if (cpu_has_avx2()) return "avx2,scalar";
#endif
    return "scalar";
}

__attribute__((constructor)) static void lc_dispatch_init(void) {
    if (lc_select_isa(getenv("ANEMOI_DATASETS_CODECS_ISA")) != 0) lc_select_isa("auto");
}

EXPORT int lc_normalize_freqs(const uint32_t *counts, uint32_t *freqs, uint32_t total) {
    return api->normalize_freqs(counts, freqs, total);
}
EXPORT size_t lc_rans_encode(const uint8_t *syms, const uint8_t *tid, const uint64_t *seg, int nseg,
                             const uint32_t *freqs, int ntab, uint8_t *out, size_t cap, int bits) {
    return api->rans_encode(syms, tid, seg, nseg, freqs, ntab, out, cap, bits);
}
EXPORT int lc_rans_decode_test(const uint8_t *in, size_t nbytes, const uint64_t *seg, int nseg, const uint32_t *freqs,
                               const uint8_t *tid, uint8_t *out, int bits) {
    return api->rans_decode_test(in, nbytes, seg, nseg, freqs, tid, out, bits);
}
EXPORT int lc_decode_field(const uint8_t *payload, size_t n, const void *S, const int32_t *const *dsrc,
                           const float *const *mf, float *out, int32_t *ival, int32_t *dout) {
    return api->decode_field(payload, n, S, dsrc, mf, out, ival, dout);
}
EXPORT void lc_mask_predict(int64_t n, const float *msrc, float thr, double c0, int nf, const float *const *A,
                            const float *const *B, const double *coef, float off, float step, int vt, int32_t *pm) {
    api->mask_predict(n, msrc, thr, c0, nf, A, B, coef, off, step, vt, pm);
}
EXPORT int lc_decode_chunk(const uint8_t *buf, size_t len, const void *S, float *out, int n_threads,
                           const float *const *static_ptr, const int16_t *const *ext_detail, double *times,
                           int32_t *status) {
    return api->decode_chunk(buf, len, S, out, n_threads, static_ptr, ext_detail, times, status);
}
EXPORT void lc_recipe_eval(int id, const double *p, const float *sla, const float *cla, const float *slo,
                           const float *clo, int64_t lo, int64_t hi, float *out) {
    api->recipe_eval(id, p, sla, cla, slo, clo, lo, hi, out);
}
EXPORT int lc_struct_check(const void *S) { return api->struct_check(S); }
EXPORT void lc_irls_scale(int64_t m, int64_t n, const double *y, const double *xw, const double *X, int64_t sr,
                          int64_t sc, double *a, double *b) {
    api->irls_scale(m, n, y, xw, X, sr, sc, a, b);
}
EXPORT void lc_release_buffers(void) { api->release_buffers(); }
EXPORT void lc_profile(int on, double *out) { api->profile(on, out); }

/* ---- CPython module stub (stable ABI) */
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", "lattice codec kernels (loaded with ctypes)",
                                    -1, NULL, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
