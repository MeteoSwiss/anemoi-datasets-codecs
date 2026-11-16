/* Function table of one ISA variant of the kernels (variant.c) and the exported entry points (dispatch.c). */
#ifndef LATTICE_CODEC_API_H
#define LATTICE_CODEC_API_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const char *name;
    int (*normalize_freqs)(const uint32_t *counts, uint32_t *freqs, uint32_t total);
    size_t (*rans_encode)(const uint8_t *syms, const uint8_t *tid, const uint64_t *seg, int nseg,
                          const uint32_t *freqs, int ntab, uint8_t *out, size_t cap, int bits);
    int (*rans_decode_test)(const uint8_t *in, size_t nbytes, const uint64_t *seg, int nseg, const uint32_t *freqs,
                            const uint8_t *tid, uint8_t *out, int bits);
    int (*decode_field)(const uint8_t *payload, size_t n, const void *S, const int32_t *const *dsrc,
                        const float *const *mf, float *out, int32_t *ival, int32_t *dout);
    void (*mask_predict)(int64_t n, const float *msrc, float thr, double c0, int nf, const float *const *A,
                         const float *const *B, const double *coef, float off, float step, int vt, int32_t *pm);
    int (*decode_chunk)(const uint8_t *buf, size_t len, const void *S, float *out, int n_threads,
                        const float *const *static_ptr, const int16_t *const *ext_detail, double *times,
                        int32_t *status);
    void (*recipe_eval)(int id, const double *p, const float *sla, const float *cla, const float *slo,
                        const float *clo, int64_t lo, int64_t hi, float *out);
    int (*struct_check)(const void *S);
    void (*irls_scale)(int64_t m, int64_t n, const double *y, const double *xw, const double *X, int64_t sr,
                       int64_t sc, double *a, double *b);
    void (*release_buffers)(void);
    void (*profile)(int on, double *out);
} lc_api_t;

#endif
