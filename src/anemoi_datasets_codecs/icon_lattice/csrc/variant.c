/* One ISA variant of the kernels, compiled once per variant (setup.py): LC_VARIANT=avx2 with -mavx2, and
 * LC_VARIANT=scalar with -DLC_SCALAR (simd.h emulates the intrinsics). Unity build: every kernel function is static
 * (LC_UNITY), so both variants link into one library; the only external symbol is the function table
 * lc_api_<variant> (hidden visibility), used by dispatch.c. */
#define LC_UNITY 1

#include "rans.c"
#include "decode.c"
#include "encode.c"
#include "api.h"

#define LC_CAT_(a, b) a##b
#define LC_CAT(a, b) LC_CAT_(a, b)
#define LC_STR_(a) #a
#define LC_STR(a) LC_STR_(a)

/* The kernels take `const lc_struct_t *`; the table uses `const void *` (same pointer, opaque to dispatch.c). */
static int decode_field_v(const uint8_t *payload, size_t n, const void *S, const int32_t *const *dsrc,
                          const float *const *mf, float *out, int32_t *ival, int32_t *dout) {
    return lc_decode_field(payload, n, (const lc_struct_t *)S, dsrc, mf, out, ival, dout);
}
static int decode_chunk_v(const uint8_t *buf, size_t len, const void *S, float *out, int n_threads,
                          const float *const *static_ptr, const int16_t *const *ext_detail, double *times,
                          int32_t *status) {
    return lc_decode_chunk(buf, len, (const lc_struct_t *)S, out, n_threads, static_ptr, ext_detail, times, status);
}
static int struct_check_v(const void *S) { return lc_struct_check((const lc_struct_t *)S); }

__attribute__((visibility("hidden"))) const lc_api_t LC_CAT(lc_api_, LC_VARIANT) = {
    .name = LC_STR(LC_VARIANT),
    .normalize_freqs = lc_normalize_freqs,
    .rans_encode = lc_rans_encode,
    .rans_decode_test = lc_rans_decode_test,
    .decode_field = decode_field_v,
    .mask_predict = lc_mask_predict,
    .decode_chunk = decode_chunk_v,
    .recipe_eval = lc_recipe_eval,
    .struct_check = struct_check_v,
    .irls_scale = lc_irls_scale,
    .release_buffers = lc_release_buffers,
    .profile = lc_profile,
};
