#ifndef HYDRA_INTERNAL_NEURAL_KERNELS_V1_H
#define HYDRA_INTERNAL_NEURAL_KERNELS_V1_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define HK_ABI_VERSION 1
#define HK_INPUTS 3u
#define HK_MLP_HIDDEN 16u
#define HK_MLP_PARAMETERS 81u
#define HK_MLP_OUTPUTS 18u
#define HK_GRU_HIDDEN 32u
#define HK_GRU_PARAMETERS 3489u
#define HK_GRU_OUTPUTS 34u

enum HkStatus {
    HK_OK=0, HK_BAD_ARGUMENT=1, HK_NONFINITE=2, HK_BAD_SCALE=3,
    HK_NEGATIVE_RAW=4, HK_OVERLAP=5, HK_ARITHMETIC=6
};

/* Exact lengths are mandatory. Out=[head,factor,hidden16/candidate_h32].
 * Out may not overlap any input range. Inputs may overlap each other.
 * Every nonzero return leaves the whole output range unchanged.
 * Caller owns valid aligned storage and synchronization for all arrays.
 */
int hk_mlp16(const double *raw,size_t raw_n,
             const double *mean,size_t mean_n,
             const double *scale,size_t scale_n,
             const double *parameters,size_t parameters_n,
             double *out,size_t out_n);

int hk_gru32(const double *raw,size_t raw_n,
             const double *mean,size_t mean_n,
             const double *scale,size_t scale_n,
             const double *parameters,size_t parameters_n,
             const double *h,size_t h_n,
             double *out,size_t out_n);

#ifdef __cplusplus
}
#endif
#endif
