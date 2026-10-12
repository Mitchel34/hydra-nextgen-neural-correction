#ifndef HYDRA_CFE_NEURAL_BRIDGE_V1_H
#define HYDRA_CFE_NEURAL_BRIDGE_V1_H
#include "bmi.h"
#include <stddef.h>
#include <stdint.h>

#define HN_ABI_VERSION 1u
#define HN_ARTIFACT_HEADER_BYTES 300u
#define HN_MLP 1u
#define HN_GRU 2u
#define HN_CATCHMENT_CAP 64
#define HN_CONFIG_SHA_CAP 65
#define HN_RAW_DEMAND "HYDRA_FEATURE_RAW_DEMAND"
#define HN_RAW_LIQUID "HYDRA_FEATURE_RAW_LIQUID"
#define HN_SOIL_PRE "HYDRA_FEATURE_SOIL_PRE_UPDATE"
#define HN_INTERVAL_START "HYDRA_FEATURE_INTERVAL_START"
#define HN_FEATURE_ID "b8bfd1126c11650800a865e80cbb1390f571c9a354b80718a9840fdcc2899e08"
#define HN_MLP_LAYOUT_ID "1afb6b21ae5fda09fee1d0ddc328c256adec32aef83ac7cbc4b2267057427a70"
#define HN_GRU_LAYOUT_ID "45230f3bb7114b41757780dffe360424e5687fd3ebfc749e16a6875c7bad969e"

typedef struct HnFrame {
    uint32_t abi_version, valid, architecture, hidden_count;
    uint64_t frame_index;
    int64_t origin_epoch;
    double interval_start_seconds;
    int64_t interval_start_epoch;
    double raw_demand_m_s, raw_liquid_mm_h, soil_before_m;
    double head, factor, applied_demand_m_s;
    double h_pre[32], h_post[32];
    char catchment_id[64], config_sha256[65], model_file_sha256[65];
    char feature_id[65], layout_id[65], model_id[65], scaler_id[65];
} HnFrame;

typedef struct HnStatus {
    uint32_t abi_version, initialized, poisoned, frame_valid;
    uint32_t ready_mask, architecture, hidden_count, reserved;
    uint64_t instance_sequence, successful_updates, capture_attempts, inference_calls;
    uint64_t native_update_calls, feature_input_sets, demand_override_calls;
    uint64_t frame_reads, pointer_exports;
    int64_t created_pid, owner_pid, parent_pid;
    uint64_t fork_generation;
    double expected_time_seconds;
    int32_t last_kernel_status;
    uint32_t reserved2;
    double committed_h[32];
} HnStatus;

Bmi *register_bmi_hydra_neural(Bmi *self);
/* Copies forensic status/last committed frame but returns FAILURE before a
 * first committed frame or after poison. Initial GRU zeros are explicit in
 * status only when initialized==1 and hidden_count==32. MLP has hidden_count0;
 * its unused fixed-size h slots are zero padding, never recurrent state. */
int hydra_neural_snapshot(Bmi *self,HnFrame *frame,HnStatus *status);
size_t hydra_neural_frame_size(void);
size_t hydra_neural_status_size(void);

/* The following source-test helper declarations have no production bodies. */
void *nn_fixture_create(const char *native_config);
void *nn_fixture_create_bridge(const char *wrapper_config);
int nn_fixture_close(void *model);
int nn_fixture_set(void *model,const char *name,double value);
int nn_fixture_set_index(void *model,const char *name,int index,double value);
int nn_fixture_update(void *model);
int nn_fixture_update_until(void *model,double target);
int nn_fixture_get(void *model,const char *name,double *value);
int nn_fixture_get_index(void *model,const char *name,int index,double *value);
int nn_fixture_get_ptr(void *model,const char *name,double **value);
int nn_fixture_snapshot(void *model,HnFrame *frame,HnStatus *status);
double nn_fixture_time(void *model);
double nn_fixture_total_storage(void *model);
double nn_fixture_min_storage(void *model);
int nn_fixture_output_count(void *model,int *count);
int nn_fixture_output_name(void *model,int index,char *name,size_t capacity);
int nn_fixture_metadata(void *model,const char *name,char *type,char *units,int *itemsize,int *nbytes);
int nn_fixture_inject(void *model,int mode);
int nn_fixture_native_counts(uint64_t *out,size_t n);
#endif
