// kernels_cond_set.cuh — Conditional scalar set via device-side pointer (CUDA only, standalone)

#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>

extern "C" __global__ void k_cond_set_f32(
    unsigned long long handle,
    const float* __restrict__ val,
    unsigned cmp,
    float threshold
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = val[0];
        unsigned cond;
        if (cmp == 0u) { cond = (v > 0.0f) ? 1u : 0u; }
        else if (cmp == 1u) { cond = (v == 0.0f) ? 1u : 0u; }
        else { cond = (v < threshold) ? 1u : 0u; }
        cudaGraphSetConditional((cudaGraphConditionalHandle)handle, cond);
    }
}

extern "C" __global__ void k_cond_set_f64(
    unsigned long long handle,
    const double* __restrict__ val,
    unsigned cmp,
    float threshold
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double v = val[0];
        unsigned cond;
        if (cmp == 0u) { cond = (v > 0.0) ? 1u : 0u; }
        else if (cmp == 1u) { cond = (v == 0.0) ? 1u : 0u; }
        else { cond = (v < (double)threshold) ? 1u : 0u; }
        cudaGraphSetConditional((cudaGraphConditionalHandle)handle, cond);
    }
}

extern "C" __global__ void k_cond_set_f16(
    unsigned long long handle,
    const __half* __restrict__ val,
    unsigned cmp,
    float threshold
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = __half2float(val[0]);
        unsigned cond;
        if (cmp == 0u) { cond = (v > 0.0f) ? 1u : 0u; }
        else if (cmp == 1u) { cond = (v == 0.0f) ? 1u : 0u; }
        else { cond = (v < threshold) ? 1u : 0u; }
        cudaGraphSetConditional((cudaGraphConditionalHandle)handle, cond);
    }
}

extern "C" __global__ void k_cond_set_bf16(
    unsigned long long handle,
    const __nv_bfloat16* __restrict__ val,
    unsigned cmp,
    float threshold
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float v = from_bf16(val[0]);
        unsigned cond;
        if (cmp == 0u) { cond = (v > 0.0f) ? 1u : 0u; }
        else if (cmp == 1u) { cond = (v == 0.0f) ? 1u : 0u; }
        else { cond = (v < threshold) ? 1u : 0u; }
        cudaGraphSetConditional((cudaGraphConditionalHandle)handle, cond);
    }
}
