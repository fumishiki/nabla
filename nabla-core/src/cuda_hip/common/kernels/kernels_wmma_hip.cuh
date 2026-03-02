// kernels_wmma_hip.cuh — WMMA f16->f32 tiled matmul (rocwmma, HIP only, standalone)

#include <rocwmma/rocwmma.hpp>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

extern "C" __global__ void k_matmul_wmma_f16(
    const _Float16* __restrict__ A,
    const _Float16* __restrict__ B,
    float* __restrict__ C,
    unsigned M, unsigned K, unsigned N
) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 64;

    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    auto a_frag = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major>();
    auto b_frag = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major>();
    auto c_frag = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (unsigned k = 0; k < K; k += WMMA_K) {
        unsigned aRow = warpM * WMMA_M;
        unsigned bCol = warpN * WMMA_N;

        if (aRow < M && k + WMMA_K <= K)
            rocwmma::load_matrix_sync(a_frag, A + aRow * K + k, K);
        if (k + WMMA_K <= K && bCol < N)
            rocwmma::load_matrix_sync(b_frag, B + k * N + bCol, N);

        rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    unsigned cRow = warpM * WMMA_M;
    unsigned cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N)
        rocwmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, rocwmma::mem_row_major);
}

